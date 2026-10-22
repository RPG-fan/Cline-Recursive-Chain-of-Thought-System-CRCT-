"""
One workflow for the explicit + structural part of dependency suggestion, for every
language. A small adapter per language supplies the language-specific steps (how
imports, links or references resolve to tracked files, and which uses are verified
against the target's symbols); everything else is shared:

    upfront (prepare_structural_index, once per run, before the parallel workers)
      - JS/TS import resolution + export index (passes 1-2, dependency_suggester)
      - SQL table index (who defines / references which table)
      - every file's (edges, AST links) through its adapter
    per file (edges_for): look the result up; compute on demand if nothing was
      prepared for this analysis object (CLI and single-file callers).

dependency_suggester.suggest_dependencies then adds the semantic step (if the
adapter enables it), combines by character priority and drops x/n pairs.

Edges are (target_path, char) in the order the old per-language functions produced
them (explicit before structural), so combine_suggestions_path_based_with_char_priority
sees the same input as before.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.path_utils import normalize_path

logger = logging.getLogger(__name__)

Edges = List[Tuple[str, str]]
Links = List[Dict[str, str]]


@dataclass
class EdgeContext:
    """What every adapter may read. Run indexes are built lazily, once per context."""

    path_to_key_info: Dict[str, KeyInfo]
    project_root: str
    file_analysis_results: Dict[str, Any]
    project_symbol_map: Dict[str, Dict[str, Any]]
    tracked_paths_globally: Set[str]
    _sql_index: Optional["SqlIndex"] = field(default=None, repr=False)

    @property
    def sql_index(self) -> "SqlIndex":
        if self._sql_index is None:
            self._sql_index = SqlIndex(self.project_symbol_map)
        return self._sql_index


@dataclass(frozen=True)
class LanguageAdapter:
    name: str
    extensions: Tuple[str, ...]
    semantic: bool  # whether the shared semantic step runs for these files
    edges: Callable[[EdgeContext, str, Dict[str, Any]], Tuple[Edges, Links]]


# --- adapters ------------------------------------------------------------------------


def _ds() -> Any:
    from . import dependency_suggester  # lazy: dependency_suggester imports this module

    return dependency_suggester


def _python_edges(ctx: EdgeContext, path: str, analysis: Dict[str, Any]) -> Tuple[Edges, Links]:
    ds = _ds()
    explicit, explicit_links = ds._identify_python_dependencies(
        path, analysis, ctx.file_analysis_results, ctx.project_root,
        ctx.path_to_key_info, ctx.project_symbol_map, ctx.tracked_paths_globally,
    )
    structural, structural_links = ds._identify_structural_dependencies(
        path, analysis, ctx.path_to_key_info, ctx.project_root,
        ctx.project_symbol_map, tracked_paths_globally=ctx.tracked_paths_globally,
    )
    return explicit + structural, explicit_links + structural_links


def _js_edges(ctx: EdgeContext, path: str, analysis: Dict[str, Any]) -> Tuple[Edges, Links]:
    ds = _ds()
    tsconfig_info = ds._tsconfig_for_file(path, ctx.project_root)
    explicit, explicit_links = ds._identify_jsts_explicit_import_dependencies(
        path, analysis, ctx.path_to_key_info, ctx.project_root, tsconfig_info,
        tracked_paths_globally=ctx.tracked_paths_globally,
    )
    structural, structural_links = ds._identify_javascript_structural_dependencies(
        path, analysis, ctx.path_to_key_info, ctx.project_root,
        ctx.project_symbol_map, tsconfig_info,
        tracked_paths_globally=ctx.tracked_paths_globally,
    )
    return explicit + structural, explicit_links + structural_links


def _link_edges(identify_name: str) -> Callable[[EdgeContext, str, Dict[str, Any]], Tuple[Edges, Links]]:
    def edges(ctx: EdgeContext, path: str, analysis: Dict[str, Any]) -> Tuple[Edges, Links]:
        identify = getattr(_ds(), identify_name)
        return (
            identify(
                path, analysis, ctx.file_analysis_results, ctx.project_root,
                ctx.path_to_key_info, tracked_paths_globally=ctx.tracked_paths_globally,
            ),
            [],
        )

    return edges


def _resolve_relative_or_root_ref(ctx: EdgeContext, path: str, url: str) -> Optional[str]:
    source_dir = os.path.dirname(path)
    if url.startswith("."):
        resolved = normalize_path(os.path.join(source_dir, url))
    elif url.startswith("/"):
        # Absolute or project-root relative
        resolved = normalize_path(os.path.join(ctx.project_root, url.lstrip("/")))
    else:
        return None
    return resolved if resolved in ctx.tracked_paths_globally else None


def _json_edges(ctx: EdgeContext, path: str, analysis: Dict[str, Any]) -> Tuple[Edges, Links]:
    edges: Edges = []
    for link in analysis.get("links", []):
        url = link.get("url") or link.get("href") or link.get("path")
        if url:
            resolved = _resolve_relative_or_root_ref(ctx, path, url)
            if resolved:
                edges.append((resolved, "<"))
    for ref in analysis.get("json_refs", []):  # structured key_path -> value refs
        value = ref.get("value")
        if value:
            resolved = _resolve_relative_or_root_ref(ctx, path, value)
            if resolved:
                edges.append((resolved, "<"))
    return edges, []


def _svelte_edges(ctx: EdgeContext, path: str, analysis: Dict[str, Any]) -> Tuple[Edges, Links]:
    ds = _ds()
    edges: Edges = []
    source_dir = os.path.dirname(path)
    # <script> imports follow JS resolution, including the project's tsconfig `paths`
    # (a .svelte file is outside TypeScript's own `include`, so the directory's first
    # config is its context) and bundler aliases such as SvelteKit's `$lib`.
    tsconfig_info = None
    for imp in analysis.get("imports", []):
        # _normalize_imports stores the specifier under "path" (it renames "source");
        # reading only "source" left every Svelte import unresolved.
        spec = imp.get("path") or imp.get("source")
        if spec:
            if tsconfig_info is None:
                tsconfig_info = ds._tsconfig_for_file(path, ctx.project_root)
            resolved = ds._resolve_js_import_path(
                spec, source_dir, ctx.project_root, ctx.path_to_key_info, tsconfig_info,
                tracked_paths_globally=ctx.tracked_paths_globally,
            )
            if resolved:
                edges.append((resolved, "<"))
    for link in analysis.get("links", []):  # markup links: relative only
        url = link.get("url") or link.get("href") or link.get("path")
        if url and url.startswith("."):
            resolved = normalize_path(os.path.join(source_dir, url))
            if resolved in ctx.path_to_key_info:
                edges.append((resolved, "<"))
    return edges, []


class SqlIndex:
    """Built once per run: which SQL file defines each table, and which files reference
    each table (excluding tables a file defines itself), in symbol-map order."""

    def __init__(self, project_symbol_map: Dict[str, Dict[str, Any]]) -> None:
        self.table_definer: Dict[str, str] = {}
        self.referencers: Dict[str, List[str]] = {}
        self.order: Dict[str, int] = {}
        for i, (other_path, symbols) in enumerate(project_symbol_map.items()):
            if not isinstance(symbols, dict) or not other_path.lower().endswith(".sql"):
                continue
            self.order[other_path] = i
            sym = cast(Dict[str, Any], symbols)
            defined = {t.lower() for t in sym.get("tables_defined", [])}
            for table in sym.get("tables_defined", []):
                self.table_definer[table.lower()] = other_path  # last definer wins
            # A table the file defines itself is a self-reference, not an inverse edge.
            for table in {t.lower() for t in sym.get("tables_referenced", [])} - defined:
                self.referencers.setdefault(table, []).append(other_path)


def _sql_edges(ctx: EdgeContext, path: str, analysis: Dict[str, Any]) -> Tuple[Edges, Links]:
    tables_defined = {t.lower() for t in analysis.get("tables_defined", [])}
    tables_referenced = {t.lower() for t in analysis.get("tables_referenced", [])}
    if not ctx.project_symbol_map:
        return [], []
    index = ctx.sql_index
    edges: Edges = []
    # Forward: this file depends on the files defining tables it references.
    for table in tables_referenced:
        if table in tables_defined:
            continue
        target = index.table_definer.get(table)
        if target and target != path:
            edges.append((target, "<"))
    # Inverse: files referencing tables this file defines depend on it.
    if tables_defined:
        dependents: Set[str] = set()
        for table in tables_defined:
            dependents.update(index.referencers.get(table, ()))
        dependents.discard(path)
        for other in sorted(dependents, key=index.order.__getitem__):
            edges.append((other, ">"))
    return edges, []


def _no_edges(ctx: EdgeContext, path: str, analysis: Dict[str, Any]) -> Tuple[Edges, Links]:
    return [], []


ADAPTERS: Tuple[LanguageAdapter, ...] = (
    LanguageAdapter("python", (".py",), True, _python_edges),
    LanguageAdapter(
        "js", (".js", ".jsx", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".tsx"), True, _js_edges
    ),
    LanguageAdapter("markdown", (".md", ".rst"), True, _link_edges("_identify_markdown_dependencies")),
    LanguageAdapter("html", (".html", ".htm"), False, _link_edges("_identify_html_dependencies")),
    LanguageAdapter("css", (".css",), False, _link_edges("_identify_css_dependencies")),
    LanguageAdapter("json", (".json",), True, _json_edges),
    LanguageAdapter("svelte", (".svelte",), True, _svelte_edges),
    # Semantic stays off for SQL (decision pending; see suggester_unification_plan.md §4).
    LanguageAdapter("sql", (".sql",), False, _sql_edges),
)
GENERIC = LanguageAdapter("generic", (), True, _no_edges)
_BY_EXTENSION: Dict[str, LanguageAdapter] = {
    ext: adapter for adapter in ADAPTERS for ext in adapter.extensions
}


def adapter_for(path: str) -> LanguageAdapter:
    return _BY_EXTENSION.get(os.path.splitext(path)[1].lower(), GENERIC)


def _valid(analysis: Any) -> bool:
    return isinstance(analysis, dict) and "error" not in analysis and "skipped" not in analysis


# --- run-scoped results ----------------------------------------------------------------

# path -> (the analysis dict it was computed from, edges, links). A caller gets the
# entry only for the identical analysis object.
_results: Dict[str, Tuple[Dict[str, Any], Edges, Links]] = {}
_run_ctx: Optional[EdgeContext] = None


def reset() -> None:
    global _run_ctx
    _results.clear()
    _run_ctx = None


def prepare_structural_index(
    file_analysis_results: Dict[str, Any],
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    project_symbol_map: Dict[str, Dict[str, Any]],
    tracked_paths_globally: Optional[Set[str]] = None,
) -> Dict[str, int]:
    """Compute every file's explicit + structural edges once, before the workers."""
    global _run_ctx
    reset()
    if tracked_paths_globally is None:
        tracked_paths_globally = set(path_to_key_info.keys())
    # JS/TS passes 1-2 (import resolution, export index, structural verification).
    _ds().prepare_js_structural_index(
        file_analysis_results, path_to_key_info, project_root, project_symbol_map,
        tracked_paths_globally,
    )
    ctx = EdgeContext(
        path_to_key_info, project_root, file_analysis_results, project_symbol_map,
        tracked_paths_globally,
    )
    start = time.perf_counter()
    per_language: Dict[str, int] = {}
    edge_count = 0
    for path, analysis in file_analysis_results.items():
        if not _valid(analysis):
            continue
        adapter = adapter_for(path)
        if adapter is GENERIC:
            continue
        try:
            edges, links = adapter.edges(ctx, path, analysis)
        except Exception as e:
            # One bad file must not abort the pass for every other file; it is
            # computed on demand (and fails there with its own error) instead.
            logger.warning(f"Structural index: {adapter.name} edges failed for {path}: {e}")
            continue
        _results[path] = (analysis, edges, links)
        per_language[adapter.name] = per_language.get(adapter.name, 0) + 1
        edge_count += len(edges)
    _run_ctx = ctx
    logger.info(
        f"Structural index: {edge_count} edges for {sum(per_language.values())} files "
        f"({', '.join(f'{k} {v}' for k, v in sorted(per_language.items()))}) in "
        f"{time.perf_counter() - start:.2f}s."
    )
    return per_language


def edges_for(
    path: str,
    analysis: Dict[str, Any],
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    file_analysis_results: Dict[str, Any],
    project_symbol_map: Dict[str, Dict[str, Any]],
    tracked_paths_globally: Set[str],
) -> Tuple[Edges, Links]:
    """The prepared (edges, links) for this file, or the same computation on demand."""
    entry = _results.get(path)
    if entry is not None and entry[0] is analysis:
        return list(entry[1]), list(entry[2])
    ctx = _run_ctx
    if (
        ctx is None
        or ctx.file_analysis_results is not file_analysis_results
        or ctx.project_symbol_map is not project_symbol_map
        or ctx.path_to_key_info is not path_to_key_info
    ):
        ctx = EdgeContext(
            path_to_key_info, project_root, file_analysis_results, project_symbol_map,
            tracked_paths_globally,
        )
    return adapter_for(path).edges(ctx, path, analysis)
