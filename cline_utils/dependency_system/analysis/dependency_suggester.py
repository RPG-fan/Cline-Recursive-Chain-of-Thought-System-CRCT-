# analysis/dependency_suggester.py

"""
Analysis module for dependency suggestion using contextual keys.
Suggests potential dependencies based on code analysis and embeddings.
Assigns specific characters based on the type of dependency found.
Outputs path-based dependencies: List[Tuple[target_norm_path, char]]
"""

import collections
import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, cast


def _has_method(methods: List[Any], member_name: str) -> bool:
    """Fast lookup for method names in a list of dicts without creating generators."""
    for m in methods:
        if isinstance(m, dict) and cast(Dict[str, Any], m).get("name") == member_name:
            return True
    return False


def _has_grid_row(grid_rows: List[Tuple[str, Any]], source_key_str: str) -> bool:
    """Fast lookup for row labels in grid rows without creating generators."""
    for r_label, _ in grid_rows:
        if r_label == source_key_str:
            return True
    return False


from cline_utils.dependency_system.core import key_manager

# Import only from lower-level modules
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.cache_manager import cached, clear_all_caches
from cline_utils.dependency_system.utils.cache_manager import (
    normalize_path_cached as normalize_path,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import get_file_type
from cline_utils.dependency_system.analysis import (
    js_package_resolution,
    similarity_index,
    structural_engine,
)
from cline_utils.dependency_system.analysis.js_tsconfig import (
    parse_jsonc,
    select_tsconfigs_for_file,
)
from cline_utils.dependency_system.io.file_io import read_file_content_safely
from cline_utils.dependency_system.utils.tracker_utils import (
    get_key_global_instance_string,
)

KEY_MANAGER_DIR = os.path.dirname(os.path.abspath(key_manager.__file__))

# NOTE: Avoid importing analyze_file here to prevent circular dependency if analyzer calls suggester


logger = logging.getLogger(__name__)


def strip_json_comments(json_str: str) -> str:
    """
    Strip standard multi-line and single-line comments from a JSON string,
    ensuring comment symbols inside double-quoted string literals are preserved.
    """
    pattern = r'("(?:\\.|[^"\\])*")|(/\*.*?\*/)|(//[^\r\n]*)'

    def replacer(match: re.Match[str]) -> str:
        # If it matches a string literal, preserve it
        g1 = match.group(1)
        if g1 is not None:
            return g1
        # Otherwise, it's a comment; discard it
        return ""

    return re.sub(pattern, replacer, json_str, flags=re.DOTALL)


# Caches for structural dependency analysis moved to cache_manager

# Character Definitions:
# <: Row depends on column.
# >: Column depends on row.
# x: Mutual dependency.
# d: Documentation dependency.
# o: Self dependency (diagonal only).
# n: Verified no dependency.
# p: Placeholder (unverified).
# s: Semantic dependency (weak .06-.07)
# S: Semantic dependency (strong .07+)

GLOBAL_SCAN_LIMIT = 1500
_PROJECT_SYMBOL_MAP_FILENAME_LOCAL = "project_symbol_map.json"

# --- Cross-file import resolution ledgers ---
# These dicts act as a shared ledger across all files processed in the same
# analysis run, so imports that appear in many files (e.g. common utilities)
# are resolved only once and reused on subsequent lookups.
#
# Key for Python ledger: (import_name, source_file_dir, project_root, relative_level)
# Value: the List[(resolved_path, item_verified)] returned by _convert_python_import_to_paths
#
# Key for JS ledger: (import_path, source_dir, project_root, applicable_tsconfig_paths)
# Value: Optional[str] – the resolved tracked path, or None
_py_import_resolution_ledger: Dict[
    Tuple[str, str, str, int], List[Tuple[str, bool]]
] = {}
_js_import_resolution_ledger: Dict[Tuple[str, str, str, Tuple[str, ...]], Optional[str]] = {}


def _get_symbol_map_path() -> str:
    """Robustly determines the path to project_symbol_map.json."""
    from cline_utils.dependency_system.core import resolve_state_path

    try:
        # Use import to find the directory of key_manager.py
        import cline_utils.dependency_system.core.key_manager as km

        _core_dir = os.path.dirname(os.path.abspath(km.__file__))
        return normalize_path(
            resolve_state_path(_PROJECT_SYMBOL_MAP_FILENAME_LOCAL, _core_dir)
        )
    except Exception:
        # Fallback relative to this file
        _core_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "core")
        return normalize_path(
            resolve_state_path(_PROJECT_SYMBOL_MAP_FILENAME_LOCAL, _core_dir)
        )


# _OLD_PROJECT_SYMBOL_MAP_FILENAME_LOCAL = "project_symbol_map_old.json" # Not used by load, only by save


def clear_caches():
    clear_all_caches()
    if hasattr(_find_and_parse_tsconfig, "_cache"):
        cast(Any, _find_and_parse_tsconfig)._cache.clear()
    if hasattr(load_project_symbol_map, "_cache"):
        cast(Any, load_project_symbol_map)._cache.clear()
    _py_import_resolution_ledger.clear()
    _js_import_resolution_ledger.clear()
    js_package_resolution.reset()
    _reset_js_structural_index()
    structural_engine.reset()
    _reset_semantic_indexes()
    similarity_index.invalidate()
    from .embedding_manager import reset_rerank_run_state

    reset_rerank_run_state()


@cached("metadata", track_path_args=[0])
def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Load metadata file with caching.

    Args:
        metadata_path: Path to the metadata file
    Returns:
        Dictionary containing metadata or empty dict on failure
    """
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata
    except FileNotFoundError:
        logger.warning(
            f"Metadata file not found at {metadata_path}. Run generate-embeddings first."
        )
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in metadata file {metadata_path}: {e}")
        return {}
    except Exception as e:
        logger.exception(f"Unexpected error reading metadata {metadata_path}: {e}")
        return {}


# --- TS/JS Config Helper ---
def _tsconfig_cache_key(start_dir: str, project_root_val: str) -> str:
    return f"tsconfig:{normalize_path(start_dir)}:{normalize_path(project_root_val)}"


@cached(
    "tsconfig_data",
    key_func=_tsconfig_cache_key,
    track_path_args=[0],
)
def _find_and_parse_tsconfig(
    start_dir: str, project_root_val: str
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Finds tsconfig.json or jsconfig.json by walking up from start_dir to project_root_val.
    Parses the first one found using jsonc-parser if available, otherwise standard json.
    Caches the result based on start_dir and project_root_val.

    Returns:
        Tuple of (config_file_path, parsed_data_dict) or None if not found/parsed.
    """
    current_dir = normalize_path(os.path.realpath(os.path.normpath(start_dir)))
    project_root_norm = normalize_path(
        os.path.realpath(os.path.normpath(project_root_val))
    )
    config_filenames = ["tsconfig.json", "jsconfig.json"]
    while True:
        for filename in config_filenames:
            config_path = os.path.join(current_dir, filename)
            if os.path.exists(config_path) and os.path.isfile(config_path):
                logger.debug(f"Found config file for JS/TS: {config_path}")
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    data: Dict[str, Any] = parse_jsonc(content)  # comments + trailing commas
                    logger.debug(
                        f"Successfully parsed {config_path} using standard json parser (comments stripped)."
                    )
                    return config_path, data
                except Exception as e:
                    logger.warning(
                        f"Error parsing {config_path}: {e}. Skipping this config."
                    )
                    return None

        if current_dir == project_root_norm or not current_dir.startswith(
            project_root_norm
        ):
            break
        parent_dir = normalize_path(
            os.path.realpath(os.path.normpath(os.path.dirname(current_dir)))
        )
        if parent_dir == current_dir:
            break
        current_dir = parent_dir
    logger.debug(
        f"No tsconfig.json or jsconfig.json found in hierarchy from {start_dir} up to {project_root_val}."
    )
    return None


def _tsconfig_for_file(file_path: str, project_root: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Applicable tsconfig/jsconfig files for one source file, in resolution order, with
    `extends` merged (see analysis.js_tsconfig). Cached in-process per config mtime.
    """
    return select_tsconfigs_for_file(file_path, project_root)


# --- MODIFIED load_project_symbol_map ---
@cached(
    "project_symbol_map_data",
    key_func=lambda: f"project_symbol_map:{os.path.getmtime(_get_symbol_map_path()) if os.path.exists(_get_symbol_map_path()) else 'missing'}",
)
def load_project_symbol_map() -> Dict[str, Dict[str, Any]]:
    """
    Loads the project_symbol_map.json file.
    The map contains information about functions, classes, globals, and exports for each file.
    Keys are normalized absolute file paths.
    It assumes the map is stored in the same directory as key_manager.py.
    """
    try:
        # Determine path relative to key_manager.py
        # Using __import__ and __file__ to robustly get key_manager's directory
        map_path = _get_symbol_map_path()

        if not os.path.exists(map_path):
            logger.warning(
                f"Project symbol map file not found at {map_path}. Symbol verification will be skipped."
            )
            return {}

        with open(map_path, "r", encoding="utf-8") as f:
            data: Dict[str, Dict[str, Any]] = json.load(f)
        logger.debug(
            f"Successfully loaded project symbol map from: {map_path} ({len(data)} entries)"
        )
        return data
    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding JSON from project symbol map file: {e}", exc_info=True
        )
        return {}
    except IOError as e:
        logger.error(f"I/O Error loading project symbol map file: {e}", exc_info=True)
        return {}
    except Exception as e:
        logger.exception(f"Unexpected error loading project symbol map: {e}")
        return {}


def should_skip_suggestion(
    source_path: str,
    target_path: str,
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]],
    path_to_key_info: Dict[str, KeyInfo],
) -> bool:
    """
    Returns True if the dependency suggestion between source_path and target_path
    should be skipped because it is already marked as 'x' (max strength) or 'n' (no dependency).
    """
    if not existing_state:
        return False

    source_path = normalize_path(source_path)
    target_path = normalize_path(target_path)

    # Primary check: Path-based lookup
    for path_pair in ((source_path, target_path), (target_path, source_path)):
        res = existing_state.get(path_pair)
        if res and res[0] in ("x", "n"):
            return True

    # Secondary check: Fallback to Key#GI lookup for compatibility
    src_ki = path_to_key_info.get(source_path)
    tgt_ki = path_to_key_info.get(target_path)
    if not src_ki or not tgt_ki:
        return False

    src_key_gi = get_key_global_instance_string(src_ki, path_to_key_info)
    tgt_key_gi = get_key_global_instance_string(tgt_ki, path_to_key_info)

    if not src_key_gi or not tgt_key_gi:
        return False

    for lookup_pair in ((src_key_gi, tgt_key_gi), (tgt_key_gi, src_key_gi)):
        res = existing_state.get(lookup_pair)
        if not res:
            continue
        char, _ = res
        if char in ("x", "n"):
            return True
    return False


# --- Main Dispatcher ---
def suggest_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    file_analysis_results: Dict[str, Any],
    threshold: float = 0.65,
    shared_scan_counter: Any = None,
    tracked_paths_globally: Optional[Set[str]] = None,
    existing_state: Optional[
        Dict[Tuple[str, str], Tuple[str, Set[str]]]
    ] = None,  # Added existing_state
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:  # MODIFIED return type
    """
    Suggest dependencies for a file, assigning appropriate characters, using contextual keys.

    Args:
        file_path: Path to the file to analyze
        path_to_key_info: Global map from normalized paths to KeyInfo objects.
        project_root: Root directory of the project
        file_analysis_results: Pre-computed analysis results for files
        threshold: Confidence threshold for *semantic* suggestions (0.0 to 1.0)
    Returns:
        List of (dependency_path, dependency_character) tuples
    """
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return [], []  # MODIFIED return

    norm_path = normalize_path(file_path)

    # --- Get the specific analysis result for the current file_path ---
    current_file_specific_analysis = file_analysis_results.get(norm_path)
    if (
        current_file_specific_analysis is None
        or "error" in current_file_specific_analysis
        or "skipped" in current_file_specific_analysis
    ):
        logger.debug(
            f"No valid analysis result for {norm_path} in file_analysis_results map. Skipping suggestions for this file."
        )
        return [], []  # MODIFIED return

    project_symbol_map = load_project_symbol_map()

    # Use passed tracked_paths_globally if available, otherwise reconstruct (O(N))
    if tracked_paths_globally is None:
        tracked_paths_globally = set(path_to_key_info.keys())

    # One workflow for every language (see _suggest_for_file / structural_engine).
    stats: Dict[str, Any] = {}
    char_suggestions, raw_ast_links_collector = _suggest_for_file(
        norm_path,
        current_file_specific_analysis,
        path_to_key_info,
        project_root,
        file_analysis_results,
        project_symbol_map,
        threshold,
        shared_scan_counter,
        tracked_paths_globally,
        existing_state,
        stats,
    )

    skipped_count = 0
    if existing_state and char_suggestions:
        original_count = len(char_suggestions)
        char_suggestions = [
            (target_path, dep_char)
            for target_path, dep_char in char_suggestions
            if not should_skip_suggestion(
                norm_path, target_path, existing_state, path_to_key_info
            )
        ]
        skipped_count = original_count - len(char_suggestions)

    if char_suggestions or skipped_count or stats.get("edges"):
        logger.debug(
            _format_suggest_summary(
                norm_path, project_root, stats, len(char_suggestions), skipped_count
            )
        )

    return char_suggestions, raw_ast_links_collector  # MODIFIED return


def _verify_class_member(
    module_symbols: Dict[str, Any], class_name: str, member_name: str
) -> bool:
    """
    Check if member_name is a method or instance attribute of class_name
    using the enriched project_symbol_map data.

    Performs deep lookup against the class's 'methods' array (including __init__
    attribute_accesses for instance attributes). Falls back to assuming valid
    if the class has no method data (backward compat for files where runtime
    inspection failed or produced no methods).

    Args:
        module_symbols: The symbol data dict for a single module file.
        class_name: The name of the class to look up.
        member_name: The method or attribute name to verify.

    Returns:
        True if the member is verified or assumed valid, False otherwise.
    """
    for cls in module_symbols.get("classes", []):
        if cls.get("name") != class_name:
            continue

        methods = cls.get("methods", [])

        # If the class has method data, do an actual lookup
        if methods:
            # Check direct method name match
            if _has_method(methods, member_name):
                return True
            # Check instance attributes (self.X set in __init__)
            init_method = next(
                (m for m in methods if m.get("name") == "__init__"), None
            )
            if init_method:
                init_attrs = init_method.get("attribute_accesses", [])
                if member_name in init_attrs:
                    return True
            # Class has methods but member not found among them
            return False
        else:
            # Fallback: class exists but no method data (runtime inspection
            # failed or class has no methods). Preserve old heuristic.
            logger.debug(
                f"_verify_class_member: Class '{class_name}' has no method data, "
                f"assuming member '{member_name}' is valid (fallback heuristic)."
            )
            return True

    # Class not found at all
    return False


def _has_symbol_name(
    module_symbols: Optional[Dict[str, Any]],
    symbol_name: str,
    keys: Optional[List[str]] = None,
) -> bool:
    """
    Fast lookup for symbol names in a module without creating generators.

    Handles a hybrid list of symbol types:
    - Dict items: Structured symbols (e.g., Python classes, functions) that contain
      a 'name' key and other metadata (like line numbers).
    - Str items: Simple symbol names or exports (e.g., JavaScript/TypeScript raw string
      exports or globals_defined) represented as plain strings.
    """
    if not module_symbols or not isinstance(module_symbols, dict):
        return False
    keys_to_check = (
        keys
        if keys is not None
        else ["functions", "classes", "globals_defined", "exports"]
    )
    for s_key in keys_to_check:
        symbol_list = module_symbols.get(s_key)
        if symbol_list and isinstance(symbol_list, list):
            for s_item in symbol_list:
                # Python symbols are typically parsed as dictionaries with a "name" key.
                if isinstance(s_item, dict):
                    if cast(Dict[str, Any], s_item).get("name") == symbol_name:
                        return True
                # JS/TS exports or other simplified symbols are represented directly as strings.
                elif isinstance(s_item, str):
                    if s_item == symbol_name:
                        return True
    return False


def _classify_role(path: str) -> str:
    """
    Classify a file as 'doc' or 'code' based on its extension.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".md", ".rst", ".txt"]:
        return "doc"
    return "code"


def _determine_dependency_direction(src_path: str, tgt_path: str) -> str:
    """
    Determine dependency direction based on file roles.
    Returns: 'x' (bidirectional), '<' (source depends on target), '>' (target depends on source)
    """
    src_role = _classify_role(src_path)
    tgt_role = _classify_role(tgt_path)

    # Doc -> Code: Documentation describes code (='>')
    if src_role == "doc" and tgt_role == "code":
        return ">"
    # Code -> Doc: Code is documented by doc (='<')
    elif src_role == "code" and tgt_role == "doc":
        return "<"
    # Same role: default to bidirectional
    return "x"


# --- Type-Specific Suggestion Functions ---
def _python_import_bindings_of(source_path: str, source_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The analyzer's `import_bindings` (every import statement, ast.walk order); for
    analyses cached before that field existed, derived from the parsed tree."""
    bindings = source_analysis.get("import_bindings")
    if isinstance(bindings, list):
        return cast(List[Dict[str, Any]], bindings)
    tree = source_analysis.get("_ast_tree")
    if tree is not None:
        from cline_utils.dependency_system.analysis.dependency_analyzer import (
            _python_import_bindings,  # pyright: ignore[reportPrivateUsage]
        )

        return _python_import_bindings(tree)
    logger.debug(f"ImportMap: no import bindings or AST for {source_path}; no names in scope.")
    return []


def _python_import_map(
    current_source_path: str,
    source_analysis: Dict[str, Any],
    project_root: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_symbol_map: Dict[str, Dict[str, Any]],
    tracked_paths_globally: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """
    Names in scope -> absolute path of the module they come from, built from the
    file's own analysis on every call. (It used to be @cached by source path alone,
    persisted, and never invalidated when the file changed: edited imports kept
    resolving through the old map.) Module resolution goes through the run-scoped
    _py_import_resolution_ledger.
    """
    norm_source_path = normalize_path(current_source_path)
    local_import_map: Dict[str, str] = {}
    current_source_dir = os.path.dirname(norm_source_path)
    try:
        for binding in _python_import_bindings_of(norm_source_path, source_analysis):
            names = binding.get("names") or []
            if binding.get("kind") == "import":
                for alias in names:
                    imported_module_string = alias.get("name") or ""
                    name_in_scope = alias.get("asname") or imported_module_string
                    resolved = _convert_python_import_to_paths(
                        import_name=imported_module_string,
                        source_file_dir=current_source_dir,
                        project_root=project_root,
                        path_to_key_info=path_to_key_info,
                        project_symbol_map=project_symbol_map,
                        specific_item_name=None,
                        relative_level=0,
                        tracked_paths_globally=tracked_paths_globally,
                    )
                    if resolved:
                        local_import_map[name_in_scope] = normalize_path(resolved[0][0])
                continue

            module_name = binding.get("module") or ""
            level = int(binding.get("level") or 0)
            # Relative imports resolve from the file's own directory, exactly as the
            # explicit-import step does: _convert_python_import_to_paths climbs
            # `level - 1` directories itself (`from . import x` = this package).
            # The structural map used to climb `level` more first, one directory too
            # high: 30 of 31 relative imports in a sample project never resolved here.
            base_dir_for_relative_resolve = current_source_dir

            resolved = _convert_python_import_to_paths(
                import_name=module_name,
                source_file_dir=base_dir_for_relative_resolve,
                project_root=project_root,
                path_to_key_info=path_to_key_info,
                project_symbol_map=project_symbol_map,
                specific_item_name=None,
                relative_level=level,
                tracked_paths_globally=tracked_paths_globally,
            )
            if resolved:
                resolved_module_file_path = normalize_path(resolved[0][0])
                for alias in names:
                    name_in_scope = alias.get("asname") or alias.get("name")
                    if name_in_scope:
                        local_import_map[name_in_scope] = resolved_module_file_path
    except Exception as e:
        logger.error(f"Error building import map for {norm_source_path}: {e}", exc_info=False)
    return local_import_map


def _resolve_name_to_path(
    name_to_resolve: Optional[str],
    source_path: str,
    current_file_import_map: Dict[str, str],
    path_to_key_info: Dict[str, KeyInfo],
) -> Optional[str]:
    """The module a (dotted) name comes from: the longest prefix that is an imported
    name in this file, e.g. for "foo.bar.baz.Item" try "foo.bar.baz", then "foo.bar",
    then "foo". None when the name is local, built-in or from an untracked module.
    (Uncached: a cheap dict walk over the file's own import map.)"""
    if not name_to_resolve:
        return None
    parts = name_to_resolve.split(".")
    for i in range(len(parts), 0, -1):
        path_from_import_map = current_file_import_map.get(".".join(parts[:i]))
        if path_from_import_map:
            if path_from_import_map in path_to_key_info:
                return path_from_import_map
            logger.warning(
                f"_resolve_name_to_path: Prefix '{'.'.join(parts[:i])}' mapped to '{path_from_import_map}', but path not in path_to_key_info. Import map might be stale or contain unresolved external refs."
            )
    return None


def _identify_structural_dependencies(
    source_path: str,
    source_analysis: Dict[str, Any],
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    project_symbol_map: Dict[str, Dict[str, Any]],
    tracked_paths_globally: Optional[Set[str]] = None,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """
    Identifies Python structural dependencies (calls, attributes, inheritance) using contextual keys.
    Returns list of tuples (dependency_path, dependency_character).
    """
    suggestions_path_based: List[Tuple[str, str]] = []
    raw_ast_verified_links: List[Dict[str, str]] = []

    if tracked_paths_globally is None:
        tracked_paths_globally = tracked_paths_globally or set(path_to_key_info.keys())

    if not source_analysis:
        return [], []

    calls = source_analysis.get("calls", [])
    attributes = source_analysis.get("attribute_accesses", [])
    inheritance = source_analysis.get("inheritance", [])
    type_references = source_analysis.get("type_references", [])
    decorators_used = source_analysis.get("decorators_used", [])
    exceptions_handled = source_analysis.get("exceptions_handled", [])
    with_contexts_used = source_analysis.get("with_contexts_used", [])

    current_file_import_map = _python_import_map(
        source_path,
        source_analysis,
        project_root,
        path_to_key_info,
        project_symbol_map,
        tracked_paths_globally,
    )

    def _resolve_name_to_path_wrapper(name: Optional[str]) -> Optional[str]:
        return _resolve_name_to_path(
            name,
            source_path,
            current_file_import_map,
            path_to_key_info,
        )

    # --- MODIFIED: Process Calls and Attributes with symbol verification ---
    for call_item in calls:
        potential_source_str = call_item.get(
            "potential_source"
        )  # e.g., "my_module_alias" or "my_module_alias.class_name"
        target_name_str = call_item.get(
            "target_name"
        )  # e.g., "my_module_alias.method_name" or "ImportedClass()"

        target_path_val = _resolve_name_to_path_wrapper(
            potential_source_str or target_name_str
        )  # Try potential_source first

        if target_path_val and target_path_val != source_path:
            # Determine the actual item name being called/accessed on the target_path_val
            actual_item_name_to_check = ""
            if target_name_str:  # "my_module_alias.method_name" -> "method_name"
                # If potential_source_str resolved, target_name_str is likely the full path to the method/function
                # We need the part of target_name_str that is relative to the resolved module (target_path_val)
                # This can be complex if target_name_str is like module.class.method
                # For now, simple split:
                if potential_source_str and target_name_str.startswith(
                    potential_source_str + "."
                ):
                    actual_item_name_to_check = target_name_str[
                        len(potential_source_str) + 1 :
                    ]
                elif not potential_source_str:  # direct call of an imported name
                    actual_item_name_to_check, _, _ = target_name_str.partition(".")
                else:  # Fallback or if potential_source_str is the same as target_name_str (e.g. direct function call)
                    _, _, actual_item_name_to_check = target_name_str.rpartition(".")

            if actual_item_name_to_check:
                # Remove "()" if it's a call representation from _get_full_name_str
                if actual_item_name_to_check.endswith("()"):
                    actual_item_name_to_check = actual_item_name_to_check[:-2]

                module_symbols = project_symbol_map.get(target_path_val, {})
                is_verified = _has_symbol_name(
                    module_symbols, actual_item_name_to_check
                )

                # Deep verification: check if the member is actually defined on the class
                # using enriched symbol map data (methods list, __init__ attributes)
                if not is_verified and potential_source_str:
                    _, _, potential_class_name = potential_source_str.rpartition(".")
                    if _verify_class_member(
                        module_symbols,
                        potential_class_name,
                        actual_item_name_to_check,
                    ):
                        is_verified = True

                if is_verified:
                    dep_char = "<"
                    suggestions_path_based.append((target_path_val, dep_char))
                    raw_ast_verified_links.append(
                        {
                            "source_path": source_path,
                            "target_path": target_path_val,
                            "char": dep_char,
                            "reason": f"Call/{target_name_str or potential_source_str}",
                        }
                    )

    for attr_item in attributes:
        potential_source_str = attr_item.get(
            "potential_source"
        )  # e.g., "my_module_alias" or "my_module_alias.instance"
        attribute_name_accessed = attr_item.get("target_name")  # e.g., "some_attribute"

        target_path_val = _resolve_name_to_path_wrapper(potential_source_str)

        if (
            target_path_val
            and target_path_val != source_path
            and attribute_name_accessed
        ):
            module_symbols = project_symbol_map.get(target_path_val, {})

            # Check if the attribute_name_accessed is a global/function/class directly in the module
            is_verified = _has_symbol_name(module_symbols, attribute_name_accessed)

            # Deep verification: check if the attribute is a method, __init__ attr, or class-level attribute
            # using enriched symbol map data (methods list, __init__ attributes)
            if not is_verified and potential_source_str:
                _, _, potential_class_name_from_source = (
                    potential_source_str.rpartition(".")
                )
                if _verify_class_member(
                    module_symbols,
                    potential_class_name_from_source,
                    attribute_name_accessed,
                ):
                    is_verified = True

            if is_verified:
                dep_char = "<"
                suggestions_path_based.append((target_path_val, dep_char))
                raw_ast_verified_links.append(
                    {
                        "source_path": source_path,
                        "target_path": target_path_val,
                        "char": dep_char,
                        "reason": f"Attribute/{potential_source_str}.{attribute_name_accessed}",
                    }
                )
    # --- END OF MODIFIED Process Calls and Attributes ---

    # Process Inheritance (from your last provided snippet, with raw_ast_links addition)
    for inh_item in inheritance:
        base_class_name_str = inh_item.get("base_class_name")
        target_path_val = _resolve_name_to_path_wrapper(base_class_name_str)
        if target_path_val and target_path_val != source_path:
            # Verify "Base" is in target_path_val's symbols
            module_symbols = project_symbol_map.get(target_path_val, {})
            _, _, actual_class_name = base_class_name_str.rpartition(".")
            if _has_symbol_name(module_symbols, actual_class_name, ["classes"]):
                dep_char = "<"
                suggestions_path_based.append((target_path_val, dep_char))
                raw_ast_verified_links.append(
                    {
                        "source_path": source_path,
                        "target_path": target_path_val,
                        "char": dep_char,
                        "reason": f"Inheritance/{actual_class_name}",
                    }
                )

    # --- MODIFIED: Process Type References with symbol map verification ---
    for type_ref_item in type_references:
        type_name_str = type_ref_item.get(
            "type_name_str"
        )  # e.g., "MyType" or "other_module.TheirType"
        target_path_val = _resolve_name_to_path_wrapper(
            type_name_str
        )  # Resolves "other_module" part if present

        if target_path_val and target_path_val != source_path:
            # Verify the actual type (e.g., "TheirType") is defined in the resolved module
            module_symbols = project_symbol_map.get(target_path_val, {})
            _, _, actual_type_to_check = type_name_str.rpartition(".")
            is_defined_type = _has_symbol_name(
                module_symbols,
                actual_type_to_check,
                ["classes", "functions", "globals_defined"],
            )
            if is_defined_type:
                if _has_symbol_name(
                    module_symbols, actual_type_to_check, ["classes"]
                ):  # Prioritize classes for type hints
                    dep_char = "<"
                    suggestions_path_based.append((target_path_val, dep_char))
                    raw_ast_verified_links.append(
                        {
                            "source_path": source_path,
                            "target_path": target_path_val,
                            "char": dep_char,
                            "reason": f"TypeHint/{actual_type_to_check}",
                        }
                    )

    # --- Process Decorators, Exceptions, With-Contexts with symbol map verification ---
    item_categories_to_process = [
        (
            decorators_used,
            "Decorator",
            "<",
        ),  # Using a decorator from another module means current file depends on it.
        (
            exceptions_handled,
            "Exception",
            "<",
        ),  # Handling an exception from another module.
        (
            with_contexts_used,
            "WithContext",
            "<",
        ),  # Using a context manager from another module.
    ]
    for item_list, item_type_log_name, dep_char in item_categories_to_process:
        for item_entry in item_list:
            # 'name' for decorators, 'type_name_str' for exceptions, 'context_expr_str' for with
            item_name_str = (
                item_entry.get("name")
                or item_entry.get("type_name_str")
                or item_entry.get("context_expr_str")
            )
            if not item_name_str:
                continue
            target_path_val = _resolve_name_to_path_wrapper(item_name_str)
            if target_path_val and target_path_val != source_path:
                module_symbols = project_symbol_map.get(target_path_val, {})
                _, _, actual_item_to_check = item_name_str.rpartition(".")

                # Check if the item exists as any kind of defined symbol in the target module
                is_defined_symbol = _has_symbol_name(
                    module_symbols,
                    actual_item_to_check,
                    ["classes", "functions", "globals_defined"],
                )
                if is_defined_symbol:
                    suggestions_path_based.append((target_path_val, dep_char))
                    raw_ast_verified_links.append(
                        {
                            "source_path": source_path,
                            "target_path": target_path_val,
                            "char": dep_char,
                            "reason": f"{item_type_log_name}/{actual_item_to_check}",
                        }
                    )

    return list(set(suggestions_path_based)), raw_ast_verified_links


# Upfront explicit + structural pass for every language (includes the JS/TS passes).
prepare_structural_index = structural_engine.prepare_structural_index


def _suggest_for_file(
    norm_path: str,
    analysis: Dict[str, Any],
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    file_analysis_results: Dict[str, Any],
    project_symbol_map: Dict[str, Dict[str, Any]],
    threshold: float,
    shared_scan_counter: Any = None,
    tracked_paths_globally: Optional[Set[str]] = None,
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """
    The one suggestion workflow, for every language: explicit + structural edges from
    the language's adapter (structural_engine), then the shared semantic step if the
    adapter enables it, then combination by character priority.
    """
    if tracked_paths_globally is None:
        tracked_paths_globally = set(path_to_key_info.keys())
    adapter = structural_engine.adapter_for(norm_path)
    edges, ast_links = structural_engine.edges_for(
        norm_path, analysis, path_to_key_info, project_root, file_analysis_results,
        project_symbol_map, tracked_paths_globally,
    )
    semantic: List[Tuple[str, str]] = []
    if adapter.semantic:
        semantic = suggest_semantic_dependencies_path_based(
            norm_path, path_to_key_info, project_root, threshold, project_symbol_map,
            shared_scan_counter, tracked_paths_globally=tracked_paths_globally,
            existing_state=existing_state,
        )
    if stats is not None:
        stats["adapter"] = adapter.name
        stats["edges"] = len(edges)
        stats["link_kinds"] = collections.Counter(
            str(link.get("reason", "?")).split("/", 1)[0] for link in ast_links
        )
        stats["semantic"] = collections.Counter(char for _, char in semantic)
    return combine_suggestions_path_based_with_char_priority(edges + semantic, norm_path), ast_links


def _format_suggest_summary(
    norm_path: str, project_root: str, stats: Dict[str, Any], kept: int, skipped: int
) -> str:
    """One line per file: what each source contributed and what survived."""
    def counts(counter: Any) -> str:
        return ", ".join(f"{k} {v}" for k, v in sorted(counter.items()))

    try:
        shown = os.path.relpath(norm_path, project_root).replace(os.sep, "/")
    except ValueError:
        shown = norm_path
    link_kinds, semantic = stats.get("link_kinds") or {}, stats.get("semantic") or {}
    parts = [f"edges {stats.get('edges', 0)}" + (f" ({counts(link_kinds)})" if link_kinds else "")]
    if semantic:
        parts.append(f"semantic {sum(semantic.values())} ({counts(semantic)})")
    tail = f"-> {kept} kept" + (f", {skipped} skipped (tracker x/n)" if skipped else "")
    return f"Suggest {shown} [{stats.get('adapter', '?')}]: {' + '.join(parts)} {tail}"


def _valid_analysis(analysis: Any) -> bool:
    return analysis is not None and "error" not in analysis and "skipped" not in analysis


# --- Per-language entry points (thin wrappers over _suggest_for_file) ----------------
# Kept with their original signatures for existing callers and tests.


def suggest_python_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    source_analysis: Optional[Dict[str, Any]],
    _all_file_analyses_map: Dict[str, Any],
    project_symbol_map: Dict[str, Dict[str, Any]],
    threshold: float,
    shared_scan_counter: Any = None,
    tracked_paths_globally: Optional[Set[str]] = None,
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    norm_file_path = normalize_path(file_path)
    if not _valid_analysis(source_analysis):
        return [], []
    return _suggest_for_file(
        norm_file_path, cast(Dict[str, Any], source_analysis), path_to_key_info,
        project_root, _all_file_analyses_map, project_symbol_map, threshold,
        shared_scan_counter, tracked_paths_globally, existing_state,
    )


def suggest_javascript_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    source_analysis: Dict[str, Any],
    project_symbol_map: Dict[str, Dict[str, Any]],
    threshold: float,
    shared_scan_counter: Any = None,
    tracked_paths_globally: Optional[Set[str]] = None,
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    norm_file_path = normalize_path(file_path)
    if not _valid_analysis(source_analysis):
        return [], []
    return _suggest_for_file(
        norm_file_path, source_analysis, path_to_key_info, project_root,
        {norm_file_path: source_analysis}, project_symbol_map, threshold,
        shared_scan_counter, tracked_paths_globally, existing_state,
    )


def _suggest_from_results(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    file_analysis_results: Dict[str, Any],
    threshold: float,
    project_symbol_map: Optional[Dict[str, Any]],
    shared_scan_counter: Any,
    tracked_paths_globally: Optional[Set[str]],
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]],
) -> List[Tuple[str, str]]:
    norm_file_path = normalize_path(file_path)
    analysis = file_analysis_results.get(norm_file_path)
    if not _valid_analysis(analysis):
        return []
    return _suggest_for_file(
        norm_file_path, analysis, path_to_key_info, project_root, file_analysis_results,
        project_symbol_map or {}, threshold, shared_scan_counter, tracked_paths_globally,
        existing_state,
    )[0]


# --- Upfront JS/TS passes ---------------------------------------------------------
# analyze-project runs prepare_js_structural_index() once, before the parallel
# suggestion workers start:
#   pass 1: resolve every import / re-export specifier of every JS-family file (fills
#           _js_import_resolution_ledger) and index every module's exported names;
#   pass 2: verify every file's calls / type references against those tables and
#           store the per-file result.
# Workers then only look the result up (_identify_javascript_structural_dependencies).
# Callers outside analyze-project (nothing prepared) compute the same result on demand.
_JS_FAMILY_EXTENSIONS = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".tsx")

# Export registry: each origin module's exported names, plus (origin, name) -> bool.
# Bound to one symbol map object; a different map resets it. clear_caches() resets too.
_js_export_registry_owner: Optional[int] = None
_js_export_names: Dict[str, frozenset] = {}
_js_export_verified: Dict[Tuple[str, str], bool] = {}

# Pass 2 results: source_path -> (the analysis dict it was computed from, suggestions,
# AST links). A worker uses an entry only for the identical analysis object.
_js_structural_results: Dict[
    str, Tuple[Dict[str, Any], List[Tuple[str, str]], List[Dict[str, str]]]
] = {}


def _reset_js_export_registry() -> None:
    global _js_export_registry_owner
    _js_export_registry_owner = None
    _js_export_names.clear()
    _js_export_verified.clear()


def _reset_js_structural_index() -> None:
    """Drop the export registry and the precomputed per-file results."""
    _reset_js_export_registry()
    _js_structural_results.clear()


def _bind_js_export_registry(project_symbol_map: Dict[str, Dict[str, Any]]) -> None:
    global _js_export_registry_owner
    if _js_export_registry_owner != id(project_symbol_map):
        _reset_js_export_registry()
        _js_export_registry_owner = id(project_symbol_map)


def _js_export_names_for(
    project_symbol_map: Dict[str, Dict[str, Any]], origin_path: str
) -> frozenset:
    names = _js_export_names.get(origin_path)
    if names is None:
        exports = (project_symbol_map.get(origin_path) or {}).get("exports") or []
        names = frozenset(
            (e.get("alias") or e.get("name"))
            for e in exports
            if isinstance(e, dict) and (e.get("alias") or e.get("name"))
        )
        _js_export_names[origin_path] = names
    return names


def _js_verified_export(
    project_symbol_map: Dict[str, Dict[str, Any]], origin_path: str, name: str
) -> bool:
    """
    Whether `origin_path` exports `name`, computed once per (origin, name) per run.
    `export {a as b}` exports 'b'; `export default` exports 'default'.
    """
    _bind_js_export_registry(project_symbol_map)
    key = (origin_path, name)
    cached = _js_export_verified.get(key)
    if cached is None:
        cached = _js_export_verified[key] = name in _js_export_names_for(
            project_symbol_map, origin_path
        )
    return cached


def prepare_js_structural_index(
    file_analysis_results: Dict[str, Any],
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    project_symbol_map: Dict[str, Dict[str, Any]],
    tracked_paths_globally: Optional[Set[str]] = None,
) -> Dict[str, int]:
    """Run both upfront JS/TS passes for every JS-family file. Returns counts."""
    import time as _time

    if tracked_paths_globally is None:
        tracked_paths_globally = set(path_to_key_info.keys())
    _reset_js_structural_index()
    _bind_js_export_registry(project_symbol_map)

    js_files = [
        (path, analysis)
        for path, analysis in file_analysis_results.items()
        if path.lower().endswith(_JS_FAMILY_EXTENSIONS)
        and isinstance(analysis, dict)
        and "error" not in analysis
        and "skipped" not in analysis
    ]

    # Pass 1: import resolution + export index.
    start = _time.perf_counter()
    tsconfigs: Dict[str, Any] = {}
    specifiers = resolved = 0
    for path, analysis in js_files:
        tsconfig_info = tsconfigs[path] = _tsconfig_for_file(path, project_root)
        source_dir = os.path.dirname(path)
        specs = [imp.get("path") for imp in analysis.get("imports", [])]
        specs += [exp["from"] for exp in analysis.get("exports", []) if "from" in exp]
        for spec in specs:
            if not spec:
                continue
            specifiers += 1
            if _resolve_js_import_path(
                spec, source_dir, project_root, path_to_key_info, tsconfig_info,
                tracked_paths_globally=tracked_paths_globally,
            ):
                resolved += 1
    for origin_path in project_symbol_map:
        _js_export_names_for(project_symbol_map, origin_path)
    logger.info(
        f"JS pass 1: resolved {resolved} of {specifiers} import specifiers in "
        f"{len(js_files)} JS/TS files and indexed the exports of "
        f"{len(_js_export_names)} modules in {_time.perf_counter() - start:.2f}s."
    )

    # Pass 2: structural verification against the tables from pass 1.
    start = _time.perf_counter()
    links = 0
    for path, analysis in js_files:
        suggestions, ast_links = _compute_js_structural_dependencies(
            path, analysis, path_to_key_info, project_root, project_symbol_map,
            tsconfigs[path], tracked_paths_globally=tracked_paths_globally,
        )
        _js_structural_results[path] = (analysis, suggestions, ast_links)
        links += len(ast_links)
    logger.info(
        f"JS pass 2: verified {links} structural links for {len(js_files)} JS/TS files "
        f"in {_time.perf_counter() - start:.2f}s."
    )
    return {"files": len(js_files), "specifiers": specifiers, "resolved": resolved, "links": links}


def _identify_javascript_structural_dependencies(
    source_path: str,
    source_analysis: Dict[str, Any],
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    project_symbol_map: Dict[str, Dict[str, Any]],
    tsconfig_info: Any,  # see _js_resolution_contexts
    tracked_paths_globally: Optional[Set[str]] = None,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """The upfront pass-2 result for this file, or the same computation on demand."""
    precomputed = _js_structural_results.get(source_path)
    if precomputed is not None and precomputed[0] is source_analysis:
        return list(precomputed[1]), list(precomputed[2])
    return _compute_js_structural_dependencies(
        source_path, source_analysis, path_to_key_info, project_root,
        project_symbol_map, tsconfig_info, tracked_paths_globally=tracked_paths_globally,
    )


def _compute_js_structural_dependencies(
    source_path: str,
    source_analysis: Dict[str, Any],
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    project_symbol_map: Dict[str, Dict[str, Any]],
    tsconfig_info: Any,  # see _js_resolution_contexts
    tracked_paths_globally: Optional[Set[str]] = None,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """
    Identifies JS/TS structural dependencies by verifying that symbols used in the source file
    are correctly imported and are present in the target module's exports.
    Logs one summary line per file.
    """
    suggestions_path_based: List[Tuple[str, str]] = []
    raw_ast_verified_links: List[Dict[str, str]] = []

    if tracked_paths_globally is None:
        tracked_paths_globally = tracked_paths_globally or set(path_to_key_info.keys())

    source_dir = os.path.dirname(source_path)
    reexport_count = 0

    # --- Step 1: Handle re-exports directly ---
    for export_item in source_analysis.get("exports", []):
        if "from" in export_item:
            resolved_path = _resolve_js_import_path(
                export_item["from"],
                source_dir,
                project_root,
                path_to_key_info,
                tsconfig_info,
                tracked_paths_globally=tracked_paths_globally,
            )
            if resolved_path and resolved_path != source_path:
                dep_char = "<"  # A re-export is a direct dependency
                reexport_count += 1
                suggestions_path_based.append((resolved_path, dep_char))
                raw_ast_verified_links.append(
                    {
                        "source_path": source_path,
                        "target_path": resolved_path,
                        "char": dep_char,
                        # `export {a} from "x"` records carry no name of their own.
                        "reason": f"JSReExport/{export_item.get('name') or export_item['from']}",
                    }
                )

    # --- Step 2: Build a reliable import map: { name_in_scope: { origin_path: "...", original_name: "..." } } ---
    import_map: Dict[str, Dict[str, str]] = {}
    for imp in source_analysis.get("imports", []):
        unresolved_path = imp.get("path")
        if not unresolved_path:
            continue

        resolved_path = _resolve_js_import_path(
            unresolved_path,
            source_dir,
            project_root,
            path_to_key_info,
            tsconfig_info,
            tracked_paths_globally=tracked_paths_globally,
        )
        if not resolved_path:
            continue

        for symbol_info in imp.get("symbols", []):
            symbol_type = symbol_info.get("type")
            name_in_scope = symbol_info.get("alias") or symbol_info.get("name")
            original_name = symbol_info.get("name")

            if not name_in_scope or name_in_scope == "*":
                continue

            if symbol_type == "namespace":
                original_name = "*"
            elif symbol_type == "default":
                original_name = "default"

            if original_name:
                import_map[name_in_scope] = {
                    "origin_path": resolved_path,
                    "original_name": original_name,
                }

    # --- Step 3: Resolve a symbol and verify it in the target module's exports. ---
    # Each distinct name is resolved once per file; (origin, name) answers are shared
    # run-wide through the export registry.
    file_results: Dict[str, Optional[str]] = {}
    unverified: Set[str] = set()

    def _find_and_verify_symbol_origin(symbol_name: str) -> Optional[str]:
        if not symbol_name:
            return None
        if symbol_name in file_results:
            return file_results[symbol_name]
        file_results[symbol_name] = result = _verify_symbol_origin_uncached(symbol_name)
        return result

    def _verify_symbol_origin_uncached(symbol_name: str) -> Optional[str]:
        base_symbol, _, member_rest = symbol_name.partition(".")
        member_access_chain = [member_rest.partition(".")[0]] if member_rest else []

        import_info = import_map.get(base_symbol)
        if not import_info:
            return None

        resolved_path = import_info["origin_path"]
        original_name_from_import = import_info["original_name"]

        # Determine what to look for in the target's exports list
        item_to_check_in_exports = original_name_from_import
        # If it was `import * as X from '...` and used as `X.item`
        if original_name_from_import == "*" and member_access_chain:
            item_to_check_in_exports = member_access_chain[0]

        if _js_verified_export(project_symbol_map, resolved_path, item_to_check_in_exports):
            return resolved_path
        unverified.add(symbol_name)
        return None

    # --- Step 4: Iterate through calls, type references and decorators. ---
    # `@Foo(...)` is already a call record; only bare `@Foo` / `@ns.Foo` add a check.
    calls = source_analysis.get("calls", [])
    called_names = {c.get("target_name") for c in calls}
    bare_decorators = [
        d for d in source_analysis.get("decorators_used", []) if d.get("name") not in called_names
    ]
    items_to_check = [
        (calls, "target_name", "JSCall"),
        (source_analysis.get("type_references", []), "type_name_str", "JSTypeRef"),
        (bare_decorators, "name", "JSDecorator"),
    ]

    verified_uses = 0
    for item_list, name_key, reason_prefix in items_to_check:
        for item in item_list:
            item_name = item.get(name_key)
            if not item_name:
                continue

            resolved_path = _find_and_verify_symbol_origin(item_name)

            if resolved_path and resolved_path != source_path:
                dep_char = "<"
                verified_uses += 1
                suggestions_path_based.append((resolved_path, dep_char))
                raw_ast_verified_links.append(
                    {
                        "source_path": source_path,
                        "target_path": resolved_path,
                        "char": dep_char,
                        "reason": f"{reason_prefix}/{item_name}",
                    }
                )

    unique_suggestions = list(set(suggestions_path_based))
    return unique_suggestions, raw_ast_verified_links


def _identify_jsts_explicit_import_dependencies(
    source_path: str,
    source_analysis: Dict[str, Any],
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    tsconfig_info: Any,  # see _js_resolution_contexts
    tracked_paths_globally: Optional[Set[str]] = None,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """
    Converts analyzer 'imports' entries for JS/TS/TSX into direct '<' dependencies.
    Also emits AST-verified link entries for each resolved import.
    Relies on _resolve_js_import_path for consistent path resolution (incl. tsconfig paths/baseUrl).
    """
    suggestions_path_based: List[Tuple[str, str]] = []
    raw_ast_verified_links: List[Dict[str, str]] = []

    if tracked_paths_globally is None:
        tracked_paths_globally = tracked_paths_globally or set(path_to_key_info.keys())
    if not source_analysis:
        return [], []

    source_dir = os.path.dirname(source_path)

    imports_list = source_analysis.get("imports", [])
    # imports format expected from analyzer:
    # [
    #   { "path": "./mod", "symbols": [ { "type": "named"/"default"/"namespace", "name": "...", "alias": "..." } ] },
    #   { "path": "lib/core", "symbols": [...] },
    #   ...
    # ]
    for imp in imports_list:
        unresolved = imp.get("path")
        if not unresolved:
            continue

        resolved = _resolve_js_import_path(
            unresolved,
            source_dir,
            project_root,
            path_to_key_info,
            tsconfig_info,
            tracked_paths_globally=tracked_paths_globally,
        )
        if not resolved:
            continue

        if resolved in tracked_paths_globally and resolved != source_path:
            dep_char = "<"
            suggestions_path_based.append((resolved, dep_char))

            # Prefer to reference the literal import path string for reason context
            reason_literal = unresolved
            raw_ast_verified_links.append(
                {
                    "source_path": source_path,
                    "target_path": resolved,
                    "char": dep_char,
                    "reason": f"ExplicitImport/{reason_literal}",
                }
            )

    # Deduplicate suggestions
    return list(set(suggestions_path_based)), raw_ast_verified_links


def suggest_documentation_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    file_analysis_results: Dict[str, Any],
    threshold: float,
    _embeddings_dir: str = "",
    _metadata_path: str = "",
    project_symbol_map: Optional[Dict[str, Any]] = None,
    shared_scan_counter: Any = None,
    tracked_paths_globally: Optional[Set[str]] = None,
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
) -> List[Tuple[str, str]]:
    return _suggest_from_results(
        file_path, path_to_key_info, project_root, file_analysis_results, threshold,
        project_symbol_map, shared_scan_counter, tracked_paths_globally, existing_state,
    )


def suggest_html_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    file_analysis_results: Dict[str, Any],
    tracked_paths_globally: Optional[Set[str]] = None,
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
) -> List[Tuple[str, str]]:
    return _suggest_from_results(
        file_path, path_to_key_info, project_root, file_analysis_results, 0.0, None,
        None, tracked_paths_globally, existing_state,
    )


def suggest_css_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    file_analysis_results: Dict[str, Any],
    tracked_paths_globally: Optional[Set[str]] = None,
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
) -> List[Tuple[str, str]]:
    return _suggest_from_results(
        file_path, path_to_key_info, project_root, file_analysis_results, 0.0, None,
        None, tracked_paths_globally, existing_state,
    )


def suggest_json_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    file_analysis_results: Dict[str, Any],
    threshold: float,
    project_symbol_map: Optional[Dict[str, Any]] = None,
    shared_scan_counter: Any = None,
    tracked_paths_globally: Optional[Set[str]] = None,
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
) -> List[Tuple[str, str]]:
    return _suggest_from_results(
        file_path, path_to_key_info, project_root, file_analysis_results, threshold,
        project_symbol_map, shared_scan_counter, tracked_paths_globally, existing_state,
    )


def suggest_svelte_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    file_analysis_results: Dict[str, Any],
    threshold: float,
    project_symbol_map: Optional[Dict[str, Any]] = None,
    shared_scan_counter: Any = None,
    tracked_paths_globally: Optional[Set[str]] = None,
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
) -> List[Tuple[str, str]]:
    return _suggest_from_results(
        file_path, path_to_key_info, project_root, file_analysis_results, threshold,
        project_symbol_map, shared_scan_counter, tracked_paths_globally, existing_state,
    )


def suggest_sql_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    file_analysis_results: Dict[str, Any],
    threshold: float,
    project_symbol_map: Optional[Dict[str, Any]] = None,
    shared_scan_counter: Any = None,
    tracked_paths_globally: Optional[Set[str]] = None,
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
) -> List[Tuple[str, str]]:
    """SQL: forward edges to the files defining referenced tables, inverse edges from
    files referencing tables this file defines (structural_engine.SqlIndex)."""
    return _suggest_from_results(
        file_path, path_to_key_info, project_root, file_analysis_results, threshold,
        project_symbol_map, shared_scan_counter, tracked_paths_globally, existing_state,
    )


def suggest_generic_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    threshold: float,
    project_symbol_map: Optional[Dict[str, Any]] = None,
    shared_scan_counter: Any = None,
    tracked_paths_globally: Optional[Set[str]] = None,
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
) -> List[Tuple[str, str]]:
    norm_file_path = normalize_path(file_path)
    return _suggest_for_file(
        norm_file_path, {}, path_to_key_info, project_root, {}, project_symbol_map or {},
        threshold, shared_scan_counter, tracked_paths_globally, existing_state,
    )[0]


def _get_ast_links_path_suggester(
    project_root: str, *args: Any, **kwargs: Any
) -> List[str]:
    return [
        os.path.join(
            project_root,
            "cline_utils",
            "dependency_system",
            "core",
            "state",
            "ast_verified_links.json",
        )
    ]


@cached(
    "ast_verified_links",
    ttl=1200,
    track_path_args=[0],
    file_deps=_get_ast_links_path_suggester,
    check_mtime=True,
)
def _load_ast_verified_links(project_root: str) -> List[Dict[str, str]]:
    """
    Load AST-verified links from project_analyzer for structural evidence
    """
    try:
        ast_links_path = os.path.join(
            project_root,
            "cline_utils",
            "dependency_system",
            "core",
            "state",
            "ast_verified_links.json",
        )
        if os.path.exists(ast_links_path):
            with open(ast_links_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            logger.warning(f"AST verified links file not found: {ast_links_path}")
            return []
    except Exception as e:
        logger.error(f"Error loading AST verified links: {e}")
        return []


def _build_ast_map(
    source_path: str, ast_verified_links: List[Dict[str, str]]
) -> Dict[str, str]:
    """
    Build a map of target paths to AST relationships for a given source file.
    """
    source_norm = os.path.normpath(source_path).replace(os.sep, "/")
    ast_map: Dict[str, str] = {}
    for link in ast_verified_links:
        link_from = link.get("source_path")
        link_to = link.get("target_path")
        # Also support 'char' key which is used in ast_verified_links.json
        relation = link.get("char") or link.get("relation", "")

        if link_from == source_norm and link_to:
            ast_map[link_to] = relation
        elif link_to == source_norm and link_from:
            # For reverse direction, store the INVERSE relationship
            reverse_key = link_from
            original_relation = relation
            inverse_relation = ""
            if original_relation == "<":
                inverse_relation = ">"
            elif original_relation == ">":
                inverse_relation = "<"
            else:  # 'x' and 'd' are symmetrical
                inverse_relation = original_relation

            if inverse_relation:
                # An existing entry for this key (the forward link) wins over the
                # inverse of a reverse link; opposed relations are not merged into
                # 'x' (see combine_suggestions_path_based_with_char_priority).
                if not ast_map.get(reverse_key):
                    ast_map[reverse_key] = inverse_relation
    return ast_map


def _enhance_semantic_suggestions_with_ast_evidence(
    semantic_suggestions: List[Tuple[str, str]],
    ast_map: Dict[str, str],
) -> List[Tuple[str, str]]:
    """
    Enhance semantic suggestions with AST-verified structural evidence
    """
    enhanced: List[Tuple[str, str]] = []

    # Track which targets we've processed to avoid duplicates when adding from ast_map
    processed_targets: Set[str] = set()

    for target_path, semantic_char in semantic_suggestions:
        target_norm = os.path.normpath(target_path).replace(os.sep, "/")
        processed_targets.add(target_norm)

        ast_relation = ast_map.get(target_norm)

        # Apply AST enhancement logic
        enhanced_char = semantic_char

        if ast_relation:
            if ast_relation in ["<", ">", "x", "d"]:
                enhanced_char = ast_relation
            elif semantic_char == "s":
                enhanced_char = "S"
            else:
                enhanced_char = semantic_char
        else:
            if semantic_char == "s":
                enhanced_char = semantic_char
            elif semantic_char == "S":
                enhanced_char = semantic_char

        enhanced.append((target_path, enhanced_char))

    # Add missing structural links
    for target_path, relation in ast_map.items():
        if target_path not in processed_targets:
            if relation in ["<", ">", "x", "d"]:
                enhanced.append((target_path, relation))

    return enhanced


# --- Semantic Suggestion (Adapted to return paths) ---


@cached("reranking", ttl=600, check_mtime=False)
def _generate_candidate_text_from_file_cached(target_norm_path: str) -> str:
    """Generates the base candidate text from file content, cached by target path."""
    from .embedding_manager import preprocess_doc_structure

    target_ext = os.path.splitext(target_norm_path)[1].lower()
    target_is_doc = target_ext in [".md", ".txt", ".rst"]

    candidate_text = ""

    if target_is_doc:
        try:
            content = read_file_content_safely(target_norm_path)
            if content is None:
                raise Exception("File read failed")
            candidate_text = preprocess_doc_structure(content)
        except Exception:
            candidate_text = f"[DOC: {os.path.basename(target_norm_path)}]"
    else:
        # Fallback for non-code or untracked files
        try:
            content = read_file_content_safely(target_norm_path)
            if content is None:
                raise Exception("File read failed")
            # Limit raw content
            candidate_text = (
                f"[FILE: {os.path.basename(target_norm_path)}]\n{content[:12800]}"
            )
        except Exception:
            candidate_text = f"[FILE: {os.path.basename(target_norm_path)}]"

    return candidate_text


# --- Upfront tracker passes for the semantic step ----------------------------------
# analyze-project builds both of these once, before the parallel suggestion workers:
#   * the tracker relations of every source key (what each file used to collect for
#     itself by re-discovering and re-reading every tracker), and
#   * the set of pairs existing trackers mark 'x' or 'n' (should_skip_suggestion's
#     answer for every pair, as set lookups).
_TRACKER_RELATION_CHARS = ("<", ">", "x", "d", "S", "s")
_TRACKER_STRONG_CHARS = ("<", ">", "x", "d")
_tracker_relations_index: Optional[Dict[str, Dict[str, str]]] = None
_tracker_relations_owner: Optional[int] = None
_skip_pairs_key: Optional[Tuple[int, int]] = None
_skip_pairs: frozenset = frozenset()
_skip_pairs_lock = threading.Lock()


def _collect_tracker_relations(
    tracker_paths: Any,
    key_str_to_ki: Dict[str, KeyInfo],
    only_source: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    source key_string -> {target norm_path: char} from the given trackers, for every
    row label (or only `only_source`). Trackers are read in the given order and
    each source's map follows the per-file rule: the first char seen for a target
    wins unless a later one is '<', '>', 'x' or 'd'. A source whose grid read fails
    gets an empty map, as the per-file loop discarded its relations on any error.
    """
    from cline_utils.dependency_system.core.dependency_grid import (
        get_dependencies_from_grid,
    )
    from cline_utils.dependency_system.utils.tracker_utils import (
        read_tracker_file_structured,
    )

    relations: Dict[str, Dict[str, str]] = {}
    failed: Set[str] = set()
    for t_path in tracker_paths:
        t_data = read_tracker_file_structured(t_path)
        grid_rows = t_data.get("grid_rows_ordered", [])
        if only_source is not None:
            sources = [only_source] if _has_grid_row(grid_rows, only_source) else []
        else:
            sources = list(dict.fromkeys(r_label for r_label, _ in grid_rows))
        if not sources:
            continue
        grid_dict = {r_label: r_data for r_label, r_data in grid_rows}
        tracker_ki_list = [
            key_str_to_ki.get(hdr)
            or KeyInfo(
                key_string=hdr,
                norm_path="",
                parent_path=None,
                tier=0,
                is_directory=False,
            )
            for hdr in t_data.get("grid_headers_ordered", [])
        ]
        for source_key_str in sources:
            if source_key_str in failed:
                continue
            try:
                deps = get_dependencies_from_grid(
                    grid_dict, source_key_str, tracker_ki_list
                )
            except Exception as e:
                failed.add(source_key_str)
                logger.warning(
                    f"Error extracting tracker dependencies of {source_key_str} from {t_path}: {e}"
                )
                continue
            source_map = relations.setdefault(source_key_str, {})
            for char, targets in deps.items():
                if char not in _TRACKER_RELATION_CHARS:
                    continue
                for tgt_key_str in targets:
                    target_ki = key_str_to_ki.get(tgt_key_str)
                    if target_ki is None:
                        continue
                    tgt_norm = target_ki.norm_path
                    if tgt_norm not in source_map or char in _TRACKER_STRONG_CHARS:
                        source_map[tgt_norm] = char
    for source_key_str in failed:
        relations[source_key_str] = {}
    return relations


def prepare_tracker_relations_index(
    config: ConfigManager, project_root: str, path_to_key_info: Dict[str, KeyInfo]
) -> int:
    """Build every source's tracker relations once (upfront). Returns the source count."""
    global _tracker_relations_index, _tracker_relations_owner
    import time as _time

    from cline_utils.dependency_system.utils.tracker_utils import find_all_tracker_paths

    start = _time.perf_counter()
    _tracker_relations_index, _tracker_relations_owner = None, None
    tracker_paths = find_all_tracker_paths(config, project_root)
    key_str_to_ki = {ki.key_string: ki for ki in path_to_key_info.values()}
    index = _collect_tracker_relations(tracker_paths, key_str_to_ki)
    _tracker_relations_index, _tracker_relations_owner = index, id(path_to_key_info)
    logger.info(
        f"Tracker relations: indexed {len(index)} source keys from "
        f"{len(tracker_paths)} trackers in {_time.perf_counter() - start:.2f}s."
    )
    return len(index)


def _tracker_relations_for(
    source_key_str: str,
    config: ConfigManager,
    project_root: str,
    path_to_key_info: Dict[str, KeyInfo],
) -> Dict[str, str]:
    """The upfront index entry, or (nothing prepared for this key map) the same
    collection for this one source."""
    if (
        _tracker_relations_index is not None
        and _tracker_relations_owner == id(path_to_key_info)
    ):
        return _tracker_relations_index.get(source_key_str, {})
    from cline_utils.dependency_system.utils.tracker_utils import find_all_tracker_paths

    key_str_to_ki = {ki.key_string: ki for ki in path_to_key_info.values()}
    return _collect_tracker_relations(
        find_all_tracker_paths(config, project_root),
        key_str_to_ki,
        only_source=source_key_str,
    ).get(source_key_str, {})


def _skip_pairs_for(
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]],
) -> frozenset:
    """Every pair (both orientations) that existing trackers mark 'x' or 'n', built
    once per existing_state object: should_skip_suggestion as set lookups."""
    global _skip_pairs_key, _skip_pairs
    if not existing_state:
        return frozenset()
    key = (id(existing_state), len(existing_state))
    if _skip_pairs_key == key:
        return _skip_pairs
    with _skip_pairs_lock:
        if _skip_pairs_key != key:
            pairs: Set[Tuple[str, str]] = set()
            for (a, b), res in existing_state.items():
                if res and res[0] in ("x", "n"):
                    pairs.add((a, b))
                    pairs.add((b, a))
            _skip_pairs = frozenset(pairs)
            _skip_pairs_key = key
        return _skip_pairs


def _reset_semantic_indexes() -> None:
    global _tracker_relations_index, _tracker_relations_owner, _skip_pairs_key, _skip_pairs
    _tracker_relations_index, _tracker_relations_owner = None, None
    _skip_pairs_key, _skip_pairs = None, frozenset()


def _log_reranker_scans(
    project_root: str, source_path: str, scored: List[Tuple[KeyInfo, float]]
) -> None:
    """Append the pairs the reranker actually scored to reranker_scans.jsonl (the
    history the next runs skip). Unscored pairs stay out, so they are retried."""
    if not scored or not source_path:
        return
    try:
        scans_file = os.path.join(
            project_root, "cline_utils", "dependency_system", "analysis",
            "reranker_scans.jsonl",
        )
        os.makedirs(os.path.dirname(scans_file), exist_ok=True)
        with open(scans_file, "a", encoding="utf-8") as f:
            for cand_ki, cand_conf in scored:
                tgt_path = getattr(cand_ki, "norm_path", None)
                if not tgt_path:
                    logger.warning(f"Skipping scan log entry: missing target for {source_path!r}")
                    continue
                json.dump(
                    {"source": source_path, "target": tgt_path, "confidence": float(cand_conf)},
                    f,
                )
                f.write("\n")
    except Exception as e:
        logger.warning(f"Failed to log reranker scans: {e}")


def suggest_semantic_dependencies_path_based(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    threshold: float,
    project_symbol_map: Optional[Dict[str, Any]] = None,
    shared_scan_counter: Any = None,
    tracked_paths_globally: Optional[Set[str]] = None,
    existing_state: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
) -> List[Tuple[str, str]]:  # Output: List[(target_norm_path, char)]
    # Check global limit FIRST to avoid unnecessary processing
    if shared_scan_counter is not None:
        # Non-blocking check (value access is atomic enough for this optimization)
        if shared_scan_counter.value >= GLOBAL_SCAN_LIMIT:
            # logger.debug("Global reranker scan limit reached (early check). Skipping semantic analysis.")
            return []

    assigned_char_semantic: Optional[str] = None
    config = ConfigManager()
    promotion_threshold = config.get_threshold("reranker_promotion_threshold")
    embeddings_dir_rel = config.get_path(
        "embeddings_dir", "cline_utils/dependency_system/analysis/embeddings"
    )
    embeddings_dir_abs = normalize_path(os.path.join(project_root, embeddings_dir_rel))
    if not os.path.exists(embeddings_dir_abs):
        logger.debug(
            f"Embeddings dir {embeddings_dir_abs} not found. No semantic suggestions for {file_path}."
        )
        return []

    source_key_info = path_to_key_info.get(file_path)
    if not source_key_info or source_key_info.is_directory:
        return []  # Only for files

    # should_skip_suggestion for every target, as lookups in the run-wide set of
    # pairs existing trackers mark 'x' or 'n' (by path, then by key string).
    skip_pairs = _skip_pairs_for(existing_state)
    source_key_str = source_key_info.key_string
    target_key_infos_list: List[KeyInfo] = [
        info
        for info in path_to_key_info.values()
        if not info.is_directory
        and info.norm_path != file_path  # Must be a file and not self
        and (file_path, info.norm_path) not in skip_pairs
        and (source_key_str, info.key_string) not in skip_pairs
    ]

    if not target_key_infos_list:
        return []

    # For calculate_similarity, it needs code_roots and doc_roots (relative to project_root)
    code_roots_rel_list = config.get_code_root_directories()
    doc_roots_rel_list = config.get_doc_directories()

    try:
        from .embedding_manager import (  # Local import to avoid top-level circularity
            calculate_similarity,
            generate_symbol_essence_string,
            preprocess_doc_structure,
        )
    except ImportError:
        logger.error(
            "embedding_manager.calculate_similarity could not be imported. Semantic suggestions disabled."
        )
        return []

    # Load AST verified links and build map early for filtering
    ast_verified_links = _load_ast_verified_links(project_root)
    ast_map = _build_ast_map(file_path, ast_verified_links)

    # Collect all candidates first for potential reranking
    candidates_with_similarity: List[Tuple[KeyInfo, float]] = []

    # One row of the run-wide similarity index (every embedding loaded once, see
    # similarity_index) instead of a calculate_similarity call per target.
    sim_scores: Optional[Dict[str, float]]
    try:
        sim_scores = similarity_index.get_scores(
            source_key_info.key_string, embeddings_dir_abs, project_root, path_to_key_info
        )
    except Exception as e_index:
        sim_scores = None
        logger.warning(
            f"Similarity index unavailable ({e_index}); scoring {file_path} pair by pair."
        )

    for target_ki in target_key_infos_list:
        if sim_scores is not None:
            confidence = sim_scores.get(target_ki.key_string, 0.0)
        else:
            try:
                # calculate_similarity expects key strings (canonical global ones)
                confidence = calculate_similarity(
                    source_key_info.key_string,
                    target_ki.key_string,
                    embeddings_dir_abs,
                    path_to_key_info,
                    project_root,
                    code_roots_rel_list,
                    doc_roots_rel_list,
                )
            except Exception as e_sim_calc:
                confidence = 0.0
                logger.warning(
                    f"Similarity calculation error between '{source_key_info.key_string}' and '{target_ki.key_string}': {e_sim_calc}",
                    exc_info=False,
                )

        # Only collect candidates that meet minimum threshold for potential reranking
        if (
            confidence >= threshold * 0.35
        ):  # Use ~third of threshold to catch potential candidates
            candidates_with_similarity.append((target_ki, confidence))

    # This variable will hold the results from either reranking or the fallback logic
    initial_suggestions: List[Tuple[str, str]] = []

    # Apply reranking if we have multiple candidates and the reranking feature is available
    if len(candidates_with_similarity) > 1:
        try:
            from .embedding_manager import (
                get_instruction_for_relation_type,
                rerank_candidates_scored,
            )
            from .reranker_history_tracker import get_historical_pairs

            # Load historical pairs to avoid re-analyzing same files
            historical_pairs = get_historical_pairs(project_root)

            # Filter candidates before reranking based on dependency character assignments
            # This filtering only applies during reranking to optimize performance
            filtered_candidates: List[Tuple[KeyInfo, float]] = []

            # Tracker relations augment existing_relation checks. They come from the
            # upfront index (prepare_tracker_relations_index) instead of every file
            # re-discovering and re-reading every tracker.
            try:
                tracker_map = _tracker_relations_for(
                    source_key_info.key_string, config, project_root, path_to_key_info
                )
                for tgt_norm, char in tracker_map.items():
                    if not ast_map.get(tgt_norm):
                        ast_map[tgt_norm] = char
            except Exception as e:
                logger.warning(
                    f"Error extracting tracker dependencies for reranking skip: {e}"
                )

            for target_ki, confidence in candidates_with_similarity:
                # Check if already verified with high-level dependency
                target_norm = target_ki.norm_path
                existing_relation = ast_map.get(target_norm)
                if existing_relation in ["<", ">", "x", "d"]:
                    # Skip reranking for already verified relations
                    continue

                # Check if pair is in history
                if (source_key_info.norm_path, target_norm) in historical_pairs:
                    # Skip reranking for historically analyzed pairs
                    continue

                sourcetype = get_file_type(source_key_info.norm_path)
                targettype = get_file_type(target_ki.norm_path)

                # Get relation-specific thresholds - use config thresholds directly
                if sourcetype == "doc" and targettype == "doc":
                    threshold_s = config.get_threshold("doc_similarity")  # 0.65
                elif (sourcetype == "doc" and targettype == "code") or (
                    sourcetype == "code" and targettype == "doc"
                ):
                    threshold_s = config.get_threshold("doc_code_similarity")  # 0.68
                else:  # code-code
                    threshold_s = config.get_threshold("code_similarity")  # 0.7

                # Only include candidates that would get 'S' or 's' assignments (not 'p' or 'n')
                if confidence >= threshold_s:
                    filtered_candidates.append((target_ki, confidence))

            candidates_with_similarity = filtered_candidates
            # logger.debug(
            #     f"Pre-reranking filter: {len(candidates_with_similarity)} candidates remaining from original collection"
            # )

            # Use shared counter if available (preferred for parallel execution)
            if shared_scan_counter is not None:
                with shared_scan_counter.get_lock():
                    current_scan_count = shared_scan_counter.value
            else:
                # Fallback to file-based check (less reliable in parallel)
                scans_file = os.path.join(
                    project_root,
                    "cline_utils",
                    "dependency_system",
                    "analysis",
                    "reranker_scans.jsonl",
                )
                current_scan_count = 0
                if os.path.exists(scans_file):
                    try:
                        with open(scans_file, "r", encoding="utf-8") as f:
                            current_scan_count = sum(1 for line in f if line.strip())
                    except Exception:
                        pass

            remaining_slots = GLOBAL_SCAN_LIMIT - current_scan_count

            # We need at least 2 slots to perform a meaningful rerank (comparison).
            # If we have 0 or 1 slot left, we consider the limit reached.
            if remaining_slots < 2:
                # Force counter to limit to ensure early exit triggers for future files
                if shared_scan_counter is not None:
                    with shared_scan_counter.get_lock():
                        if shared_scan_counter.value < GLOBAL_SCAN_LIMIT:
                            shared_scan_counter.value = GLOBAL_SCAN_LIMIT

                logger.debug(
                    "Global reranker scan limit reached. Skipping reranking for this file."
                )
                top_candidates = []
            else:
                # All candidates (including doc-code) share the same budget
                top_candidates = sorted(
                    candidates_with_similarity, key=lambda x: x[1], reverse=True
                )[:remaining_slots]

            if len(top_candidates) > 1:
                # Update shared counter immediately
                if shared_scan_counter is not None:
                    with shared_scan_counter.get_lock():
                        shared_scan_counter.value += 1

                # Scanned pairs are logged after reranking, and only the ones the
                # reranker actually scored (_log_reranker_scans): a pair logged
                # here before scoring entered history and was skipped for up to
                # 10 runs even when its batch was never scored.

            if len(top_candidates) > 1:
                # Get the source text for reranking
                try:
                    ext = os.path.splitext(file_path)[1].lower()
                    is_doc = ext in [".md", ".txt", ".rst"]

                    # Prepare query text (source file)
                    query_text = ""

                    if is_doc:
                        # For docs, we prefer preprocess_doc_structure or the new doc-aware SES
                        if project_symbol_map and file_path in project_symbol_map:
                            query_text = generate_symbol_essence_string(
                                file_path,
                                project_symbol_map[file_path],
                                symbol_map=project_symbol_map,
                            )
                        else:
                            try:
                                content = read_file_content_safely(file_path)
                                if content is None:
                                    raise Exception("File read failed")
                                query_text = preprocess_doc_structure(content)
                            except Exception:
                                query_text = f"[DOC: {os.path.basename(file_path)}]"
                    elif project_symbol_map and file_path in project_symbol_map:
                        # Code files in symbol map
                        query_text = generate_symbol_essence_string(
                            file_path,
                            project_symbol_map[file_path],
                            symbol_map=project_symbol_map,
                        )
                    else:
                        # Fallback for untracked code or other files
                        try:
                            content = read_file_content_safely(file_path)
                            if content is None:
                                raise Exception("File read failed")
                            query_text = f"[FILE: {os.path.basename(file_path)}]\n{content[:12800]}"
                        except Exception:
                            query_text = f"[FILE: {os.path.basename(file_path)}]"
                except Exception as e:
                    logger.warning(
                        f"Could not read source file {file_path} for reranking: {e}"
                    )
                    # Fallback to original similarity scores if source can't be read
                    # This part can be simplified since the fallback is handled later
                    pass  # Fall through to the original similarity logic
                else:
                    # Prepare candidate texts for reranking
                    candidate_texts: List[str] = []
                    valid_candidates: List[Tuple[KeyInfo, float]] = []

                    for target_ki, original_confidence in top_candidates:
                        try:
                            # Get instruction for this relation type and prepare text
                            instruction = get_instruction_for_relation_type(
                                file_path, target_ki.norm_path
                            )

                            # Read the actual content for reranking
                            # Use SES if available to reduce context size
                            candidate_text = ""

                            target_ext = os.path.splitext(target_ki.norm_path)[
                                1
                            ].lower()
                            target_is_doc = target_ext in [".md", ".txt", ".rst"]

                            if target_is_doc:
                                if (
                                    project_symbol_map
                                    and target_ki.norm_path in project_symbol_map
                                ):
                                    candidate_text = generate_symbol_essence_string(
                                        target_ki.norm_path,
                                        project_symbol_map[target_ki.norm_path],
                                        symbol_map=project_symbol_map,
                                    )
                                else:
                                    try:
                                        content = (
                                            _generate_candidate_text_from_file_cached(
                                                target_ki.norm_path
                                            )
                                        )
                                        if not content:
                                            raise Exception("File read failed")
                                        candidate_text = preprocess_doc_structure(
                                            content
                                        )
                                    except Exception:
                                        candidate_text = f"[DOC: {os.path.basename(target_ki.norm_path)}]"
                            elif (
                                project_symbol_map
                                and target_ki.norm_path in project_symbol_map
                            ):
                                candidate_text = generate_symbol_essence_string(
                                    target_ki.norm_path,
                                    project_symbol_map[target_ki.norm_path],
                                    symbol_map=project_symbol_map,
                                )
                            else:
                                # Fallback for non-code or untracked files
                                try:
                                    content = read_file_content_safely(
                                        target_ki.norm_path
                                    )
                                    if content is None:
                                        raise Exception("File read failed")
                                    # Limit raw content
                                    candidate_text = f"[FILE: {os.path.basename(target_ki.norm_path)}]\n{content[:12800]}"
                                except Exception:
                                    candidate_text = f"[FILE: {os.path.basename(target_ki.norm_path)}]"

                            candidate_text = instruction + candidate_text
                            candidate_texts.append(candidate_text)
                            valid_candidates.append((target_ki, original_confidence))
                        except Exception:
                            # If we can't read candidate content, skip it
                            continue

                    if len(candidate_texts) > 1:
                        # Perform reranking
                        reranked_results, scored_indices = rerank_candidates_scored(
                            query_text,
                            candidate_texts,
                            top_k=min(10, len(candidate_texts)),
                            source_file_path=file_path,
                        )
                        _log_reranker_scans(
                            project_root,
                            source_key_info.norm_path,
                            [
                                valid_candidates[i]
                                for i in sorted(scored_indices)
                                if i < len(valid_candidates)
                            ],
                        )

                        # Process reranked results
                        for _, (candidate_idx, rerank_score) in enumerate(
                            reranked_results
                        ):
                            # Per candidate: a score below every threshold must not
                            # inherit the previous candidate's character.
                            assigned_char_semantic = None
                            if candidate_idx < len(valid_candidates):
                                target_ki, original_confidence = valid_candidates[
                                    candidate_idx
                                ]

                                # Determine relation type for reranking threshold calibration
                                source_type = get_file_type(source_key_info.norm_path)
                                target_type = get_file_type(target_ki.norm_path)

                                # Get relation-specific thresholds for reranking
                                if source_type == "doc" and target_type == "doc":
                                    threshold_S_strong_semantic = config.get_threshold(
                                        "doc_similarity"
                                    )
                                    threshold_s_weak_semantic = (
                                        threshold * 0.8
                                    )  # Slightly lower for doc-doc
                                elif (
                                    source_type == "doc" and target_type == "code"
                                ) or (source_type == "code" and target_type == "doc"):
                                    threshold_S_strong_semantic = config.get_threshold(
                                        "doc_code_similarity"
                                    )
                                    threshold_s_weak_semantic = threshold
                                else:  # code-code
                                    threshold_S_strong_semantic = config.get_threshold(
                                        "code_similarity"
                                    )
                                    threshold_s_weak_semantic = threshold

                                # HIGH CONFIDENCE: Promote to verified dependency character
                                if rerank_score >= promotion_threshold:
                                    direction = _determine_dependency_direction(
                                        source_key_info.norm_path, target_ki.norm_path
                                    )
                                    assigned_char_semantic = direction
                                elif rerank_score >= threshold_S_strong_semantic:
                                    assigned_char_semantic = (
                                        "S"  # Strong semantic match after reranking
                                    )
                                elif rerank_score >= threshold_s_weak_semantic:
                                    assigned_char_semantic = (
                                        "s"  # Weak semantic match after reranking
                                    )

                                if assigned_char_semantic:
                                    initial_suggestions.append(
                                        (target_ki.norm_path, assigned_char_semantic)
                                    )

                        # CORRECTED: After processing reranked results, DO NOT return.
                        # The function will now proceed to the enhancement step below.

        except ImportError as e:
            logger.debug(
                f"Reranking not available ({e}), using original similarity scores"
            )
        except Exception as e:
            logger.warning(f"Reranking error, using original similarity scores: {e}")

    # CORRECTED: This fallback logic now only runs if reranking did not produce results.
    if not initial_suggestions:
        # Process candidates with original similarity scores
        for target_ki, confidence in candidates_with_similarity:
            assigned_char_semantic = None  # per candidate (see the reranked loop)
            # Determine relation type for threshold calibration
            source_type = get_file_type(source_key_info.norm_path)
            target_type = get_file_type(target_ki.norm_path)

            # Get relation-specific thresholds
            if source_type == "doc" and target_type == "doc":
                threshold_S_strong_semantic = config.get_threshold("doc_similarity")
                threshold_s_weak_semantic = (
                    threshold * 0.8
                )  # Slightly lower for doc-doc
            elif (source_type == "doc" and target_type == "code") or (
                source_type == "code" and target_type == "doc"
            ):
                threshold_S_strong_semantic = config.get_threshold(
                    "doc_code_similarity"
                )
                threshold_s_weak_semantic = threshold
            else:  # code-code
                threshold_S_strong_semantic = config.get_threshold("code_similarity")
                threshold_s_weak_semantic = threshold

            # HIGH CONFIDENCE: Promote to verified dependency character
            if confidence >= promotion_threshold:
                direction = _determine_dependency_direction(
                    source_key_info.norm_path, target_ki.norm_path
                )
                assigned_char_semantic = direction
            elif confidence >= threshold_S_strong_semantic:
                assigned_char_semantic = "S"
            elif confidence >= threshold_s_weak_semantic:
                assigned_char_semantic = "s"

            if assigned_char_semantic:
                initial_suggestions.append(
                    (target_ki.norm_path, assigned_char_semantic)
                )

    # Use the refactored enhancement helper
    enhanced_suggestions = _enhance_semantic_suggestions_with_ast_evidence(
        initial_suggestions, ast_map
    )

    return enhanced_suggestions


# --- Helper Functions ---
def combine_suggestions_path_based_with_char_priority(
    suggestions_path_based: List[Tuple[str, str]],  # List[(target_norm_path, char)]
    source_path_for_log: str,
) -> List[Tuple[str, str]]:  # Output: List[(target_norm_path, char)]
    combined_by_path: Dict[str, str] = {}  # target_norm_path -> char
    config = ConfigManager()
    get_priority = config.get_char_priority

    for target_path, char_val in suggestions_path_based:
        if not target_path or target_path == source_path_for_log:
            continue

        current_char_for_path = combined_by_path.get(target_path)
        current_priority_val = (
            get_priority(current_char_for_path) if current_char_for_path else -1
        )
        new_priority_val = get_priority(char_val)

        if new_priority_val > current_priority_val:
            combined_by_path[target_path] = char_val
        # Equal priority: keep the existing character. An opposed '<' and '>' for
        # the same target are two different claims about ONE cell ("source depends
        # on target" vs "target depends on source"), not mutual evidence. A mutual
        # 'x' needs both FILES to report the same character from their own side;
        # that is decided per pair in tracker_io, not here.
    return list(combined_by_path.items())


# --- Dependency Identification Helpers ---


def _convert_python_import_to_paths(
    import_name: str,  # The module part, e.g., "foo" in "from .foo import X", or "os.path"
    source_file_dir: str,  # Absolute, normalized dir of the file containing the import
    project_root: str,  # Absolute, normalized project root
    path_to_key_info: Dict[str, KeyInfo],  # The global map of tracked files
    project_symbol_map: Dict[str, Dict[str, Any]],  # For verifying specific_item_name
    specific_item_name: Optional[str] = None,  # e.g., "X" in "from .foo import X"
    _is_from_import: bool = False,  # Kept for signature, could be used for nuanced logic
    relative_level: int = 0,  # 0 for absolute, 1 for '.', 2 for '..'
    tracked_paths_globally: Optional[Set[str]] = None,
) -> List[
    Tuple[str, bool]
]:  # Returns List[(resolved_module_path, item_verified_in_module_symbols)]

    # --- Ledger check: path resolution is purely structural and independent of
    # specific_item_name, so we cache at the module-path level only.  Symbol
    # verification (specific_item_name) is a cheap in-memory lookup done after
    # the path is found and is NOT stored in the ledger.
    _ledger_key = (import_name, source_file_dir, project_root, relative_level)
    if _ledger_key in _py_import_resolution_ledger:
        return _py_import_resolution_ledger[_ledger_key]

    potential_paths_abs_info: List[Tuple[str, bool]] = []  # (path, item_verified_flag)
    normalized_project_root = normalize_path(project_root)

    candidate_module_file_paths_to_check_in_map: List[str] = []

    if relative_level > 0:
        current_search_base = normalize_path(source_file_dir)
        for _ in range(relative_level - 1):
            parent_dir = os.path.dirname(current_search_base)
            if (
                not parent_dir
                or parent_dir == current_search_base
                or (
                    not parent_dir.startswith(normalized_project_root)
                    and parent_dir != normalized_project_root
                )
            ):
                current_search_base = None
                break
            current_search_base = parent_dir

        if current_search_base:
            module_path_part_rel = (
                import_name.replace(".", os.sep) if import_name else ""
            )
            base_path_for_relative_import = normalize_path(
                os.path.join(current_search_base, module_path_part_rel)
            )
            candidate_module_file_paths_to_check_in_map.append(
                f"{base_path_for_relative_import}.py"
            )
            candidate_module_file_paths_to_check_in_map.append(
                normalize_path(
                    os.path.join(base_path_for_relative_import, "__init__.py")
                )
            )

    elif import_name:  # Absolute import (relative_level == 0)
        fs_like_import_path = import_name.replace(".", os.sep)

        # 1. Try relative to project_root
        base_abs_from_proj_root = normalize_path(
            os.path.join(normalized_project_root, fs_like_import_path)
        )
        candidate_module_file_paths_to_check_in_map.append(
            f"{base_abs_from_proj_root}.py"
        )
        candidate_module_file_paths_to_check_in_map.append(
            normalize_path(os.path.join(base_abs_from_proj_root, "__init__.py"))
        )

        # 2. Try relative to each configured code_root
        try:
            configured_code_roots_rel = ConfigManager().get_code_root_directories()
            for cr_rel in configured_code_roots_rel:
                abs_code_root = normalize_path(
                    os.path.join(normalized_project_root, cr_rel)
                )
                # Check if import_name starts with the code_root's relative name (e.g., "src.")
                # If import_name = "src.module.foo" and cr_rel = "src"
                if import_name.startswith(cr_rel + "."):
                    # The "project_root" case above already handles this correctly because
                    # fs_like_import_path would be "src/module/foo".
                    # os.path.join(project_root, "src/module/foo") is the correct base.
                    # No special action needed here to avoid double prefixing "src/src/...".
                    pass
                else:
                    # Handles imports like "my_module" when "src" is a code root,
                    # looking for "project_root/src/my_module.py"
                    base_abs_from_code_root = normalize_path(
                        os.path.join(abs_code_root, fs_like_import_path)
                    )
                    candidate_module_file_paths_to_check_in_map.append(
                        f"{base_abs_from_code_root}.py"
                    )
                    candidate_module_file_paths_to_check_in_map.append(
                        normalize_path(
                            os.path.join(base_abs_from_code_root, "__init__.py")
                        )
                    )
        except Exception as e_cfg:
            logger.warning(
                f"Error getting code_root_directories for absolute import fallback: {e_cfg}"
            )

        candidate_module_file_paths_to_check_in_map = list(
            set(candidate_module_file_paths_to_check_in_map)
        )  # Deduplicate

    # --- Check generated candidates against path_to_key_info and verify specific_item_name ---
    if tracked_paths_globally is None:
        tracked_paths_globally = tracked_paths_globally or set(path_to_key_info.keys())
    seen_paths: Set[str] = set()
    for p_candidate_str in candidate_module_file_paths_to_check_in_map:
        # p_candidate_str is already a normalized absolute path from the generation logic
        if (
            p_candidate_str in tracked_paths_globally
        ):  # Direct check against tracked files
            if p_candidate_str not in seen_paths:
                item_verified_in_symbols = (
                    True  # Default to true if no specific item to check
                )
                if specific_item_name:
                    module_symbols = project_symbol_map.get(p_candidate_str, {})
                    is_defined = _has_symbol_name(
                        module_symbols,
                        specific_item_name,
                        ["functions", "classes", "globals_defined"],
                    )

                    if (
                        not is_defined
                        and os.path.basename(p_candidate_str) == "__init__.py"
                    ):
                        # Check for re-exported submodule/subpackage if the resolved module is an __init__.py
                        package_dir_of_init = os.path.dirname(p_candidate_str)
                        potential_submodule_file = normalize_path(
                            os.path.join(
                                package_dir_of_init, specific_item_name + ".py"
                            )
                        )
                        potential_subpackage_init = normalize_path(
                            os.path.join(
                                package_dir_of_init, specific_item_name, "__init__.py"
                            )
                        )

                        # Crucially, check if these potential submodule paths are ALSO tracked
                        if (
                            potential_submodule_file in tracked_paths_globally
                            or potential_subpackage_init in tracked_paths_globally
                        ):
                            is_defined = True

                    if not is_defined:
                        item_verified_in_symbols = False

                potential_paths_abs_info.append(
                    (p_candidate_str, item_verified_in_symbols)
                )
                seen_paths.add(p_candidate_str)

                # For absolute imports, Python's import machinery typically stops at the first match.
                # We replicate this behavior by returning after the first successful resolution.
                if relative_level == 0 and potential_paths_abs_info:
                    _py_import_resolution_ledger[_ledger_key] = potential_paths_abs_info
                    return potential_paths_abs_info  # Return immediately

    # Store in ledger before returning so future files with the same import skip work.
    _py_import_resolution_ledger[_ledger_key] = potential_paths_abs_info
    return potential_paths_abs_info


# ---


def _identify_python_dependencies(
    source_path: str,
    source_analysis: Dict[str, Any],
    _file_analysis_results: Dict[str, Dict[str, Any]],
    project_root: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_symbol_map: Dict[str, Dict[str, Any]],
    tracked_paths_globally: Set[str],
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:  # MODIFIED return type
    dependencies_paths: List[Tuple[str, str]] = []
    raw_ast_links: List[Dict[str, str]] = []  # NEW: For collecting AST-derived links

    imports_in_source = source_analysis.get("imports", [])
    source_dir_norm = os.path.dirname(source_path)

    for import_item in imports_in_source:
        if isinstance(import_item, dict):
            import_dict = cast(Dict[str, Any], import_item)
            import_name_str_from_ast = cast(str, import_dict.get("path", ""))
        else:
            import_name_str_from_ast = cast(str, import_item)

        if not import_name_str_from_ast:
            continue

        temp_import_name = import_name_str_from_ast
        level_for_convert = 0
        while temp_import_name.startswith("."):
            level_for_convert += 1
            temp_import_name = temp_import_name[1:]
        module_to_resolve_for_convert = temp_import_name

        #  Pass tracked_paths_globally to avoid O(N) reconstruction
        # Impact: Significantly reduces time complexity in parallel workers from O(N * M) to O(M)
        # Measurement: Check task processing times in _handle_dependency_task
        resolved_path_infos = _convert_python_import_to_paths(
            import_name=module_to_resolve_for_convert,
            source_file_dir=source_dir_norm,
            project_root=project_root,
            path_to_key_info=path_to_key_info,
            project_symbol_map=project_symbol_map,
            specific_item_name=None,
            _is_from_import=True,
            relative_level=level_for_convert,
            tracked_paths_globally=tracked_paths_globally,
        )
        for path_abs_val, item_verified_flag in resolved_path_infos:
            if path_abs_val in tracked_paths_globally and path_abs_val != source_path:
                dep_char = "<"  # Explicit imports are a direct dependency
                dependencies_paths.append((path_abs_val, dep_char))

                # NEW: Collect this resolved import as an AST-verified link
                raw_ast_links.append(
                    {
                        "source_path": source_path,
                        "target_path": path_abs_val,
                        "char": dep_char,
                        "reason": f"ExplicitImport/{import_name_str_from_ast}",
                    }
                )
                # ---

                break
    return list(set(dependencies_paths)), list(raw_ast_links)


# Module-resolution candidates, following TypeScript: a `.js`-family specifier maps to
# its TS source first (`./x.js` -> x.ts, x.tsx, x.d.ts, then x.js, x.jsx), so compiled
# output beside a source file is never chosen over the source.
_JS_SOURCE_SUBSTITUTIONS: Dict[str, Tuple[str, ...]] = {
    ".js": (".ts", ".tsx", ".d.ts", ".js", ".jsx"),
    ".jsx": (".tsx", ".d.ts", ".jsx"),
    ".mjs": (".mts", ".d.mts", ".mjs"),
    ".cjs": (".cts", ".d.cts", ".cjs"),
}
# Extensions that name the target file itself. Any other "extension" is part of a
# dotted basename (`./x.browser`, `./foo.service`) and gets the probe extensions appended.
_JS_EXACT_EXTENSIONS = (
    ".ts", ".tsx", ".mts", ".cts", ".json",
    ".css", ".scss", ".sass", ".less", ".svg", ".png", ".jpg", ".jpeg", ".gif",
    ".webp", ".vue", ".svelte", ".html", ".wasm", ".node",
)
_JS_PROBE_EXTENSIONS = (".ts", ".tsx", ".d.ts", ".js", ".jsx")


def _js_resolution_candidates(base_path: str) -> List[str]:
    """Candidate files for an absolute, not yet extension-resolved import base path."""
    lower = base_path.lower()
    if lower.endswith((".d.ts", ".d.mts", ".d.cts")):
        return [base_path]
    stem, ext = os.path.splitext(base_path)
    ext = ext.lower()
    if ext in _JS_SOURCE_SUBSTITUTIONS:
        return [stem + e for e in _JS_SOURCE_SUBSTITUTIONS[ext]]
    if ext in _JS_EXACT_EXTENSIONS:
        return [base_path]
    return [base_path + e for e in _JS_PROBE_EXTENSIONS] + [
        f"{base_path}/index{e}" for e in _JS_PROBE_EXTENSIONS
    ]


def _first_tracked_js_candidate(base_path: str, tracked_paths: Set[str]) -> Optional[str]:
    for candidate in _js_resolution_candidates(base_path):
        candidate = normalize_path(candidate)
        if candidate in tracked_paths:
            return candidate
    return None


def _js_resolution_contexts(
    tsconfig_info: Any,
) -> Tuple[Tuple[str, ...], List[Tuple[Optional[str], Dict[str, Any], Optional[str]]]]:
    """
    Normalizes tsconfig input to (config_paths, [(base_url, paths, paths_base), ...]).
    Accepts the list from _tsconfig_for_file (merged configs, js_tsconfig.load_tsconfig)
    or the legacy single (config_path, raw_config) tuple from _find_and_parse_tsconfig.
    """
    if not tsconfig_info:
        return (), []
    if (
        isinstance(tsconfig_info, tuple)
        and len(tsconfig_info) == 2
        and isinstance(tsconfig_info[0], str)
    ):
        entries = [tsconfig_info]
    else:
        entries = list(tsconfig_info)
    contexts: List[Tuple[Optional[str], Dict[str, Any], Optional[str]]] = []
    for config_path, config in entries:
        if "paths_base" in config:  # merged config
            contexts.append((config.get("base_url"), config.get("paths") or {}, config.get("paths_base")))
        else:  # raw tsconfig dict
            options = config.get("compilerOptions", {}) or {}
            config_dir = os.path.dirname(config_path)
            base_url = (
                normalize_path(os.path.join(config_dir, options["baseUrl"]))
                if isinstance(options.get("baseUrl"), str)
                else None
            )
            contexts.append((base_url, options.get("paths") or {}, base_url or config_dir))
    return tuple(normalize_path(p) for p, _ in entries), contexts


def _match_tsconfig_paths(
    import_path: str, paths: Dict[str, Any]
) -> Optional[Tuple[List[str], str]]:
    """TypeScript's `paths` matching: an exact key wins, else the wildcard pattern with
    the longest prefix. Returns (mappings, captured_text) or None."""
    exact = paths.get(import_path)
    if isinstance(exact, list) and "*" not in import_path:
        return [m for m in exact if isinstance(m, str)], ""
    best: Optional[Tuple[str, str, List[str]]] = None
    for pattern, mappings in paths.items():
        if pattern.count("*") != 1 or not isinstance(mappings, list):
            continue
        prefix, suffix = pattern.split("*")
        if (
            import_path.startswith(prefix)
            and import_path.endswith(suffix)
            and len(import_path) >= len(prefix) + len(suffix)
            and (best is None or len(prefix) > len(best[0]))
        ):
            best = (prefix, suffix, [m for m in mappings if isinstance(m, str)])
    if best is None:
        return None
    prefix, suffix, mappings = best
    return mappings, import_path[len(prefix) : len(import_path) - len(suffix)]


def _resolve_js_import_path(
    import_path: str,
    source_dir: str,
    project_root: str,
    path_to_key_info: Dict[str, KeyInfo],
    tsconfig_info: Any,
    tracked_paths_globally: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    Resolves a JS/TS import specifier to a tracked file path.

    Relative specifiers resolve against source_dir. Non-relative ones go through each
    applicable tsconfig in order (`paths`, then `baseUrl`); packages stay unresolved.
    Candidate files follow TypeScript (see _js_resolution_candidates).
    """
    if tracked_paths_globally is None:
        tracked_paths_globally = tracked_paths_globally or set(path_to_key_info.keys())

    config_paths, contexts = _js_resolution_contexts(tsconfig_info)
    # --- Ledger check: JS import resolution depends on the import string, the
    # source directory, the project root, and the applicable tsconfig files.
    _js_ledger_key = (import_path, source_dir, project_root, config_paths)
    if _js_ledger_key in _js_import_resolution_ledger:
        return _js_import_resolution_ledger[_js_ledger_key]

    candidate: Optional[str] = None
    if import_path.startswith("."):
        base_path = normalize_path(os.path.abspath(os.path.join(source_dir, import_path)))
        candidate = _first_tracked_js_candidate(base_path, tracked_paths_globally)
    else:
        for base_url, paths, paths_base in contexts:
            matched = _match_tsconfig_paths(import_path, paths) if paths and paths_base else None
            if matched:
                mappings, captured = matched
                for mapping in mappings:
                    candidate_base = normalize_path(
                        os.path.join(paths_base, mapping.replace("*", captured))
                    )
                    candidate = _first_tracked_js_candidate(candidate_base, tracked_paths_globally)
                    if candidate:
                        break
            # As in TypeScript, baseUrl is only consulted when no `paths` pattern matched.
            elif base_url:
                candidate = _first_tracked_js_candidate(
                    normalize_path(os.path.join(base_url, import_path)), tracked_paths_globally
                )
            if candidate:
                break
        if not candidate:  # bundler aliases, then workspace packages
            for base in js_package_resolution.resolution_bases(import_path, source_dir, project_root):
                candidate = _first_tracked_js_candidate(base, tracked_paths_globally)
                if candidate:
                    break

    # Misses are stored too, so repeated unresolvable paths are skipped immediately.
    _js_import_resolution_ledger[_js_ledger_key] = candidate
    return candidate


def _identify_markdown_dependencies(
    source_path: str,
    source_analysis: Dict[str, Any],
    _file_analyses: Dict[str, Dict[str, Any]],
    project_root: str,
    path_to_key_info: Dict[str, KeyInfo],
    tracked_paths_globally: Set[str],
) -> List[Tuple[str, str]]:
    dependencies_paths: List[Tuple[str, str]] = []
    links_in_source = source_analysis.get("links", [])
    source_dir_norm = os.path.dirname(source_path)

    norm_project_root = normalize_path(project_root)
    for link_item in links_in_source:
        url_val = link_item.get("url", "")
        if not url_val or url_val.startswith(
            ("#", "mailto:", "tel:", "http:", "https:", "//", "data:")
        ):
            continue
        url_cleaned_val = url_val.split("#")[0].split("?")[0]
        if not url_cleaned_val:
            continue
        if os.path.isabs(url_cleaned_val):
            logger.debug(
                f"MD Link: Skipping absolute-looking URL '{url_cleaned_val}' in '{source_path}'."
            )
            continue
        resolved_base_path_abs = normalize_path(
            os.path.abspath(os.path.join(source_dir_norm, url_cleaned_val))
        )

        # Ensure resolved path is within the project
        if not resolved_base_path_abs.startswith(norm_project_root):
            logger.debug(
                f"MD Link: Resolved path '{resolved_base_path_abs}' for link '{url_cleaned_val}' in '{source_path}' is outside project. Skipping."
            )
            continue
        possible_target_paths_check = [resolved_base_path_abs]
        _base_name_md, base_ext_md = os.path.splitext(resolved_base_path_abs)
        if not base_ext_md or os.path.isdir(resolved_base_path_abs):
            possible_target_paths_check.extend(
                [
                    f"{resolved_base_path_abs}.md",
                    f"{resolved_base_path_abs}.rst",
                    normalize_path(os.path.join(resolved_base_path_abs, "index.md")),
                    normalize_path(os.path.join(resolved_base_path_abs, "README.md")),
                ]
            )
        found_target_path_md: Optional[str] = None
        for target_path_try in possible_target_paths_check:
            if (
                os.path.isfile(target_path_try)
                and target_path_try in tracked_paths_globally
            ):
                found_target_path_md = target_path_try
                break
        if found_target_path_md and found_target_path_md != source_path:
            dependencies_paths.append((found_target_path_md, "d"))
    return list(set(dependencies_paths))


def _identify_html_dependencies(
    source_path: str,
    source_analysis: Dict[str, Any],
    _file_analyses: Dict[str, Dict[str, Any]],
    project_root: str,
    path_to_key_info: Dict[str, KeyInfo],
    tracked_paths_globally: Set[str],
) -> List[Tuple[str, str]]:
    dependencies_paths: List[Tuple[str, str]] = []
    source_dir_norm = os.path.dirname(source_path)

    norm_project_root = normalize_path(project_root)

    # --- ADDED: Get doc_roots from ConfigManager ---
    config = ConfigManager()
    # get_doc_directories returns relative paths, convert them to absolute
    abs_doc_roots = [
        normalize_path(os.path.join(norm_project_root, dr))
        for dr in config.get_doc_directories()
    ]
    if not abs_doc_roots:  # Fallback to project_root if no doc_roots are configured
        abs_doc_roots = [norm_project_root]
    urls_to_check_html: List[Tuple[Optional[str], str]] = []
    for link_item in source_analysis.get("links", []):
        urls_to_check_html.append((link_item.get("url"), "link"))
    for script_item in source_analysis.get("scripts", []):
        urls_to_check_html.append((script_item.get("url"), "script"))
    for style_item in source_analysis.get("stylesheets", []):
        urls_to_check_html.append((style_item.get("url"), "style"))
    for img_item in source_analysis.get("images", []):
        urls_to_check_html.append((img_item.get("url"), "image"))
    for url_val_html, resource_type_hint_html in urls_to_check_html:
        if not url_val_html or url_val_html.startswith(
            ("#", "mailto:", "tel:", "http:", "https:", "//", "data:")
        ):
            continue
        url_cleaned_html = url_val_html.split("#")[0].split("?")[0]
        if not url_cleaned_html:
            continue
        resolved_path_abs_html: Optional[str] = None
        if url_cleaned_html.startswith("/"):
            # --- MODIFIED: Try resolving against each doc_root ---
            path_relative_to_root = url_cleaned_html.lstrip("/")
            for doc_root_abs in abs_doc_roots:
                potential_path = normalize_path(
                    os.path.abspath(os.path.join(doc_root_abs, path_relative_to_root))
                )
                if potential_path.startswith(norm_project_root) and os.path.isfile(
                    potential_path
                ):
                    resolved_path_abs_html = potential_path
                    logger.debug(
                        f"HTML Link: Root-relative '{url_cleaned_html}' resolved to '{resolved_path_abs_html}' via doc_root '{doc_root_abs}'."
                    )
                    break
            if not resolved_path_abs_html:
                logger.debug(
                    f"HTML Link: Root-relative '{url_cleaned_html}' in '{source_path}' could not be resolved against configured doc_roots: {abs_doc_roots}."
                )
        else:
            # Relative to the current HTML file's directory
            potential_path_rel = normalize_path(
                os.path.abspath(os.path.join(source_dir_norm, url_cleaned_html))
            )
            if potential_path_rel.startswith(norm_project_root) and os.path.isfile(
                potential_path_rel
            ):  # Ensure within project
                resolved_path_abs_html = potential_path_rel
            else:
                logger.debug(
                    f"HTML Link: Relative '{url_cleaned_html}' in '{source_path}' resolved to '{potential_path_rel}' which is outside project or not a file."
                )
        if not resolved_path_abs_html:
            logger.debug(
                f"HTML Link: Path for link '{url_val_html}' in '{source_path}' could not be resolved to an existing project file."
            )
            continue

        # Ensure it's not outside the project root (double check, though resolution logic should handle it)
        if not resolved_path_abs_html.startswith(norm_project_root):
            logger.debug(
                f"HTML Link: Resolved path '{resolved_path_abs_html}' for link '{url_val_html}' in '{source_path}' is outside project. Skipping."
            )
            continue
        if (
            resolved_path_abs_html in tracked_paths_globally
            and resolved_path_abs_html != source_path
        ):
            dep_char_html = "d"
            target_ext_html = os.path.splitext(resolved_path_abs_html)[1].lower()
            if resource_type_hint_html == "style" or target_ext_html == ".css":
                dep_char_html = "d"
            elif resource_type_hint_html == "script" or target_ext_html in [
                ".js",
                ".ts",
                ".tsx",
                ".mjs",
                ".cjs",
            ]:
                dep_char_html = "d"
            elif resource_type_hint_html == "link" and target_ext_html in [
                ".html",
                ".htm",
                ".md",
                ".rst",
            ]:
                dep_char_html = "d"
            elif resource_type_hint_html == "image" and target_ext_html in [
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".svg",
                ".webp",
            ]:
                dep_char_html = "d"
            dependencies_paths.append((resolved_path_abs_html, dep_char_html))
    return list(set(dependencies_paths))


def _identify_css_dependencies(
    source_path: str,
    source_analysis: Dict[str, Any],
    _file_analyses: Dict[str, Dict[str, Any]],
    project_root: str,
    path_to_key_info: Dict[str, KeyInfo],
    tracked_paths_globally: Set[str],
) -> List[Tuple[str, str]]:
    dependencies_paths: List[Tuple[str, str]] = []
    imports_in_css = source_analysis.get("imports", [])
    source_dir_norm = os.path.dirname(source_path)

    norm_project_root = normalize_path(project_root)
    for import_item_css in imports_in_css:
        url_val_css = import_item_css.get("url", "")  # CSS @import url(...)
        if not url_val_css or url_val_css.startswith(
            ("#", "http:", "https:", "//", "data:")
        ):
            continue
        url_val_css = url_val_css.strip("'\"")
        url_cleaned_css = url_val_css.split("#")[0].split("?")[0]
        if not url_cleaned_css:
            continue
        resolved_path_abs_css = normalize_path(
            os.path.abspath(os.path.join(source_dir_norm, url_cleaned_css))
        )
        if not resolved_path_abs_css.startswith(norm_project_root):
            logger.debug(
                f"CSS Import: Resolved path '{resolved_path_abs_css}' for import '{url_val_css}' in '{source_path}' is outside project. Skipping."
            )
            continue
        if (
            os.path.isfile(resolved_path_abs_css)
            and resolved_path_abs_css in tracked_paths_globally
            and resolved_path_abs_css != source_path
        ):
            dependencies_paths.append((resolved_path_abs_css, "<"))
    return list(set(dependencies_paths))


# --- END OF FILE dependency_suggester.py ---
