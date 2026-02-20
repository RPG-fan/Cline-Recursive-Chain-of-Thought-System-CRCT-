# analysis/dependency_analyzer.py

"""
Analysis module for dependency detection and code analysis.
Parses files to identify imports, function calls, and other dependency indicators.
"""

import ast
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

# Global tree-sitter variables and their corresponding language and parser instances.
# These are imported and initialized directly at module load time.
import tree_sitter
import tree_sitter_css as tscss
import tree_sitter_html as tshtml
import tree_sitter_javascript as tsjavascript
import tree_sitter_language_pack as tslp
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript

Language = tree_sitter.Language
Parser = tree_sitter.Parser
Query = tree_sitter.Query
QueryCursor = tree_sitter.QueryCursor
Node = tree_sitter.Node

JS_LANGUAGE = Language(tsjavascript.language())
CSS_LANGUAGE = Language(tscss.language())
HTML_LANGUAGE = Language(tshtml.language())
TS_LANGUAGE = Language(tstypescript.language_typescript())
TSX_LANGUAGE = Language(tstypescript.language_tsx())
PY_LANGUAGE = Language(tspython.language())

# Extended languages via language pack
JSON_LANGUAGE = tslp.get_language("json")
MARKDOWN_LANGUAGE = tslp.get_language("markdown")
MARKDOWN_INLINE_LANGUAGE = tslp.get_language("markdown_inline")
SVELTE_LANGUAGE = tslp.get_language("svelte")
SQL_LANGUAGE = tslp.get_language("sql")

# --- FIX (MAJOR): Remove global Parser instances. They are NOT thread-safe. ---
# Parsers will now be created locally within each analysis function.
# CSS_PARSER = Parser(CSS_LANGUAGE) # REMOVED
# HTML_PARSER = Parser(HTML_LANGUAGE) # REMOVED
# JS_PARSER = Parser(JS_LANGUAGE) # REMOVED
# TS_PARSER = Parser(TS_LANGUAGE) # REMOVED
# TSX_PARSER = Parser(TSX_LANGUAGE) # REMOVED

from cline_utils.dependency_system.utils.cache_manager import cache_manager, cached
from cline_utils.dependency_system.utils.cache_manager import (
    get_project_root_cached as get_project_root,
)
from cline_utils.dependency_system.utils.cache_manager import (
    invalidate_dependent_entries,
)
from cline_utils.dependency_system.utils.cache_manager import (
    normalize_path_cached as normalize_path,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager

# Import only from utils, core, and io layers
from cline_utils.dependency_system.utils.path_utils import (
    get_file_type as util_get_file_type,
)
from cline_utils.dependency_system.utils.path_utils import is_subpath

logger = logging.getLogger(__name__)

# Regular expressions
PYTHON_IMPORT_PATTERN = re.compile(
    r"^\s*from\s+([.\w]+)\s+import\s+(?:\(|\*|\w+)", re.MULTILINE
)
PYTHON_IMPORT_MODULE_PATTERN = re.compile(
    r"^\s*import\s+([.\w]+(?:\s*,\s*[.\w]+)*)", re.MULTILINE
)
JAVASCRIPT_IMPORT_PATTERN = re.compile(
    r'import(?:["\'\s]*(?:[\w*{}\n\r\s,]+)from\s*)?["\']([^"\']+)["\']'
    + r'|\brequire\s*\(\s*["\']([^"\']+)["\']\s*\)'
    + r'|import\s*\(\s*["\']([^"\']+)["\']\s*\)'
)
MARKDOWN_LINK_PATTERN = re.compile(r"\[(?:[^\]]+)\]\(([^)]+)\)")
HTML_A_HREF_PATTERN = re.compile(
    r'<a\s+(?:[^>]*?\s+)?href=(["\'])(?P<url>[^"\']+?)\1', re.IGNORECASE
)
HTML_SCRIPT_SRC_PATTERN = re.compile(
    r'<script\s+(?:[^>]*?\s+)?src=(["\'])(?P<url>[^"\']+?)\1', re.IGNORECASE
)
HTML_LINK_HREF_PATTERN = re.compile(
    r'<link\s+(?:[^>]*?\s+)?href=(["\'])(?P<url>[^"\']+?)\1', re.IGNORECASE
)
HTML_IMG_SRC_PATTERN = re.compile(
    r'<img\s+(?:[^>]*?\s+)?src=(["\'])(?P<url>[^"\']+?)\1', re.IGNORECASE
)
CSS_IMPORT_PATTERN = re.compile(
    r'@import\s+(?:url\s*\(\s*)?["\']?([^"\')\s]+[^"\')]*?)["\']?(?:\s*\))?;',
    re.IGNORECASE,
)

# Common external libraries/modules to ignore calls from
IGNORED_CALL_SOURCES = {
    "logger",
    "logging",
    "os",
    "sys",
    "json",
    "re",
    "math",
    "datetime",
    "time",
    "random",
    "subprocess",
    "shutil",
    "pathlib",
    "typing",
    "argparse",
    "psycopg",
    "psycopg2",
    "asyncio",
    "sql",
    "Literal",
}

# Common generic method names to ignore if source is unknown or generic
GENERIC_CALL_NAMES = {
    "get",
    "set",
    "update",
    "append",
    "extend",
    "pop",
    "remove",
    "clear",
    "copy",
    "keys",
    "values",
    "items",
    "split",
    "join",
    "strip",
    "replace",
    "format",
    "startswith",
    "endswith",
    "lower",
    "upper",
    "find",
    "count",
    "index",
    "sort",
    "reverse",
    "close",
    "read",
    "write",
    "open",
    "create",
    "delete",
    "save",
    "load",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
}

# Keep a small blacklist for things that are hard to resolve (like logger variables)
# or are definitely external but might not be imported explicitly in a way we catch.
MANUAL_IGNORED_SOURCES = {"logger", "console"}


@cached("is_internal_module", key_func=lambda module_name: f"is_internal:{module_name}")
def _is_internal_module(module_name: str) -> bool:
    """
    Checks if a module name corresponds to an internal project file.
    Uses ConfigManager to get code roots.
    """
    if not module_name:
        return False

    # Built-ins are external
    if module_name in sys.builtin_module_names:
        return False

    config_manager = ConfigManager()
    code_roots = config_manager.get_code_root_directories()

    # Split to get top-level package (e.g. 'psycopg.sql' -> 'psycopg')
    root_module = module_name.split(".")[0]

    for root in code_roots:
        # Check for directory (package)
        if os.path.isdir(os.path.join(root, root_module)):
            return True
        # Check for .py file (module)
        if os.path.isfile(os.path.join(root, root_module + ".py")):
            return True

    return False


def _is_useful_call(
    target_name: str,
    potential_source: Optional[str],
    imports_map: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Determines if a call is 'useful' for dependency analysis.
    Filters out calls to external libraries based on path resolution and generic methods.

    Args:
        target_name: The name of the function/method called.
        potential_source: The variable/module the method is called on (e.g. 'logger' in 'logger.info').
        imports_map: A dictionary mapping local names to full module paths (e.g. {'plt': 'matplotlib.pyplot'}).
    """
    if not target_name:
        return False

    # 1. Check Manual Blacklist (fastest)
    if potential_source in MANUAL_IGNORED_SOURCES:
        return False

    # 2. Check Path-Based Resolution (Strict Filtering)
    if potential_source:
        # If we have an imports map, try to resolve the source
        resolved_source = potential_source
        if imports_map and potential_source in imports_map:
            resolved_source = imports_map[potential_source]

        # --- STRICT FILTERING ---
        # If it looks like a module (contains dot or is in imports), check if it's internal.
        # If it's NOT internal, we filter it out.

        is_imported = imports_map and potential_source in imports_map
        is_dotted = "." in potential_source

        if is_imported or is_dotted:
            if not _is_internal_module(resolved_source):
                return False

        # Also check against known external libraries explicitly to be safe
        root_source = resolved_source.split(".")[0]
        if root_source in IGNORED_CALL_SOURCES:
            return False

    # 3. Check Generic Methods
    # Strategy: If generic name AND (source is None OR source is generic-looking variable), maybe filter?
    if target_name.split(".")[-1] in GENERIC_CALL_NAMES:
        if not potential_source:
            pass
        elif potential_source in {
            "str",
            "dict",
            "list",
            "set",
            "tuple",
            "int",
            "float",
            "object",
        }:
            return False

    return True


def _consolidate_list_of_dicts(
    items: List[Dict[str, Any]], group_by_keys: List[str]
) -> List[Dict[str, Any]]:
    """
    Consolidates a list of dictionaries by grouping them by the specified keys.
    Merges 'line' fields into a list of integers.
    Preserves other fields from the first occurrence.
    """
    if not items:
        return []

    # Helper to get key tuple
    def get_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
        return tuple(item.get(k) for k in group_by_keys)

    grouped: Dict[Tuple[Any, ...], List[int]] = {}
    first_items: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for item in items:
        # Skip if any grouping key is missing (optional, but safer for strict grouping)
        # Or treat missing as None (which item.get does).
        key = get_key(item)
        line = item.get("line", -1)

        if key not in grouped:
            grouped[key] = []
            first_items[key] = item

        if isinstance(line, list):
            grouped[key].extend(cast(List[int], line))
        else:
            grouped[key].append(line)

    consolidated: List[Dict[str, Any]] = []
    for key, lines in grouped.items():
        unique_lines = sorted(list(set(lines)))

        # Create new item based on the first one found
        new_item = first_items[key].copy()
        new_item["line"] = unique_lines
        consolidated.append(new_item)

    return consolidated


def _get_ts_node_text(node: Any, content_bytes: bytes) -> str:
    """Safely decodes the text of a tree-sitter node."""
    return content_bytes[node.start_byte : node.end_byte].decode(
        "utf8", errors="ignore"
    )


def _normalize_sql_identifier_text(text: str) -> str:
    """
    Normalizes an SQL identifier string.
    Handles schema qualification and common quoting styles.
    """
    text = text.strip()

    # Handle schema-qualified names: schema.table -> table
    if "." in text:
        parts = text.split(".")
        text = parts[-1].strip()

    # Strip matching quote/bracket pairs
    while len(text) >= 2:
        if (
            (text[0] == '"' and text[-1] == '"')
            or (text[0] == "'" and text[-1] == "'")
            or (text[0] == "`" and text[-1] == "`")
            or (text[0] == "[" and text[-1] == "]")
        ):
            text = text[1:-1]
        else:
            break

    return text.strip()


def _extract_sql_identifier(node: Any, content_bytes: bytes) -> str:
    """
    Robustly extracts an SQL identifier from a node.
    Handles quotes (", ', `, []), schema qualifications, and special characters.
    e.g. "public"."users" -> "users", public.users -> users, [dbo].[items] -> items
    """
    text = _get_ts_node_text(node, content_bytes)
    return _normalize_sql_identifier_text(text)


def _normalize_imports(result: Dict[str, Any]) -> None:
    """Normalizes import structures to a consistent List[Dict[str, Any]]."""
    raw_imports = cast(List[Any], result.get("imports", []) or [])
    norm_imports: List[Dict[str, Any]] = []
    for imp in raw_imports:
        if isinstance(imp, dict):
            imp_dict = cast(Dict[str, Any], imp)
            if "path" in imp_dict:
                norm_imports.append(imp_dict)
            elif "source" in imp_dict:
                rest_dict = {k: v for k, v in imp_dict.items() if k != "source"}
                item = {"path": imp_dict["source"]}
                item.update(rest_dict)
                norm_imports.append(item)
        elif isinstance(imp, str):
            norm_imports.append({"path": imp})
    result["imports"] = norm_imports


# --- Main Analysis Function ---
@cached(
    "file_analysis",
    key_func=lambda file_path, force=False: f"analyze_file:{normalize_path(str(file_path))}:{(os.path.getmtime(str(file_path)) if os.path.exists(str(file_path)) else 0)}:{force}",
    track_path_args=[0],
)
def analyze_file(file_path: str, force: bool = False) -> Dict[str, Any]:
    """
    Analyzes a file to identify dependencies, imports, and other metadata.
    Uses caching based on file path, modification time, and force flag.
    Skips binary files before attempting text-based analysis.
    Python ASTs are stored separately in "ast_cache".
    For JavaScript/TypeScript, 'tree-sitter' ASTs are stored in "ts_ast_cache".

    Args:
        file_path: Path to the file to analyze
        force: If True, bypass the cache for this specific file analysis.
    Returns:
        Dictionary containing analysis results (without AST for Python files) or error/skipped status.
    """
    norm_file_path = normalize_path(file_path)
    if not os.path.exists(norm_file_path) or not os.path.isfile(norm_file_path):
        return {"error": "File not found or not a file", "file_path": norm_file_path}

    config_manager = ConfigManager()
    project_root = get_project_root()
    excluded_dirs_rel = config_manager.get_excluded_dirs()
    # get_excluded_paths() from config_manager now returns a list of absolute normalized paths
    # including resolved file patterns.
    all_excluded_paths_abs = set(config_manager.get_excluded_paths())  # Fetch once
    excluded_extensions = set(config_manager.get_excluded_extensions())

    # Check against pre-normalized absolute excluded paths
    if (
        norm_file_path in all_excluded_paths_abs
        or any(
            is_subpath(norm_file_path, excluded_dir_abs)
            for excluded_dir_abs in {
                normalize_path(os.path.join(project_root, p)) for p in excluded_dirs_rel
            }
        )
        or os.path.splitext(norm_file_path)[1].lower() in excluded_extensions
        or os.path.basename(norm_file_path).endswith("_module.md")
    ):  # Check tracker file name pattern
        logger.debug(f"Skipping analysis of excluded/tracker file: {norm_file_path}")
        return {
            "skipped": True,
            "reason": "Excluded path, extension, or tracker file",
            "file_path": norm_file_path,
        }

    # --- Binary File Check ---
    try:
        with open(norm_file_path, "rb") as f_check_binary:
            # Read a small chunk to check for null bytes, common in many binary files
            # This is a heuristic, not a perfect binary detector.
            if b"\0" in f_check_binary.read(1024):
                logger.debug(f"Skipping analysis of binary file: {norm_file_path}")
                return {
                    "skipped": True,
                    "reason": "Binary file detected",
                    "file_path": norm_file_path,
                    "size": os.path.getsize(norm_file_path),
                }
    except FileNotFoundError:
        return {
            "error": "File disappeared before binary check",
            "file_path": norm_file_path,
        }
    except Exception as e_bin_check:
        logger.warning(
            f"Error during binary check for {norm_file_path}: {e_bin_check}. Proceeding with text analysis attempt."
        )

    try:
        file_type = util_get_file_type(norm_file_path)
        # Initialize with all potential keys to ensure consistent structure
        analysis_result: Dict[str, Any] = {
            "file_path": norm_file_path,
            "file_type": file_type,
            "imports": [],
            "links": [],
            "functions": [],
            "classes": [],
            "calls": [],
            "attribute_accesses": [],
            "inheritance": [],
            "type_references": [],
            "globals_defined": [],
            "exports": [],
            "code_blocks": [],
            "scripts": [],
            "stylesheets": [],
            "images": [],
            "decorators_used": [],
            "exceptions_handled": [],
            "with_contexts_used": [],
        }
        try:
            with open(norm_file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return {
                "error": "File disappeared during analysis",
                "file_path": norm_file_path,
            }
        except UnicodeDecodeError as e:
            logger.warning(
                f"Encoding error reading {norm_file_path} as UTF-8: {e}. File might be non-text or use different encoding."
            )
            return {
                "error": "Encoding error",
                "details": str(e),
                "file_path": norm_file_path,
            }
        except Exception as e:
            logger.error(f"Error reading file {norm_file_path}: {e}", exc_info=True)
            return {
                "error": "File read error",
                "details": str(e),
                "file_path": norm_file_path,
            }

        if file_type == "py":
            _analyze_python_file(norm_file_path, content, analysis_result)
            # --- FIX (MAJOR): Do not pop the AST. Keep it in the result for explicit passing. ---
            # The AST is still cached for other potential uses, but it's no longer removed
            # from the main analysis result.
            ast_object = analysis_result.get("_ast_tree")
            if ast_object:
                ast_cache = cache_manager.get_cache("ast_cache")
                ast_cache.set(norm_file_path, ast_object)

            # --- ADDED: Tree-sitter analysis for Python ---
            ts_result: dict[str, Any] = {
                "file_path": norm_file_path,
                "file_type": file_type,
                "imports": [],
                "functions": [],
                "classes": [],
                "calls": [],
                "globals_defined": [],
                "_ts_tree": None,
            }
            _analyze_python_file_ts(norm_file_path, content, ts_result)

            # Merge tree-sitter results into the main analysis_result (AST preferred)
            _merge_analysis_results(analysis_result, ts_result)

            # Cache the TS tree if present
            ts_tree_object = ts_result.get("_ts_tree")
            if ts_tree_object:
                ts_ast_cache = cache_manager.get_cache("ts_ast_cache")
                ts_ast_cache.set(norm_file_path, ts_tree_object)
            # ----------------------------------------------

        elif file_type == "js":
            # Strict separation of concerns: use JavaScript-specific analyzer only for .js
            _analyze_javascript_file_ts(norm_file_path, content, analysis_result)
            # --- FIX (MAJOR): Do not pop the tree-sitter tree. ---
            ts_tree_object = analysis_result.get("_ts_tree")
            if ts_tree_object:
                ts_ast_cache = cache_manager.get_cache("ts_ast_cache")
                ts_ast_cache.set(norm_file_path, ts_tree_object)

        # (Repeat the fix for all other tree-sitter based analyses)
        elif file_type == "ts":
            # Strict separation of concerns: use TypeScript-specific analyzer only for .ts
            _analyze_typescript_file_ts(norm_file_path, content, analysis_result)
            ts_tree_object = analysis_result.get("_ts_tree")
            if ts_tree_object:
                ts_ast_cache = cache_manager.get_cache("ts_ast_cache")
                ts_ast_cache.set(norm_file_path, ts_tree_object)

        elif file_type == "tsx":
            # Strict separation of concerns: use TSX-specific analyzer only for .tsx
            _analyze_tsx_file_ts(norm_file_path, content, analysis_result)
            ts_tree_object = analysis_result.get("_ts_tree")
            if ts_tree_object:
                ts_ast_cache = cache_manager.get_cache("ts_ast_cache")
                ts_ast_cache.set(norm_file_path, ts_tree_object)

        elif file_type == "md":
            _analyze_markdown_file_ts(norm_file_path, content, analysis_result)
            ts_tree_object = analysis_result.get("_ts_tree")
            if ts_tree_object:
                ts_ast_cache = cache_manager.get_cache("ts_ast_cache")
                ts_ast_cache.set(norm_file_path, ts_tree_object)

        elif file_type == "html":
            _analyze_html_file_ts(norm_file_path, content, analysis_result)
            ts_tree_object = analysis_result.get("_ts_tree")
            if ts_tree_object:
                ts_ast_cache = cache_manager.get_cache("ts_ast_cache")
                ts_ast_cache.set(norm_file_path, ts_tree_object)

        elif file_type == "css":
            _analyze_css_file_ts(norm_file_path, content, analysis_result)
            ts_tree_object = analysis_result.get("_ts_tree")
            if ts_tree_object:
                ts_ast_cache = cache_manager.get_cache("ts_ast_cache")
                ts_ast_cache.set(norm_file_path, ts_tree_object)

        elif file_type == "json":
            _analyze_json_file_ts(norm_file_path, content, analysis_result)
            ts_tree_object = analysis_result.get("_ts_tree")
            if ts_tree_object:
                ts_ast_cache = cache_manager.get_cache("ts_ast_cache")
                ts_ast_cache.set(norm_file_path, ts_tree_object)            

        elif file_type == "svelte":
            _analyze_svelte_file_ts(norm_file_path, content, analysis_result)
            ts_tree_object = analysis_result.get("_ts_tree")
            if ts_tree_object:
                ts_ast_cache = cache_manager.get_cache("ts_ast_cache")
                ts_ast_cache.set(norm_file_path, ts_tree_object)            

        elif file_type == "sql":
            _analyze_sql_file_ts(norm_file_path, content, analysis_result)
            ts_tree_object = analysis_result.get("_ts_tree")
            if ts_tree_object:
                ts_ast_cache = cache_manager.get_cache("ts_ast_cache")
                ts_ast_cache.set(norm_file_path, ts_tree_object)

        # --- ADDED: Consolidate all fields (Global) ---
        # Define grouping keys for each field
        consolidation_map = {
            "calls": ["target_name", "potential_source"],
            "functions": ["name"],
            "classes": ["name"],
            "globals_defined": ["name"],
            "attribute_accesses": ["target_name", "potential_source"],
            "inheritance": ["class_name", "base_class_name"],
            "type_references": ["type_name_str", "context", "target_name"],
            "decorators_used": ["name", "target_type", "target_name"],
            "exceptions_handled": ["type_name_str"],
            "with_contexts_used": ["context_expr_str"],
            "code_blocks": ["language", "content"],  # MD
            "links": ["url"],  # HTML/MD
            "headers": ["level", "text"],  # MD
            "scripts": ["content", "url"],  # HTML/Svelte
            "stylesheets": ["content", "url"],  # HTML/Svelte
            "images": ["url", "src"],  # HTML/MD
            "definitions": ["type", "summary"],  # SQL
            "columns": ["name", "type"],  # SQL
            "relationships": ["source_col", "target_table", "target_col"],  # SQL
            "json_keys": ["path"],  # JSON
            "json_refs": ["key_path", "value"],  # JSON
        }

        # --- FIX: robustly normalize imports from all analyzers ---
        _normalize_imports(analysis_result)

        for field, keys in consolidation_map.items():
            if analysis_result.get(field):
                analysis_result[field] = _consolidate_list_of_dicts(
                    analysis_result[field], keys
                )

        try:
            analysis_result["size"] = os.path.getsize(norm_file_path)
        except FileNotFoundError:
            analysis_result["size"] = -1
        except OSError:
            analysis_result["size"] = -2

        # Emit a normalized cross-language summary used by downstream suggester/linker
        try:
            summary: Dict[str, Any] = {
                "file_path": norm_file_path,
                "file_type": file_type,
                "imports": analysis_result.get("imports", []) or [],
                "exports": analysis_result.get("exports", []) or [],
                "functions": analysis_result.get("functions", []) or [],
                "classes": analysis_result.get("classes", []) or [],
                "calls": analysis_result.get("calls", []) or [],
                "type_references": analysis_result.get("type_references", []) or [],
                # SQL-specific
                "tables_defined": analysis_result.get("tables_defined", []) or [],
                "tables_referenced": analysis_result.get("tables_referenced", []) or [],
                "relationships": analysis_result.get("relationships", []) or [],
            }
            analysis_result["symbol_summary"] = summary
        except Exception as e_summary:
            logger.warning(
                f"Failed to build symbol_summary for {norm_file_path}: {e_summary}"
            )

        # NEW: Minimal link emission for AST-verified links pipeline
        # We convert import paths discovered by analyzers into basic link hints
        # Downstream resolver will resolve to absolute file keys and write to ast_verified_links.json
        try:
            if "ast_verified_links" not in analysis_result:
                analysis_result["ast_verified_links"] = []
            imports_list: List[Dict[str, Any]] = cast(
                List[Dict[str, Any]], analysis_result.get("imports", []) or []
            )
            for imp in imports_list:
                if imp and imp.get("path"):
                    cast(
                        List[Dict[str, Any]], analysis_result["ast_verified_links"]
                    ).append(
                        {
                            "source_file": norm_file_path,
                            "target_spec": imp.get("path"),
                            "line": imp.get("line"),
                            "via": "import",
                            "confidence": 0.9,
                        }
                    )
            # Also emit links from export re-exports like `export ... from 'x'`
            exports_list: List[Dict[str, Any]] = cast(
                List[Dict[str, Any]], analysis_result.get("exports", []) or []
            )
            for ex in exports_list:
                if ex and ex.get("from"):
                    cast(
                        List[Dict[str, Any]], analysis_result["ast_verified_links"]
                    ).append(
                        {
                            "source_file": norm_file_path,
                            "target_spec": ex.get("from"),
                            "line": ex.get("line"),
                            "via": "export_from",
                            "confidence": 0.9,
                        }
                    )
        except Exception as e_links:
            logger.warning(
                f"Failed to emit ast_verified_links for {norm_file_path}: {e_links}"
            )

        return analysis_result  # This result contains _ast_tree and _ts_tree for downstream use
    except Exception as e:
        logger.exception(f"Unexpected error analyzing {norm_file_path}: {e}")
        return {
            "error": "Unexpected analysis error",
            "details": str(e),
            "file_path": norm_file_path,
        }


# --- Analysis Helper Functions ---


def _analyze_python_file(file_path: str, content: str, result: Dict[str, Any]) -> None:
    # Ensure lists are initialized (caller already does this, but good for safety)
    result.setdefault("imports", [])
    result.setdefault("functions", [])
    result.setdefault("classes", [])
    result.setdefault("calls", [])
    result.setdefault("attribute_accesses", [])
    result.setdefault("inheritance", [])
    result.setdefault("type_references", [])
    result.setdefault("globals_defined", [])
    result.setdefault("decorators_used", [])
    result.setdefault("exceptions_handled", [])
    result.setdefault("with_contexts_used", [])
    # --- ADDED: Key for storing the AST tree ---
    result.setdefault("_ast_tree", None)
    # ---

    def _capture_significant_assignment(node: Union[ast.Assign, ast.AnnAssign], result: Dict[str, Any]):
        """Helper to capture significant variable/attribute assignments for logical essence."""
        # Check if there is a value to capture (AnnAssign might not have one)
        value_node = getattr(node, "value", None)
        if value_node is None:
            return

        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            target_name = _get_full_name_str(target)
            if not target_name:
                continue
                
            # User Request: Capture ALL assignments, minimal filtering.
            
            extracted_val: Optional[str] = None
            
            # Generalize extraction using ast.unparse for ANY node type
            try:
                val_repr = ast.unparse(value_node)
                # Collapse whitespace for cleaner storage
                val_repr = " ".join(val_repr.split())
                extracted_val = val_repr
            except Exception:
                # Fallback or ignore if unparse fails
                pass
                    
            if extracted_val:
                if "literal_assignments" not in result:
                    result["literal_assignments"] = []
                # Avoid duplicates
                if not any(a["name"] == target_name and a["value"] == extracted_val for a in result["literal_assignments"]):
                    result["literal_assignments"].append({
                        "name": target_name,
                        "value": extracted_val,
                        "line": node.lineno
                    })
                # USER DEMAND: ALL ASSIGNMENTS. NO EXCEPTIONS. NO LIMITS.

    # _get_full_name_str and _extract_type_names_from_annotation helpers
    def _get_full_name_str(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            # Recursively get the base part
            base = _get_full_name_str(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        if isinstance(node, ast.Subscript):
            base = _get_full_name_str(node.value)
            index_repr = "..."
            slice_node = node.slice
            if isinstance(slice_node, ast.Constant):
                index_repr = repr(slice_node.value)
            elif isinstance(slice_node, ast.Name):
                index_repr = slice_node.id
            elif isinstance(slice_node, ast.Tuple):
                elts_str = ", ".join(
                    [_get_full_name_str(e) or "..." for e in slice_node.elts]
                )
                index_repr = f"Tuple[{elts_str}]"
            elif isinstance(slice_node, ast.Slice):
                lower = _get_full_name_str(slice_node.lower) if slice_node.lower else ""
                upper = _get_full_name_str(slice_node.upper) if slice_node.upper else ""
                step = _get_full_name_str(slice_node.step) if slice_node.step else ""
                index_repr = f"{lower}:{upper}:{step}".rstrip(":")
            # Fallback for ast.Index which wraps the actual slice value in older Python versions
            elif hasattr(slice_node, "value"):
                index_value_node = getattr(slice_node, "value")
                if isinstance(index_value_node, ast.Constant):
                    index_repr = repr(index_value_node.value)
                elif isinstance(index_value_node, ast.Name):
                    index_repr = index_value_node.id
                # Could add more complex slice representations here if needed
            return f"{base}[{index_repr}]" if base else f"[{index_repr}]"
        if isinstance(node, ast.Call):
            base = _get_full_name_str(node.func)
            return f"{base}()" if base else "()"
        if isinstance(node, ast.Constant):
            return repr(node.value)
        return None

    def _get_source_object_str(
        node: ast.AST,
    ) -> Optional[str]:  # Included for completeness
        if isinstance(node, ast.Attribute):
            return _get_full_name_str(node.value)
        if isinstance(node, ast.Call):
            return _get_full_name_str(node.func)
        if isinstance(node, ast.Subscript):
            return _get_full_name_str(node.value)
        return None

    def _extract_type_names_from_annotation(
        annotation_node: Optional[ast.AST],
    ) -> Set[str]:  # Included for completeness
        names: Set[str] = set()
        if not annotation_node:
            return names
        nodes_to_visit = [annotation_node]
        while nodes_to_visit:
            node = nodes_to_visit.pop(0)
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                full_name = _get_full_name_str(node)
                if full_name:
                    names.add(full_name)
            elif isinstance(node, ast.Subscript):
                if node.value:
                    nodes_to_visit.append(node.value)
                current_slice = node.slice
                # For Python < 3.9, slice is often ast.Index(value=actual_slice_node)
                if hasattr(current_slice, "value") and not isinstance(
                    current_slice,
                    (ast.Name, ast.Attribute, ast.Tuple, ast.Constant, ast.BinOp),
                ):
                    current_slice = getattr(current_slice, "value")
                if isinstance(
                    current_slice, (ast.Name, ast.Attribute, ast.Constant, ast.BinOp)
                ):
                    nodes_to_visit.append(current_slice)
                elif isinstance(
                    current_slice, ast.Tuple
                ):  # e.g., (str, int) in Dict[str, int]
                    for elt in current_slice.elts:
                        nodes_to_visit.append(elt)
            elif isinstance(node, ast.Constant) and isinstance(
                node.value, str
            ):  # Forward reference: 'MyClass'
                names.add(node.value)
            elif isinstance(node, ast.BinOp) and isinstance(
                node.op, ast.BitOr
            ):  # For X | Y syntax (Python 3.10+)
                nodes_to_visit.append(node.left)
                nodes_to_visit.append(node.right)
        return names

    result.setdefault("_ast_tree", None)  # Initialize key in result
    tree_obj_for_debug: Optional[ast.AST] = None

    try:
        tree = ast.parse(content, filename=file_path)
        result["_ast_tree"] = tree
        tree_obj_for_debug = tree

        logger.debug(
            f"DEBUG DA: Parsed {file_path}. AST tree assigned to result['_ast_tree']. Type: {type(result['_ast_tree'])}"
        )

        for node_with_parent in ast.walk(tree):
            for child in ast.iter_child_nodes(node_with_parent):
                setattr(child, "_parent", node_with_parent)
        logger.debug(f"DEBUG DA: Parent pointers added for {file_path}.")

        # Pass 1: Populate top-level definitions
        imports_map = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append(alias.name)
                    imports_map[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                relative_prefix = "." * node.level
                full_import_source = f"{relative_prefix}{module_name}"
                result["imports"].append(full_import_source)
                for alias in node.names:
                    full_path = (
                        f"{full_import_source}.{alias.name}"
                        if full_import_source
                        else alias.name
                    )
                    imports_map[alias.asname or alias.name] = full_path
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_data = {"name": node.name, "line": node.lineno}
                if isinstance(node, ast.AsyncFunctionDef):
                    func_data["async"] = True
                # Avoid duplicates if somehow processed differently (though tree.body is one pass)
                if not any(
                    f["name"] == node.name and f["line"] == node.lineno
                    for f in result["functions"]
                ):
                    result["functions"].append(func_data)
            elif isinstance(node, ast.ClassDef):
                # Add TOP-LEVEL classes to result["classes"]
                if not any(
                    c["name"] == node.name and c["line"] == node.lineno
                    for c in result["classes"]
                ):
                    result["classes"].append({"name": node.name, "line": node.lineno})
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):  # Simple assignment: MY_VAR = 1
                        result["globals_defined"].append(
                            {"name": target.id, "line": node.lineno}
                        )
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):  # MY_VAR: int = 1
                    result["globals_defined"].append(
                        {"name": node.target.id, "line": node.lineno, "annotated": True}
                    )
                _capture_significant_assignment(node, result)
            elif isinstance(node, ast.Assign):
                _capture_significant_assignment(node, result)

        logger.debug(f"DEBUG DA: tree.body processed for {file_path}.")

        # Pass 2: ast.walk for detailed analysis
        for node in ast.walk(tree):
            # ENHANCEMENT: Capture function-level imports into imports_map
            # This enables resolution of calls like create_agent imported inside a function
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Only add if not already present (top-level imports take precedence)
                    if (alias.asname or alias.name) not in imports_map:
                        imports_map[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                relative_prefix = "." * node.level
                full_import_source = f"{relative_prefix}{module_name}"
                for alias in node.names:
                    full_path = (
                        f"{full_import_source}.{alias.name}"
                        if full_import_source
                        else alias.name
                    )
                    # Only add if not already present
                    if (alias.asname or alias.name) not in imports_map:
                        imports_map[alias.asname or alias.name] = full_path

            # Decorators (for all functions/classes, top-level or nested)
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                parent = getattr(node, "_parent", None)
                target_type = "unknown"
                is_top_level = parent is tree

                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    target_type = (
                        "function"
                        if is_top_level
                        else (
                            "method"
                            if isinstance(parent, ast.ClassDef)
                            else "nested_function"
                        )
                    )
                elif node:
                    target_type = "class" if is_top_level else "nested_class"
                    # --- ENHANCED ANALYSIS: Class Docstrings ---
                    docstring = ast.get_docstring(node)
                    doc_summary = docstring.split("\n")[0] if docstring else None

                    # Update existing class entry or create new
                    class_entry = next(
                        (
                            c
                            for c in result["classes"]
                            if c["name"] == node.name and c["line"] == node.lineno
                        ),
                        None,
                    )
                    if class_entry:
                        if doc_summary:
                            class_entry["docstring"] = doc_summary
                    else:
                        new_entry = {"name": node.name, "line": node.lineno}
                        if doc_summary:
                            new_entry["docstring"] = doc_summary
                        result["classes"].append(new_entry)

                for dec_node in node.decorator_list:
                    dec_name = _get_full_name_str(dec_node)
                    if dec_name:
                        result["decorators_used"].append(
                            {
                                "name": dec_name,
                                "target_type": target_type,
                                "target_name": node.name,
                                "line": dec_node.lineno,
                            }
                        )
            
            # Significant Assignments (Top-level, Class-level, or Method-level)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                parent = getattr(node, "_parent", None)
                # Allow assignments at module level, class level, or inside functions/methods
                if parent is tree or isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    _capture_significant_assignment(node, result)

            # Type References
            if isinstance(node, ast.AnnAssign):
                if node.annotation:
                    target_name_val = _get_full_name_str(node.target)
                    context = "variable_annotation"
                    parent = getattr(node, "_parent", None)
                    if parent is tree:
                        context = "global_variable_annotation"
                    elif isinstance(parent, ast.ClassDef):
                        context = "class_variable_annotation"
                    elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        context = "local_variable_annotation"
                    for type_name_str in _extract_type_names_from_annotation(
                        node.annotation
                    ):
                        result["type_references"].append(
                            {
                                "type_name_str": type_name_str,
                                "context": context,
                                "target_name": (
                                    target_name_val
                                    if target_name_val
                                    else "_unknown_target_"
                                ),
                                "line": node.lineno,
                            }
                        )

            # --- ENHANCED ANALYSIS: Functions and Methods ---
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                doc_summary = docstring.split("\n")[0] if docstring else None

                params: List[str] = []
                for arg in node.args.args:
                    params.append(arg.arg)
                if node.args.vararg:
                    params.append(f"*{node.args.vararg.arg}")
                if node.args.kwarg:
                    params.append(f"**{node.args.kwarg.arg}")

                # Create function/method entry
                new_entry = {"name": node.name, "line": node.lineno, "params": params}
                if doc_summary:
                    new_entry["docstring"] = doc_summary
                if isinstance(node, ast.AsyncFunctionDef):
                    new_entry["async"] = True

                # Determine if it's a method or top-level function
                parent = getattr(node, "_parent", None)
                if isinstance(parent, ast.ClassDef):
                    # It's a method - add to parent class
                    class_entry = next(
                        (
                            c
                            for c in result["classes"]
                            if c["name"] == parent.name and c["line"] == parent.lineno
                        ),
                        None,
                    )
                    if class_entry:
                        if "methods" not in class_entry:
                            class_entry["methods"] = []
                        class_entry["methods"].append(new_entry)
                    else:
                        # Fallback
                        result["functions"].append(new_entry)
                else:
                    # Top-level function (or nested in another function)
                    result["functions"].append(new_entry)

                is_top_level_func = any(item is node for item in tree.body)
                if not is_top_level_func:
                    for dec_node in node.decorator_list:
                        dec_name = _get_full_name_str(dec_node)
                        if dec_name:
                            result["decorators_used"].append(
                                {
                                    "name": dec_name,
                                    "target_type": (
                                        "method"
                                        if isinstance(
                                            getattr(node, "_parent", None), ast.ClassDef
                                        )
                                        else "nested_function"
                                    ),
                                    "target_name": node.name,
                                    "line": dec_node.lineno,
                                }
                            )
                for arg_node_type in [
                    node.args.args,
                    node.args.posonlyargs,
                    node.args.kwonlyargs,
                ]:
                    for arg in arg_node_type:
                        if arg.annotation:
                            for type_name_str in _extract_type_names_from_annotation(
                                arg.annotation
                            ):
                                result["type_references"].append(
                                    {
                                        "type_name_str": type_name_str,
                                        "context": "arg_annotation",
                                        "function_name": node.name,
                                        "target_name": arg.arg,
                                        "line": getattr(
                                            arg.annotation, "lineno", node.lineno
                                        ),
                                    }
                                )
                if node.args.vararg and node.args.vararg.annotation:
                    for type_name_str in _extract_type_names_from_annotation(
                        node.args.vararg.annotation
                    ):
                        result["type_references"].append(
                            {
                                "type_name_str": type_name_str,
                                "context": "vararg_annotation",
                                "function_name": node.name,
                                "target_name": node.args.vararg.arg,
                                "line": getattr(
                                    node.args.vararg.annotation, "lineno", node.lineno
                                ),
                            }
                        )
                if node.args.kwarg and node.args.kwarg.annotation:
                    for type_name_str in _extract_type_names_from_annotation(
                        node.args.kwarg.annotation
                    ):
                        result["type_references"].append(
                            {
                                "type_name_str": type_name_str,
                                "context": "kwarg_annotation",
                                "function_name": node.name,
                                "target_name": node.args.kwarg.arg,
                                "line": getattr(
                                    node.args.kwarg.annotation, "lineno", node.lineno
                                ),
                            }
                        )
                if node.returns:
                    for type_name_str in _extract_type_names_from_annotation(
                        node.returns
                    ):
                        result["type_references"].append(
                            {
                                "type_name_str": type_name_str,
                                "context": "return_annotation",
                                "function_name": node.name,
                                "line": getattr(node.returns, "lineno", node.lineno),
                            }
                        )
            # Inheritance
            elif isinstance(node, ast.ClassDef):
                is_top_level_class = any(item is node for item in tree.body)
                if not is_top_level_class:
                    for dec_node in node.decorator_list:
                        dec_name = _get_full_name_str(dec_node)
                        if dec_name:
                            result["decorators_used"].append(
                                {
                                    "name": dec_name,
                                    "target_type": "nested_class",
                                    "target_name": node.name,
                                    "line": dec_node.lineno,
                                }
                            )
                for base in node.bases:
                    base_full_name = _get_full_name_str(base)
                    if base_full_name:
                        # Avoid duplicates if inheritance was somehow processed differently before
                        if not any(
                            inh["class_name"] == node.name
                            and inh["base_class_name"] == base_full_name
                            for inh in result["inheritance"]
                        ):
                            result["inheritance"].append(
                                {
                                    "class_name": node.name,
                                    "base_class_name": base_full_name,
                                    "potential_source": base_full_name,
                                    "line": getattr(base, "lineno", node.lineno),
                                }
                            )
                
                # Scan class body for significant assignments (NEW)
                for class_item in node.body:
                    if isinstance(class_item, (ast.Assign, ast.AnnAssign)):
                        _capture_significant_assignment(class_item, result)

            # --- NEW ENHANCEMENT: Capture Significant Assignments/Literals ---
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                _capture_significant_assignment(node, result)

            # --- PYTHON 3.10+ MATCH STATEMENTS ---
            elif hasattr(ast, "Match") and isinstance(node, ast.Match):
                # Capture dependencies in the subject
                # (Dependencies in cases are handled by recursive walk, but we can explicitly note the pattern types if needed)
                pass

            # --- PYTHON 3.12+ TYPE ALIASES ---
            elif hasattr(ast, "TypeAlias") and isinstance(node, ast.TypeAlias):
                # type MyType = int
                result["globals_defined"].append(
                    {"name": node.name.id, "line": node.lineno, "type": "TypeAlias"}
                )
                # Extract dependencies from the value
                for type_name in _extract_type_names_from_annotation(node.value):
                    result["type_references"].append(
                        {
                            "type_name_str": type_name,
                            "context": "type_alias_definition",
                            "target_name": node.name.id,
                            "line": node.lineno,
                        }
                    )

            # --- PYTHON 3.11+ EXCEPTION GROUPS (TryStar) ---
            elif hasattr(ast, "TryStar") and isinstance(node, ast.TryStar):
                # Handled implicitly by walk for body, but we can note the exception types in handlers
                for handler in node.handlers:
                    if handler.type:
                        type_name = _get_full_name_str(handler.type)
                        if type_name:
                            result["type_references"].append(
                                {
                                    "type_name_str": type_name,
                                    "context": "exception_group_handler",
                                    "target_name": "try_star_block",
                                    "line": handler.lineno,
                                }
                            )

            # Calls
            elif isinstance(node, ast.Call):
                target_full_name = _get_full_name_str(node.func)
                potential_source = _get_source_object_str(node.func)

                # ENHANCEMENT: If potential_source is None (direct call like ConfigManager()),
                # try to resolve from imports_map to capture cross-file relationships
                if potential_source is None and target_full_name:
                    # For direct calls, the base name is the function/class being called
                    base_name = target_full_name.split("(")[0].split(".")[0]
                    if base_name in imports_map:
                        potential_source = cast(str, imports_map[base_name])

                if target_full_name and _is_useful_call(
                    target_full_name,
                    potential_source,
                    cast(Dict[str, str], imports_map),
                ):
                    result["calls"].append(
                        {
                            "target_name": target_full_name,
                            "potential_source": potential_source,
                            "line": node.lineno,
                        }
                    )
            # Attribute Accesses
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                attribute_name = node.attr
                potential_source = _get_full_name_str(node.value)
                if potential_source:
                    result["attribute_accesses"].append(
                        {
                            "target_name": attribute_name,
                            "potential_source": potential_source,
                            "line": node.lineno,
                        }
                    )
            # Exceptions Handled
            elif isinstance(node, ast.ExceptHandler):
                if node.type:  # node.type can be None for a bare except
                    exception_type_name = _get_full_name_str(node.type)
                    if exception_type_name:
                        result["exceptions_handled"].append(
                            {"type_name_str": exception_type_name, "line": node.lineno}
                        )
            # With Contexts
            elif isinstance(node, ast.With):
                for item in node.items:
                    context_expr_name = _get_full_name_str(item.context_expr)
                    if context_expr_name:
                        result["with_contexts_used"].append(
                            {
                                "context_expr_str": context_expr_name,
                                "line": item.context_expr.lineno,
                            }
                        )

        logger.debug(f"DEBUG DA: Second ast.walk completed for {file_path}.")

    except SyntaxError as e:
        logger.warning(
            f"AST Syntax Error in {file_path}: {e}. Analysis may be incomplete."
        )
        result["error"] = f"AST Syntax Error: {e}"
        # result["_ast_tree"] remains None if parsing failed, or holds tree if parsing succeeded but later step failed
    except Exception as e:
        # Log with full traceback for unexpected errors during AST processing
        logger.exception(
            f"Unexpected AST analysis error IN TRY BLOCK for {file_path}: {e}"
        )
        result["error"] = f"Unexpected AST analysis error: {e}"

    is_tree_none_at_end = result.get("_ast_tree") is None
    logger.debug(
        f"DEBUG DA: End of _analyze_python_file for {file_path}. result['_ast_tree'] is None: {is_tree_none_at_end}. tree_obj_for_debug type: {type(tree_obj_for_debug)}. Keys: {list(result.keys())}"
    )


def _analyze_javascript_file_ts(
    file_path: str, content: str, result: Dict[str, Any]
) -> None:
    """
    Tree-sitter based analysis for JavaScript (.js) files using minimal, grammar-safe patterns:
      - import paths
      - function declarations
      - class declarations
      - simple identifier call expressions
    """
    result.setdefault("imports", [])
    result.setdefault("functions", [])
    result.setdefault("classes", [])
    result.setdefault("calls", [])
    result.setdefault("exports", [])
    result.setdefault("_ts_tree", None)

    try:
        content_bytes = content.encode("utf-8", errors="ignore")
        # --- FIX (MAJOR): Create a local, thread-safe parser instance. ---
        parser = Parser(JS_LANGUAGE)
        tree = parser.parse(content_bytes)
        result["_ts_tree"] = tree

        # Expand JS tree-sitter queries: imports (incl. require), exports, functions, classes, calls (identifier, member)
        imports_query = """
        [
          (import_statement source: (string) @path)
          (call_expression
            function: (identifier) @req.fn
            arguments: (arguments (string) @path)
          ) @require
            (#match? @req.fn "^(require|import)$")
        ]
        """
        functions_query = "(function_declaration name: (identifier) @function.name)"
        classes_query = "(class_declaration name: (identifier) @class.name)"
        calls_query = """
        [
          (call_expression function: (identifier) @call.name)
          (call_expression function: (member_expression property: (property_identifier) @call.name))
        ]
        """
        # Capture comments and string literals for SES
        extra_query = """
        [
          (comment) @comment
          (string) @literal
        ]
        """
        exports_query = """
        [
        (export_statement
            (export_clause (export_specifier name: (identifier) @export.name))
        )
        (export_statement
            (export_clause (export_specifier name: (identifier) @export.orig alias: (identifier) @export.alias))
        )
        (export_statement
            declaration: (variable_declaration
            (variable_declarator
                name: (identifier) @export.default))
        ) @default.export
        (export_statement
            declaration: (function_declaration name: (identifier) @export.func.name)
        )
        (export_statement
            declaration: (class_declaration name: (identifier) @export.class.name)
        )
        ]
        """

        def run_query_js(query_str: str) -> List[Tuple[Any, str]]:
            # Use QueryCursor.matches API per tree_sitter/__init__.pyi
            q = Query(JS_LANGUAGE, query_str)
            captures: List[Tuple[Any, str]] = []
            cursor = QueryCursor(q)
            matches = cursor.matches(tree.root_node)
            for _pattern_index, captures_dict in matches:
                for cap_name, nodes in captures_dict.items():
                    for node in nodes:
                        captures.append((node, cap_name))
            return captures

        # Imports (ESM and require/import() calls)
        for node, cap in run_query_js(imports_query):
            if cap == "path":
                path_text = _get_ts_node_text(node, content_bytes)
                if (
                    len(path_text) >= 2
                    and path_text[0] in ('"', "'")
                    and path_text[-1] == path_text[0]
                ):
                    path_text = path_text[1:-1]
                result["imports"].append(
                    {"path": path_text, "line": node.start_point[0] + 1, "symbols": []}
                )

        # Functions
        for node, cap in run_query_js(functions_query):
            if cap == "function.name":
                result["functions"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )

        # Classes
        for node, cap in run_query_js(classes_query):
            if cap == "class.name":
                result["classes"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )

        # Calls (identifier and member-expression calls)
        for node, cap in run_query_js(calls_query):
            if cap == "call.name":
                result["calls"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )

        # Extra literals and comments
        result.setdefault("comments", [])
        result.setdefault("literals", [])
        for node, cap in run_query_js(extra_query):
            text = _get_ts_node_text(node, content_bytes)
            if cap == "comment":
                # Clean up comment markers
                clean = text.strip("/ \n*")
                if len(clean) > 5 and clean not in result["comments"]:
                    result["comments"].append(clean)
            elif cap == "literal":
                # Clean quotes
                if len(text) >= 2 and text[0] in ("'", '"', "`") and text[-1] == text[0]:
                    text = text[1:-1]
                
                # Filter for "meaningful" literals:
                # - Paths or URLs (contains /)
                # - Descriptive identifiers (long, contains space/underscore, or capital letters)
                is_meaningful = (
                    len(text) > 5 and (
                        "/" in text or 
                        " " in text or 
                        "_" in text or 
                        any(c.isupper() for c in text)
                    )
                )
                if is_meaningful and text not in result["literals"]:
                    result["literals"].append(text)

        # Exports (collect names and re-exports)
        result.setdefault("exports", [])
        for node, cap in run_query_js(exports_query):
            if cap == "export.name":
                result["exports"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )
            elif cap == "export.orig":
                # will be paired with alias cap in same match; capture as name, alias separately if present
                result["exports"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )
            elif cap == "export.alias":
                # attach alias to last export if appropriate
                if result["exports"]:
                    result["exports"][-1]["alias"] = _get_ts_node_text(
                        node, content_bytes
                    )
            elif cap == "export.from":
                from_text = _get_ts_node_text(node, content_bytes)
                if (
                    len(from_text) >= 2
                    and from_text[0] in ('"', "'")
                    and from_text[-1] == from_text[0]
                ):
                    from_text = from_text[1:-1]
                result["exports"].append(
                    {"from": from_text, "line": node.start_point[0] + 1}
                )
            elif cap == "export.default":
                result["exports"].append(
                    {
                        "name": "default",
                        "alias": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )

    except Exception as e:
        logger.exception(f"JS analysis error for {file_path}: {e}")
        result["error"] = f"JS analysis error: {e}"
    finally:
        # Normalization to ensure downstream suggestion pipeline has consistent shapes
        result.setdefault("imports", [])
        result.setdefault("functions", [])
        result.setdefault("classes", [])
        result.setdefault("calls", [])
        result.setdefault("exports", [])
        # ensure import dicts have "path" key
        _normalize_imports(result)
        # Tag analysis kind for consumers
        result["analysis_kind"] = "js"
        # Build exports (JS minimal): handle `export ... from "path"` re-exports if present
        try:
            if "exports" not in result:
                result["exports"] = []
            # A very light pattern using tree-sitter captures already built elsewhere can be added later.
            # Keep placeholder structure consistent.
        except Exception:
            pass


def _analyze_typescript_file_ts(
    file_path: str, content: str, result: Dict[str, Any]
) -> None:
    """
    Tree-sitter based analysis for TypeScript (.ts) files using minimal, grammar-safe patterns:
      - import paths
      - function declarations
      - class declarations
      - simple identifier call expressions
      - type references (type_identifier in type_annotation/generic_type)
    """
    result.setdefault("imports", [])
    result.setdefault("functions", [])
    result.setdefault("classes", [])
    result.setdefault("calls", [])
    result.setdefault("type_references", [])
    result.setdefault("_ts_tree", None)

    try:
        content_bytes = content.encode("utf-8", errors="ignore")
        # --- FIX (MAJOR): Create a local, thread-safe parser instance. ---
        parser = Parser(TS_LANGUAGE)
        tree = parser.parse(content_bytes)
        result["_ts_tree"] = tree

        # Expand TS queries: imports (incl. require), exports, richer calls, plus types
        imports_query = """
        [
          (import_statement source: (string) @path)
          (call_expression
            function: (identifier) @req.fn
            arguments: (arguments (string) @path))
            (#match? @req.fn "^(require|import)$")
        ]
        """
        functions_query = "(function_declaration name: (identifier) @function.name)"
        classes_query = "(class_declaration name: (identifier) @class.name)"
        calls_query = """
        [
          (call_expression function: (identifier) @call.name)
          (call_expression function: (member_expression property: (property_identifier) @call.name))
        ]
        """
        type_ann_query = "(type_annotation (type_identifier) @type.name)"
        generic_type_query = "(generic_type (type_identifier) @type.name)"
        exports_query = """
        [
          (export_statement
            (export_clause (export_specifier name: (identifier) @export.name))
          )
          (export_statement
            (export_clause (export_specifier name: (identifier) @export.orig alias: (identifier) @export.alias))
          )
          (export_statement
            (export_clause (export_from_clause source: (string) @export.from))
          )
          (export_statement
            (export_default_declaration (identifier) @export.default)
          )
        ]
        """

        def run_query_ts(query_str: str) -> List[Tuple[Any, str]]:
            q = Query(TS_LANGUAGE, query_str)
            captures: List[Tuple[Any, str]] = []
            cursor = QueryCursor(q)
            matches = cursor.matches(tree.root_node)
            for _pattern_index, captures_dict in matches:
                for cap_name, nodes in captures_dict.items():
                    for node in nodes:
                        captures.append((node, cap_name))
            return captures

        # Imports
        for node, cap in run_query_ts(imports_query):
            if cap == "path":
                path_text = _get_ts_node_text(node, content_bytes)
                if (
                    len(path_text) >= 2
                    and path_text[0] in ('"', "'")
                    and path_text[-1] == path_text[0]
                ):
                    path_text = path_text[1:-1]
                result["imports"].append(
                    {"path": path_text, "line": node.start_point[0] + 1, "symbols": []}
                )

        # Functions
        for node, cap in run_query_ts(functions_query):
            if cap == "function.name":
                result["functions"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )

        # Classes
        for node, cap in run_query_ts(classes_query):
            if cap == "class.name":
                result["classes"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )

        # Calls
        for node, cap in run_query_ts(calls_query):
            if cap == "call.name":
                result["calls"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )

        # Type references
        for node, cap in run_query_ts(type_ann_query):
            if cap == "type.name":
                result["type_references"].append(
                    {
                        "type_name_str": _get_ts_node_text(node, content_bytes),
                        "context": "type_annotation",
                        "line": node.start_point[0] + 1,
                    }
                )
        for node, cap in run_query_ts(generic_type_query):
            if cap == "type.name":
                result["type_references"].append(
                    {
                        "type_name_str": _get_ts_node_text(node, content_bytes),
                        "context": "generic_type",
                        "line": node.start_point[0] + 1,
                    }
                )

        # Exports
        result.setdefault("exports", [])
        for node, cap in run_query_ts(exports_query):
            if cap == "export.name":
                result["exports"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )
            elif cap == "export.orig":
                result["exports"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )
            elif cap == "export.alias":
                if result["exports"]:
                    result["exports"][-1]["alias"] = _get_ts_node_text(
                        node, content_bytes
                    )
            elif cap == "export.from":
                from_text = _get_ts_node_text(node, content_bytes)
                if (
                    len(from_text) >= 2
                    and from_text[0] in ('"', "'")
                    and from_text[-1] == from_text[0]
                ):
                    from_text = from_text[1:-1]
                result["exports"].append(
                    {"from": from_text, "line": node.start_point[0] + 1}
                )
            elif cap == "export.default":
                result["exports"].append(
                    {
                        "name": "default",
                        "alias": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )

    except Exception as e:
        logger.exception(f"TS analysis error for {file_path}: {e}")
        result["error"] = f"TS analysis error: {e}"
    finally:
        # Normalization for suggestion pipeline
        result.setdefault("imports", [])
        result.setdefault("functions", [])
        result.setdefault("classes", [])
        result.setdefault("calls", [])
        result.setdefault("type_references", [])
        result.setdefault("exports", [])
        _normalize_imports(result)
        result["analysis_kind"] = "ts"
        # Ensure exports list exists for re-export links
        try:
            result.setdefault("exports", [])
        except Exception:
            pass


def _analyze_tsx_file_ts(file_path: str, content: str, result: Dict[str, Any]) -> None:
    """
    Tree-sitter based analysis for TSX (.tsx) files using minimal, grammar-safe patterns:
      - import paths
      - function declarations
      - class declarations
      - simple identifier call expressions
      - type references (type_identifier in type_annotation/generic_type)
    """
    result.setdefault("imports", [])
    result.setdefault("functions", [])
    result.setdefault("classes", [])
    result.setdefault("calls", [])
    result.setdefault("type_references", [])
    result.setdefault("_ts_tree", None)

    try:
        content_bytes = content.encode("utf-8", errors="ignore")
        # --- FIX (MAJOR): Create a local, thread-safe parser instance. ---
        parser = Parser(TSX_LANGUAGE)
        tree = parser.parse(content_bytes)
        result["_ts_tree"] = tree

        # Expand TSX queries similar to TS
        imports_query = """
        [
          (import_statement source: (string) @path)
          (call_expression
            function: (identifier) @req.fn
            arguments: (arguments (string) @path))
            (#match? @req.fn "^(require|import)$")
        ]
        """
        functions_query = "(function_declaration name: (identifier) @function.name)"
        classes_query = "(class_declaration name: (identifier) @class.name)"
        calls_query = """
        [
          (call_expression function: (identifier) @call.name)
          (call_expression function: (member_expression property: (property_identifier) @call.name))
        ]
        """
        type_ann_query = "(type_annotation (type_identifier) @type.name)"
        generic_type_query = "(generic_type (type_identifier) @type.name)"
        exports_query = """
        [
          (export_statement
            (export_clause (export_specifier name: (identifier) @export.name))
          )
          (export_statement
            (export_clause (export_specifier name: (identifier) @export.orig alias: (identifier) @export.alias))
          )
          (export_statement
            (export_clause (export_from_clause source: (string) @export.from))
          )
          (export_statement
            (export_default_declaration (identifier) @export.default)
          )
        ]
        """

        def run_query_tsx(query_str: str) -> List[Tuple[Any, str]]:
            q = Query(TSX_LANGUAGE, query_str)
            captures: List[Tuple[Any, str]] = []
            cursor = QueryCursor(q)
            matches = cursor.matches(tree.root_node)
            for _pattern_index, captures_dict in matches:
                for cap_name, nodes in captures_dict.items():
                    for node in nodes:
                        captures.append((node, cap_name))
            return captures

        # Imports
        for node, cap in run_query_tsx(imports_query):
            if cap == "path":
                path_text = _get_ts_node_text(node, content_bytes)
                if (
                    len(path_text) >= 2
                    and path_text[0] in ('"', "'")
                    and path_text[-1] == path_text[0]
                ):
                    path_text = path_text[1:-1]
                result["imports"].append(
                    {"path": path_text, "line": node.start_point[0] + 1, "symbols": []}
                )

        # Functions
        for node, cap in run_query_tsx(functions_query):
            if cap == "function.name":
                result["functions"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )

        # Classes
        for node, cap in run_query_tsx(classes_query):
            if cap == "class.name":
                result["classes"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )

        # Calls
        for node, cap in run_query_tsx(calls_query):
            if cap == "call.name":
                result["calls"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )

        # Type references
        for node, cap in run_query_tsx(type_ann_query):
            if cap == "type.name":
                result["type_references"].append(
                    {
                        "type_name_str": _get_ts_node_text(node, content_bytes),
                        "context": "type_annotation",
                        "line": node.start_point[0] + 1,
                    }
                )
        for node, cap in run_query_tsx(generic_type_query):
            if cap == "type.name":
                result["type_references"].append(
                    {
                        "type_name_str": _get_ts_node_text(node, content_bytes),
                        "context": "generic_type",
                        "line": node.start_point[0] + 1,
                    }
                )

        # Exports
        result.setdefault("exports", [])
        for node, cap in run_query_tsx(exports_query):
            if cap == "export.name":
                result["exports"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )
            elif cap == "export.orig":
                result["exports"].append(
                    {
                        "name": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )
            elif cap == "export.alias":
                if result["exports"]:
                    result["exports"][-1]["alias"] = _get_ts_node_text(
                        node, content_bytes
                    )
            elif cap == "export.from":
                from_text = _get_ts_node_text(node, content_bytes)
                if (
                    len(from_text) >= 2
                    and from_text[0] in ('"', "'")
                    and from_text[-1] == from_text[0]
                ):
                    from_text = from_text[1:-1]
                result["exports"].append(
                    {"from": from_text, "line": node.start_point[0] + 1}
                )
            elif cap == "export.default":
                result["exports"].append(
                    {
                        "name": "default",
                        "alias": _get_ts_node_text(node, content_bytes),
                        "line": node.start_point[0] + 1,
                    }
                )

    except Exception as e:
        logger.exception(f"TSX analysis error for {file_path}: {e}")
        result["error"] = f"TSX analysis error: {e}"
    finally:
        # Normalization for suggestion pipeline
        result.setdefault("imports", [])
        result.setdefault("functions", [])
        result.setdefault("classes", [])
        result.setdefault("calls", [])
        result.setdefault("type_references", [])
        result.setdefault("exports", [])
        _normalize_imports(result)
        result["analysis_kind"] = "tsx"
        # Ensure exports list exists for re-export links
        try:
            result.setdefault("exports", [])
        except Exception:
            pass


def _analyze_json_file_ts(file_path: str, content: str, result: Dict[str, Any]) -> None:
    """Analyzes JSON file content using tree-sitter."""
    for key in ["links", "json_keys", "json_refs", "_ts_tree"]:
        result.setdefault(key, [] if key != "_ts_tree" else None)

    try:
        content_bytes = content.encode("utf8")
        parser = Parser(JSON_LANGUAGE)
        tree = parser.parse(content_bytes)
        result["_ts_tree"] = tree
        root_node = tree.root_node

        # Look for strings that might be paths
        query = Query(JSON_LANGUAGE, "(string) @str")
        cursor = QueryCursor(query)
        seen_links: Set[str] = set()
        for _, captures in cursor.matches(root_node):
            for node in captures.get("str", []):
                text = _get_ts_node_text(node, content_bytes).strip("\"'")
                # Heuristic for likely file references
                ext = os.path.splitext(text)[1].lower()
                looks_like_file = ext in {
                    ".json",
                    ".yaml",
                    ".yml",
                    ".sql",
                    ".md",
                    ".rst",
                    ".txt",
                    ".js",
                    ".ts",
                    ".tsx",
                    ".svelte",
                    ".py",
                    ".css",
                    ".html",
                }
                if (
                    ("/" in text or "\\" in text or looks_like_file)
                    and not text.startswith(("http:", "https:", "#", "mailto:", "tel:"))
                    and text not in seen_links
                ):
                    seen_links.add(text)
                    result["links"].append(
                        {"url": text, "line": node.start_point[0] + 1}
                    )

        # Extract structured key paths for SES and traceability
        seen_key_paths: Set[str] = set()
        seen_ref_pairs: Set[str] = set()
        max_keys = 2000
        max_refs = 500

        def _extract_json_string(node: Any) -> str:
            return _get_ts_node_text(node, content_bytes).strip().strip("\"'")

        def _walk_json(node: Any, parent_path: List[str]) -> None:
            if len(cast(List[Dict[str, Any]], result["json_keys"])) >= max_keys:
                return

            if node.type == "pair":
                key_node = node.child_by_field_name("key")
                value_node = node.child_by_field_name("value")

                if key_node is not None:
                    key_text = _extract_json_string(key_node)
                    if key_text:
                        current_path = parent_path + [key_text]
                        path_str = ".".join(current_path)
                        if path_str not in seen_key_paths:
                            seen_key_paths.add(path_str)
                            cast(List[Dict[str, Any]], result["json_keys"]).append(
                                {
                                    "key": key_text,
                                    "path": path_str,
                                    "line": key_node.start_point[0] + 1,
                                }
                            )

                        if (
                            value_node is not None
                            and value_node.type == "string"
                            and len(cast(List[Dict[str, Any]], result["json_refs"]))
                            < max_refs
                        ):
                            raw_value = _extract_json_string(value_node)
                            raw_ext = os.path.splitext(raw_value)[1].lower()
                            raw_looks_like_file = raw_ext in {
                                ".json",
                                ".yaml",
                                ".yml",
                                ".sql",
                                ".md",
                                ".rst",
                                ".txt",
                                ".js",
                                ".ts",
                                ".tsx",
                                ".svelte",
                                ".py",
                                ".css",
                                ".html",
                            }
                            if (
                                raw_value
                                and (
                                    "/" in raw_value
                                    or "\\" in raw_value
                                    or raw_looks_like_file
                                )
                                and not raw_value.startswith(
                                    ("http:", "https:", "#", "mailto:", "tel:")
                                )
                            ):
                                ref_key = f"{path_str}|{raw_value}"
                                if ref_key not in seen_ref_pairs:
                                    seen_ref_pairs.add(ref_key)
                                    cast(
                                        List[Dict[str, Any]], result["json_refs"]
                                    ).append(
                                        {
                                            "key_path": path_str,
                                            "value": raw_value,
                                            "line": value_node.start_point[0] + 1,
                                        }
                                    )

                        if value_node is not None:
                            _walk_json(value_node, current_path)
                        return

            for child in node.children:
                if child.is_named:
                    _walk_json(child, parent_path)

        _walk_json(root_node, [])
        result["analysis_kind"] = "json"
    except Exception as e:
        logger.error(f"Error parsing JSON {file_path} with tree-sitter: {e}")
        result["error"] = f"Tree-sitter JSON parsing error: {e}"


def _analyze_markdown_file_regex(
    file_path: str, content: str, result: Dict[str, Any]
) -> None:
    """Analyzes Markdown file content using regex (legacy/fallback)."""
    result.setdefault("links", [])
    result.setdefault("code_blocks", [])
    try:
        for match in MARKDOWN_LINK_PATTERN.finditer(content):
            url = match.group(1)
            if url and not url.startswith(("#", "http:", "https:", "mailto:", "tel:")):
                result["links"].append(
                    {"url": url, "line": content[: match.start()].count("\n") + 1}
                )
    except Exception as e:
        logger.warning(f"Regex error during MD link analysis in {file_path}: {e}")
    try:
        code_block_pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
        for match in code_block_pattern.finditer(content):
            lang = match.group(1) or "text"
            result["code_blocks"].append(
                {
                    "language": lang.lower(),
                    "line": content[: match.start()].count("\n") + 1,
                    "content": match.group(2),
                }
            )
    except Exception as e:
        logger.warning(f"Regex error during MD code block analysis in {file_path}: {e}")

def _analyze_markdown_file_ts(
    file_path: str, content: str, result: Dict[str, Any]
) -> None:
    """Analyzes Markdown file content using tree-sitter for blocks and inline for links."""
    for key in ["links", "code_blocks", "headers", "images", "_ts_tree"]:
        result.setdefault(key, [] if key != "_ts_tree" else None)

    try:
        content_bytes = content.encode("utf8")

        # 1. Parse Block Structure (Code Blocks, Headers)
        parser = Parser(MARKDOWN_LANGUAGE)
        tree = parser.parse(content_bytes)
        result["_ts_tree"] = tree
        root_node = tree.root_node

        # Extract Code Blocks
        cb_query_str = (
            "(fenced_code_block (info_string) @lang (code_fence_content) @content)"
        )
        cb_query = Query(MARKDOWN_LANGUAGE, cb_query_str)
        cb_cursor = QueryCursor(cb_query)
        for _, captures in cb_cursor.matches(root_node):
            langs = captures.get("lang", [])
            contents = captures.get("content", [])
            if contents:
                lang = _get_ts_node_text(langs[0], content_bytes) if langs else "text"
                result["code_blocks"].append(
                    {
                        "language": lang.strip().lower(),
                        "line": contents[0].start_point[0] + 1,
                        "content": _get_ts_node_text(contents[0], content_bytes),
                    }
                )

        # Extract Headers (ATX)
        h_query_str = "(atx_heading [ (atx_h1_marker) (atx_h2_marker) (atx_h3_marker) (atx_h4_marker) (atx_h5_marker) (atx_h6_marker) ] @marker (inline) @text)"
        try:
            h_query = Query(MARKDOWN_LANGUAGE, h_query_str)
            h_cursor = QueryCursor(h_query)
            for _, captures in h_cursor.matches(root_node):
                marker_node = captures["marker"][0]
                text_node = captures["text"][0]
                marker_text = _get_ts_node_text(marker_node, content_bytes)
                level = marker_text.count("#")
                result["headers"].append(
                    {
                        "level": level,
                        "text": _get_ts_node_text(text_node, content_bytes).strip(),
                        "line": marker_node.start_point[0] + 1,
                    }
                )
        except Exception as qe:
            logger.debug(f"Markdown header query error: {qe}")

        # 2. Parse Inline Content (Links, Images)
        inline_parser = Parser(MARKDOWN_INLINE_LANGUAGE)
        inline_tree = inline_parser.parse(content_bytes)
        inline_root = inline_tree.root_node

        # Query for links and images
        inline_query_str = """
        [
          (inline_link (link_destination) @link.url)
          (image (link_text) @img.alt (link_destination) @img.url)
          (image (link_destination) @img.url)
        ]
        """
        try:
            inline_query = Query(MARKDOWN_INLINE_LANGUAGE, inline_query_str)
            inline_cursor = QueryCursor(inline_query)
            for _, captures in inline_cursor.matches(inline_root):
                # Handle Links
                for node in captures.get("link.url", []):
                    url = _get_ts_node_text(node, content_bytes)
                    if url and not url.startswith(
                        ("#", "http:", "https:", "mailto:", "tel:")
                    ):
                        result["links"].append(
                            {"url": url, "line": node.start_point[0] + 1}
                        )

                # Handle Images
                img_urls = captures.get("img.url", [])
                img_alts = captures.get("img.alt", [])
                for i, node in enumerate(img_urls):
                    url = _get_ts_node_text(node, content_bytes)
                    alt = ""
                    if i < len(img_alts):
                        alt = _get_ts_node_text(img_alts[i], content_bytes)
                    result["images"].append(
                        {
                            "url": url,
                            "src": url,
                            "alt": alt,
                            "line": node.start_point[0] + 1,
                        }
                    )
        except Exception as qe:
            logger.debug(f"Markdown inline query error: {qe}")

        result["analysis_kind"] = "markdown_ts"

    except Exception as e:
        logger.error(f"Error parsing Markdown {file_path} with tree-sitter: {e}")
        # Fallback to regex
        _analyze_markdown_file_regex(file_path, content, result)


def _extract_template_tree(node: Any, source_bytes: bytes) -> Dict[str, Any]:
    """
    Recursively extracts the FULL structure of the Svelte template.
    Captures type, content, attributes, and children without truncation.
    """
    node_type = node.type

    # Base dict
    data = {
        "type": node_type,
        "start_point": node.start_point,
        "end_point": node.end_point,
    }

    # Helper to get full text
    # User said "FULL DATA", so let's store text for everything that isn't the root
    if node_type != "document":
        data["content"] = _get_ts_node_text(node, source_bytes)

    # Specific handling for attributes to make them accessible
    if node_type == "attribute":
        # extract name and value
        attr_name = ""
        attr_value = ""
        for child in node.children:
            if child.type == "attribute_name":
                attr_name = _get_ts_node_text(child, source_bytes)
            elif child.type in ("attribute_value", "quoted_attribute_value"):
                attr_value = _get_ts_node_text(child, source_bytes)
        data["name"] = attr_name
        data["value"] = attr_value

    children: List[Dict[str, Any]] = []
    for child in node.children:
        # Filter out purely syntactic noise? User said "FULL DATA".
        # But maybe skip unnamed nodes that aren't text?
        # Tree-sitter 'named' nodes are usually what we want.
        if child.is_named or child.type == "text":
            children.append(_extract_template_tree(child, source_bytes))

    if children:
        data["children"] = children

    return data


def _analyze_ts_script_content(
    script_content: str, result: Dict[str, Any], is_module: bool = False
) -> None:
    """
    Analyzes TypeScript/JavaScript content from a Svelte script block.
    Extracts imports, exports, functions, props, state, and calls.
    """
    try:
        ts_parser = Parser(TS_LANGUAGE)
        ts_tree = ts_parser.parse(script_content.encode("utf-8", errors="ignore"))
        ts_root = ts_tree.root_node

        # Helper to run queries using correct cursor pattern
        def run_q(query_str: str, language: Any) -> list[Any]:
            try:
                q = Query(language, query_str)
                cursor = QueryCursor(q)
                matches = cursor.matches(ts_root)
                return matches
            except Exception:
                # logger.debug(f"DEBUG: Query error: {e_q}")
                return []

        # 1. Imports
        imports_query_str = """
        [
            (import_statement source: (string) @path)
            (call_expression
                function: (identifier) @req.fn
                arguments: (arguments (string) @path)
                (#match? @req.fn "^(require|import)$"))
        ]
        """
        for _, caps in run_q(imports_query_str, TS_LANGUAGE):
            for node in caps.get("path", []):
                path = _get_ts_node_text(node, script_content.encode("utf-8")).strip(
                    "\"'"
                )
                line = node.start_point[0] + 1
                cast(List[Dict[str, Any]], result.setdefault("imports", [])).append(
                    {"path": path, "line": line}
                )

        # 2. Exports (Robust)
        # We query for the export statement and then inspect its children
        # This avoids grammar specific node naming issues in the query itself
        export_queries = ["(export_statement) @exp"]

        for q_str in export_queries:
            for _, caps in run_q(q_str, TS_LANGUAGE):
                for node in caps.get("exp", []):
                    # Inspect children to find the name
                    name_node = None

                    # Search for identifier in children
                    def find_id(n: Optional[Node]) -> Optional[Node]:
                        if n is None:
                            return None
                        if n.type == "identifier":
                            return n
                        if n.type == "type_identifier":
                            return n
                        for i in range(n.child_count):
                            res = find_id(n.child(i))
                            if res:
                                return res
                        return None

                    # We want the identifier of the declaration, not the export keyword
                    # usually export -> declaration -> declarator -> identifier
                    # or export -> declaration -> identifier

                    # Skip the 'export' keyword node
                    start_index = 0
                    for i in range(node.child_count):
                        if node.child(i).type == "export":
                            start_index = i + 1
                            break

                    # Search in the remaining siblings (the declaration part)
                    for i in range(start_index, node.child_count):
                        child = node.child(i)
                        name_node = find_id(child)
                        if name_node:
                            break

                    if name_node:
                        name = _get_ts_node_text(
                            name_node, script_content.encode("utf-8")
                        )
                        line = name_node.start_point[0] + 1

                        if name not in [e["name"] for e in result.get("exports", [])]:
                            cast(
                                List[Dict[str, Any]], result.setdefault("exports", [])
                            ).append(
                                {
                                    "name": name,
                                    "line": line,
                                    "context": "module" if is_module else "instance",
                                }
                            )

                        # If it's a prop (export let/var in instance), also add to props
                        if not is_module:
                            if name not in [p["name"] for p in result.get("props", [])]:
                                cast(
                                    List[Dict[str, Any]], result.setdefault("props", [])
                                ).append({"name": name, "line": line})

        # 3. Functions (Robust)
        # Similar to exports, we use broad queries and inspect in Python
        functions_query_str = """
        [
            (function_declaration) @func
            (generator_function_declaration) @func
            (method_definition) @func
            (lexical_declaration) @func
        ]
        """

        try:
            ts_q_funcs = Query(TS_LANGUAGE, functions_query_str)
            ts_cursor_funcs = QueryCursor(ts_q_funcs)
            for _, caps in ts_cursor_funcs.matches(ts_root):
                for node in caps.get("func", []):
                    f_name = None

                    # Helper to extract name and params from function-like nodes
                    def extract_func_info(n: Node) -> Optional[str]:
                        name = None

                        # Direct function declarations
                        if n.type in (
                            "function_declaration",
                            "generator_function_declaration",
                            "async_function_declaration",
                            "method_definition",
                        ):
                            for i in range(n.child_count):
                                child = n.child(i)
                                if child and (
                                    child.type == "identifier"
                                    or child.type == "property_identifier"
                                ):
                                    name = _get_ts_node_text(
                                        child, script_content.encode("utf-8")
                                    )
                                elif child and child.type == "formal_parameters":
                                    # Extract params
                                    pass  # For now just identifying the function is enough

                        # Arrow functions / Function expressions in variables
                        elif n.type == "lexical_declaration":
                            # lexical_declaration -> variable_declarator -> name, value -> arrow_function
                            for i in range(n.child_count):
                                child = n.child(i)
                                if child and child.type == "variable_declarator":
                                    vd = child
                                    var_name = None
                                    is_func = False
                                    for j in range(vd.child_count):
                                        c = vd.child(j)
                                        if c and c.type == "identifier":
                                            var_name = _get_ts_node_text(
                                                c, script_content.encode("utf-8")
                                            )
                                        elif c and c.type in (
                                            "arrow_function",
                                            "function_expression",
                                        ):
                                            is_func = True
                                    if is_func and var_name:
                                        name = var_name

                        return name

                    f_name = extract_func_info(node)

                    if f_name:
                        line = node.start_point[0] + 1
                        # Avoid duplicates
                        if f_name not in [
                            f["name"] for f in result.get("functions", [])
                        ]:
                            cast(
                                List[Dict[str, Any]], result.setdefault("functions", [])
                            ).append({"name": f_name, "line": line})
        except Exception:
            # Fallback or just log, but the broad query should be safe
            pass

        # 4. State (Top-level variables, excluding exports/props)
        state_query_str = """
        (lexical_declaration
            (variable_declarator
                name: (identifier) @var.name))
        """
        known_exports = set(
            e["name"]
            for e in result.get("exports", [])
            if e.get("context") == ("module" if is_module else "instance")
        )

        for _, caps in run_q(state_query_str, TS_LANGUAGE):
            for node in caps.get("var.name", []):
                var_name = _get_ts_node_text(node, script_content.encode("utf-8"))
                # Exclude if it was exported (already captured as prop or export)
                if var_name not in known_exports:
                    cast(List[Dict[str, Any]], result.setdefault("state", [])).append(
                        {
                            "name": var_name,
                            "line": node.start_point[0] + 1,
                            "context": "module" if is_module else "instance",
                        }
                    )

        # 5. Function Calls (Significant ones?)
        # Let's capture all calls for now as requested
        calls_query_str = """
        (call_expression
            function: [
                (identifier) @call.id
                (member_expression property: (property_identifier) @call.prop)
            ]
        )
        """
        ts_q_calls = Query(TS_LANGUAGE, calls_query_str)
        for _, caps in QueryCursor(ts_q_calls).matches(ts_root):
            for _key, nodes in caps.items():
                for node in nodes:
                    call_name = _get_ts_node_text(node, script_content.encode("utf-8"))
                    cast(List[Dict[str, Any]], result.setdefault("calls", [])).append(
                        {
                            "name": call_name,
                            "line": node.start_point[0] + 1,
                            "context": "module" if is_module else "instance",
                        }
                    )

        # 6. Reactive Statements ($: ...) - Only if instance
        if not is_module:
            # Fix: labeled_statement in TS/JS does not have a 'statement' field.
            # It matches (labeled_statement (statement_identifier) @label (_) @stmt)
            reactive_query_str = """
             (labeled_statement
                 label: (statement_identifier) @label
                 (_) @stmt
                 (#eq? @label "$"))
             """
            try:
                ts_q_reactive = Query(TS_LANGUAGE, reactive_query_str.strip())
                for _, caps in QueryCursor(ts_q_reactive).matches(ts_root):
                    for node in caps.get("stmt", []):
                        stmt_text = _get_ts_node_text(
                            node, script_content.encode("utf-8")
                        )
                        cast(
                            List[Dict[str, Any]], result.setdefault("reactive", [])
                        ).append(
                            {"content": stmt_text, "line": node.start_point[0] + 1}
                        )
            except Exception as e:
                logger.debug(f"Reactive statement analysis failed: {e}")

    except Exception as e_inner:
        logger.debug(f"Inner TS analysis failed: {e_inner}")


def _analyze_svelte_file_ts(
    file_path: str, content: str, result: Dict[str, Any]
) -> None:
    """Analyzes Svelte file content using tree-sitter."""

    for key in [
        "imports",
        "exports",
        "functions",
        "props",
        "state",
        "calls",
        "reactive",
        "scripts",
        "stylesheets",
        "links",
        "images",
        "components",
        "_ts_tree",
        "template_tree",
        "template_outline",
        "logic",
    ]:
        result.setdefault(key, [] if key != "_ts_tree" else None)

    try:
        content_bytes = content.encode("utf8")
        parser = Parser(SVELTE_LANGUAGE)
        tree = parser.parse(content_bytes)
        result["_ts_tree"] = tree
        root_node = tree.root_node

        # 1. Extract and Analyze Scripts (Instance vs Module)
        script_query = Query(SVELTE_LANGUAGE, "(script_element) @script")
        cursor = QueryCursor(script_query)

        for _, captures in cursor.matches(root_node):
            for script_node in captures.get("script", []):
                # Check for context="module"
                is_module = False
                script_content = ""

                # Iterate children to find attributes and content
                for child in script_node.children:
                    if child.type == "start_tag":
                        # Check attributes
                        for sub in child.children:
                            if sub.type == "attribute":
                                attr_name = ""
                                attr_val = ""
                                for part in sub.children:
                                    if part.type == "attribute_name":
                                        attr_name = _get_ts_node_text(
                                            part, content_bytes
                                        )
                                    elif part.type == "quoted_attribute_value":
                                        attr_val = _get_ts_node_text(
                                            part, content_bytes
                                        )

                                if attr_name == "context" and "module" in attr_val:
                                    is_module = True
                    elif child.type in ("script_body", "raw_text", "text"):
                        if child.text:
                            script_content = child.text.decode("utf-8", errors="ignore")

                # Robust fallback: if no content found but node has text, it might be the content
                if not script_content.strip() and script_node.text:
                    full_text = script_node.text.decode("utf-8", errors="ignore")
                    # Strip tags if they are included in .text
                    import re

                    script_content = re.sub(r"^<script[^>]*>", "", full_text)
                    script_content = re.sub(r"</script>$", "", script_content)

                if script_content.strip():
                    # Find start line of the script content for offset
                    offset = 0
                    found_body = False
                    for child in script_node.children:
                        if child.type in ("script_body", "raw_text", "text"):
                            offset = child.start_point[0]
                            found_body = True
                            break
                    if not found_body:
                        offset = script_node.start_point[0]

                    temp_result: Dict[str, List[Any]] = {}
                    _analyze_ts_script_content(
                        script_content, cast(Dict[str, Any], temp_result), is_module
                    )

                    # Merge and adjust lines
                    for key, items in temp_result.items():
                        if key == "scripts":
                            continue  # Skip scripts list if any
                        for item in items:
                            if isinstance(item, dict) and "line" in item:
                                item["line"] += offset
                            cast(
                                List[Dict[str, Any]], result.setdefault(key, [])
                            ).append(cast(Dict[str, Any], item))

                # Store script content (NEW) - Avoid duplicates if script block matches multiple patterns
                # We can key by start line to avoid dups
                script_key = f"{script_node.start_point[0]}"
                if script_key not in result.setdefault("_seen_scripts", set()):
                    result["_seen_scripts"].add(script_key)
                    result["scripts"].append(
                        {
                            "content": script_content,
                            "line": script_node.start_point[0] + 1,
                        }
                    )

        # 2. Extract Styles
        style_query = Query(
            SVELTE_LANGUAGE,
            "((style_element (raw_text) @content)) ((style_element (start_tag) (raw_text) @content))",
        )
        cursor_style = QueryCursor(style_query)
        for _, captures in cursor_style.matches(root_node):
            for node in captures.get("content", []):
                style_content = _get_ts_node_text(node, content_bytes)
                result["stylesheets"].append(
                    {"content": style_content, "line": node.start_point[0]}
                )

        # 3. Extract Links (HTML-style)
        link_queries = {
            "links": '(element (start_tag (tag_name) @tag (#eq? @tag "a") (attribute (attribute_name) @name (#eq? @name "href") (quoted_attribute_value (attribute_value) @path))))',
            "images": '(element (start_tag (tag_name) @tag (#eq? @tag "img") (attribute (attribute_name) @name (#eq? @name "src") (quoted_attribute_value (attribute_value) @path))))',
        }
        for q_name, q_str in link_queries.items():
            query = Query(SVELTE_LANGUAGE, q_str)
            cursor = QueryCursor(query)
            for _, captures in cursor.matches(root_node):
                for node in captures.get("path", []):
                    url = _get_ts_node_text(node, content_bytes)
                    if url and not url.startswith(
                        ("#", "http:", "https:", "mailto:", "tel:", "data:")
                    ):
                        line = node.start_point[0]
                        if q_name == "links":
                            result["links"].append({"url": url, "line": line})
                        elif q_name == "images":
                            result["images"].append({"url": url, "line": line})

        # 4. Extract Components (Capitalized Tags)
        # Matches: <MyComponent ... /> or <MyComponent>...</MyComponent>
        # We query both start_tag (for <Comp>...</Comp>) and self_closing_tag (for <Comp />)
        # and filter by capitalization.
        component_query_str = """
        (start_tag (tag_name) @tag)
        (self_closing_tag (tag_name) @tag)
        """
        try:
            comp_query = Query(SVELTE_LANGUAGE, component_query_str)
            comp_cursor = QueryCursor(comp_query)
            result.setdefault("components", [])
            seen_comps: Set[str] = set()
            for _, captures in comp_cursor.matches(root_node):
                for node in captures.get("tag", []):
                    tag_name = _get_ts_node_text(node, content_bytes)
                    # Heuristic: Components usually start with Uppercase
                    if tag_name and tag_name[0].isupper():
                        if tag_name not in seen_comps:
                            seen_comps.add(tag_name)
                            result["components"].append(
                                {"name": tag_name, "line": node.start_point[0] + 1}
                            )
        except Exception as e_comp:
            logger.debug(f"Svelte component extraction failed: {e_comp}")

        # 5. Extract Logic Blocks (Structure)
        # Identify if/each/await blocks to show template control flow
        logic_queries = {
            "if": "(if_statement) @block",
            "each": "(each_statement) @block",
            "await": "(await_statement) @block",
        }
        for logic_type, q_str in logic_queries.items():
            try:
                l_query = Query(SVELTE_LANGUAGE, q_str)
                l_cursor = QueryCursor(l_query)
                for _, captures in l_cursor.matches(root_node):
                    for node in captures.get("block", []):
                        # Extract the condition/expression if possible
                        # For 'if', it's usually the first child after 'if'
                        # For 'each', it's after 'each'
                        # We'll just grab the text of the opening part to be safe and simple
                        block_text = _get_ts_node_text(node, content_bytes)
                        # Truncate to first line or reasonable length to show "essence"
                        first_line = block_text.split("\n")[0].strip()

                        result.setdefault("logic", []).append(
                            {
                                "type": logic_type,
                                "content": first_line,
                                "line": node.start_point[0] + 1,
                            }
                        )
            except Exception:
                pass

        # 6. Full Template Tree (NEW)
        try:
            # Extract children of the root document, excluding script/style
            template_children: List[Dict[str, Any]] = []
            for child in root_node.children:
                if child.type not in ("script_element", "style_element"):
                    template_children.append(
                        _extract_template_tree(child, content_bytes)
                    )

            if template_children:
                result["template_tree"] = template_children

                # Build a compact outline for SES/token-efficient map storage.
                outline: List[str] = []
                max_outline_lines = 500

                def _outline(nodes: List[Dict[str, Any]], depth: int = 0) -> None:
                    if len(outline) >= max_outline_lines:
                        return
                    indent = "  " * depth
                    for node_data in nodes:
                        if len(outline) >= max_outline_lines:
                            return
                        n_type = cast(str, node_data.get("type", ""))
                        children = cast(
                            List[Dict[str, Any]], node_data.get("children", [])
                        )

                        if n_type == "element":
                            tag = ""
                            attrs = ""
                            for child_data in children:
                                c_type = child_data.get("type")
                                if c_type in ("start_tag", "self_closing_tag"):
                                    for sub in cast(
                                        List[Dict[str, Any]],
                                        child_data.get("children", []),
                                    ):
                                        s_type = sub.get("type")
                                        if s_type == "tag_name":
                                            tag = cast(str, sub.get("content", ""))
                                        elif s_type == "attribute":
                                            a_name = cast(str, sub.get("name", ""))
                                            a_val = cast(str, sub.get("value", "")).strip(
                                                "\"'"
                                            )
                                            if a_name == "id" and a_val:
                                                attrs += f"#{a_val}"
                                            elif a_name == "class" and a_val:
                                                attrs += "." + ".".join(a_val.split())
                                    break
                            if tag:
                                outline.append(f"{indent}<{tag}{attrs}>")
                            _outline(children, depth + 1)
                        elif n_type in (
                            "if_statement",
                            "each_statement",
                            "await_statement",
                            "key_statement",
                        ):
                            # Try to get the first line of content for logic blocks
                            head = cast(str, node_data.get("content", "")).split("\n")[0].strip()
                            if head:
                                outline.append(f"{indent}{head}")
                            _outline(children, depth + 1)
                        elif n_type == "text":
                            text_content = cast(str, node_data.get("content", "")).strip()
                            if text_content and len(text_content) > 3:
                                # Escape newlines for compact outline
                                text_content = " ".join(text_content.split())
                                if len(text_content) > 4000:
                                    text_content = text_content[:4000] + "..."
                                outline.append(f"{indent}{text_content}")
                        else:
                            _outline(children, depth)

                _outline(template_children)
                if outline:
                    result["template_outline"] = outline
        except Exception as e_tree:
            logger.debug(f"Svelte template tree extraction failed: {e_tree}")

        result["analysis_kind"] = "svelte"
    except Exception as e:
        logger.error(f"Error parsing Svelte {file_path} with tree-sitter: {e}")
        result["error"] = f"Tree-sitter Svelte parsing error: {e}"


def _analyze_sql_file_ts(file_path: str, content: str, result: Dict[str, Any]) -> None:
    """Analyzes SQL file content using tree-sitter."""
    result.setdefault("_ts_tree", None)
    result.setdefault("links", [])
    result.setdefault("definitions", [])  # New field for SQL statements
    result.setdefault("columns", [])  # New field for column info
    result.setdefault("relationships", [])  # New field for FKs
    result.setdefault("tables_defined", [])  # EXPLICIT: Tables created in this file
    result.setdefault("tables_referenced", [])  # EXPLICIT: Tables used in this file

    try:
        content_bytes = content.encode("utf8")
        parser = Parser(SQL_LANGUAGE)
        tree = parser.parse(content_bytes)
        result["_ts_tree"] = tree
        root_node = tree.root_node

        def _add_table_defined(raw_name: str) -> None:
            normalized = _normalize_sql_identifier_text(raw_name).lower()
            if normalized and normalized not in result["tables_defined"]:
                result["tables_defined"].append(normalized)

        def _add_table_referenced(raw_name: str) -> None:
            normalized = _normalize_sql_identifier_text(raw_name).lower()
            if normalized and normalized not in result["tables_referenced"]:
                result["tables_referenced"].append(normalized)

        # 1. Capture Top-level Statements (Definitions and References)
        # Using specific positional patterns to find table names properly
        # Fixed syntax and capture names based on user feedback and AST inspection.
        # "update" table is in a "relation" child.
        # "delete" table is in a sibling "from" node.
        sql_query_str = """
        [
            (create_table (object_reference) @table_defined) @create_stmt
            (create_view (object_reference) @table_defined) @view_stmt
            (cte (identifier) @table_defined) @cte_stmt
            
            (insert (object_reference) @table_used) @insert_stmt
            (update (object_reference) @table_used) @update_stmt
            (update (relation (object_reference) @table_used)) @update_stmt
            
            (statement (delete) (from (object_reference) @table_used)) @delete_stmt
            
            (statement (keyword_truncate) (keyword_table)? (object_reference) @table_used) @truncate_stmt
            (statement (keyword_merge) (keyword_into)? (object_reference) @table_used) @merge_stmt
            
            (alter_table (object_reference) @table_used) @alter_stmt
            (create_index (object_reference) @table_used) @index_stmt
            (drop_table (object_reference) @table_used) @drop_stmt
            
            (from (object_reference) @table_used) @from_clause
            (from (relation (object_reference) @table_used)) @from_clause
            (join (relation (object_reference) @table_used)) @join_clause
        ]
        """
        try:
            query = Query(SQL_LANGUAGE, sql_query_str.strip())
            cursor = QueryCursor(query)
            for _, captures in cursor.matches(root_node):
                # Definitions (CREATE TABLE, CREATE VIEW, CTE)
                if "table_defined" in captures:
                    for node in captures["table_defined"]:
                        table = _extract_sql_identifier(node, content_bytes)
                        if table:
                            _add_table_defined(table)

                # References (INSERT, UPDATE, DELETE, ALTER, INDEX, SELECT, DROP, TRUNCATE, MERGE, JOIN)
                if "table_used" in captures:
                    for node in captures["table_used"]:
                        table = _extract_sql_identifier(node, content_bytes)
                        if table:
                            _add_table_referenced(table)

                # Generic summary for definitions list (backward compatibility)
                for name, nodes in captures.items():
                    if name.endswith("_stmt") or name.endswith("_clause"):
                        for node in nodes:
                            stmt_text = _get_ts_node_text(node, content_bytes)
                            summary = " ".join(
                                [l.strip() for l in stmt_text.split("\n") if l.strip()]
                            )

                            cast(List[Dict[str, Any]], result["definitions"]).append(
                                {
                                    "type": name.replace("_stmt", "").replace(
                                        "_clause", ""
                                    ),
                                    "summary": summary,
                                    "line": node.start_point[0] + 1,
                                }
                            )
        except Exception as qe:
            logger.debug(f"SQL statement query error: {qe}")

        # 1b. Regex fallback for dialect-specific statements (e.g. PostgreSQL dump COPY)
        # This supplements tree-sitter coverage for statements that may parse inconsistently.
        regex_fallbacks: List[Tuple[str, str, str]] = [
            (
                "create",
                "table_defined",
                r"(?im)^\s*create\s+(?:or\s+replace\s+)?table\s+(?:if\s+not\s+exists\s+)?([^\s(;]+)",
            ),
            (
                "view",
                "table_defined",
                r"(?im)^\s*create\s+(?:or\s+replace\s+)?view\s+([^\s(;]+)",
            ),
            ("insert", "table_used", r"(?im)^\s*insert\s+into\s+([^\s(;]+)"),
            ("update", "table_used", r"(?im)^\s*update\s+([^\s(;]+)"),
            ("delete", "table_used", r"(?im)^\s*delete\s+from\s+([^\s(;]+)"),
            ("alter", "table_used", r"(?im)^\s*alter\s+table\s+([^\s(;]+)"),
            ("drop", "table_used", r"(?im)^\s*drop\s+table(?:\s+if\s+exists)?\s+([^\s(;]+)"),
            ("truncate", "table_used", r"(?im)^\s*truncate\s+(?:table\s+)?([^\s(;]+)"),
            ("copy", "table_used", r"(?im)^\s*copy\s+([^\s(]+)\s*\("),
        ]
        try:
            for stmt_type, bucket, pattern in regex_fallbacks:
                for match in re.finditer(pattern, content):
                    raw_table = match.group(1).strip()
                    if not raw_table:
                        continue

                    if bucket == "table_defined":
                        _add_table_defined(raw_table)
                    else:
                        _add_table_referenced(raw_table)

                    cast(List[Dict[str, Any]], result["definitions"]).append(
                        {
                            "type": stmt_type,
                            "summary": f"{stmt_type.upper()} {raw_table}",
                            "line": content[: match.start()].count("\n") + 1,
                        }
                    )
        except Exception as re_fallback_err:
            logger.debug(f"SQL regex fallback error: {re_fallback_err}")

        # 2. Capture Column Definitions
        col_query_str = """
        (column_definition
            name: (_) @name
            type: (_) @type
        ) @col
        """
        try:
            col_query = Query(SQL_LANGUAGE, col_query_str)
            col_cursor = QueryCursor(col_query)
            for _, captures in col_cursor.matches(root_node):
                node_name = _extract_sql_identifier(captures["name"][0], content_bytes)
                node_type = _get_ts_node_text(captures["type"][0], content_bytes)
                result["columns"].append(
                    {
                        "name": node_name,
                        "type": node_type,
                        "line": captures["col"][0].start_point[0] + 1,
                    }
                )
        except Exception as qe:
            logger.debug(f"SQL column query error: {qe}")

        # 3. Capture Foreign Keys / Relationships
        # Supports both inline (column_definition) and table-level (add_constraint)
        fk_query_str = """
        [
          (column_definition
            (identifier) @name
            (keyword_references)
            (object_reference) @ref_table
            (identifier) @ref_col
          )
          (add_constraint
            (identifier) @constraint_name
            (constraint
              (keyword_foreign)
              (keyword_key)
              (ordered_columns (column (identifier) @name))
              (keyword_references)
              (object_reference) @ref_table
              (identifier) @ref_col
            )
          )
        ]
        """
        try:
            fk_query = Query(SQL_LANGUAGE, fk_query_str)
            fk_cursor = QueryCursor(fk_query)
            for _, captures in fk_cursor.matches(root_node):
                name_nodes = captures.get("name")
                ref_table_node = captures.get("ref_table")
                ref_col_nodes = captures.get("ref_col")

                if name_nodes and ref_table_node:
                    col_name = _extract_sql_identifier(name_nodes[0], content_bytes)
                    target_table = _extract_sql_identifier(
                        ref_table_node[0], content_bytes
                    )
                    target_table_lower = target_table.lower() if target_table else ""

                    target_col = ""
                    if ref_col_nodes:
                        target_col = _extract_sql_identifier(
                            ref_col_nodes[0], content_bytes
                        )

                    result["relationships"].append(
                        {
                            "source_col": col_name,
                            "target_table": target_table_lower,
                            "target_col": target_col,
                        }
                    )
                    # Add as reference as well
                    if (
                        target_table_lower
                        and target_table_lower not in result["tables_referenced"]
                    ):
                        result["tables_referenced"].append(target_table_lower)
        except Exception as qe:
            logger.debug(f"SQL relationship query error: {qe}")

        # 4. Extract INSERT Data Samples (for SES)
        try:
            # We want to capture string literals from INSERT statements.
            max_values_per_file = 50
            captured_values = 0
            if "inserts" not in result:
                result["inserts"] = []
            
            # Helper to extract relevant strings from a node tree
            def _extract_strings_from_node(n: Any):
                nonlocal captured_values
                if captured_values >= max_values_per_file:
                    return

                # 1. Standard strings are aliased as "literal" in DerekStride/tree-sitter-sql
                # 2. Postgres-style dollar-quoted strings are anonymous nodes like $tag$ content $tag$
                #    We need to check the text content for the pattern or use a fallback check if node type matches
                
                is_standard_literal = n.type == "literal"
                # Check for dollar quoted string (anonymous node check heuristic)
                is_dollar_quoted = False
                if not is_standard_literal and n.type == "string": # Some grammars map it to string
                     is_dollar_quoted = True
                
                # Check text content for dollar quotes if type check fails or is ambiguous
                text = _get_ts_node_text(n, content_bytes)
                if not is_dollar_quoted and len(text) > 4 and text.startswith("$") and "$" in text[1:]:
                     # Basic check for $...$...
                     import re
                     if re.match(r"^\$[^\$]*\$.*\$[^\$]*\$$", text, re.DOTALL):
                         is_dollar_quoted = True

                if is_standard_literal or is_dollar_quoted:
                    clean_text = text
                    if is_standard_literal:
                         # Check if it starts/ends with quotes
                        if len(text) >= 2 and text[0] in ("'", '"') and text[-1] == text[0]:
                            clean_text = text[1:-1]
                        else:
                            # Might be a number or boolean, skip
                            return
                    
                    if len(clean_text) > 3 or is_dollar_quoted:
                        # Try to find table name if we are in an INSERT
                        current = n
                        table_name = "unknown"
                        
                        # Traverse up to find 'insert' node
                        # Node name is 'insert', not 'insert_statement' in this grammar
                        while current and current.type != "insert":
                            current = current.parent
                        
                        if current:
                            # In tree-sitter-sql (DerekStride), table is an object_reference child
                            # There is no 'table' field.
                            table_node = None
                            for child in current.children:
                                if child.type == "object_reference":
                                    table_node = child
                                    break
                            
                            if table_node:
                                table_name = _extract_sql_identifier(table_node, content_bytes)

                        # Find/Create entry for this table
                        inserts_list = cast(List[Dict[str, Any]], result.get("inserts", []))
                        
                        # Ensure we work with the referenced list in the result dict
                        if "inserts" not in result:
                             result["inserts"] = []
                             inserts_list = cast(List[Dict[str, Any]], result["inserts"])

                        # Check if we already have an entry for this table
                        table_entry: Optional[Dict[str, Any]] = next((i for i in inserts_list if i.get("table") == table_name), None)
                        
                        if not table_entry:
                            table_entry = {"table": table_name, "columns": {}, "values": []} 
                            inserts_list.append(table_entry)
                        
                        # Add value to the 'values' list for this table, for SES generation
                        # We use 'columns' in embedding_manager, but here we just capturing raw values
                        # Let's map it to a 'literals' key or similar that embedding_manager understands
                        cols_dict = cast(Dict[str, str], table_entry.get("columns", {}))
                        
                        # If we just have values, maybe we can store them as "value_N": "literal"
                        pk = f"val_{len(cols_dict)}"
                        if "columns" not in table_entry:
                            table_entry["columns"] = cols_dict
                        
                        # Avoid duplicates in the same table entry
                        if clean_text not in cols_dict.values():
                             cols_dict[pk] = clean_text
                             captured_values += 1

                for child in n.children:
                    _extract_strings_from_node(child)

            # Query for INSERT nodes
            insert_nodes_query = Query(SQL_LANGUAGE, "(insert) @ins")
            ins_cursor = QueryCursor(insert_nodes_query)
            
            for _, captures in ins_cursor.matches(root_node):
                if captured_values >= max_values_per_file:
                    break
                for insert_node in captures.get("ins", []):
                    _extract_strings_from_node(insert_node)

        except Exception as e_data:
            pass

        result["analysis_kind"] = "sql"
    except Exception as e:
        logger.error(f"Error parsing SQL {file_path} with tree-sitter: {e}")
        result["error"] = f"Tree-sitter SQL parsing error: {e}"


def _analyze_html_file_ts(file_path: str, content: str, result: Dict[str, Any]) -> None:
    """Analyzes HTML file content using tree-sitter."""
    for key in ["links", "scripts", "stylesheets", "images", "_ts_tree"]:
        result.setdefault(key, [] if key != "_ts_tree" else None)

    queries = {
        "scripts": '(script_element (start_tag (attribute (attribute_name) @name (#eq? @name "src") (quoted_attribute_value (attribute_value) @path))))',
        "stylesheets": '(element (start_tag (tag_name) @tag (#eq? @tag "link") (attribute (attribute_name) @name (#eq? @name "href") (quoted_attribute_value (attribute_value) @path))))',
        "images": '(element (start_tag (tag_name) @tag (#eq? @tag "img") (attribute (attribute_name) @name (#eq? @name "src") (quoted_attribute_value (attribute_value) @path))))',
        "links": '(element (start_tag (tag_name) @tag (#eq? @tag "a") (attribute (attribute_name) @name (#eq? @name "href") (quoted_attribute_value (attribute_value) @path))))',
    }

    lang_queries = {
        name: Query(HTML_LANGUAGE, q_str) for name, q_str in queries.items()
    }

    try:
        content_bytes = content.encode("utf8")
        # --- FIX (MAJOR): Create a local, thread-safe parser instance. ---
        parser = Parser(HTML_LANGUAGE)
        tree = parser.parse(content_bytes)
        result["_ts_tree"] = tree
        root_node = tree.root_node

        for query_name, query in lang_queries.items():
            cursor = QueryCursor(query)
            for _pattern_index, captures_dict in cursor.matches(root_node):
                path_nodes = captures_dict.get("path")
                if not path_nodes:
                    continue

                for node in path_nodes:
                    line = node.start_point[0] + 1
                    url = _get_ts_node_text(node, content_bytes)
                    if url and not url.startswith(
                        ("#", "http:", "https:", "mailto:", "tel:", "data:")
                    ):
                        if query_name == "scripts":
                            result["scripts"].append({"url": url, "line": line})
                        elif query_name == "stylesheets":
                            result["stylesheets"].append({"url": url, "line": line})
                        elif query_name == "images":
                            result["images"].append({"url": url, "line": line})
                        elif query_name == "links":
                            result["links"].append({"url": url, "line": line})

        for key in ["links", "scripts", "stylesheets", "images"]:
            if result.get(key):
                unique_items = {frozenset(d.items()) for d in result[key]}
                result[key] = [dict(fs) for fs in unique_items]

    except Exception as e:
        logger.error(
            f"Error parsing HTML {file_path} with tree-sitter: {e}", exc_info=True
        )
        result["error"] = f"Tree-sitter HTML parsing error: {e}"


def _analyze_css_file_ts(file_path: str, content: str, result: Dict[str, Any]) -> None:
    """Analyzes CSS file content using tree-sitter."""
    result.setdefault("imports", [])
    result.setdefault("_ts_tree", None)

    # CSS_PARSER, CSS_LANGUAGE, Query, QueryCursor are directly imported and initialized at module load.
    # If they are None, a hard import error would have occurred.
    # No explicit check for TREE_SITTER_AVAILABLE as imports are now direct.

    query_str = """
    (import_statement (string_value) @path)
    """

    query = Query(CSS_LANGUAGE, query_str)

    try:
        content_bytes = content.encode("utf8")
        # --- FIX (MAJOR): Create a local, thread-safe parser instance. ---
        parser = Parser(CSS_LANGUAGE)
        tree = parser.parse(content_bytes)
        result["_ts_tree"] = tree
        root_node = tree.root_node

        cursor = QueryCursor(query)
        for _pattern_index, captures_dict in cursor.matches(root_node):
            path_nodes = captures_dict.get("path")
            if not path_nodes:
                continue
            for node in path_nodes:
                line = node.start_point[0] + 1
                url = _get_ts_node_text(node, content_bytes).strip("'\"")
                if url and not url.startswith(("#", "http:", "https:", "data:")):
                    result["imports"].append({"url": url, "line": line})

        if result.get("imports"):
            unique_items = {frozenset(d.items()) for d in result["imports"]}
            result["imports"] = [dict(fs) for fs in unique_items]

    except Exception as e:
        logger.error(
            f"Error parsing CSS {file_path} with tree-sitter: {e}", exc_info=True
        )
        result["error"] = f"Tree-sitter CSS parsing error: {e}"


def _merge_analysis_results(primary: Dict[str, Any], secondary: Dict[str, Any]) -> None:
    """
    Merges secondary analysis results into primary results with deduplication.
    Primary results (e.g., from Native AST) are preferred.
    Only adds items from secondary (e.g., Tree-Sitter) if they are not already in primary.
    """
    # Merge lists of dictionaries based on 'name' (and 'line' if available/relevant)
    keys_to_merge = ["functions", "classes", "globals_defined", "imports", "calls"]

    for key in keys_to_merge:
        if key not in secondary or not secondary[key]:
            continue

        if key not in primary:
            primary[key] = []

        primary_items = primary[key]
        secondary_items = secondary[key]

        # Create a set of existing identifiers in primary for O(1) lookup
        # For imports, it's usually a list of strings. For others, it's a list of dicts.
        existing_ids: Set[str] = set()

        if key == "imports":
            # Imports can be strings or dicts (if detailed)
            for item in primary_items:
                if isinstance(item, str):
                    existing_ids.add(item)
                elif isinstance(item, dict) and "path" in item:
                    existing_ids.add(cast(str, item["path"]))
        elif key == "calls":
            for item in primary_items:
                if isinstance(item, dict) and "target_name" in item:
                    # Use target_name + line as unique key to avoid merging same call on same line
                    # But if line differs, it might be a different call.
                    # However, AST usually captures all calls.
                    # If TS captures a call AST missed, we want it.
                    # If TS captures same call, we skip.
                    # Let's use target_name + line approximation.
                    dict_item = cast(Dict[str, Any], item)
                    line = dict_item.get("line", -1)
                    existing_ids.add(f"{dict_item['target_name']}:{line}")
        else:
            # functions, classes, globals_defined
            for item in primary_items:
                if isinstance(item, dict) and "name" in item:
                    # Use name + line to be specific, or just name?
                    # If we have multiple functions with same name (overloads?), line helps.
                    dict_item = cast(Dict[str, Any], item)
                    line = dict_item.get("line", -1)
                    existing_ids.add(f"{dict_item['name']}:{line}")

        # Merge unique items from secondary
        for item in secondary_items:
            if key == "imports":
                item_id = item if isinstance(item, str) else item.get("path")
                if item_id and item_id not in existing_ids:
                    primary_items.append(item)
            elif key == "calls":
                if isinstance(item, dict) and "target_name" in item:
                    d_item = cast(Dict[str, Any], item)
                    line = d_item.get("line", -1)
                    unique_key = f"{d_item['target_name']}:{line}"
                    if unique_key not in existing_ids:
                        primary_items.append(item)
            else:
                if isinstance(item, dict) and "name" in item:
                    dict_item = cast(Dict[str, Any], item)
                    line = dict_item.get("line", -1)
                    unique_key = f"{dict_item['name']}:{line}"
                    if unique_key not in existing_ids:
                        primary_items.append(item)

def _analyze_python_file_ts(
    file_path: str, content: str, result: Dict[str, Any]
) -> None:
    """
    Tree-sitter based analysis for Python (.py) files.
    Extracts imports, functions, classes, and calls.
    """
    result.setdefault("imports", [])
    result.setdefault("functions", [])
    result.setdefault("classes", [])
    result.setdefault("calls", [])
    result.setdefault("globals_defined", [])
    result.setdefault("_ts_tree", None)

    try:
        content_bytes = content.encode("utf-8", errors="ignore")
        parser = Parser(PY_LANGUAGE)
        tree = parser.parse(content_bytes)
        result["_ts_tree"] = tree

        # --- Queries ---

        # Imports
        imports_query_str = """
        [
            (import_statement
                name: (dotted_name) @import_name)
            (import_from_statement
                module_name: (dotted_name) @module_name)
        ]
        """

        # Functions
        functions_query_str = """
        (function_definition
            name: (identifier) @func_name) @function
        """

        # Classes
        classes_query_str = """
        (class_definition
            name: (identifier) @class_name) @class
        """

        # Calls
        # Query to capture function calls.
        # We want to capture:
        # 1. Simple calls: func() -> func
        # 2. Attribute calls: obj.method() -> method, potential_source=obj
        calls_query_str = """
        (call
            function: (identifier) @call_func
        )
        (call
            function: (attribute 
                object: (_) @call_obj
                attribute: (identifier) @call_attr
            )
        )
        """

        # Execute Queries

        # 1. Imports
        # Use tree_sitter.Query constructor to avoid deprecation warning
        query = tree_sitter.Query(PY_LANGUAGE, imports_query_str)
        cursor = tree_sitter.QueryCursor(query)
        captures = cursor.captures(tree.root_node)

        for name, nodes in captures.items():
            for node in nodes:
                text = _get_ts_node_text(node, content_bytes)
                if name == "import_name":
                    result["imports"].append(text)
                elif name == "module_name":
                    result["imports"].append(text)

        # 2. Functions
        query = tree_sitter.Query(PY_LANGUAGE, functions_query_str)
        cursor = tree_sitter.QueryCursor(query)
        captures = cursor.captures(tree.root_node)

        for name, nodes in captures.items():
            if name == "func_name":
                for node in nodes:
                    func_name = _get_ts_node_text(node, content_bytes)
                    # Find parent function_definition for line number
                    parent = node.parent
                    while parent and parent.type != "function_definition":
                        parent = parent.parent
                    line = (
                        parent.start_point[0] + 1 if parent else node.start_point[0] + 1
                    )
                    result["functions"].append({"name": func_name, "line": line})

        # 3. Classes
        query = tree_sitter.Query(PY_LANGUAGE, classes_query_str)
        cursor = tree_sitter.QueryCursor(query)
        captures = cursor.captures(tree.root_node)

        for name, nodes in captures.items():
            if name == "class_name":
                for node in nodes:
                    class_name = _get_ts_node_text(node, content_bytes)
                    parent = node.parent
                    while parent and parent.type != "class_definition":
                        parent = parent.parent
                    line = (
                        parent.start_point[0] + 1 if parent else node.start_point[0] + 1
                    )
                    result["classes"].append({"name": class_name, "line": line})

        # 4. Calls
        calls_query = tree_sitter.Query(PY_LANGUAGE, calls_query_str)
        cursor_calls = tree_sitter.QueryCursor(calls_query)

        matches_calls = cursor_calls.matches(tree.root_node)

        for _, match_captures in matches_calls:
            # match_captures is { "capture_name": [Node, ...] }
            # Usually one node per capture name in a single match for these patterns.

            target_name = None
            potential_source = None
            line = -1

            if "call_func" in match_captures:
                node = match_captures["call_func"][0]
                target_name = _get_ts_node_text(node, content_bytes)
                line = node.start_point[0] + 1
            elif "call_attr" in match_captures:
                node = match_captures["call_attr"][0]
                target_name = _get_ts_node_text(node, content_bytes)
                line = node.start_point[0] + 1

                if "call_obj" in match_captures:
                    obj_node = match_captures["call_obj"][0]
                    potential_source = _get_ts_node_text(obj_node, content_bytes)
                    # Construct full name if possible: obj.method
                    if potential_source:
                        target_name = f"{potential_source}.{target_name}"

            if target_name and _is_useful_call(target_name, potential_source):
                # Avoid duplicates if already in result["calls"] (from AST merge or other)
                # But here we are just building the TS result list.
                result["calls"].append(
                    {
                        "target_name": target_name,
                        "potential_source": potential_source,
                        "line": line,
                    }
                )

    except Exception as e:
        logger.warning(f"Tree-sitter Python analysis failed for {file_path}: {e}")


# --- End of dependency_analyzer.py ---
