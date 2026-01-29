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
from typing import Any, Dict, List, Optional, Set, Tuple

# Global tree-sitter variables and their corresponding language and parser instances.
# These are imported and initialized directly at module load time.
import tree_sitter
import tree_sitter_css as tscss
import tree_sitter_html as tshtml
import tree_sitter_javascript as tsjavascript
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

# --- FIX (MAJOR): Remove global Parser instances. They are NOT thread-safe. ---
# Parsers will now be created locally within each analysis function.
# CSS_PARSER = Parser(CSS_LANGUAGE) # REMOVED
# HTML_PARSER = Parser(HTML_LANGUAGE) # REMOVED
# JS_PARSER = Parser(JS_LANGUAGE) # REMOVED
# TS_PARSER = Parser(TS_LANGUAGE) # REMOVED
# TSX_PARSER = Parser(TSX_LANGUAGE) # REMOVED

from cline_utils.dependency_system.utils.cache_manager import (
    cache_manager,
    cached,
    invalidate_dependent_entries,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager

# Import only from utils, core, and io layers
from cline_utils.dependency_system.utils.path_utils import (
    get_file_type as util_get_file_type,
)
from cline_utils.dependency_system.utils.path_utils import (
    get_project_root,
    is_subpath,
    normalize_path,
)

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
    def get_key(item):
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
            grouped[key].extend(line)
        else:
            grouped[key].append(line)

    consolidated = []
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


# --- Main Analysis Function ---
@cached(
    "file_analysis",
    key_func=lambda file_path, force=False: f"analyze_file:{normalize_path(str(file_path))}:{(os.path.getmtime(str(file_path)) if os.path.exists(str(file_path)) else 0)}:{force}",
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
            ts_result = {
                "file_path": norm_file_path,
                "file_type": file_type,
                "imports": [],
                "functions": [],
                "classes": [],
                "calls": [],
                "globals_defined": [],
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
            _analyze_markdown_file(norm_file_path, content, analysis_result)

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
            "scripts": ["url"],  # HTML
            "stylesheets": ["url"],  # HTML
            "images": ["url"],  # HTML
        }

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
            imports_list: List[Any] = analysis_result.get("imports", []) or []
            for imp in imports_list:
                if isinstance(imp, dict) and imp.get("path"):
                    analysis_result["ast_verified_links"].append(
                        {
                            "source_file": norm_file_path,
                            "target_spec": imp.get("path"),
                            "line": imp.get("line"),
                            "via": "import",
                            "confidence": 0.9,
                        }
                    )
            # Also emit links from export re-exports like `export ... from 'x'`
            exports_list: List[Any] = analysis_result.get("exports", []) or []
            for ex in exports_list:
                if isinstance(ex, dict) and ex.get("from"):
                    analysis_result["ast_verified_links"].append(
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

        logger.debug(f"DEBUG DA: tree.body processed for {file_path}.")

        # Pass 2: ast.walk for detailed analysis
        for node in ast.walk(tree):
            # Decorators (for all functions/classes, top-level or nested)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
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

                params = []
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
                if target_full_name and _is_useful_call(
                    target_full_name, potential_source, imports_map
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
        norm_imports: List[Dict[str, Any]] = []
        for imp in result["imports"]:
            if isinstance(imp, dict):
                if "path" in imp:
                    norm_imports.append(imp)
                elif "source" in imp:
                    norm_imports.append(
                        {
                            "path": imp["source"],
                            **{k: v for k, v in imp.items() if k != "source"},
                        }
                    )
            elif isinstance(imp, str):
                norm_imports.append({"path": imp})
        result["imports"] = norm_imports
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
        norm_imports: List[Dict[str, Any]] = []
        for imp in result["imports"]:
            if isinstance(imp, dict):
                if "path" in imp:
                    norm_imports.append(imp)
                elif "source" in imp:
                    norm_imports.append(
                        {
                            "path": imp["source"],
                            **{k: v for k, v in imp.items() if k != "source"},
                        }
                    )
            elif isinstance(imp, str):
                norm_imports.append({"path": imp})
        result["imports"] = norm_imports
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
        norm_imports: List[Dict[str, Any]] = []
        for imp in result["imports"]:
            if isinstance(imp, dict):
                if "path" in imp:
                    norm_imports.append(imp)
                elif "source" in imp:
                    norm_imports.append(
                        {
                            "path": imp["source"],
                            **{k: v for k, v in imp.items() if k != "source"},
                        }
                    )
            elif isinstance(imp, str):
                norm_imports.append({"path": imp})
        result["imports"] = norm_imports
        result["analysis_kind"] = "tsx"
        # Ensure exports list exists for re-export links
        try:
            result.setdefault("exports", [])
        except Exception:
            pass


def _analyze_javascript_file_regex_fallback(
    file_path: str, content: str, result: Dict[str, Any]
) -> None:
    """
    Fallback analysis for JS files using regex (when tree-sitter unavailable or errors occur).
    """
    import_matches = JAVASCRIPT_IMPORT_PATTERN.finditer(content)
    # Ensure imports are not duplicated if fallback is called after partial AST analysis
    existing_imports = set(d["path"] for d in result.get("imports", []))
    new_imports = [
        m.group(1) or m.group(2) or m.group(3)
        for m in import_matches
        if m and (m.group(1) or m.group(2) or m.group(3))
    ]
    for imp in new_imports:
        if imp not in existing_imports:
            result["imports"].append(
                {"path": imp, "line": -1, "symbols": []}
            )  # Added symbols for consistency


def _analyze_markdown_file(
    file_path: str, content: str, result: Dict[str, Any]
) -> None:
    """Analyzes Markdown file content using regex."""
    result["links"] = []
    result["code_blocks"] = []
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


def _analyze_html_file_ts(file_path: str, content: str, result: Dict[str, Any]) -> None:
    """Analyzes HTML file content using tree-sitter."""
    for key in ["links", "scripts", "stylesheets", "images", "_ts_tree"]:
        result.setdefault(key, [] if key != "_ts_tree" else None)

    # HTML_PARSER, HTML_LANGUAGE, Query, QueryCursor are directly imported and initialized at module load.
    # If they are None, a hard import error would have occurred.
    # No explicit check for TREE_SITTER_AVAILABLE as imports are now direct.

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
        existing_ids = set()

        if key == "imports":
            # Imports can be strings or dicts (if detailed)
            for item in primary_items:
                if isinstance(item, str):
                    existing_ids.add(item)
                elif isinstance(item, dict) and "path" in item:
                    existing_ids.add(item["path"])
        elif key == "calls":
            for item in primary_items:
                if isinstance(item, dict) and "target_name" in item:
                    # Use target_name + line as unique key to avoid merging same call on same line
                    # But if line differs, it might be a different call.
                    # However, AST usually captures all calls.
                    # If TS captures a call AST missed, we want it.
                    # If TS captures same call, we skip.
                    # Let's use target_name + line approximation.
                    line = item.get("line", -1)
                    existing_ids.add(f"{item['target_name']}:{line}")
        else:
            # functions, classes, globals_defined
            for item in primary_items:
                if isinstance(item, dict) and "name" in item:
                    # Use name + line to be specific, or just name?
                    # If we have multiple functions with same name (overloads?), line helps.
                    line = item.get("line", -1)
                    existing_ids.add(f"{item['name']}:{line}")

        # Merge unique items from secondary
        for item in secondary_items:
            if key == "imports":
                item_id = item if isinstance(item, str) else item.get("path")
                if item_id and item_id not in existing_ids:
                    primary_items.append(item)
            elif key == "calls":
                if isinstance(item, dict) and "target_name" in item:
                    line = item.get("line", -1)
                    unique_key = f"{item['target_name']}:{line}"
                    if unique_key not in existing_ids:
                        primary_items.append(item)
            else:
                if isinstance(item, dict) and "name" in item:
                    line = item.get("line", -1)
                    unique_key = f"{item['name']}:{line}"
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

        # Process captures.
        # captures_calls is a dict: { "capture_name": [Node, Node, ...] }
        # But we need to group them by match to associate obj with attr.
        # Since `captures()` returns a flat dict of all captures, we can't easily group by match pattern instance
        # if there are multiple matches.
        # HOWEVER, `QueryCursor.matches()` returns a list of matches, where each match is (pattern_index, captures_dict).
        # This is better for associating @call_obj with @call_attr.

        matches_calls = cursor_calls.matches(tree.root_node)

        for pattern_index, match_captures in matches_calls:
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
