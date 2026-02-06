# analysis/dependency_suggester.py

"""
Analysis module for dependency suggestion using contextual keys.
Suggests potential dependencies based on code analysis and embeddings.
Assigns specific characters based on the type of dependency found.
Outputs path-based dependencies: List[Tuple[target_norm_path, char]]
"""

import ast
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Attempt to import jsonc-parser
try:
    from jsonc_parser.parser import JsoncParser

    JSONC_PARSER_AVAILABLE = True
except ImportError:
    JSONC_PARSER_AVAILABLE = False
    JsoncParser = None
    FileError = ParserError = JsoncFunctionParameterError = Exception

import logging

from cline_utils.dependency_system.core import key_manager

# Import only from lower-level modules
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.cache_manager import (
    cache_manager,
    cached,
    clear_all_caches,
)
from cline_utils.dependency_system.utils.cache_manager import (
    normalize_path_cached as normalize_path,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import get_file_type

KEY_MANAGER_DIR = os.path.dirname(os.path.abspath(key_manager.__file__))

# NOTE: Avoid importing analyze_file here to prevent circular dependency if analyzer calls suggester


logger = logging.getLogger(__name__)

# Caches for structural dependency analysis to avoid using function attributes
_structural_import_map_cache: Dict[str, Dict[str, str]] = {}
_structural_resolved_path_cache: Dict[Tuple[str, Optional[str]], Optional[str]] = {}

# Character Definitions:
# <: Row depends on column.
# >: Column depends on row.
# x: Mutual dependency.
# d: Documentation dependency.
# o: Self dependency (diagonal only).
# n: Verified no dependency.
# p: Placeholder (unverified).
# s: Semantic dependency (weak .06-.07) - Adjusted based on .clinerules
# S: Semantic dependency (strong .07+) - Added based on .clinerules

_PROJECT_SYMBOL_MAP_FILENAME_LOCAL = "project_symbol_map.json"


# _OLD_PROJECT_SYMBOL_MAP_FILENAME_LOCAL = "project_symbol_map_old.json" # Not used by load, only by save
def clear_caches():
    clear_all_caches()
    if hasattr(_find_and_parse_tsconfig, "_cache"):
        _find_and_parse_tsconfig._cache.clear()  # type: ignore
    _structural_import_map_cache.clear()
    _structural_resolved_path_cache.clear()
    if hasattr(load_project_symbol_map, "_cache"):
        load_project_symbol_map._cache.clear()  # type: ignore


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
@cached(
    "tsconfig_data",
    key_func=lambda start_dir, project_root_val: f"tsconfig:{normalize_path(start_dir)}:{normalize_path(project_root_val)}",
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
    current_dir = normalize_path(start_dir)
    project_root_norm = normalize_path(project_root_val)
    config_filenames = ["tsconfig.json", "jsconfig.json"]
    while True:
        for filename in config_filenames:
            config_path = os.path.join(current_dir, filename)
            if os.path.exists(config_path) and os.path.isfile(config_path):
                logger.debug(f"Found config file for JS/TS: {config_path}")
                try:
                    if (
                        JSONC_PARSER_AVAILABLE and JsoncParser
                    ):  # Check JsoncParser is not None
                        # Ensure JsoncParser.parse_file is called correctly
                        data = JsoncParser.parse_file(config_path)  # type: ignore
                        logger.debug(
                            f"Successfully parsed {config_path} using jsonc-parser."
                        )
                        return config_path, data
                    else:
                        with open(config_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        logger.debug(
                            f"Successfully parsed {config_path} using standard json parser."
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
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir
    logger.debug(
        f"No tsconfig.json or jsconfig.json found in hierarchy from {start_dir} up to {project_root_val}."
    )
    return None


# --- MODIFIED load_project_symbol_map ---
@cached(
    "project_symbol_map_data",
    key_func=lambda: f"project_symbol_map:{os.path.getmtime(os.path.join(KEY_MANAGER_DIR, _PROJECT_SYMBOL_MAP_FILENAME_LOCAL)) if os.path.exists(os.path.join(KEY_MANAGER_DIR, _PROJECT_SYMBOL_MAP_FILENAME_LOCAL)) else 'missing'}",
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
        key_manager_module = __import__(
            "cline_utils.dependency_system.core.key_manager", fromlist=[""]
        )
        core_dir = os.path.dirname(os.path.abspath(key_manager_module.__file__))
        map_path = normalize_path(
            os.path.join(core_dir, _PROJECT_SYMBOL_MAP_FILENAME_LOCAL)
        )

        if not os.path.exists(map_path):
            logger.warning(
                f"Project symbol map file not found at {map_path}. Symbol verification will be skipped."
            )
            return {}

        with open(map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
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


# --- Main Dispatcher ---
def suggest_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    file_analysis_results: Dict[str, Any],
    threshold: float = 0.65,
    shared_scan_counter: Any = None,
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
    file_ext = os.path.splitext(norm_path)[1].lower()

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

    # Initialize collectors
    char_suggestions: List[Tuple[str, str]] = []
    raw_ast_links_collector: List[Dict[str, str]] = []  # NEW: For Python AST links

    if file_ext == ".py":
        # suggest_python_dependencies now returns two lists
        char_suggestions, py_ast_links = (
            suggest_python_dependencies(  # MODIFIED: Unpack two values
                norm_path,
                path_to_key_info,
                project_root,
                current_file_specific_analysis,
                file_analysis_results,
                project_symbol_map,
                threshold,
                shared_scan_counter,
            )
        )
        raw_ast_links_collector.extend(py_ast_links)  # NEW: Collect AST links

    elif file_ext in (".js", ".ts", ".tsx", ".mjs", ".cjs"):
        char_suggestions, js_ast_links = suggest_javascript_dependencies(
            norm_path,
            path_to_key_info,
            project_root,
            current_file_specific_analysis,
            project_symbol_map,
            threshold,
            shared_scan_counter,
        )
        raw_ast_links_collector.extend(js_ast_links)

    elif file_ext in (".md", ".rst"):
        config = ConfigManager()
        embeddings_dir_rel = config.get_path(
            "embeddings_dir", "cline_utils/dependency_system/analysis/embeddings"
        )
        embeddings_dir = normalize_path(os.path.join(project_root, embeddings_dir_rel))
        metadata_path = os.path.join(embeddings_dir, "metadata.json")
        char_suggestions = suggest_documentation_dependencies(
            norm_path,
            path_to_key_info,
            project_root,
            file_analysis_results,  # Pass the BIG map
            threshold,
            embeddings_dir,
            metadata_path,
            project_symbol_map,
            shared_scan_counter,
        )
        # No AST links from Markdown

    elif file_ext in (".html", ".htm"):
        char_suggestions = suggest_html_dependencies(
            norm_path,
            path_to_key_info,
            project_root,
            file_analysis_results,  # Pass the BIG map
        )
        # No AST links from HTML

    elif file_ext == ".css":
        char_suggestions = suggest_css_dependencies(
            norm_path,
            path_to_key_info,
            project_root,
            file_analysis_results,  # Pass the BIG map
        )
        # No AST links from CSS

    else:  # Generic
        char_suggestions = suggest_generic_dependencies(
            norm_path,
            path_to_key_info,
            project_root,
            threshold,
            project_symbol_map,
            shared_scan_counter,
        )
        # No AST links from generic

    return char_suggestions, raw_ast_links_collector  # MODIFIED return


# --- Type-Specific Suggestion Functions ---
def _identify_structural_dependencies(
    source_path: str,
    source_analysis: Dict[str, Any],
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    project_symbol_map: Dict[str, Dict[str, Any]],
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """
    Identifies Python structural dependencies (calls, attributes, inheritance) using contextual keys.
    Returns list of tuples (dependency_path, dependency_character).
    """
    suggestions_path_based: List[Tuple[str, str]] = []
    raw_ast_verified_links: List[Dict[str, str]] = []

    if not source_analysis:
        return [], []

    calls = source_analysis.get("calls", [])
    attributes = source_analysis.get("attribute_accesses", [])
    inheritance = source_analysis.get("inheritance", [])
    type_references = source_analysis.get("type_references", [])
    decorators_used = source_analysis.get("decorators_used", [])
    exceptions_handled = source_analysis.get("exceptions_handled", [])
    with_contexts_used = source_analysis.get("with_contexts_used", [])

    def _build_import_map(
        current_source_path: str, tree: Optional[ast.AST]
    ) -> Dict[str, str]:  # Add tree parameter
        """
        Builds a map of names available in the current scope to the absolute path
        of the module file they were imported from or represent, using the AST
        retrieved from the dedicated 'ast_cache'.

        Example:
        - `import my_module.sub` -> map `{"my_module.sub": "/abs/path/to/my_module/sub.py"}`
        - `import my_module.sub as s` -> map `{"s": "/abs/path/to/my_module/sub.py"}`
        - `from my_package import specific_item` -> map `{"specific_item": "/abs/path/to/my_package/__init__.py"}`
        - `from my_package.another_module import specific_item as si` -> map `{"si": "/abs/path/to/my_package/another_module.py"}`
        """
        norm_source_path = normalize_path(current_source_path)
        if norm_source_path in _structural_import_map_cache:
            return _structural_import_map_cache[norm_source_path]

        local_import_map: Dict[str, str] = {}

        # ast_cache = cache_manager.get_cache("ast_cache")
        # tree = ast_cache.get(norm_source_path)  # norm_source_path is the key

        if not tree:
            logger.error(
                f"ImportMap: AST tree not found in 'ast_cache' for {norm_source_path}. Cannot build import map accurately. This may indicate a parsing failure during the analysis phase or a cache miss/eviction."
            )
            _structural_import_map_cache[norm_source_path] = local_import_map
            return local_import_map  # Return empty map if AST is not available

        try:
            current_source_dir = os.path.dirname(norm_source_path)
            # project_root is available from the outer scope of _identify_structural_dependencies

            for node in ast.walk(tree):  # Iterate through the provided AST tree
                if isinstance(node, ast.Import):
                    for alias_node in node.names:
                        imported_module_string = alias_node.name
                        name_in_scope = alias_node.asname or alias_node.name

                        resolved_paths_info_list = _convert_python_import_to_paths(
                            import_name=imported_module_string,
                            source_file_dir=current_source_dir,
                            project_root=project_root,
                            path_to_key_info=path_to_key_info,
                            project_symbol_map=project_symbol_map,
                            specific_item_name=None,
                            relative_level=0,
                        )

                        if resolved_paths_info_list:
                            resolved_module_file_path = normalize_path(
                                resolved_paths_info_list[0][0]
                            )
                            item_verified = resolved_paths_info_list[0][1]
                            local_import_map[name_in_scope] = resolved_module_file_path
                            logger.debug(
                                f"ImportMap (ast.Import): Mapped '{name_in_scope}' to module path '{resolved_module_file_path}'. Item verified: {item_verified}"
                            )
                        else:
                            logger.debug(
                                f"ImportMap (ast.Import): Could not resolve module string '{imported_module_string}' (imported as '{name_in_scope}') from '{current_source_path}'."
                            )

                elif isinstance(node, ast.ImportFrom):
                    module_name_from_ast = node.module or ""
                    level = node.level

                    base_dir_for_relative_resolve = current_source_dir
                    if level > 0:  # Only adjust base_dir if it's a relative import
                        temp_base = current_source_dir
                        parent = None
                        for _ in range(level):
                            parent = os.path.dirname(temp_base)
                            if (
                                not parent
                                or parent == temp_base
                                or not parent.startswith(normalize_path(project_root))
                            ):
                                logger.warning(
                                    f"Relative import level {level} for '{module_name_from_ast}' in '{current_source_path}' went too high or out of project. Resolution base fallback to project root."
                                )
                                temp_base = normalize_path(project_root)
                                break
                            temp_base = parent
                        base_dir_for_relative_resolve = temp_base

                    module_resolved_paths_info_list = _convert_python_import_to_paths(
                        import_name=module_name_from_ast,
                        source_file_dir=base_dir_for_relative_resolve,  # This is the directory from which relative pathing starts
                        project_root=project_root,
                        path_to_key_info=path_to_key_info,
                        project_symbol_map=project_symbol_map,
                        specific_item_name=None,
                        relative_level=level,
                    )

                    if module_resolved_paths_info_list:
                        resolved_module_file_path = normalize_path(
                            module_resolved_paths_info_list[0][0]
                        )
                        module_symbols = project_symbol_map.get(
                            resolved_module_file_path, {}
                        )

                        for alias in node.names:
                            item_name_actually_imported = alias.name
                            name_in_scope = alias.asname or alias.name

                            # Verify if item_name_actually_imported exists in resolved_module_file_path's symbols
                            is_defined_in_module_symbols = (
                                any(
                                    f.get("name") == item_name_actually_imported
                                    for f in module_symbols.get("functions", [])
                                )
                                or any(
                                    c.get("name") == item_name_actually_imported
                                    for c in module_symbols.get("classes", [])
                                )
                                or any(
                                    g.get("name") == item_name_actually_imported
                                    for g in module_symbols.get("globals_defined", [])
                                )
                            )

                            is_submodule_or_package = False
                            if not is_defined_in_module_symbols:
                                # If not a direct symbol, check if it's a submodule/package re-exported
                                # by an __init__.py, and if that submodule/package is tracked.
                                if (
                                    os.path.basename(resolved_module_file_path)
                                    == "__init__.py"
                                ):
                                    package_dir_for_submodule_check = os.path.dirname(
                                        resolved_module_file_path
                                    )
                                    potential_submodule_path_py = normalize_path(
                                        os.path.join(
                                            package_dir_for_submodule_check,
                                            item_name_actually_imported + ".py",
                                        )
                                    )
                                    potential_subpackage_path_init = normalize_path(
                                        os.path.join(
                                            package_dir_for_submodule_check,
                                            item_name_actually_imported,
                                            "__init__.py",
                                        )
                                    )

                                    if (
                                        potential_submodule_path_py in path_to_key_info
                                        or potential_subpackage_path_init
                                        in path_to_key_info
                                    ):
                                        is_submodule_or_package = True
                                        logger.debug(
                                            f"ImportMap: Item '{item_name_actually_imported}' (imported as '{name_in_scope}') from module '{module_name_from_ast}' appears to be a tracked submodule/package within '{package_dir_for_submodule_check}'."
                                        )

                            local_import_map[name_in_scope] = resolved_module_file_path

                            if is_defined_in_module_symbols or is_submodule_or_package:
                                logger.debug(
                                    f"ImportMap (ast.ImportFrom): Mapped '{name_in_scope}' (item '{item_name_actually_imported}') to module '{resolved_module_file_path}' (verified in symbols or as tracked submodule)."
                                )
                            else:
                                logger.debug(
                                    f"ImportMap (ast.ImportFrom): Item '{item_name_actually_imported}' (imported as '{name_in_scope}') from module '{module_name_from_ast}' (path '{resolved_module_file_path}') not directly verified in its symbols or as a tracked submodule. Mapping to module path as fallback."
                                )
                    else:
                        logger.debug(
                            f"ImportMap: Module '{module_name_from_ast}' (level {level}) itself could not be resolved from '{current_source_path}'. Items like '{', '.join(a.name for a in node.names)}' not mapped."
                        )

        except Exception as e:
            logger.error(
                f"Error building import map for {norm_source_path} using AST: {e}",
                exc_info=False,
            )

        _structural_import_map_cache[norm_source_path] = local_import_map
        return local_import_map

    source_ast_tree = source_analysis.get("_ast_tree")
    current_file_import_map = _build_import_map(source_path, source_ast_tree)

    def _resolve_name_to_path(name_to_resolve: Optional[str]) -> Optional[str]:
        if not name_to_resolve:
            return None

        # Use a cache specific to this run of _identify_structural_dependencies for this source_path
        # The cache key should remain the same as it's for the (source_path, name_to_resolve) pair.
        cache_key_res = (source_path, name_to_resolve)
        if cache_key_res in _structural_resolved_path_cache:
            return _structural_resolved_path_cache[cache_key_res]

        parts = name_to_resolve.split(".")
        resolved_module_path_val: Optional[str] = None

        # Iterate from the longest possible prefix down to the shortest (first part)
        # e.g., for "foo.bar.baz.Item", try "foo.bar.baz", then "foo.bar", then "foo"
        for i in range(len(parts), 0, -1):
            current_prefix_to_check = ".".join(parts[:i])

            # Check if this prefix exists in our import map
            path_from_import_map = current_file_import_map.get(current_prefix_to_check)

            if path_from_import_map:
                # Ensure the path from the import map is a tracked file
                if path_from_import_map in path_to_key_info:
                    resolved_module_path_val = path_from_import_map
                    logger.debug(
                        f"_resolve_name_to_path: Resolved prefix '{current_prefix_to_check}' (from '{name_to_resolve}') to module path '{resolved_module_path_val}' via import map."
                    )
                    break  # Found the longest matching prefix that's an imported module
                else:
                    # This case should be rare if _build_import_map only stores paths present in path_to_key_info
                    logger.warning(
                        f"_resolve_name_to_path: Prefix '{current_prefix_to_check}' mapped to '{path_from_import_map}', but path not in path_to_key_info. Import map might be stale or contain unresolved external refs."
                    )

        if not resolved_module_path_val:
            # If no prefix was found in the import map, it means the name_to_resolve
            # was not from an import statement (e.g., it's a local variable, global in current file, or built-in).
            # In this context, for finding *external module dependencies*, we return None.
            logger.debug(
                f"_resolve_name_to_path: Name '{name_to_resolve}' or its prefixes not found in import map for '{source_path}'. Assumed local or built-in."
            )

        _structural_resolved_path_cache[cache_key_res] = resolved_module_path_val
        return resolved_module_path_val

    # --- MODIFIED: Process Calls and Attributes with symbol verification ---
    for call_item in calls:
        potential_source_str = call_item.get(
            "potential_source"
        )  # e.g., "my_module_alias" or "my_module_alias.class_name"
        target_name_str = call_item.get(
            "target_name"
        )  # e.g., "my_module_alias.method_name" or "ImportedClass()"

        target_path_val = _resolve_name_to_path(
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
                parts = target_name_str.split(".")
                if (
                    len(parts) > 1 and ".".join(parts[:-1]) == potential_source_str
                ):  # my_alias.item -> item
                    actual_item_name_to_check = parts[-1]
                elif not potential_source_str:  # direct call of an imported name
                    actual_item_name_to_check = target_name_str.split(".")[
                        0
                    ]  # if target_name_str itself is dotted
                else:  # Fallback or if potential_source_str is the same as target_name_str (e.g. direct function call)
                    actual_item_name_to_check = target_name_str.split(".")[-1]

            if actual_item_name_to_check:
                # Remove "()" if it's a call representation from _get_full_name_str
                if actual_item_name_to_check.endswith("()"):
                    actual_item_name_to_check = actual_item_name_to_check[:-2]

                module_symbols = project_symbol_map.get(target_path_val, {})
                is_verified = any(
                    s_item.get("name") == actual_item_name_to_check
                    for s_list_key in ["functions", "classes", "globals_defined"]
                    for s_item in module_symbols.get(s_list_key, [])
                )

                # Heuristic: if it's a class method call like MyClass.method(), check if MyClass is in symbols,
                # but we don't have easy access to method names within classes in project_symbol_map yet.
                # For now, if potential_source (e.g., MyClass) is a class in target_path_val, assume method call is valid.
                if not is_verified and potential_source_str:
                    potential_class_name = potential_source_str.split(".")[-1]
                    if any(
                        c.get("name") == potential_class_name
                        for c in module_symbols.get("classes", [])
                    ):
                        is_verified = True  # Assume valid method call if the class itself is verified
                        logger.debug(
                            f"StructuralDep/Call: Assuming valid method call '{actual_item_name_to_check}' on verified class '{potential_class_name}' from {target_path_val}"
                        )

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
                    logger.debug(
                        f"StructuralDep/Call: Verified {source_path} < uses '{target_name_str or potential_source_str}' from {target_path_val}"
                    )
                else:
                    logger.debug(
                        f"StructuralDep/Call: Item '{actual_item_name_to_check}' (from call '{target_name_str}') not verified in symbols of '{target_path_val}'."
                    )
            else:
                logger.debug(
                    f"StructuralDep/Call: Could not determine specific item for call '{target_name_str}' with potential source '{potential_source_str}' resolved to '{target_path_val}'."
                )

    for attr_item in attributes:
        potential_source_str = attr_item.get(
            "potential_source"
        )  # e.g., "my_module_alias" or "my_module_alias.instance"
        attribute_name_accessed = attr_item.get("target_name")  # e.g., "some_attribute"

        target_path_val = _resolve_name_to_path(potential_source_str)

        if (
            target_path_val
            and target_path_val != source_path
            and attribute_name_accessed
        ):
            module_symbols = project_symbol_map.get(target_path_val, {})

            # Check if the attribute_name_accessed is a global/function/class directly in the module
            is_verified = any(
                s_item.get("name") == attribute_name_accessed
                for s_list_key in ["functions", "classes", "globals_defined"]
                for s_item in module_symbols.get(s_list_key, [])
            )

            # Heuristic: if potential_source_str (e.g., MyClass or my_instance_of_MyClass) resolves to a class,
            # we assume the attribute access is valid for now (we don't have instance/class attributes in project_symbol_map easily).
            if not is_verified and potential_source_str:
                potential_class_name_from_source = potential_source_str.split(".")[-1]
                if any(
                    c.get("name") == potential_class_name_from_source
                    for c in module_symbols.get("classes", [])
                ):
                    is_verified = True  # Assume valid attribute access if the base object is a known class
                    logger.debug(
                        f"StructuralDep/Attribute: Assuming valid attribute '{attribute_name_accessed}' access on object/class '{potential_source_str}' (resolved to class in {target_path_val})"
                    )

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
                logger.debug(
                    f"StructuralDep/Attribute: Verified {source_path} < uses attribute '{attribute_name_accessed}' from source '{potential_source_str}' (module: {target_path_val})"
                )
            else:
                logger.debug(
                    f"StructuralDep/Attribute: Attribute '{attribute_name_accessed}' from source '{potential_source_str}' not directly verified as defined symbol in '{target_path_val}'."
                )
    # --- END OF MODIFIED Process Calls and Attributes ---

    # Process Inheritance (from your last provided snippet, with raw_ast_links addition)
    for inh_item in inheritance:
        base_class_name_str = inh_item.get("base_class_name")
        target_path_val = _resolve_name_to_path(base_class_name_str)
        if target_path_val and target_path_val != source_path:
            # Verify "Base" is in target_path_val's symbols
            module_symbols = project_symbol_map.get(target_path_val, {})
            actual_class_name = (
                base_class_name_str.split(".")[-1]
                if "." in base_class_name_str
                else base_class_name_str
            )
            if any(
                c["name"] == actual_class_name
                for c in module_symbols.get("classes", [])
            ):
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
                logger.debug(
                    f"StructuralDep/Inheritance: Verified {source_path} < inherits '{actual_class_name}' from {target_path_val}"
                )
            else:
                logger.debug(
                    f"StructuralDep/Inheritance: Base class '{actual_class_name}' (from '{base_class_name_str}') not found in symbols of resolved module '{target_path_val}'. Skipping dep suggestion for this inheritance."
                )

    # --- MODIFIED: Process Type References with symbol map verification ---
    for type_ref_item in type_references:
        type_name_str = type_ref_item.get(
            "type_name_str"
        )  # e.g., "MyType" or "other_module.TheirType"
        target_path_val = _resolve_name_to_path(
            type_name_str
        )  # Resolves "other_module" part if present

        if target_path_val and target_path_val != source_path:
            # Verify the actual type (e.g., "TheirType") is defined in the resolved module
            module_symbols = project_symbol_map.get(target_path_val, {})
            actual_type_to_check = (
                type_name_str.split(".")[-1] if "." in type_name_str else type_name_str
            )
            is_defined_type = (
                any(
                    c["name"] == actual_type_to_check
                    for c in module_symbols.get("classes", [])
                )
                or any(
                    f["name"] == actual_type_to_check
                    for f in module_symbols.get("functions", [])
                )
                or any(
                    g["name"] == actual_type_to_check
                    for g in module_symbols.get("globals_defined", [])
                )
            )
            if is_defined_type:
                if any(
                    c["name"] == actual_type_to_check
                    for c in module_symbols.get("classes", [])
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
                    logger.debug(
                        f"StructuralDep/TypeHint: Verified {source_path} < uses type '{actual_type_to_check}' from {target_path_val}"
                    )
            else:
                logger.debug(
                    f"StructuralDep/TypeHint: Type '{actual_type_to_check}' (from '{type_name_str}') not found in symbols of resolved module '{target_path_val}'. Skipping dep suggestion."
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
            target_path_val = _resolve_name_to_path(item_name_str)
            if target_path_val and target_path_val != source_path:
                module_symbols = project_symbol_map.get(target_path_val, {})
                actual_item_to_check = (
                    item_name_str.split(".")[-1]
                    if "." in item_name_str
                    else item_name_str
                )

                # Check if the item exists as any kind of defined symbol in the target module
                is_defined_symbol = (
                    any(
                        c["name"] == actual_item_to_check
                        for c in module_symbols.get("classes", [])
                    )
                    or any(
                        f["name"] == actual_item_to_check
                        for f in module_symbols.get("functions", [])
                    )
                    or any(
                        g["name"] == actual_item_to_check
                        for g in module_symbols.get("globals_defined", [])
                    )
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
                    logger.debug(
                        f"StructuralDep/{item_type_log_name}: Verified {source_path} {dep_char} uses '{actual_item_to_check}' from {target_path_val}"
                    )
                else:
                    logger.debug(
                        f"StructuralDep/{item_type_log_name}: Item '{actual_item_to_check}' (from '{item_name_str}') not found in symbols of resolved module '{target_path_val}'. Skipping dep suggestion."
                    )

    return list(set(suggestions_path_based)), raw_ast_verified_links


def suggest_python_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    source_analysis: Dict[
        str, Any
    ],  # MODIFIED: This is now the specific analysis for file_path
    _all_file_analyses_map: Dict[
        str, Any
    ],  # MODIFIED: This is the full map of all file analyses
    project_symbol_map: Dict[str, Dict[str, Any]],
    threshold: float,
    shared_scan_counter: Any = None,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:  # MODIFIED return type
    norm_file_path = normalize_path(file_path)
    # source_analysis is already the specific analysis for norm_file_path
    if (
        source_analysis is None
        or "error" in source_analysis
        or "skipped" in source_analysis
    ):
        logger.debug(
            f"No valid analysis for {norm_file_path}, skipping Python suggestions."
        )
        return [], []  # MODIFIED return

    explicit_deps_paths, explicit_raw_ast_links = _identify_python_dependencies(
        norm_file_path,
        source_analysis,
        _all_file_analyses_map,
        project_root,
        path_to_key_info,
        project_symbol_map,
    )
    structural_suggestions_paths, structural_raw_ast_links = (
        _identify_structural_dependencies(
            norm_file_path,
            source_analysis,
            path_to_key_info,
            project_root,
            project_symbol_map,
        )
    )
    semantic_suggestions_paths = suggest_semantic_dependencies_path_based(
        norm_file_path,
        path_to_key_info,
        project_root,
        threshold,
        project_symbol_map,
        shared_scan_counter,
    )

    all_suggestions_paths = (
        explicit_deps_paths + structural_suggestions_paths + semantic_suggestions_paths
    )

    # Combine all raw AST links
    all_raw_ast_links = explicit_raw_ast_links + structural_raw_ast_links  # NEW

    # Apply priority to character-based suggestions
    combined_suggestions = combine_suggestions_path_based_with_char_priority(
        all_suggestions_paths, norm_file_path
    )

    if combined_suggestions:
        logger.debug(f"Final Combined Dependencies for {norm_file_path}:")
        for target_path, char_code in combined_suggestions:
            logger.debug(f"  -> {target_path} ({char_code})")
    else:
        logger.debug(f"No combined dependencies found for {norm_file_path}.")

    if all_raw_ast_links:
        logger.debug(
            f"Raw AST-Verified Links Collected for {norm_file_path}: {len(all_raw_ast_links)} links"
        )

    return combined_suggestions, all_raw_ast_links


def suggest_javascript_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    source_analysis: Dict[str, Any],
    project_symbol_map: Dict[str, Dict[str, Any]],
    threshold: float,
    shared_scan_counter: Any = None,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """
    Suggest dependencies for JS/TS/TSX by combining:
      1) Explicit import edges from analyzer 'imports'
      2) Structural verifications (calls, type refs, re-exports)
      3) Semantic similarities
    Returns combined suggestions and AST-verified links.
    """
    norm_file_path = normalize_path(file_path)
    if (
        not source_analysis
        or "error" in source_analysis
        or "skipped" in source_analysis
    ):
        return [], []

    source_file_dir = os.path.dirname(norm_file_path)
    tsconfig_info = _find_and_parse_tsconfig(source_file_dir, project_root)

    # 1) Explicit imports -> direct '<' deps + AST links
    explicit_suggestions_paths, explicit_raw_ast_links = (
        _identify_jsts_explicit_import_dependencies(
            norm_file_path,
            source_analysis,
            path_to_key_info,
            project_root,
            tsconfig_info,
        )
    )

    # 2) Structural dependencies verified against exports/symbol map
    structural_suggestions_paths, structural_raw_ast_links = (
        _identify_javascript_structural_dependencies(
            norm_file_path,
            source_analysis,
            path_to_key_info,
            project_root,
            project_symbol_map,
            tsconfig_info,
        )
    )

    # 3) Semantic suggestions
    semantic_suggestions_paths = suggest_semantic_dependencies_path_based(
        norm_file_path,
        path_to_key_info,
        project_root,
        threshold,
        project_symbol_map,
        shared_scan_counter,
    )

    # Combine suggestions then return with union of AST links
    all_suggestions_paths = (
        explicit_suggestions_paths
        + structural_suggestions_paths
        + semantic_suggestions_paths
    )
    combined_suggestions = combine_suggestions_path_based_with_char_priority(
        all_suggestions_paths, norm_file_path
    )

    all_raw_ast_links = explicit_raw_ast_links + structural_raw_ast_links
    return combined_suggestions, all_raw_ast_links


def _identify_javascript_structural_dependencies(
    source_path: str,
    source_analysis: Dict[str, Any],
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    project_symbol_map: Dict[str, Dict[str, Any]],
    tsconfig_info: Optional[Tuple[str, Dict[str, Any]]],
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """
    Identifies JS/TS structural dependencies by verifying that symbols used in the source file
    are correctly imported and are present in the target module's exports.
    """
    suggestions_path_based: List[Tuple[str, str]] = []
    raw_ast_verified_links: List[Dict[str, str]] = []
    source_dir = os.path.dirname(source_path)

    # --- Step 1: Handle re-exports directly ---
    for export_item in source_analysis.get("exports", []):
        if "from" in export_item:
            resolved_path = _resolve_js_import_path(
                export_item["from"],
                source_dir,
                project_root,
                path_to_key_info,
                tsconfig_info,
            )
            if resolved_path and resolved_path != source_path:
                dep_char = "<"  # A re-export is a direct dependency
                suggestions_path_based.append((resolved_path, dep_char))
                raw_ast_verified_links.append(
                    {
                        "source_path": source_path,
                        "target_path": resolved_path,
                        "char": dep_char,
                        "reason": f"JSReExport/{export_item['name']}",
                    }
                )

    # --- Step 2: Build a reliable import map: { name_in_scope: { origin_path: "...", original_name: "..." } } ---
    import_map: Dict[str, Dict[str, str]] = {}
    for imp in source_analysis.get("imports", []):
        unresolved_path = imp.get("path")
        if not unresolved_path:
            continue

        resolved_path = _resolve_js_import_path(
            unresolved_path, source_dir, project_root, path_to_key_info, tsconfig_info
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
                logger.debug(
                    f"JSImportMap: Mapped '{name_in_scope}' to {{origin:'{resolved_path}', original:'{original_name}'}}"
                )

    # --- Step 3: Helper function to resolve a symbol and verify it in the target module's exports. ---
    def _find_and_verify_symbol_origin(symbol_name: str) -> Optional[str]:
        if not symbol_name:
            return None

        parts = symbol_name.split(".")
        base_symbol = parts[0]
        member_access_chain = parts[1:]

        import_info = import_map.get(base_symbol)
        if not import_info:
            return None

        resolved_path = import_info["origin_path"]
        original_name_from_import = import_info["original_name"]

        target_module_symbols = project_symbol_map.get(resolved_path)
        if not target_module_symbols or not target_module_symbols.get("exports"):
            logger.debug(
                f"JSVerify: No exports found in project_symbol_map for resolved path '{resolved_path}'."
            )
            return None

        # Determine what to look for in the target's exports list
        item_to_check_in_exports = original_name_from_import
        # If it was `import * as X from '...` and used as `X.item`
        if original_name_from_import == "*" and member_access_chain:
            item_to_check_in_exports = member_access_chain[0]

        for exported_item in target_module_symbols["exports"]:
            exported_name = exported_item.get("name")
            # `export default ...` results in name: 'default' or the actual name.
            # `export {a}` results in name: 'a'.
            # `export {a as b}` results in name: 'a', alias: 'b'. The exported name is 'b'.
            public_facing_export_name = exported_item.get("alias") or exported_name

            if public_facing_export_name == item_to_check_in_exports:
                logger.debug(
                    f"JSVerify: Verified symbol '{symbol_name}' (checking for '{item_to_check_in_exports}') is exported from '{resolved_path}'."
                )
                return resolved_path

        logger.debug(
            f"JSVerify: Symbol '{symbol_name}' (checking for '{item_to_check_in_exports}') is NOT exported from '{resolved_path}'."
        )
        return None

    # --- Step 4: Iterate through calls and type references to find dependencies. ---
    items_to_check = [
        (source_analysis.get("calls", []), "target_name", "JSCall"),
        (source_analysis.get("type_references", []), "type_name_str", "JSTypeRef"),
    ]

    for item_list, name_key, reason_prefix in items_to_check:
        for item in item_list:
            item_name = item.get(name_key)
            if not item_name:
                continue

            resolved_path = _find_and_verify_symbol_origin(item_name)

            if resolved_path and resolved_path != source_path:
                dep_char = "<"
                suggestions_path_based.append((resolved_path, dep_char))
                raw_ast_verified_links.append(
                    {
                        "source_path": source_path,
                        "target_path": resolved_path,
                        "char": dep_char,
                        "reason": f"{reason_prefix}/{item_name}",
                    }
                )
                logger.debug(
                    f"JSStructural/{reason_prefix}: Verified {source_path} < uses '{item_name}' from {resolved_path}"
                )

    return list(set(suggestions_path_based)), raw_ast_verified_links


def _identify_jsts_explicit_import_dependencies(
    source_path: str,
    source_analysis: Dict[str, Any],
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    tsconfig_info: Optional[Tuple[str, Dict[str, Any]]],
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """
    Converts analyzer 'imports' entries for JS/TS/TSX into direct '<' dependencies.
    Also emits AST-verified link entries for each resolved import.
    Relies on _resolve_js_import_path for consistent path resolution (incl. tsconfig paths/baseUrl).
    """
    suggestions_path_based: List[Tuple[str, str]] = []
    raw_ast_verified_links: List[Dict[str, str]] = []

    if not source_analysis:
        return [], []

    source_dir = os.path.dirname(source_path)
    tracked_paths_globally = set(path_to_key_info.keys())

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
            unresolved, source_dir, project_root, path_to_key_info, tsconfig_info
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
    _embeddings_dir: str,
    _metadata_path: str,
    project_symbol_map: Optional[Dict[str, Any]] = None,
    shared_scan_counter: Any = None,
) -> List[Tuple[str, str]]:
    norm_file_path = normalize_path(file_path)
    analysis = file_analysis_results.get(norm_file_path)
    if analysis is None or "error" in analysis or "skipped" in analysis:
        return []

    explicit_deps_paths = _identify_markdown_dependencies(
        norm_file_path, analysis, file_analysis_results, project_root, path_to_key_info
    )
    semantic_suggestions_paths = suggest_semantic_dependencies_path_based(
        norm_file_path,
        path_to_key_info,
        project_root,
        threshold,
        project_symbol_map,
        shared_scan_counter,
    )

    all_suggestions_paths = explicit_deps_paths + semantic_suggestions_paths
    return combine_suggestions_path_based_with_char_priority(
        all_suggestions_paths, norm_file_path
    )


def suggest_html_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    file_analysis_results: Dict[str, Any],
) -> List[Tuple[str, str]]:  # Output: List[(target_norm_path, char)]
    norm_file_path = normalize_path(file_path)
    analysis = file_analysis_results.get(norm_file_path)
    if analysis is None or "error" in analysis or "skipped" in analysis:
        return []

    explicit_deps_paths = _identify_html_dependencies(
        norm_file_path, analysis, file_analysis_results, project_root, path_to_key_info
    )
    # Optionally add semantic for HTML if meaningful:
    # semantic_suggestions_paths = suggest_semantic_dependencies_path_based(norm_file_path, path_to_key_info, project_root, some_html_threshold)
    # all_suggestions_paths = explicit_deps_paths + semantic_suggestions_paths
    return combine_suggestions_path_based_with_char_priority(
        explicit_deps_paths, norm_file_path
    )


def suggest_css_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    file_analysis_results: Dict[str, Any],
) -> List[Tuple[str, str]]:  # Output: List[(target_norm_path, char)]
    norm_file_path = normalize_path(file_path)
    analysis = file_analysis_results.get(norm_file_path)
    if analysis is None or "error" in analysis or "skipped" in analysis:
        return []

    explicit_deps_paths = _identify_css_dependencies(
        norm_file_path, analysis, file_analysis_results, project_root, path_to_key_info
    )
    return combine_suggestions_path_based_with_char_priority(
        explicit_deps_paths, norm_file_path
    )


def suggest_generic_dependencies(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    threshold: float,
    project_symbol_map: Optional[Dict[str, Any]] = None,
    shared_scan_counter: Any = None,
) -> List[Tuple[str, str]]:  # Output: List[(target_norm_path, char)]
    norm_file_path = normalize_path(file_path)
    semantic_suggestions_paths = suggest_semantic_dependencies_path_based(
        norm_file_path,
        path_to_key_info,
        project_root,
        threshold,
        project_symbol_map,
        shared_scan_counter,
    )
    return combine_suggestions_path_based_with_char_priority(
        semantic_suggestions_paths, norm_file_path
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
    ast_map = {}
    for link in ast_verified_links:
        link_from = link.get("source_path")
        link_to = link.get("target_path")
        # Also support 'char' key which is used in ast_verified_links.json
        relation = link.get("char") or link.get("relation", "")

        if link_from == source_norm:
            ast_map[link_to] = relation
        elif link_to == source_norm:
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
                existing_relation = ast_map.get(reverse_key)
                if not existing_relation:
                    ast_map[reverse_key] = inverse_relation
                # If relations are opposite, merge them into 'x' for mutual dependency.
                elif {existing_relation, inverse_relation} == {"<", ">"}:
                    ast_map[reverse_key] = "x"
    return ast_map


def _enhance_semantic_suggestions_with_ast_evidence(
    semantic_suggestions: List[Tuple[str, str]],
    ast_map: Dict[str, str],
) -> List[Tuple[str, str]]:
    """
    Enhance semantic suggestions with AST-verified structural evidence
    """
    enhanced = []

    # Track which targets we've processed to avoid duplicates when adding from ast_map
    processed_targets = set()

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
def suggest_semantic_dependencies_path_based(
    file_path: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    threshold: float,
    project_symbol_map: Optional[Dict[str, Any]] = None,
    shared_scan_counter: Any = None,
) -> List[Tuple[str, str]]:  # Output: List[(target_norm_path, char)]
    # Check global limit FIRST to avoid unnecessary processing
    GLOBAL_SCAN_LIMIT = 20
    if shared_scan_counter is not None:
        # Non-blocking check (value access is atomic enough for this optimization)
        if shared_scan_counter.value >= GLOBAL_SCAN_LIMIT:
            # logger.debug("Global reranker scan limit reached (early check). Skipping semantic analysis.")
            return []

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

    suggested_deps_path_based: List[Tuple[str, str]] = []
    target_key_infos_list: List[KeyInfo] = [
        info
        for info in path_to_key_info.values()
        if not info.is_directory
        and info.norm_path != file_path  # Must be a file and not self
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

    for target_ki in target_key_infos_list:
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
            confidence >= threshold * 0.5
        ):  # Use half threshold to catch potential candidates
            candidates_with_similarity.append((target_ki, confidence))

    # This variable will hold the results from either reranking or the fallback logic
    initial_suggestions = []

    # Apply reranking if we have multiple candidates and the reranking feature is available
    if len(candidates_with_similarity) > 1:
        try:
            from .embedding_manager import (
                get_instruction_for_relation_type,
                rerank_candidates_with_qwen3,
            )
            from .reranker_history_tracker import get_historical_pairs

            # Load historical pairs to avoid re-analyzing same files
            historical_pairs = get_historical_pairs(project_root)

            # Filter candidates before reranking based on dependency character assignments
            # This filtering only applies during reranking to optimize performance
            filtered_candidates = []
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
                # if confidence >= threshold_s:
                filtered_candidates.append((target_ki, confidence))

            candidates_with_similarity = filtered_candidates
            logger.debug(
                f"Pre-reranking filter: {len(candidates_with_similarity)} candidates remaining from original collection"
            )

            # Check global scan limit to ensure "curated selection" of ~20 files per run
            GLOBAL_SCAN_LIMIT = 20

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
                    "Global reranker scan limit (20) reached. Skipping reranking for this file."
                )
                top_candidates = []
            else:
                # Prepare for reranking - get top candidates
                top_candidates = sorted(
                    candidates_with_similarity, key=lambda x: x[1], reverse=True
                )[:remaining_slots]

                if len(top_candidates) > 1:
                    # Update shared counter immediately
                    if shared_scan_counter is not None:
                        with shared_scan_counter.get_lock():
                            shared_scan_counter.value += 1

                    # Log to file
                    try:
                        scans_file = os.path.join(
                            project_root,
                            "cline_utils",
                            "dependency_system",
                            "analysis",
                            "reranker_scans.jsonl",
                        )
                        os.makedirs(os.path.dirname(scans_file), exist_ok=True)
                        with open(scans_file, "a", encoding="utf-8") as f:
                            for cand_ki, cand_conf in top_candidates:
                                json.dump(
                                    {
                                        "source": source_key_info.norm_path,
                                        "target": cand_ki.norm_path,
                                        "confidence": float(cand_conf),
                                    },
                                    f,
                                )
                                f.write("\n")
                    except Exception as e:
                        logger.warning(f"Failed to log reranker scans: {e}")

            if len(top_candidates) > 1:

                # Get the source text for reranking
                try:
                    # Prepare query text (source file)
                    query_text = ""
                    if project_symbol_map and file_path in project_symbol_map:
                        query_text = generate_symbol_essence_string(
                            file_path, project_symbol_map[file_path]
                        )
                    else:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        ext = os.path.splitext(file_path)[1].lower()
                        if ext in [".md", ".txt", ".rst"]:
                            query_text = preprocess_doc_structure(content)
                        else:
                            query_text = f"[FILE: {os.path.basename(file_path)}]\n{content[:8192]}"
                except Exception as e:
                    logger.warning(
                        f"Could not read source file {file_path} for reranking: {e}"
                    )
                    # Fallback to original similarity scores if source can't be read
                    # This part can be simplified since the fallback is handled later
                    pass  # Fall through to the original similarity logic
                else:
                    # Prepare candidate texts for reranking
                    candidate_texts = []
                    valid_candidates = []

                    for target_ki, original_confidence in top_candidates:
                        try:
                            # Get instruction for this relation type and prepare text
                            instruction = get_instruction_for_relation_type(
                                file_path, target_ki.norm_path
                            )

                            # Read the actual content for reranking
                            # Use SES if available to reduce context size
                            candidate_text = ""

                            # Check if target is in symbol map
                            if (
                                project_symbol_map
                                and target_ki.norm_path in project_symbol_map
                            ):
                                candidate_text = generate_symbol_essence_string(
                                    target_ki.norm_path,
                                    project_symbol_map[target_ki.norm_path],
                                )
                            else:
                                # Fallback for non-code or untracked files
                                try:
                                    with open(
                                        target_ki.norm_path, "r", encoding="utf-8"
                                    ) as f:
                                        content = f.read()

                                    ext = os.path.splitext(target_ki.norm_path)[
                                        1
                                    ].lower()
                                    if ext in [".md", ".txt", ".rst"]:
                                        candidate_text = preprocess_doc_structure(
                                            content
                                        )
                                    else:
                                        # Limit raw content
                                        candidate_text = f"[FILE: {os.path.basename(target_ki.norm_path)}]\n{content[:8192]}"
                                except Exception:
                                    continue

                            candidate_text = instruction + candidate_text
                            candidate_texts.append(candidate_text)
                            valid_candidates.append((target_ki, original_confidence))
                        except Exception:
                            # If we can't read candidate content, skip it
                            continue

                    if len(candidate_texts) > 1:
                        # Perform reranking
                        reranked_results = rerank_candidates_with_qwen3(
                            query_text,
                            candidate_texts,
                            top_k=min(10, len(candidate_texts)),
                            source_file_path=file_path,
                        )

                        # Process reranked results
                        for i, (candidate_idx, rerank_score) in enumerate(
                            reranked_results
                        ):
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

                                assigned_char_semantic: Optional[str] = None

                                # HIGH CONFIDENCE: Promote to verified dependency character
                                if rerank_score >= promotion_threshold:

                                    def _classify_role(path: str) -> str:
                                        ext = os.path.splitext(path)[1].lower()
                                        if ext in [".md", ".rst", ".txt"]:
                                            return "doc"
                                        return "code"

                                    def determine_dependency_direction(
                                        src_path: str, tgt_path: str
                                    ) -> str:
                                        """
                                        Determine dependency direction based on file roles.
                                        Returns: 'x' (bidirectional), '<' (source depends on target), '>' (target depends on source)
                                        """
                                        src_role = _classify_role(src_path)
                                        tgt_role = _classify_role(tgt_path)

                                        # Doc -> Code: Documentation describes code ='>')
                                        if src_role == "doc" and tgt_role == "code":
                                            return ">"
                                        # Code -> Doc: Code is documented by doc (='<')
                                        elif src_role == "code" and tgt_role == "doc":
                                            return "<"
                                        # Same role: default to bidirectional
                                        return "x"

                                    direction = determine_dependency_direction(
                                        source_key_info.norm_path, target_ki.norm_path
                                    )
                                    assigned_char_semantic = direction
                                    logger.debug(
                                        f"HIGH CONFIDENCE (>= {promotion_threshold:.2f}): {source_key_info.norm_path} -> {target_ki.norm_path} "
                                        f"promoted to '{direction}' (score: {rerank_score:.3f})"
                                    )
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

            assigned_char_semantic: Optional[str] = None

            # HIGH CONFIDENCE: Promote to verified dependency character
            if confidence >= promotion_threshold:

                def _classify_role(path: str) -> str:
                    ext = os.path.splitext(path)[1].lower()
                    if ext in [".md", ".rst", ".txt"]:
                        return "doc"
                    return "code"

                def determine_fallback_dependency_direction(
                    src_path: str, tgt_path: str
                ) -> str:
                    """Determine dependency direction for high-confidence fallback assignments."""
                    src_role = _classify_role(src_path)
                    tgt_role = _classify_role(tgt_path)

                    if src_role == "doc" and tgt_role == "code":
                        return ">"
                    elif src_role == "code" and tgt_role == "doc":
                        return "<"
                    return "x"

                direction = determine_fallback_dependency_direction(
                    source_key_info.norm_path, target_ki.norm_path
                )
                assigned_char_semantic = direction
                logger.debug(
                    f"HIGH CONFIDENCE (>= {promotion_threshold:.2f}) [fallback]: {source_key_info.norm_path} -> {target_ki.norm_path} "
                    f"promoted to '{direction}' (conf: {confidence:.3f}, rel: {source_type}->{target_type})"
                )
            elif confidence >= threshold_S_strong_semantic:
                assigned_char_semantic = "S"
            elif confidence >= threshold_s_weak_semantic:
                assigned_char_semantic = "s"

            if assigned_char_semantic:
                initial_suggestions.append(
                    (target_ki.norm_path, assigned_char_semantic)
                )
                if (
                    confidence < promotion_threshold
                ):  # Only log non-promoted assignments
                    logger.debug(
                        f"Initial semantic: {source_key_info.norm_path} -> {target_ki.norm_path} ('{assigned_char_semantic}') conf: {confidence:.3f} (rel: {source_type}->{target_type})"
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
        elif (
            new_priority_val == current_priority_val
            and char_val != current_char_for_path
            and current_char_for_path is not None
        ):
            # Handle specific merge cases like < and > to x
            if {char_val, current_char_for_path} == {"<", ">"}:
                if combined_by_path.get(target_path) != "x":
                    combined_by_path[target_path] = "x"
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
) -> List[
    Tuple[str, bool]
]:  # Returns List[(resolved_module_path, item_verified_in_module_symbols)]

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
        logger.debug(
            f"Absolute import '{import_name}': Candidate file paths to check: {candidate_module_file_paths_to_check_in_map}"
        )

    # --- Check generated candidates against path_to_key_info and verify specific_item_name ---
    seen_paths = set()
    for p_candidate_str in candidate_module_file_paths_to_check_in_map:
        # p_candidate_str is already a normalized absolute path from the generation logic
        if p_candidate_str in path_to_key_info:  # Direct check against tracked files
            if p_candidate_str not in seen_paths:
                item_verified_in_symbols = (
                    True  # Default to true if no specific item to check
                )
                if specific_item_name:
                    module_symbols = project_symbol_map.get(p_candidate_str, {})
                    is_defined = (
                        any(
                            f.get("name") == specific_item_name
                            for f in module_symbols.get("functions", [])
                        )
                        or any(
                            c.get("name") == specific_item_name
                            for c in module_symbols.get("classes", [])
                        )
                        or any(
                            g.get("name") == specific_item_name
                            for g in module_symbols.get("globals_defined", [])
                        )
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
                            potential_submodule_file in path_to_key_info
                            or potential_subpackage_init in path_to_key_info
                        ):
                            is_defined = True
                            logger.debug(
                                f"Import Check: Item '{specific_item_name}' imported from '{import_name}' (via package '{p_candidate_str}') appears to be a tracked re-exported submodule/package."
                            )

                    if not is_defined:
                        item_verified_in_symbols = False
                        logger.debug(
                            f"Import Check: Item '{specific_item_name}' imported from '{import_name}' (resolved to module '{p_candidate_str}') not found in its defined symbols or as a tracked submodule/package."
                        )

                potential_paths_abs_info.append(
                    (p_candidate_str, item_verified_in_symbols)
                )
                seen_paths.add(p_candidate_str)

                # For absolute imports, Python's import machinery typically stops at the first match.
                # We replicate this behavior by returning after the first successful resolution.
                if relative_level == 0 and potential_paths_abs_info:
                    logger.debug(
                        f"Absolute import '{import_name}' resolved to '{p_candidate_str}'. Taking first match."
                    )
                    return potential_paths_abs_info  # Return immediately

    if not potential_paths_abs_info and import_name:
        logger.debug(
            f"Could not resolve import '{import_name}' (item: {specific_item_name}, level:{relative_level}) from source '{source_file_dir}' to any *tracked* project files. Candidates checked (sample): {candidate_module_file_paths_to_check_in_map[:3]}"
        )
    return potential_paths_abs_info


# ---


def _identify_python_dependencies(
    source_path: str,
    source_analysis: Dict[str, Any],
    _file_analysis_results: Dict[str, Dict[str, Any]],
    project_root: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_symbol_map: Dict[str, Dict[str, Any]],
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:  # MODIFIED return type
    dependencies_paths: List[Tuple[str, str]] = []
    raw_ast_links: List[Dict[str, str]] = []  # NEW: For collecting AST-derived links

    imports_in_source = source_analysis.get("imports", [])
    source_dir_norm = os.path.dirname(source_path)
    tracked_paths_globally = set(path_to_key_info.keys())

    for (
        import_name_str_from_ast
    ) in imports_in_source:  # This is like "module" or ".module" or "..module.sub"
        temp_import_name = import_name_str_from_ast
        level_for_convert = 0
        while temp_import_name.startswith("."):
            level_for_convert += 1
            temp_import_name = temp_import_name[1:]
        module_to_resolve_for_convert = temp_import_name

        resolved_path_infos = _convert_python_import_to_paths(
            import_name=module_to_resolve_for_convert,
            source_file_dir=source_dir_norm,
            project_root=project_root,
            path_to_key_info=path_to_key_info,
            project_symbol_map=project_symbol_map,
            specific_item_name=None,
            _is_from_import=True,
            relative_level=level_for_convert,
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

                if not item_verified_flag and module_to_resolve_for_convert:
                    logger.debug(
                        f"PythonDep: Import of '{module_to_resolve_for_convert}' from '{source_path}' resolved to module '{path_abs_val}', but specific item verification status: {item_verified_flag}."
                    )
                break
    return list(set(dependencies_paths)), list(raw_ast_links)


def _resolve_js_import_path(
    import_path: str,
    source_dir: str,
    project_root: str,
    path_to_key_info: Dict[str, KeyInfo],
    tsconfig_info: Optional[Tuple[str, Dict[str, Any]]],
) -> Optional[str]:
    """Resolves a JS/TS import path to a tracked file path."""
    if import_path.startswith("."):
        base_path = normalize_path(
            os.path.abspath(os.path.join(source_dir, import_path))
        )
        extensions = [
            "",
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            "/index.ts",
            "/index.tsx",
            "/index.js",
            "/index.jsx",
        ]
        for ext in extensions:
            candidate = normalize_path(base_path + ext)
            if candidate in path_to_key_info:
                return candidate
        return None

    if tsconfig_info:
        config_path, config = tsconfig_info
        compiler_options = config.get("compilerOptions", {})
        base_url_rel = compiler_options.get("baseUrl", ".")
        base_url_abs = normalize_path(
            os.path.abspath(os.path.join(os.path.dirname(config_path), base_url_rel))
        )
        paths = compiler_options.get("paths", {})

        for alias, mappings in paths.items():
            alias_pattern_str = "^" + re.escape(alias).replace(r"\*", "(.*)") + "$"
            match = re.match(alias_pattern_str, import_path)
            if match:
                for mapping in mappings:
                    captured_part = match.group(1) if "*" in alias else ""
                    expanded_path = mapping.replace("*", captured_part)
                    candidate_base = normalize_path(
                        os.path.join(base_url_abs, expanded_path)
                    )
                    extensions = [
                        "",
                        ".ts",
                        ".tsx",
                        ".js",
                        ".jsx",
                        "/index.ts",
                        "/index.tsx",
                        "/index.js",
                        "/index.jsx",
                    ]
                    for ext in extensions:
                        candidate = normalize_path(candidate_base + ext)
                        if candidate in path_to_key_info:
                            return candidate

    # Fallback for non-aliased, non-relative paths (e.g. from baseUrl)
    if tsconfig_info:
        config_path, config = tsconfig_info
        compiler_options = config.get("compilerOptions", {})
        base_url_rel = compiler_options.get("baseUrl", ".")
        base_url_abs = normalize_path(
            os.path.abspath(os.path.join(os.path.dirname(config_path), base_url_rel))
        )
        candidate_base = normalize_path(os.path.join(base_url_abs, import_path))
        extensions = [
            "",
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            "/index.ts",
            "/index.tsx",
            "/index.js",
            "/index.jsx",
        ]
        for ext in extensions:
            candidate = normalize_path(candidate_base + ext)
            if candidate in path_to_key_info:
                return candidate

    return None


def _identify_markdown_dependencies(
    source_path: str,
    source_analysis: Dict[str, Any],
    _file_analyses: Dict[str, Dict[str, Any]],
    project_root: str,
    path_to_key_info: Dict[str, KeyInfo],
) -> List[Tuple[str, str]]:
    dependencies_paths: List[Tuple[str, str]] = []
    links_in_source = source_analysis.get("links", [])
    source_dir_norm = os.path.dirname(source_path)
    tracked_paths_globally = set(path_to_key_info.keys())
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
) -> List[Tuple[str, str]]:
    dependencies_paths: List[Tuple[str, str]] = []
    source_dir_norm = os.path.dirname(source_path)
    tracked_paths_globally = set(path_to_key_info.keys())
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
) -> List[Tuple[str, str]]:
    dependencies_paths: List[Tuple[str, str]] = []
    imports_in_css = source_analysis.get("imports", [])
    source_dir_norm = os.path.dirname(source_path)
    tracked_paths_globally = set(path_to_key_info.keys())
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
