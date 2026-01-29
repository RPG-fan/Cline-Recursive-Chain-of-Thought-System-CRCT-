# io/update_main_tracker.py
"""
IO module for main tracker specific data, including key filtering
and dependency aggregation logic using contextual keys.
"""
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Import necessary functions from other modules
from cline_utils.dependency_system.core.dependency_grid import (
    DIAGONAL_CHAR,
    PLACEHOLDER_CHAR,
    decompress,
)
from cline_utils.dependency_system.core.key_manager import (
    KeyInfo,
    get_key_from_path,
    sort_key_strings_hierarchically,
    sort_keys,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import (
    get_project_root,
    is_subpath,
    join_paths,
    normalize_path,
)
from cline_utils.dependency_system.utils.tracker_utils import (
    read_tracker_file_structured,
)

logger = logging.getLogger(__name__)

# --- Main Tracker Path ---


def get_main_tracker_path(project_root: str) -> str:
    """Gets the path to the main tracker file (module_relationship_tracker.md)."""
    config_manager = ConfigManager()
    # Default relative path, can be overridden in .clinerules
    memory_dir_rel = config_manager.get_path("memory_dir", "cline_docs/memory")
    memory_dir_abs = join_paths(project_root, memory_dir_rel)
    # *** Use get_path for the filename as well, providing a default ***
    # This assumes you might want to configure 'main_tracker_filename' in the [paths] section of .clinerules
    tracker_filename = config_manager.get_path(
        "main_tracker_filename", "module_relationship_tracker.md"
    )
    # Ensure tracker_filename is just the filename, not potentially a nested path
    tracker_filename = os.path.basename(tracker_filename)
    return join_paths(memory_dir_abs, tracker_filename)


# --- Key Filtering Logic ---


# <<< *** MODIFIED FUNCTION SIGNATURE AND LOGIC *** >>>
def main_key_filter(
    project_root: str, path_to_key_info: Dict[str, KeyInfo]
) -> Dict[str, KeyInfo]:
    """
    Logic for determining which modules (directories represented by KeyInfo)
    to include in the main tracker. Includes only directories within configured code roots.

    Args:
        project_root: Absolute path to the project root.
        path_to_key_info: The global map from normalized paths to KeyInfo objects.

    Returns:
        A dictionary mapping the normalized path of each filtered module (directory)
        to its corresponding KeyInfo object.
    """
    config_manager = ConfigManager()
    root_directories_rel: List[str] = config_manager.get_code_root_directories()
    filtered_modules: Dict[str, KeyInfo] = {}  # {norm_path: KeyInfo}
    abs_code_roots: Set[str] = {
        normalize_path(os.path.join(project_root, p)) for p in root_directories_rel
    }

    if not abs_code_roots:
        logger.warning(
            "No code root directories defined for main tracker key filtering."
        )
        return {}

    # Iterate through all KeyInfo objects
    for norm_path, key_info in path_to_key_info.items():
        # Check if the path represents a directory
        if key_info.is_directory:
            # Check if the directory is equal to or within any of the code roots
            if any(
                norm_path == root_dir or is_subpath(norm_path, root_dir)
                for root_dir in abs_code_roots
            ):
                filtered_modules[norm_path] = key_info  # Add the KeyInfo object

    logger.info(
        f"Main key filter selected {len(filtered_modules)} module paths for the main tracker."
    )
    # Example: Log first 5 selected module paths and their keys
    # sample_keys = {ki.key_string for ki in list(filtered_modules.values())[:5]}
    # logger.debug(f"Sample main tracker keys: {sample_keys}")
    return filtered_modules


# --- Hierarchical Helper (Adapted for Paths) ---
def _get_descendants_paths(
    parent_path: str, hierarchy: Dict[str, List[str]]
) -> Set[str]:
    """Helper to get all descendant paths (INCLUDING self) for hierarchical check."""
    # Ensure paths are normalized from the start if not guaranteed by caller
    norm_parent_path = normalize_path(parent_path)
    descendants = {norm_parent_path}
    # Use a proper queue/stack for BFS/DFS
    queue = [normalize_path(p) for p in hierarchy.get(norm_parent_path, [])]
    processed = {norm_parent_path}
    while queue:
        child_path = queue.pop(0)  # BFS style
        if child_path not in processed:
            descendants.add(child_path)
            processed.add(child_path)
            # Add grandchildren (normalized) to the queue
            grandchildren = [normalize_path(p) for p in hierarchy.get(child_path, [])]
            queue.extend(grandchildren)
    return descendants


# --- Dependency Aggregation Logic (Adapted for Paths & Contextual Keys) ---
def aggregate_dependencies_contextual(
    project_root: str,
    path_to_key_info: Dict[str, KeyInfo],  # Global map: norm_path -> KeyInfo
    filtered_modules: Dict[
        str, KeyInfo
    ],  # Modules for the main tracker: {norm_path: KeyInfo}
    file_to_module: Optional[
        Dict[str, str]
    ] = None,  # Map: {norm_file_path: norm_module_path}
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Aggregates dependencies from mini-trackers to the main tracker,
    using paths as primary identifiers and handling contextual keys. Includes
    hierarchical rollup.

    Args:
        project_root: Absolute path to the project root.
        path_to_key_info: Global mapping of normalized paths to KeyInfo objects.
        filtered_modules: The paths and KeyInfo for modules (directories) in the main tracker.
        file_to_module: Mapping from normalized file paths to their containing module's normalized path.

    Returns:
        A dictionary where keys are source module paths and values are lists of
        (target_module_path, aggregated_dependency_char) tuples.
    """

    # Local helper to get mini tracker path without importing from tracker_io
    # This avoids a circular import cycle
    def _get_mini_tracker_path_local(module_path: str) -> str:
        module_name = os.path.basename(module_path)
        if not module_name and module_path:
            drive, _ = os.path.splitdrive(module_path)
            module_name = (
                drive.replace(":", "") + "_drive" if drive else "unknown_module"
            )
        elif not module_name:
            module_name = "unknown_root_module"
        return normalize_path(os.path.join(module_path, f"{module_name}_module.md"))

    if not file_to_module:
        logger.error(
            "File-to-module mapping missing, cannot perform main tracker aggregation."
        )
        return {}
    if not filtered_modules:
        logger.warning("No module paths/keys provided for main tracker aggregation.")
        return {}

    config = ConfigManager()
    get_priority = config.get_char_priority

    # Stores module_path -> target_module_path -> (highest_priority_char, highest_priority)
    aggregated_deps_prio = defaultdict(
        lambda: defaultdict(lambda: (PLACEHOLDER_CHAR, -1))
    )
    logger.info(
        f"Starting aggregation for {len(filtered_modules)} main tracker modules..."
    )

    # --- Step 1: Gather direct foreign dependencies from all relevant mini-trackers ---
    processed_mini_trackers = 0
    # Iterate through the modules designated for the main tracker
    for source_module_path, _ in filtered_modules.items():
        # Paths in filtered_modules should already be normalized
        # norm_source_module_path = normalize_path(source_module_path) # Already normalized
        norm_source_module_path = source_module_path
        mini_tracker_path = _get_mini_tracker_path_local(norm_source_module_path)
        if not os.path.exists(mini_tracker_path):
            continue

        processed_mini_trackers += 1
        try:
            # read_tracker_file returns data based on the *file content*, keys are strings
            mini_data = read_tracker_file_structured(mini_tracker_path)
            mini_grid = mini_data.get("grid", {})
            # Key definitions LOCAL to this mini-tracker: {key_string: path_string (might not be normalized)}
            mini_keys_defined_raw = mini_data.get("keys", {})
            # Normalize paths defined in the mini-tracker for consistent lookup
            mini_keys_defined = {
                k: normalize_path(p) for k, p in mini_keys_defined_raw.items()
            }

            if not mini_grid or not mini_keys_defined:
                logger.debug(
                    f"Mini tracker {os.path.basename(mini_tracker_path)} grid/keys empty."
                )
                continue
            # Get the list of key strings defined in this mini-tracker and sort them simply
            mini_grid_key_strings = sort_key_strings_hierarchically(
                list(mini_keys_defined.keys())
            )
            # Note: If natural sort order (like key_manager.sort_keys) is TRULY needed
            # for string keys here, a separate natural sort utility would be required.
            # Standard sort is usually sufficient for grid consistency.

            key_string_to_idx_mini = {k: i for i, k in enumerate(mini_grid_key_strings)}

            # Iterate through rows (sources) of the mini-tracker grid using key strings
            for mini_source_key_string, compressed_row in mini_grid.items():
                if mini_source_key_string not in key_string_to_idx_mini:
                    continue
                # Find the path corresponding to the key string *within this mini-tracker*
                mini_source_path = mini_keys_defined.get(mini_source_key_string)
                if not mini_source_path:
                    continue  # Path must be defined locally
                # Determine the module the source path belongs to using the global map
                # mini_source_path should already be normalized from mini_keys_defined creation
                actual_source_module_path = file_to_module.get(mini_source_path)
                if not actual_source_module_path:
                    # Log if a file/dir defined in a mini-tracker isn't in the file_to_module map
                    # logger.warning(f"Path '{mini_source_path}' (key '{mini_source_key_string}') from {mini_tracker_path} not found in file_to_module map.")
                    continue
                # IMPORTANT CHECK: Aggregate only if the source's module *is* the module this mini-tracker represents.
                if actual_source_module_path != norm_source_module_path:
                    # logger.debug(...)
                    continue
                # Process the columns (targets) for this valid source row
                try:
                    decompressed_row = list(decompress(compressed_row))
                    if len(decompressed_row) != len(mini_grid_key_strings):
                        logger.warning(
                            f"Row length mismatch for '{mini_source_key_string}' in {mini_tracker_path}."
                        )
                        continue
                    for col_idx, dep_char in enumerate(decompressed_row):
                        if dep_char in (PLACEHOLDER_CHAR, DIAGONAL_CHAR):
                            continue
                        mini_target_key_string = mini_grid_key_strings[col_idx]
                        # Find the path for the target key string defined locally
                        target_path = mini_keys_defined.get(mini_target_key_string)
                        if not target_path:
                            continue  # Target path must be defined locally
                        # Find the module the target path belongs to using the global map
                        # target_path should already be normalized
                        target_module_path = file_to_module.get(target_path)
                        if not target_module_path:
                            # logger.warning(f"Target path '{target_path}' (key '{mini_target_key_string}') from {mini_tracker_path} not found in file_to_module map.")
                            continue
                        # --- Check for FOREIGN relationship (source module != target module) ---
                        if target_module_path != actual_source_module_path:
                            # Use module paths as keys in aggregated_deps_prio
                            current_priority = get_priority(dep_char)
                            _stored_char, stored_priority = aggregated_deps_prio[
                                actual_source_module_path
                            ][target_module_path]
                            if current_priority > stored_priority:
                                # logger.debug(...) # Log using paths
                                aggregated_deps_prio[actual_source_module_path][
                                    target_module_path
                                ] = (dep_char, current_priority)
                            elif (
                                current_priority == stored_priority
                                and current_priority > -1
                            ):
                                # Handle equal priority conflicts (<> -> x)
                                if {dep_char, _stored_char} == {"<", ">"}:
                                    if (
                                        aggregated_deps_prio[actual_source_module_path][
                                            target_module_path
                                        ][0]
                                        != "x"
                                    ):
                                        # logger.debug(...) # Log using paths
                                        aggregated_deps_prio[actual_source_module_path][
                                            target_module_path
                                        ] = ("x", current_priority)
                                # else: Keep existing on other equal priority conflicts
                except Exception as decomp_err:
                    logger.warning(
                        f"Error decompressing/processing row for '{mini_source_key_string}' in {mini_tracker_path}: {decomp_err}"
                    )
        except Exception as read_err:
            logger.error(
                f"Error reading or processing mini tracker {mini_tracker_path} during aggregation: {read_err}",
                exc_info=True,
            )

    logger.info(
        f"Processed {processed_mini_trackers} mini-trackers for direct dependencies."
    )

    # --- Step 2: Perform Hierarchical Rollup (Using Paths) ---
    logger.info("Performing hierarchical rollup...")
    # Build parent -> direct children map using normalized paths from filtered_modules
    hierarchy: Dict[str, List[str]] = defaultdict(list)
    module_paths_list = sorted(
        list(filtered_modules.keys())
    )  # Use paths from filtered_modules
    for p_path in module_paths_list:
        # p_path is already normalized
        for c_path in module_paths_list:
            # c_path is already normalized
            if p_path == c_path:
                continue
            # Check if c_path is DIRECTLY inside p_path
            # Ensure trailing slash consistency might be needed if paths mix formats, but normalize_path should handle this.
            if (
                c_path.startswith(p_path + os.sep)
                and normalize_path(os.path.dirname(c_path)) == p_path
            ):
                hierarchy[p_path].append(c_path)

    # Iteratively propagate dependencies up the hierarchy
    changed_in_pass = True
    max_passes = len(module_paths_list)  # Safety break
    current_pass = 0
    while changed_in_pass and current_pass < max_passes:
        changed_in_pass = False
        current_pass += 1
        logger.debug(f"Hierarchy Rollup Pass {current_pass}")
        for parent_path in module_paths_list:  # Iterate through all potential parents
            # Calculate all descendants ONCE per parent per pass
            all_descendants_paths = _get_descendants_paths(
                parent_path, hierarchy
            )  # Use the path-based helper
            # Check direct children for inheritance
            for child_path in hierarchy.get(parent_path, []):
                # Inherit dependencies *from* the child
                child_deps = list(aggregated_deps_prio.get(child_path, {}).items())
                for target_path, (dep_char, priority) in child_deps:
                    # Inherit if:
                    # 1. Dependency is meaningful (priority > -1)
                    # 2. Target is NOT the parent itself
                    # 3. Target is NOT another descendant of the parent
                    if (
                        priority > -1
                        and target_path != parent_path
                        and target_path not in all_descendants_paths
                    ):
                        _parent_stored_char, parent_stored_priority = (
                            aggregated_deps_prio[parent_path][target_path]
                        )
                        if priority > parent_stored_priority:
                            aggregated_deps_prio[parent_path][target_path] = (
                                dep_char,
                                priority,
                            )
                            changed_in_pass = True
                        elif priority == parent_stored_priority and priority > -1:
                            # Merge < > to x on equal priority during rollup
                            if {_parent_stored_char, dep_char} == {"<", ">"}:
                                if (
                                    aggregated_deps_prio[parent_path][target_path][0]
                                    != "x"
                                ):
                                    # logger.debug(...) # Log using paths
                                    aggregated_deps_prio[parent_path][target_path] = (
                                        "x",
                                        priority,
                                    )
                                    changed_in_pass = True
                            # else: keep existing if priorities equal and not < > conflict

    if current_pass == max_passes and changed_in_pass:
        logger.warning(
            "Hierarchical rollup reached max passes, potentially indicating a cycle or very deep nesting."
        )

    # --- Step 3: Convert to final output format (Using Paths) ---
    final_suggestions = defaultdict(list)
    # Sort source paths for deterministic output order
    sorted_source_paths = sorted(aggregated_deps_prio.keys())
    for source_path in sorted_source_paths:
        # Ensure source_path is actually a module path we care about (should be by construction, but check is safe)
        if source_path not in filtered_modules:
            continue

        targets = aggregated_deps_prio[source_path]
        # Sort target paths for deterministic order within each source
        sorted_target_paths = sorted(targets.keys())
        for target_path in sorted_target_paths:
            dep_char, _priority = targets[target_path]
            # Ensure target path is also a module path and dependency is not placeholder
            if target_path in filtered_modules and dep_char != PLACEHOLDER_CHAR:
                final_suggestions[source_path].append((target_path, dep_char))
        # Sorting of tuples (target_path, dep_char) was implicitly handled by iterating sorted target paths
        # If explicit sort by tuple needed: final_suggestions[source_path].sort()

    logger.info("Main tracker aggregation finished.")
    return dict(final_suggestions)


# --- Data Structure Export ---
# This structure defines how the main tracker is updated.
# It should be imported and used by tracker_io.update_tracker when tracker_type is 'main'.

main_tracker_data = {
    # <<< *** MODIFIED SIGNATURE *** >>>
    "key_filter": main_key_filter,  # Now takes path_to_key_info, returns Dict[str, KeyInfo]
    "dependency_aggregation": aggregate_dependencies_contextual,  # Already adapted
    "get_tracker_path": get_main_tracker_path,
}

# EoF
