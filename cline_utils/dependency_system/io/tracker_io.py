# io/tracker_io.py

"""
IO module for tracker file operations using contextual keys.
Handles reading, writing, merging and exporting tracker files.
Relies on ordered lists of KeyInfo objects, where key strings can be duplicated
if their associated paths are different. The order is determined by
hierarchical key string sort, then by normalized path.
"""

import datetime
import io
import json
import logging
import os
import re
import shutil
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from cline_utils.dependency_system.core.dependency_grid import (
    DIAGONAL_CHAR,
    EMPTY_CHAR,
    PLACEHOLDER_CHAR,
    compress,
    create_initial_grid,
    decompress,
    validate_grid,
)

# --- Core Imports ---
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.core.key_manager import (
    get_key_from_path as get_key_string_from_path_global,
)  # string from global map for a path
from cline_utils.dependency_system.core.key_manager import (
    get_sortable_parts_for_key,
    load_global_key_map,
    load_old_global_key_map,
    sort_key_strings_hierarchically,
)

from cline_utils.dependency_system.core.key_manager import (
    validate_key as validate_key_format,
)

# --- IO Imports (Specific tracker data for paths/filters) ---
from cline_utils.dependency_system.io.update_doc_tracker import doc_tracker_data
from cline_utils.dependency_system.io.update_main_tracker import main_tracker_data
from cline_utils.dependency_system.io.update_mini_tracker import get_mini_tracker_data
from cline_utils.dependency_system.utils.cache_manager import (
    cached,
    check_file_modified,
    invalidate_dependent_entries,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager

# --- Utility Imports ---
from cline_utils.dependency_system.utils.path_utils import (
    get_project_root,
    is_subpath,
    join_paths,
    normalize_path,
)
from cline_utils.dependency_system.utils.tracker_utils import (
    aggregate_all_dependencies,
    find_all_tracker_paths,
    get_key_global_instance_string,
    read_grid_from_lines,
    read_key_definitions_from_lines,
    read_tracker_file_structured,
    resolve_key_global_instance_to_ki,
)

logger = logging.getLogger(__name__)

PathMigrationInfo = Dict[str, Tuple[Optional[str], Optional[str]]]

# --- Constant for AST verified links file (as provided) ---
AST_VERIFIED_LINKS_FILENAME = "ast_verified_links.json"
_CORE_DIR_FOR_AST_LINKS: Optional[str] = None
try:
    _key_manager_module_for_path = __import__(
        "cline_utils.dependency_system.core.key_manager", fromlist=[""]
    )
    _CORE_DIR_FOR_AST_LINKS = os.path.dirname(
        os.path.abspath(_key_manager_module_for_path.__file__)
    )
except ImportError:
    logger.error(
        "TrackerIO: Could not determine core directory for AST links file. Path may be incorrect."
    )


# --- build_path_migration_map (Keep existing as provided) ---
def build_path_migration_map(
    old_global_map: Optional[Dict[str, KeyInfo]], new_global_map: Dict[str, KeyInfo]
) -> PathMigrationInfo:
    """
    Compares old and new global key maps to build a migration map based on paths.

    Args:
        old_global_map: The loaded KeyInfo map from the previous state (or None).
        new_global_map: The loaded KeyInfo map for the current state.

    Returns:
        A dictionary mapping normalized paths to tuples (old_key, new_key).
        Returns keys as None if the path didn't exist in that state or if maps are missing.
    """
    path_migration_info: PathMigrationInfo = {}
    logger.debug("Building path migration map based on global key maps...")
    # Create reverse lookups (Path -> Key)
    old_path_to_key: Dict[str, str] = {}
    new_path_to_key: Dict[str, str] = {}
    old_paths: Set[str] = set()
    new_paths: Set[str] = set()

    if old_global_map:
        duplicates_old = set()
        for path, info in old_global_map.items():
            norm_p = normalize_path(path)
            if norm_p in old_path_to_key:
                if norm_p not in duplicates_old:  # Log only once per duplicate
                    logger.critical(
                        f"CRITICAL ERROR: Duplicate path '{norm_p}' found in OLD global key map! Keys: '{old_path_to_key[norm_p]}' and '{info.key_string}'. Aborting migration map build."
                    )
                    duplicates_old.add(norm_p)
            else:
                old_path_to_key[norm_p] = info.key_string
                old_paths.add(norm_p)
        if duplicates_old:
            raise ValueError(
                "Duplicate paths found in old global key map. Cannot proceed."
            )
        logger.debug(f"Old global map: Found {len(old_paths)} unique paths.")
    else:
        logger.warning(
            "Old global map not provided or loaded. Path stability relies only on new map."
        )

    duplicates_new = set()
    for path, info in new_global_map.items():
        norm_p = normalize_path(path)
        if norm_p in new_path_to_key:
            if norm_p not in duplicates_new:
                logger.critical(
                    f"CRITICAL ERROR: Duplicate path '{norm_p}' found in NEW global key map! Keys: '{new_path_to_key[norm_p]}' and '{info.key_string}'. Aborting migration map build."
                )
                duplicates_new.add(norm_p)
        else:
            new_path_to_key[norm_p] = info.key_string
            new_paths.add(norm_p)
    if duplicates_new:
        raise ValueError("Duplicate paths found in new global key map. Cannot proceed.")
    logger.debug(f"New global map: Found {len(new_paths)} unique paths.")

    # Determine stable, removed, added paths
    stable_paths = (
        old_paths.intersection(new_paths) if old_global_map else new_paths
    )  # If no old map, only "new" paths are considered stable relative to themselves
    removed_paths = old_paths - new_paths if old_global_map else set()
    added_paths = (
        new_paths - old_paths if old_global_map else set()
    )  # If no old map, all new paths are 'added' contextually

    logger.debug(
        f"Path comparison: Stable={len(stable_paths)}, Removed={len(removed_paths)}, Added={len(added_paths)}"
    )

    # Populate the migration map
    for path in stable_paths:
        old_key = old_path_to_key.get(path)  # Will be None if old_global_map is None
        new_key = new_path_to_key.get(
            path
        )  # Should always exist if path is in new_paths
        path_migration_info[path] = (old_key, new_key)
    for path in removed_paths:
        old_key = old_path_to_key.get(path)
        path_migration_info[path] = (old_key, None)
    for path in added_paths:
        new_key = new_path_to_key.get(path)
        path_migration_info[path] = (None, new_key)

    logger.debug(
        f"Path migration map built with {len(path_migration_info)} total path entries."
    )
    return path_migration_info


# --- Path Finding ---
# Caching for get_tracker_path (consider config mtime)
@cached(
    "tracker_paths",
    key_func=lambda project_root, tracker_type="main", module_path=None: f"tracker_path:{normalize_path(project_root)}:{tracker_type}:{normalize_path(module_path) if module_path else 'none'}:{(os.path.getmtime(ConfigManager().config_path) if ConfigManager().config_path and os.path.exists(ConfigManager().config_path) else 0)}",
)
def get_tracker_path(
    project_root: str, tracker_type: str = "main", module_path: Optional[str] = None
) -> str:
    """
    Get the path to the appropriate tracker file based on type. Ensures path uses forward slashes.

    Args:
        project_root: Project root directory
        tracker_type: Type of tracker ('main', 'doc', or 'mini')
        module_path: The module path (required for mini-trackers)
    Returns:
        Normalized path to the tracker file using forward slashes
    """
    project_root = normalize_path(project_root)
    norm_module_path = normalize_path(module_path) if module_path else None

    if tracker_type == "main":
        return normalize_path(main_tracker_data["get_tracker_path"](project_root))
    elif tracker_type == "doc":
        return normalize_path(doc_tracker_data["get_tracker_path"](project_root))
    elif tracker_type == "mini":
        if not norm_module_path:
            raise ValueError("module_path must be provided for mini-trackers")
        mini_data_config = get_mini_tracker_data()
        if "get_tracker_path" in mini_data_config and callable(
            mini_data_config["get_tracker_path"]
        ):  # Check if dedicated function exists
            path_func = mini_data_config["get_tracker_path"]
            return normalize_path(str(path_func(norm_module_path)))
        else:
            module_name = os.path.basename(norm_module_path)
            if (
                not module_name and norm_module_path
            ):  # Handle drive root case, e.g. "H:/"
                drive, _ = os.path.splitdrive(norm_module_path)
                module_name = (
                    drive.replace(":", "") + "_drive" if drive else "unknown_module"
                )
            elif (
                not module_name
            ):  # Handle case like "" or "." if they somehow reach here
                module_name = "unknown_root_module"
            return normalize_path(
                os.path.join(norm_module_path, f"{module_name}_module.md")
            )
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")


# --- Writing Section Helpers (now use display keys) ---
def _get_display_key_for_tracker(
    ki: KeyInfo, global_map: Dict[str, KeyInfo], global_key_counts: Dict[str, int]
) -> str:
    if global_key_counts.get(ki.key_string, 0) > 1:
        # Globally duplicated base key, so use its KEY#GI
        gi_str = get_key_global_instance_string(ki, global_map)  # From tracker_utils
        if gi_str:
            return gi_str
        else:  # Fallback if GI string can't be formed (should not happen for valid KI)
            logger.error(
                f"WriteHelper: Could not get GI string for KI {ki}. Using base key."
            )
            return ki.key_string
    return ki.key_string  # Not globally duplicated, use base key


def _write_key_definitions_section(
    file_obj: io.TextIOBase,
    key_info_list: List[KeyInfo],
    current_global_map: Dict[str, KeyInfo],  # Needed for display key determination
    global_key_counts: Dict[str, int],  # Precomputed counts
):
    """Writes the key definitions section using the pre-sorted list of KeyInfo."""
    file_obj.write("---KEY_DEFINITIONS_START---\n")
    file_obj.write("Key Definitions:\n")
    for ki in key_info_list:
        display_key = _get_display_key_for_tracker(
            ki, current_global_map, global_key_counts
        )
        file_obj.write(f"{display_key}: {normalize_path(ki.norm_path)}\n")
    file_obj.write("---KEY_DEFINITIONS_END---\n")


def _write_grid_section(
    file_obj: io.TextIOBase,
    key_info_list: List[KeyInfo],  # Defines order and provides KIs
    grid_compressed_rows: List[str],
    current_global_map: Dict[str, KeyInfo],  # Needed for display key
    global_key_counts: Dict[str, int],  # Precomputed counts
):
    """Writes the grid section. Assumes lists are correctly ordered and sized by caller."""
    file_obj.write("---GRID_START---\n")

    # Get display keys for header and row labels
    display_keys = [
        _get_display_key_for_tracker(ki, current_global_map, global_key_counts)
        for ki in key_info_list
    ]

    if not display_keys:
        file_obj.write("X \n")
    else:
        file_obj.write(f"X {' '.join(display_keys)}\n")
        if len(display_keys) != len(grid_compressed_rows):
            logger.critical(
                f"CRITICAL WRITE GRID: Display key count {len(display_keys)} != row count {len(grid_compressed_rows)}."
            )

        for i in range(min(len(display_keys), len(grid_compressed_rows))):
            file_obj.write(f"{display_keys[i]} = {grid_compressed_rows[i]}\n")
    file_obj.write("---GRID_END---\n")


# --- Adapted Utility: validate_grid_ordered ---
def validate_grid_ordered(
    grid_rows_compressed: List[str],  # List of compressed rows
    expected_length: int,  # Expected number of items (rows/columns)
) -> bool:
    """
    Validates a grid represented by a list of compressed rows.
    Checks for consistent length and valid characters.
    """
    if not grid_rows_compressed and expected_length == 0:
        return True  # Empty grid is valid if expected length is 0
    if len(grid_rows_compressed) != expected_length:
        logger.error(
            f"Grid validation failed: Expected {expected_length} rows, but got {len(grid_rows_compressed)}."
        )
        return False

    # Ensure ConfigManager is initialized to get allowed_dependency_chars
    try:
        config_mgr_instance = ConfigManager()
        allowed_dep_chars = config_mgr_instance.get_allowed_dependency_chars()
    except (
        Exception
    ) as e_conf:  # Catch potential errors during ConfigManager init if used early
        logger.error(
            f"Could not load config for grid validation: {e_conf}. Using fallback chars."
        )
        allowed_dep_chars = ["<", ">", "x", "d", "s", "S"]

    # Add 'n' to the set of valid characters
    valid_chars = {PLACEHOLDER_CHAR, EMPTY_CHAR, DIAGONAL_CHAR, "n"}.union(
        set(allowed_dep_chars)
    )  # MODIFIED
    # Or, if you have NO_DEPENDENCY_CHAR constant:
    # valid_chars = {PLACEHOLDER_CHAR, EMPTY_CHAR, DIAGONAL_CHAR, NO_DEPENDENCY_CHAR}.union(set(allowed_dep_chars))

    for i, compressed_row in enumerate(grid_rows_compressed):
        try:
            decompressed_row = decompress(compressed_row)
            if len(decompressed_row) != expected_length:
                logger.error(
                    f"Grid validation failed: Row {i} has length {len(decompressed_row)}, expected {expected_length}."
                )
                return False
            for char_idx, char_val in enumerate(decompressed_row):
                if char_val not in valid_chars:
                    logger.error(
                        f"Grid validation failed: Row {i}, Col {char_idx} has invalid character '{char_val}'."
                    )
                    return False
                # Diagonal check: only strictly enforce if not an empty placeholder during initial build.
                if (
                    i == char_idx
                    and char_val != DIAGONAL_CHAR
                    and char_val != PLACEHOLDER_CHAR
                    and expected_length > 0
                ):
                    logger.warning(
                        f"Grid validation: Row {i}, Col {char_idx} (diagonal) is '{char_val}', expected '{DIAGONAL_CHAR}' or '{PLACEHOLDER_CHAR}'."
                    )
                    # Allowing PLACEHOLDER_CHAR on diagonal during intermediate states before finalization.
        except Exception as e:
            logger.error(f"Grid validation failed: Error decompressing row {i}: {e}")
            return False
    return True


def write_mini_tracker_with_template_preservation(
    output_file: str,
    lines_from_old_file: List[str],  # For preserving header/footer
    key_info_list_to_write: List[KeyInfo],  # The KIs defining the new tracker content
    grid_compressed_rows_to_write: List[str],  # The grid data for the new content
    last_key_edit_msg: str,
    last_grid_edit_msg: str,
    template_string: str,  # The module's template string
    template_markers: Tuple[str, str],  # (start_marker, end_marker)
    current_global_map: Dict[str, KeyInfo],  # Passed in for display key logic
    module_path_for_template: str,  # Actual module path for formatting {module_name}
):
    logger.debug(f"Writing mini tracker: {output_file}")
    start_marker, end_marker = template_markers
    existing_content_start_idx, existing_content_end_idx = -1, -1

    module_name_for_format = os.path.basename(module_path_for_template)
    if not module_name_for_format and module_path_for_template:
        drive, _ = os.path.splitdrive(module_path_for_template)
        module_name_for_format = (
            drive.replace(":", "") + "_drive" if drive else "UnknownModule"
        )
    elif not module_name_for_format:
        module_name_for_format = "UnknownRootModule"

    if lines_from_old_file:
        try:
            existing_content_start_idx = next(
                i
                for i, line in enumerate(lines_from_old_file)
                if line.strip() == start_marker
            )
            existing_content_end_idx = next(
                i
                for i, line in enumerate(lines_from_old_file)
                if line.strip() == end_marker and i > existing_content_start_idx
            )
        except StopIteration:
            logger.debug(
                f"Mini-tracker {os.path.basename(output_file)}: Markers not found/invalid in existing lines. Full template overwrite."
            )
            existing_content_start_idx, existing_content_end_idx = -1, -1
            lines_from_old_file = (
                []
            )  # Treat as if no old lines to preserve header/footer from

    # Precompute global key counts for display key determination by section writers
    global_key_counts_for_display = defaultdict(int)
    for ki_global in current_global_map.values():
        global_key_counts_for_display[ki_global.key_string] += 1

    try:
        with open(output_file, "w", encoding="utf-8", newline="\n") as f:
            # 1. Write header (content before start marker or full template)
            if existing_content_start_idx != -1:  # Markers found, preserve header
                for i in range(
                    existing_content_start_idx + 1
                ):  # Include the start marker line
                    f.write(lines_from_old_file[i])
                if not lines_from_old_file[existing_content_start_idx].endswith("\n"):
                    f.write("\n")
            else:  # No valid old markers, or new file: write formatted template
                try:
                    formatted_template = template_string.format(
                        module_name=module_name_for_format
                    )
                    f.write(formatted_template)
                except KeyError:  # If template doesn't use {module_name}
                    f.write(template_string)
                # Ensure start marker is present if template didn't include it
                # Check if template content *already ends with* the start marker (common pattern)
                template_lines_stripped = [
                    line.strip() for line in template_string.splitlines()
                ]
                if (
                    not template_lines_stripped
                    or template_lines_stripped[-1] != start_marker
                ):
                    if not template_string.endswith("\n"):
                        f.write("\n")
                    f.write(
                        f"\n{start_marker}\n"
                    )  # Add blank line before for separation

            f.write("\n")  # Ensure blank line after header/start_marker

            # 2. Write Key Definitions (uses current_global_map and global_key_counts)
            _write_key_definitions_section(
                f,
                key_info_list_to_write,
                current_global_map,
                global_key_counts_for_display,
            )
            f.write("\n")

            # 3. Write Metadata
            f.write(f"last_KEY_edit: {last_key_edit_msg}\n")
            f.write(f"last_GRID_edit: {last_grid_edit_msg}\n\n")

            # 4. Write Grid Data (uses current_global_map and global_key_counts)
            _write_grid_section(
                f,
                key_info_list_to_write,
                grid_compressed_rows_to_write,
                current_global_map,
                global_key_counts_for_display,
            )
            f.write("\n")

            # 5. Write footer (content after end marker or just the end marker)
            if existing_content_end_idx != -1:  # Markers found, preserve footer
                # Write the end marker line itself from the old content
                f.write(lines_from_old_file[existing_content_end_idx])
                if not lines_from_old_file[existing_content_end_idx].endswith("\n"):
                    f.write("\n")
                # Write content after the original end marker
                for i in range(existing_content_end_idx + 1, len(lines_from_old_file)):
                    f.write(lines_from_old_file[i])
                if lines_from_old_file and not lines_from_old_file[-1].endswith(
                    "\n"
                ):  # Ensure final newline
                    f.write("\n")
            else:  # No valid old markers or new file: ensure end marker if template didn't include it
                template_lines_stripped = [
                    line.strip() for line in template_string.splitlines()
                ]
                if (
                    not template_lines_stripped
                    or template_lines_stripped[-1] != end_marker
                ):
                    # Check if the last written part was the grid's end marker
                    # This is tricky; easier to just ensure it's there if not in template.
                    # A robust check would see if f.tell() is right after GRID_END---
                    f.write(f"{end_marker}\n")

        logger.debug(f"Successfully wrote mini tracker content to: {output_file}")
    except Exception as e_write_mini:
        logger.error(
            f"Error during write_mini_tracker_with_template_preservation for {output_file}: {e_write_mini}",
            exc_info=True,
        )
        # Consider if this should raise to stop cache invalidation etc. For now, just logs.


# --- Patched: Top-level write_tracker_file ---
def write_tracker_file(
    tracker_path: str,
    key_info_to_write: List[KeyInfo],
    grid_rows_ordered: List[str],
    last_key_edit: str,
    last_grid_edit: str,
    current_global_map: Dict[str, KeyInfo],  # NEW - needed for display key decisions
) -> bool:
    tracker_path = normalize_path(tracker_path)
    try:
        # Precompute global key counts
        global_key_counts = defaultdict(int)
        for ki_global in current_global_map.values():
            global_key_counts[ki_global.key_string] += 1

        dirname = os.path.dirname(tracker_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        grid_key_strings_for_header_and_rows = [
            ki.key_string for ki in key_info_to_write
        ]
        expected_grid_size = len(key_info_to_write)

        # --- Validate grid before writing ---
        # `validate_grid_ordered` now checks overall row count and individual row lengths.
        if not validate_grid_ordered(grid_rows_ordered, expected_grid_size):
            logger.error(
                f"Aborting write to {tracker_path} due to grid validation failure. "
                + f"Expected {expected_grid_size} items, received {len(grid_rows_ordered)} grid rows."
            )
            return False

        # --- Ensure grid_rows_ordered matches expected_grid_size ---
        # This is a final safety check; validation should ideally catch inconsistencies.
        final_grid_rows_to_write = list(grid_rows_ordered)  # Mutable copy
        if len(final_grid_rows_to_write) != expected_grid_size:
            # This case should ideally be caught by upstream logic or validation.
            # If we reach here, it's a fallback.
            logger.warning(
                f"Correcting grid row count for {tracker_path}. Expected {expected_grid_size}, got {len(grid_rows_ordered)} data rows. Rebuilding problematic rows."
            )
            temp_decomp_rows = []
            # Try to decompress what we have
            for i in range(expected_grid_size):
                if i < len(grid_rows_ordered):
                    try:
                        decomp_row = list(decompress(grid_rows_ordered[i]))
                        if len(decomp_row) == expected_grid_size:
                            temp_decomp_rows.append(decomp_row)
                            continue
                    except Exception:
                        pass  # Fall through to re-initialize
                # Re-initialize row
                new_row_chars = [PLACEHOLDER_CHAR] * expected_grid_size
                new_row_chars[i] = DIAGONAL_CHAR
                temp_decomp_rows.append(new_row_chars)
            final_grid_rows_to_write = [compress("".join(r)) for r in temp_decomp_rows]
        else:
            final_grid_rows_to_write = grid_rows_ordered

        # --- Write Content ---
        with open(tracker_path, "w", encoding="utf-8", newline="\n") as f:
            _write_key_definitions_section(
                f, key_info_to_write, current_global_map, global_key_counts
            )
            f.write("\n")
            f.write(f"last_KEY_edit: {last_key_edit}\n")
            f.write(f"last_GRID_edit: {last_grid_edit}\n\n")
            _write_grid_section(
                f,
                key_info_to_write,
                grid_rows_ordered,
                current_global_map,
                global_key_counts,
            )
        logger.debug(
            f"Successfully wrote tracker file: {tracker_path} with {len(key_info_to_write)} key instances."
        )
        # Invalidate cache for this specific tracker file after writing
        invalidate_dependent_entries("tracker_data", f"tracker_data:{tracker_path}:.*")
        return True
    except IOError as e:
        logger.error(f"I/O Error writing {tracker_path}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.exception(f"Unexpected error writing {tracker_path}: {e}")
        return False


# --- Backup ---
def backup_tracker_file(tracker_path: str) -> str:
    """
    Create a backup of a tracker file, keeping the 2 most recent backups.

    Args:
        tracker_path: Path to the tracker file
    Returns:
        Path to the backup file or empty string on failure
    """
    tracker_path = normalize_path(tracker_path)
    if not os.path.exists(tracker_path):
        logger.warning(f"Tracker file not found for backup: {tracker_path}")
        return ""
    try:
        config = ConfigManager()
        project_root = get_project_root()
        backup_dir_rel = config.get_path(
            "backups_dir", "cline_docs/backups"
        )  # Default if not in config
        backup_dir_abs = normalize_path(os.path.join(project_root, backup_dir_rel))
        os.makedirs(backup_dir_abs, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = os.path.basename(tracker_path)
        backup_filename = f"{base_name}.{timestamp}.bak"
        backup_path = os.path.join(backup_dir_abs, backup_filename)
        shutil.copy2(tracker_path, backup_path)
        logger.debug(
            f"Backed up tracker '{base_name}' to: {os.path.basename(backup_path)}"
        )

        # --- Cleanup old backups (keep 2 most recent) ---
        try:
            backup_files_for_base: List[Tuple[datetime.datetime, str]] = []
            for filename_in_backup_dir in os.listdir(backup_dir_abs):
                if filename_in_backup_dir.startswith(
                    base_name + "."
                ) and filename_in_backup_dir.endswith(".bak"):
                    # Extract timestamp part of filename
                    # Format: base_name.YYYYMMDD_HHMMSS_ffffff.bak
                    match = re.search(
                        r"\.(\d{8}_\d{6}_\d{6})\.bak$", filename_in_backup_dir
                    )
                    if match:
                        timestamp_str_from_name = match.group(1)
                        try:
                            file_dt_obj = datetime.datetime.strptime(
                                timestamp_str_from_name, "%Y%m%d_%H%M%S_%f"
                            )
                            backup_files_for_base.append(
                                (
                                    file_dt_obj,
                                    os.path.join(
                                        backup_dir_abs, filename_in_backup_dir
                                    ),
                                )
                            )
                        except ValueError:
                            logger.warning(
                                f"Could not parse timestamp for backup file: {filename_in_backup_dir}"
                            )

            backup_files_for_base.sort(
                key=lambda x: x[0], reverse=True
            )  # Sort newest first

            if len(backup_files_for_base) > 2:
                files_to_delete = backup_files_for_base[2:]  # All except the two newest
                logger.debug(
                    f"Cleaning up {len(files_to_delete)} older backups for '{base_name}'."
                )
                for _, file_path_to_delete in files_to_delete:
                    try:
                        os.remove(file_path_to_delete)
                    except OSError as delete_error:
                        logger.error(
                            f"Error deleting old backup {file_path_to_delete}: {delete_error}"
                        )
        except Exception as cleanup_error:
            logger.error(
                f"Error during backup cleanup for {base_name}: {cleanup_error}",
                exc_info=True,
            )

        return backup_path
    except Exception as e:
        logger.error(
            f"Error backing up tracker file {tracker_path}: {e}", exc_info=True
        )
        return ""


# --- Merge Helpers (Patched) ---
def _merge_grids(
    primary_grid_rows_compressed: List[str],  # Ordered list
    secondary_grid_rows_compressed: List[str],  # Ordered list
    primary_key_info_list: List[KeyInfo],  # Defines order for primary_grid_rows
    secondary_key_info_list: List[KeyInfo],  # Defines order for secondary_grid_rows
    merged_key_info_list: List[KeyInfo],  # Defines order for the output grid
) -> List[str]:  # Returns merged list of compressed rows
    """Merges two grids. Primary overwrites secondary. Grids are based on ordered KeyInfo lists."""

    merged_size = len(merged_key_info_list)
    # Initialize merged grid (decompressed)
    merged_decompressed_grid_rows: List[List[str]] = [
        [PLACEHOLDER_CHAR] * merged_size for _ in range(merged_size)
    ]
    for i in range(merged_size):
        if i < merged_size:
            merged_decompressed_grid_rows[i][i] = DIAGONAL_CHAR  # Safety check

    config = ConfigManager()  # For priority
    get_priority = config.get_char_priority

    # Helper to decompress a list of rows, robustly
    def safe_decompress_rows(
        grid_rows_compressed: List[str], expected_ki_list: List[KeyInfo]
    ) -> List[Optional[List[str]]]:
        decomp_rows_list: List[Optional[List[str]]] = []
        expected_len = len(expected_ki_list)
        if not grid_rows_compressed and expected_len == 0:
            return []  # Handle empty grid case
        if len(grid_rows_compressed) != expected_len:
            logger.warning(
                f"Merge Prep: Grid row count {len(grid_rows_compressed)} != expected KI list len {expected_len}. Grid data may be misaligned."
            )
            # Pad with None if grid_rows_compressed is shorter, or truncate if longer for processing loop
            # This indicates an upstream issue if counts don't match.

        for idx in range(expected_len):  # Iterate based on expected_len
            if idx >= len(grid_rows_compressed):  # Not enough rows in compressed data
                logger.warning(
                    f"Merge Prep: Missing compressed row for KI at index {idx} ({expected_ki_list[idx].norm_path}). Treating as invalid."
                )
                decomp_rows_list.append(None)
                continue

            compressed_row_str = grid_rows_compressed[idx]
            row_ki_context = expected_ki_list[idx]
            try:
                decomp_chars = list(decompress(compressed_row_str))
                if len(decomp_chars) == expected_len:
                    decomp_rows_list.append(decomp_chars)
                else:
                    logger.warning(
                        f"Merge Prep: Incorrect decompressed length for row {idx} (key '{row_ki_context.key_string}', path '{row_ki_context.norm_path}'). Expected {expected_len}, got {len(decomp_chars)}. Row skipped."
                    )
                    decomp_rows_list.append(None)
            except Exception as e:
                logger.warning(
                    f"Merge Prep: Failed to decompress row {idx} (key '{row_ki_context.key_string}', path '{row_ki_context.norm_path}'): {e}. Row skipped."
                )
                decomp_rows_list.append(None)
        return decomp_rows_list

    primary_decomp_rows = safe_decompress_rows(
        primary_grid_rows_compressed, primary_key_info_list
    )
    secondary_decomp_rows = safe_decompress_rows(
        secondary_grid_rows_compressed, secondary_key_info_list
    )

    # Map paths to their indices in each list for efficient lookup during merge
    primary_path_to_idx = {
        ki.norm_path: i for i, ki in enumerate(primary_key_info_list)
    }
    secondary_path_to_idx = {
        ki.norm_path: i for i, ki in enumerate(secondary_key_info_list)
    }
    # merged_path_to_idx = {ki.norm_path: i for i, ki in enumerate(merged_key_info_list)} # Not strictly needed if iterating merged_key_info_list

    for merged_row_idx, merged_row_ki in enumerate(merged_key_info_list):
        for merged_col_idx, merged_col_ki in enumerate(merged_key_info_list):
            if merged_row_idx == merged_col_idx:
                continue  # Diagonal already set

            primary_val: Optional[str] = None
            pri_row_original_idx = primary_path_to_idx.get(merged_row_ki.norm_path)
            pri_col_original_idx = primary_path_to_idx.get(merged_col_ki.norm_path)
            if pri_row_original_idx is not None and pri_col_original_idx is not None:
                if (
                    pri_row_original_idx < len(primary_decomp_rows)
                    and primary_decomp_rows[pri_row_original_idx] is not None
                ):  # Check row was valid
                    decomp_pri_row = primary_decomp_rows[pri_row_original_idx]
                    if decomp_pri_row and pri_col_original_idx < len(decomp_pri_row):
                        primary_val = decomp_pri_row[pri_col_original_idx]

            secondary_val: Optional[str] = None
            sec_row_original_idx = secondary_path_to_idx.get(merged_row_ki.norm_path)
            sec_col_original_idx = secondary_path_to_idx.get(merged_col_ki.norm_path)
            if sec_row_original_idx is not None and sec_col_original_idx is not None:
                if (
                    sec_row_original_idx < len(secondary_decomp_rows)
                    and secondary_decomp_rows[sec_row_original_idx] is not None
                ):  # Check row was valid
                    decomp_sec_row = secondary_decomp_rows[sec_row_original_idx]
                    if decomp_sec_row and sec_col_original_idx < len(decomp_sec_row):
                        secondary_val = decomp_sec_row[sec_col_original_idx]

            final_val_to_set = PLACEHOLDER_CHAR  # Default
            if primary_val is not None and primary_val != PLACEHOLDER_CHAR:
                final_val_to_set = primary_val
            elif secondary_val is not None and secondary_val != PLACEHOLDER_CHAR:
                final_val_to_set = secondary_val

            merged_decompressed_grid_rows[merged_row_idx][
                merged_col_idx
            ] = final_val_to_set

    final_merged_compressed_rows = [
        compress("".join(row_list)) for row_list in merged_decompressed_grid_rows
    ]
    return final_merged_compressed_rows


# --- Patched: merge_trackers ---
def merge_trackers(
    primary_tracker_path: str,
    secondary_tracker_path: str,
    output_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Merge two tracker files. Primary takes precedence for path definitions (if key strings conflict).
    Grid merging is path-based.
    Output format uses List[KeyInfo] for definitions.
    """
    primary_tracker_path = normalize_path(primary_tracker_path)
    secondary_tracker_path = normalize_path(secondary_tracker_path)
    output_path = normalize_path(output_path) if output_path else primary_tracker_path
    logger.info(
        f"Merging '{os.path.basename(primary_tracker_path)}' & '{os.path.basename(secondary_tracker_path)}' into '{os.path.basename(output_path)}'"
    )

    # --- Load CURRENT Global Key Map (path_to_key_info) ---
    # This is essential for ensuring merged KeyInfo objects are up-to-date.
    current_global_path_to_key_info = load_global_key_map()
    if not current_global_path_to_key_info:
        logger.error("Merge failed: Cannot load current global key map.")
        return None

    # _parse_tracker_for_merge (helper from previous patch)
    def _parse_tracker_for_merge(
        file_path: str, global_map: Dict[str, KeyInfo]
    ) -> Dict[str, Any]:
        parsed_data = {
            "key_info_list": [],
            "grid_header_keys": [],
            "grid_rows_compressed": [],
            "last_key_edit": "",
            "last_grid_edit": "",
        }
        if not os.path.exists(file_path):
            return parsed_data
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            key_path_pairs = read_key_definitions_from_lines(
                lines
            )  # List[(key_str, path_str)]

            temp_ki_list: List[KeyInfo] = []
            for k_str, p_str in key_path_pairs:
                ki = global_map.get(
                    p_str
                )  # Get KeyInfo from current global map for the path
                if ki:  # Path exists in current global map
                    if (
                        ki.key_string == k_str
                    ):  # Key string in file matches current global key for that path
                        temp_ki_list.append(ki)
                    else:  # Path exists, but its global key string has changed
                        logger.warning(
                            f"Merge Parse {os.path.basename(file_path)}: Path '{p_str}' has key '{k_str}' in file, but global key is '{ki.key_string}'. Using global KeyInfo."
                        )
                        temp_ki_list.append(ki)  # Add the current KeyInfo for the path
                else:  # Path from tracker definition not in current global map
                    logger.warning(
                        f"Merge Parse {os.path.basename(file_path)}: Path '{p_str}' (key '{k_str}') not found in current global map. Skipping this definition."
                    )
            parsed_data["key_info_list"] = (
                temp_ki_list  # List of current KeyInfo objects for paths defined in tracker
            )

            grid_headers, grid_rows_tuples = read_grid_from_lines(
                lines
            )  # grid_rows_tuples: List[(label_str, comp_data_str)]
            parsed_data["grid_header_keys"] = grid_headers  # Key strings from X line

            # Align grid_rows_compressed with key_info_list. This is crucial.
            # The grid rows must correspond to the paths (and thus KeyInfo objects) in key_info_list.
            # If key_info_list was filtered (e.g. path removed from global map), grid must reflect this.
            aligned_grid_rows_comp: List[str] = []
            if len(temp_ki_list) == len(grid_rows_tuples):  # Ideal case: counts match
                # Further check if key strings also match, assuming order is consistent
                all_match = True
                for i in range(len(temp_ki_list)):
                    # Ensure the display key from _get_display_key_for_tracker matches the grid row label
                    # This requires global_map and global_key_counts for _get_display_key_for_tracker
                    # For simplicity here, assume the key_string from KeyInfo is what's in the file if it's an older file
                    # or the full display key if it's a newer one. The read_grid_from_lines already gives KEY#GI or KEY.
                    if (
                        temp_ki_list[i].key_string != grid_rows_tuples[i][0]
                        and get_key_global_instance_string(temp_ki_list[i], global_map)
                        != grid_rows_tuples[i][0]
                    ):  # Check both base and GI
                        all_match = False
                        break
                if all_match:
                    aligned_grid_rows_comp = [tpl[1] for tpl in grid_rows_tuples]
                else:
                    logger.warning(
                        f"Merge Parse {os.path.basename(file_path)}: Key string labels in grid rows do not match order of resolved definitions. Grid data might be misaligned."
                    )
                    # Fallback: use as is, but this is risky. Better to try to realign if possible, or discard grid.
                    # For now, if labels mismatch but counts are same, take the data.
                    aligned_grid_rows_comp = [tpl[1] for tpl in grid_rows_tuples]
            elif grid_rows_tuples:  # Counts mismatch
                logger.warning(
                    f"Merge Parse {os.path.basename(file_path)}: Grid row count ({len(grid_rows_tuples)}) != resolved definition count ({len(temp_ki_list)}). Grid data likely misaligned/incomplete."
                )
                # Try to salvage if header count matches definitions, otherwise grid is unusable.
                if len(grid_headers) == len(temp_ki_list) and len(
                    grid_rows_tuples
                ) >= len(temp_ki_list):
                    aligned_grid_rows_comp = [
                        grid_rows_tuples[i][1] for i in range(len(temp_ki_list))
                    ]
                elif len(grid_headers) == len(
                    grid_rows_tuples
                ):  # Header matches rows but not defs
                    aligned_grid_rows_comp = [
                        tpl[1] for tpl in grid_rows_tuples
                    ]  # Use all rows, but header might be wrong for merged_grid
                # Else, aligned_grid_rows_comp remains empty, meaning grid is effectively discarded.
            parsed_data["grid_rows_compressed"] = aligned_grid_rows_comp

            # Metadata reading (can use read_tracker_file_raw_dict for this simple part if it's reliable)
            raw_meta = read_tracker_file_structured(
                file_path
            )  # Assumes this util gives the old dict format for metadata
            parsed_data["last_key_edit"] = raw_meta.get("last_key_edit", "")
            parsed_data["last_grid_edit"] = raw_meta.get("last_grid_edit", "")
        except Exception as e:
            logger.error(
                f"Error parsing tracker file {file_path} for merge: {e}", exc_info=True
            )
        return parsed_data

    pri_data = _parse_tracker_for_merge(
        primary_tracker_path, current_global_path_to_key_info
    )
    sec_data = _parse_tracker_for_merge(
        secondary_tracker_path, current_global_path_to_key_info
    )

    pri_ki_list: List[KeyInfo] = pri_data["key_info_list"]
    pri_grid_comp: List[str] = pri_data["grid_rows_compressed"]
    sec_ki_list: List[KeyInfo] = sec_data["key_info_list"]
    sec_grid_comp: List[str] = sec_data["grid_rows_compressed"]

    if (
        not pri_ki_list and not sec_ki_list
    ):  # Both effectively empty after filtering against global map
        logger.warning(
            "Both trackers are empty or unreadable after parsing against current global map. Cannot merge."
        )
        return None

    # Path -> KeyInfo map for merged definitions, primary overwrites secondary for same path.
    merged_key_info_objects_by_path: Dict[str, KeyInfo] = {}
    final_merged_grid_comp: List[str]
    final_merged_last_key_edit: str

    if not pri_ki_list:  # Primary is empty
        logger.info(
            f"Primary tracker {os.path.basename(primary_tracker_path)} content is empty/invalid. Using secondary tracker."
        )
        for ki_s in sec_ki_list:
            merged_key_info_objects_by_path[ki_s.norm_path] = ki_s
        final_merged_grid_comp = sec_grid_comp
        final_merged_last_key_edit = sec_data["last_key_edit"]
    elif not sec_ki_list:  # Secondary is empty
        logger.info(
            f"Secondary tracker {os.path.basename(secondary_tracker_path)} content is empty/invalid. Using primary tracker."
        )
        for ki_p in pri_ki_list:
            merged_key_info_objects_by_path[ki_p.norm_path] = ki_p
        final_merged_grid_comp = pri_grid_comp
        final_merged_last_key_edit = pri_data["last_key_edit"]
    else:  # Both have content
        logger.debug(
            f"Merging {len(pri_ki_list)} primary items and {len(sec_ki_list)} secondary items."
        )
        for ki_s in sec_ki_list:
            merged_key_info_objects_by_path[ki_s.norm_path] = ki_s
        for ki_p in pri_ki_list:
            merged_key_info_objects_by_path[ki_p.norm_path] = (
                ki_p  # Primary path defs overwrite
            )

        # The final sorted list of KeyInfo for the merged tracker's structure
        temp_merged_ki_list_sorted = sorted(
            list(merged_key_info_objects_by_path.values()),
            key=lambda ki_lambda: (
                (
                    get_sortable_parts_for_key(ki_lambda.key_string)
                    if ki_lambda.key_string
                    else []
                ),
                ki_lambda.norm_path,
            ),
        )
        # Ensure input grids align with their KI lists before merging
        # This check should ideally be inside _parse_tracker_for_merge or _merge_grids
        if len(pri_grid_comp) != len(pri_ki_list):
            logger.warning(
                f"Primary grid for merge has {len(pri_grid_comp)} rows, but {len(pri_ki_list)} defs. Merge may be flawed."
            )
            # --- MODIFIED CALL for Error 2 fix ---
            initial_grid_dict_pri_pad = create_initial_grid(
                pri_ki_list
            )  # Pass List[KeyInfo]
            pri_grid_comp_pad_rows = [
                initial_grid_dict_pri_pad[ki.key_string] for ki in pri_ki_list
            ]
            # Ensure pri_grid_comp has the correct number of rows by padding or truncating
            if len(pri_grid_comp) < len(pri_ki_list):
                pri_grid_comp.extend(pri_grid_comp_pad_rows[len(pri_grid_comp) :])
            elif len(pri_grid_comp) > len(pri_ki_list):
                pri_grid_comp = pri_grid_comp[: len(pri_ki_list)]
            # --- END OF MODIFICATION ---

        if len(sec_grid_comp) != len(sec_ki_list):
            logger.warning(
                f"Secondary grid for merge has {len(sec_grid_comp)} rows, but {len(sec_ki_list)} defs. Merge may be flawed."
            )
            # --- MODIFIED CALL for Error 2 fix ---
            initial_grid_dict_sec_pad = create_initial_grid(
                sec_ki_list
            )  # Pass List[KeyInfo]
            sec_grid_comp_pad_rows = [
                initial_grid_dict_sec_pad[ki.key_string] for ki in sec_ki_list
            ]
            # Ensure sec_grid_comp has the correct number of rows
            if len(sec_grid_comp) < len(sec_ki_list):
                sec_grid_comp.extend(sec_grid_comp_pad_rows[len(sec_grid_comp) :])
            elif len(sec_grid_comp) > len(sec_ki_list):
                sec_grid_comp = sec_grid_comp[: len(sec_ki_list)]

        final_merged_grid_comp = _merge_grids(
            pri_grid_comp,
            sec_grid_comp,
            pri_ki_list,
            sec_ki_list,  # These are the KI lists corresponding to the grid data passed
            temp_merged_ki_list_sorted,  # This is the structure of the output grid
        )
        final_merged_last_key_edit = (
            pri_data["last_key_edit"] or sec_data["last_key_edit"]
        )

    final_merged_key_info_list_for_write = sorted(
        list(merged_key_info_objects_by_path.values()),
        key=lambda ki_lambda_2: (
            (
                get_sortable_parts_for_key(ki_lambda_2.key_string)
                if ki_lambda_2.key_string
                else []
            ),
            ki_lambda_2.norm_path,
        ),
    )

    final_merged_last_grid_edit = f"Merged from {os.path.basename(primary_tracker_path)} and {os.path.basename(secondary_tracker_path)} on {datetime.datetime.now().isoformat()}"

    # Backup target file before writing merged content
    if os.path.exists(output_path):
        backup_tracker_file(output_path)
        logger.info(
            f"Backed up target file before merge: {os.path.basename(output_path)}"
        )

    # write_tracker_file expects current_global_map as its last arg now.
    if write_tracker_file(
        output_path,
        final_merged_key_info_list_for_write,
        final_merged_grid_comp,  # This must be List[str] of compressed rows for merged_ki_list_sorted
        final_merged_last_key_edit,
        final_merged_last_grid_edit,
        current_global_path_to_key_info,
    ):  # Pass current_global_path_to_key_info
        logger.info(f"Successfully merged trackers into: {output_path}")
        # Invalidate relevant caches
        invalidate_dependent_entries("tracker_data", f"tracker_data:{output_path}:.*")
        if output_path == primary_tracker_path:
            invalidate_dependent_entries(
                "tracker_data", f"tracker_data:{primary_tracker_path}:.*"
            )
        if output_path == secondary_tracker_path:
            invalidate_dependent_entries(
                "tracker_data", f"tracker_data:{secondary_tracker_path}:.*"
            )
        invalidate_dependent_entries("grid_decompress", ".*")
        invalidate_dependent_entries("grid_validation", ".*")
        invalidate_dependent_entries("grid_dependencies", ".*")
        # Return data in the new format if needed by caller, or just status
        # For now, return a dict that might be useful, mirroring roughly old `merged_data` but with new structures
        return {
            "key_info_list": final_merged_key_info_list_for_write,
            "grid_rows_compressed": final_merged_grid_comp,
            "last_key_edit": final_merged_last_key_edit,
            "last_grid_edit": final_merged_last_grid_edit,
        }
    else:
        logger.error(f"Failed to write merged tracker to: {output_path}")
        return None


# --- _is_file_key ---
def _is_file_key(key_string: str) -> bool:
    if not key_string:
        return False
    return bool(re.search(r"\d+$", key_string))


# --- Mini Tracker Specific Functions ---
def get_mini_tracker_path(module_path: str) -> str:
    norm_module_path = normalize_path(module_path)
    mini_data_config = get_mini_tracker_data()  # From config/defaults
    if "get_tracker_path" in mini_data_config and callable(
        mini_data_config["get_tracker_path"]
    ):
        return normalize_path(
            str(mini_data_config["get_tracker_path"](norm_module_path))
        )
    else:  # Fallback standard naming convention
        module_name = os.path.basename(norm_module_path)
        raw_path = os.path.join(norm_module_path, f"{module_name}_module.md")
        return normalize_path(raw_path)


def create_mini_tracker(
    module_path: str,
    path_to_key_info_global: Dict[str, KeyInfo],
    key_info_list_for_grid: List[
        KeyInfo
    ],  # This is the List[KeyInfo] for the mini-tracker's structure
    new_key_strings_for_this_tracker: Optional[List[str]] = None,
):
    mini_tracker_config = get_mini_tracker_data()
    template_content = mini_tracker_config["template"]
    marker_start, marker_end = mini_tracker_config["markers"]

    norm_module_path = normalize_path(module_path)
    module_name_for_template = os.path.basename(norm_module_path)
    output_file = get_mini_tracker_path(norm_module_path)

    # Precompute global key counts for writing
    global_key_counts = defaultdict(int)
    for ki_global in path_to_key_info_global.values():
        global_key_counts[ki_global.key_string] += 1

    # Call updated create_initial_grid from dependency_grid.py
    # It now expects List[KeyInfo] and returns Dict[str, str] (keyed by ki.key_string)
    initial_grid_dict_mini = create_initial_grid(key_info_list_for_grid)

    # Convert dict to ordered list of compressed rows, matching key_info_list_for_grid order
    # The keys for the dict are ki.key_string, which is what we need.
    initial_grid_compressed_rows = [
        initial_grid_dict_mini[ki.key_string] for ki in key_info_list_for_grid
    ]

    try:
        dirname = os.path.dirname(output_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(output_file, "w", encoding="utf-8", newline="\n") as f:
            # Write template content, potentially formatted
            try:
                f.write(template_content.format(module_name=module_name_for_template))
            except KeyError:
                f.write(template_content)

            # Ensure start marker is present if not in template
            if marker_start not in template_content:
                if not template_content.endswith(
                    "\n\n"
                ) and not template_content.endswith("\n"):
                    f.write("\n")  # Ensure separation
                if not template_content.endswith("\n"):
                    f.write("\n")
                f.write(f"{marker_start}\n")

            f.write(
                "\n"
            )  # Blank line after marker or template content, before definitions

            _write_key_definitions_section(
                f, key_info_list_for_grid, path_to_key_info_global, global_key_counts
            )
            f.write("\n")

            # Determine module's own key string label for metadata message
            module_own_key_label = ""
            found_module_as_ki = next(
                (
                    ki
                    for ki in key_info_list_for_grid
                    if ki.norm_path == norm_module_path
                ),
                None,
            )
            if found_module_as_ki:
                module_own_key_label = _get_display_key_for_tracker(
                    found_module_as_ki, path_to_key_info_global, global_key_counts
                )

            # Use get_sortable_parts_for_key for correct sorting of keys in metadata message
            sorted_new_keys_for_msg = sorted(
                new_key_strings_for_this_tracker or [], key=get_sortable_parts_for_key
            )

            last_key_edit_message = (
                f"Assigned keys: {', '.join(sorted_new_keys_for_msg)}"
                if sorted_new_keys_for_msg
                else (
                    f"Initial key: {module_own_key_label}"
                    if module_own_key_label
                    else "Initial creation"
                )
            )
            f.write(f"last_KEY_edit: {last_key_edit_message}\n")
            f.write(f"last_GRID_edit: Initial creation\n\n")

            _write_grid_section(
                f,
                key_info_list_for_grid,
                initial_grid_compressed_rows,
                path_to_key_info_global,
                global_key_counts,
            )
            f.write("\n")

            # Ensure end marker is present if not in template
            if marker_end not in template_content:
                f.write(f"{marker_end}\n")

        logger.info(f"Created new mini tracker: {output_file}")
        return True
    except IOError as e_io:
        logger.error(
            f"I/O Error creating mini tracker {output_file}: {e_io}", exc_info=True
        )
        return False
    except Exception as e_exc:
        logger.exception(
            f"Unexpected error creating mini tracker {output_file}: {e_exc}"
        )
        return False


# --- Helper for Import Relationships: Reads a specific cell from another tracker ---
# (Defined here for use by update_tracker's import logic; non-cached version)
def _get_char_from_specific_tracker(
    source_path_lookup: str,
    target_path_lookup: str,
    tracker_file_to_read: str,
    global_map_for_context: Dict[str, KeyInfo],  # Current global map
) -> Optional[str]:
    if not os.path.exists(tracker_file_to_read):
        return None
    try:
        # Use read_tracker_file_structured to get its ordered definitions and grid
        data = read_tracker_file_structured(tracker_file_to_read)
        if (
            not data
            or not data.get("definitions_ordered")
            or not data.get("grid_rows_ordered")
        ):
            return None

        defs_ordered_in_other_tracker = data[
            "definitions_ordered"
        ]  # List[Tuple[key_str, path_str]]
        grid_rows_ordered_in_other_tracker = data[
            "grid_rows_ordered"
        ]  # List[Tuple[label, compressed_row]]

        # Create path -> index mapping for the definitions in the tracker_file_to_read
        path_to_idx_in_other_tracker: Dict[str, int] = {}
        for i, (key_str, path_str) in enumerate(defs_ordered_in_other_tracker):
            if (
                path_str not in path_to_idx_in_other_tracker
            ):  # First occurrence if path duplicated
                path_to_idx_in_other_tracker[path_str] = i

        source_idx_in_other = path_to_idx_in_other_tracker.get(source_path_lookup)
        target_idx_in_other = path_to_idx_in_other_tracker.get(target_path_lookup)

        if source_idx_in_other is not None and target_idx_in_other is not None:
            if source_idx_in_other < len(grid_rows_ordered_in_other_tracker):
                _row_label, compressed_row = grid_rows_ordered_in_other_tracker[
                    source_idx_in_other
                ]
                # Check consistency: row label from grid should match key from definition
                if defs_ordered_in_other_tracker[source_idx_in_other][0] != _row_label:
                    logger.warning(
                        f"HomeTrackerRead: Key mismatch for row {source_idx_in_other} in {os.path.basename(tracker_file_to_read)}. Def key: {defs_ordered_in_other_tracker[source_idx_in_other][0]}, Grid label: {_row_label}."
                    )
                    # Proceed cautiously or return None if strict consistency is required

                decomp_row_chars = decompress(compressed_row)
                if len(decomp_row_chars) == len(
                    defs_ordered_in_other_tracker
                ):  # Row length must match total defs
                    if target_idx_in_other < len(decomp_row_chars):
                        return decomp_row_chars[target_idx_in_other]
    except Exception as e_read_home:
        logger.debug(
            f"HomeTrackerRead: Error reading/parsing {os.path.basename(tracker_file_to_read)} for char lookup: {e_read_home}",
            exc_info=False,
        )
    return None


def _load_ast_verified_links() -> List[Dict[str, str]]:
    """
    Loads the ast_verified_links.json file from the core directory.
    Returns a list of AST-verified link dictionaries, or an empty list on error.
    """
    if _CORE_DIR_FOR_AST_LINKS is None:
        logger.error(
            "TrackerIO: _CORE_DIR_FOR_AST_LINKS not set. Cannot load AST verified links."
        )
        return []

    ast_links_path = normalize_path(
        os.path.join(_CORE_DIR_FOR_AST_LINKS, AST_VERIFIED_LINKS_FILENAME)
    )

    if not os.path.exists(ast_links_path):
        logger.info(
            f"AST verified links file not found at {ast_links_path}. No AST overrides will be applied from file."
        )
        return []

    try:
        with open(ast_links_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error(
                f"AST verified links file {ast_links_path} does not contain a list. Format error."
            )
            return []
        # Normalize and deduplicate by (source_path, target_path, char)
        normed: List[Dict[str, str]] = []
        seen: Set[Tuple[str, str, str]] = set()
        for entry in data:
            if not isinstance(entry, dict):
                continue
            sp = normalize_path(entry.get("source_path", "") or "")
            tp = normalize_path(entry.get("target_path", "") or "")
            ch = (entry.get("char", "") or "").strip()
            if not sp or not tp or not ch:
                continue
            key = (sp, tp, ch)
            if key in seen:
                continue
            seen.add(key)
            # keep the same shape but with normalized paths
            new_e = dict(entry)
            new_e["source_path"] = sp
            new_e["target_path"] = tp
            new_e["char"] = ch
            normed.append(new_e)
        logger.debug(
            f"Successfully loaded {len(normed)} AST-verified links from {ast_links_path} (deduplicated)."
        )
        return normed
    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding JSON from AST verified links file {ast_links_path}: {e}"
        )
        return []  # Corrected from pass to return []
    except Exception as e:  # Catch other potential errors like IOError
        logger.error(
            f"Error loading AST verified links file {ast_links_path}: {e}",
            exc_info=True,
        )
        return []


# --- End of Helper ---


def update_tracker(
    output_file_suggestion: str,
    path_to_key_info: Dict[str, KeyInfo],
    tracker_type: str = "main",
    suggestions_external: Optional[
        Dict[str, List[Tuple[str, str]]]
    ] = None,  # ALWAYS KEY#global_instance
    file_to_module: Optional[Dict[str, str]] = None,
    new_keys: Optional[List[KeyInfo]] = None,
    force_apply_suggestions: bool = False,
    keys_to_explicitly_remove: Optional[Set[str]] = None,
    use_old_map_for_migration: bool = True,
    apply_ast_overrides: bool = True,
    return_update: bool = False,
) -> Optional[
    Dict[str, Any]
]:  # Returns Dict if return_update is True, else writes directly.
    # --- AST Override for Doc Tracker ---
    if tracker_type == "doc":
        apply_ast_overrides = False
        logger.debug("AST overrides are disabled for doc_tracker updates.")
    """
    Updates or creates a tracker file based on type using contextual keys.
    Invalidates cache on changes.
    Performs path stability checks before migrating grid data.
    Calls tracker-specific logic for filtering, aggregation (main), and path determination.
    Uses hierarchical sorting for key strings.
    """
    logger.debug(
        f"--- update_tracker CALLED --- Suggestion: '{output_file_suggestion}', Type: '{tracker_type}', ForceSugg: {force_apply_suggestions}, ApplyAST: {apply_ast_overrides}"
    )
    # --- Initialize counters and flags ---
    project_root = get_project_root()
    config = ConfigManager()
    get_priority = config.get_char_priority
    min_positive_priority = max(2, get_priority("s"))
    abs_doc_roots_set = {
        normalize_path(os.path.join(project_root, p))
        for p in config.get_doc_directories()
    }

    # Shared helper used by consolidation and pruning: determines whether a path
    # is inside any configured documentation root. Declared early so nested
    # functions (like key_in_doc_root) can safely reference it during consolidation.
    def is_path_in_doc_roots(path: str, doc_roots_set: Set[str]) -> bool:
        norm_item_p = normalize_path(path)
        return any(
            is_subpath(norm_item_p, dr) or norm_item_p == dr for dr in doc_roots_set
        )

    # Mini-tracker import counters/flags
    native_foreign_import_ct, foreign_foreign_import_ct = 0, 0
    grid_content_changed_by_imports = False
    # Suggestion flags/vars
    suggestion_applied_flag = False
    applied_manual_source_path: Optional[str] = None
    applied_manual_target_paths: List[Tuple[str, str]] = []
    applied_manual_dep_type: Optional[str] = None
    consolidation_changes_ct = 0
    # Structural dependency counters/flags
    structural_deps_applied_count = 0
    grid_content_changed_by_structural = False
    # Overall metadata flags
    grid_structure_changed_flag = False  # Tracks if items added/removed or pruned
    output_file: str = ""
    module_path_for_mini: str = ""
    relevant_key_infos_for_type: List[KeyInfo] = []
    suggestions_to_process_for_this_tracker: Dict[str, List[Tuple[str, str]]] = (
        defaultdict(list)
    )
    any_forced_suggestion_processed_for_metadata = False  # For Stage 1
    ast_overrides_applied_count = 0
    # Precompute global key counts for _get_display_key_for_tracker if used for metadata messages
    global_key_counts_for_update_tracker = defaultdict(int)
    for ki_global_counts in path_to_key_info.values():
        global_key_counts_for_update_tracker[ki_global_counts.key_string] += 1

    # --- 1. Type-Specific Logic Block ---
    if tracker_type == "main":
        output_file = main_tracker_data["get_tracker_path"](project_root)
        filtered_modules_map: Dict[str, KeyInfo] = main_tracker_data["key_filter"](
            project_root, path_to_key_info
        )
        relevant_key_infos_for_type = list(filtered_modules_map.values())

        relevant_gis_for_this_tracker = {
            gis
            for ki in relevant_key_infos_for_type
            if (gis := get_key_global_instance_string(ki, path_to_key_info)) is not None
        }

        processed_suggestions_for_main: Dict[str, List[Tuple[str, str]]] = defaultdict(
            list
        )
        if (
            suggestions_external
        ):  # Already KEY#global_instance from project_analyzer or CLI
            for src_gi_str, deps_gi_list in suggestions_external.items():
                if src_gi_str in relevant_gis_for_this_tracker:
                    valid_targets = [
                        (tgt_gi_str, char)
                        for tgt_gi_str, char in deps_gi_list
                        if tgt_gi_str in relevant_gis_for_this_tracker
                    ]
                    if valid_targets:
                        processed_suggestions_for_main[src_gi_str].extend(valid_targets)

        # If not forcing and no (or few) external suggestions were relevant, main tracker performs its own aggregation.
        # The results of aggregation are then merged with any already processed external suggestions.
        if not force_apply_suggestions and (
            not suggestions_external or not processed_suggestions_for_main
        ):
            logger.debug(
                f"Main Tracker '{os.path.basename(output_file)}': Performing internal aggregation (either no external suggestions or they were not relevant)."
            )
            try:
                aggregated_path_deps = main_tracker_data["dependency_aggregation"](
                    project_root, path_to_key_info, filtered_modules_map, file_to_module
                )
                for src_path_agg, tgt_deps_agg_list in aggregated_path_deps.items():
                    src_ki_agg = path_to_key_info.get(src_path_agg)
                    if not src_ki_agg:
                        continue
                    src_key_gi_str = get_key_global_instance_string(
                        src_ki_agg, path_to_key_info
                    )
                    if (
                        not src_key_gi_str
                        or src_key_gi_str not in relevant_gis_for_this_tracker
                    ):
                        continue

                    for tgt_path_agg, char_agg in tgt_deps_agg_list:
                        tgt_ki_agg = path_to_key_info.get(tgt_path_agg)
                        if not tgt_ki_agg:
                            continue
                        tgt_key_gi_str = get_key_global_instance_string(
                            tgt_ki_agg, path_to_key_info
                        )
                        if (
                            not tgt_key_gi_str
                            or tgt_key_gi_str not in relevant_gis_for_this_tracker
                        ):
                            continue

                        # Add aggregated link if not already present from suggestions_external
                        if (
                            tgt_key_gi_str,
                            char_agg,
                        ) not in processed_suggestions_for_main.get(src_key_gi_str, []):
                            processed_suggestions_for_main[src_key_gi_str].append(
                                (tgt_key_gi_str, char_agg)
                            )
            except Exception as e_agg_main:
                logger.error(
                    f"Main Tracker: Aggregation or conversion to global instances failed: {e_agg_main}",
                    exc_info=True,
                )
        suggestions_to_process_for_this_tracker = processed_suggestions_for_main
        logger.info(
            f"Main Tracker: Populated suggestions_to_process (sources: {len(suggestions_to_process_for_this_tracker)}, total links: {sum(len(v) for v in suggestions_to_process_for_this_tracker.values())})"
        )

    elif tracker_type == "doc":
        output_file = doc_tracker_data["get_tracker_path"](project_root)
        filtered_items_map: Dict[str, KeyInfo] = doc_tracker_data["file_inclusion"](
            project_root, path_to_key_info
        )
        relevant_key_infos_for_type = list(filtered_items_map.values())
        if suggestions_external:  # Already KEY#global_instance
            relevant_gis_for_this_tracker = {
                get_key_global_instance_string(ki, path_to_key_info)
                for ki in relevant_key_infos_for_type
                if get_key_global_instance_string(ki, path_to_key_info)
            }
            for src_gi_str, deps_gi_list in suggestions_external.items():
                if src_gi_str in relevant_gis_for_this_tracker:
                    valid_targets = [
                        (tgt_gi_str, char)
                        for tgt_gi_str, char in deps_gi_list
                        if tgt_gi_str in relevant_gis_for_this_tracker
                    ]
                    if valid_targets:
                        suggestions_to_process_for_this_tracker[src_gi_str].extend(
                            valid_targets
                        )
        logger.info(
            f"Doc Tracker: Populated suggestions_to_process (count: {sum(len(v) for v in suggestions_to_process_for_this_tracker.values())})"
        )
    elif tracker_type == "mini":
        # --- Derive module_path_for_mini and output_file ---
        current_path_check_input = normalize_path(output_file_suggestion)
        module_ki_obj_for_path: Optional[KeyInfo] = None
        if os.path.isdir(current_path_check_input):
            module_ki_obj_for_path = path_to_key_info.get(current_path_check_input)
        if not (module_ki_obj_for_path and module_ki_obj_for_path.is_directory):
            parent_dir = os.path.dirname(current_path_check_input)
            if os.path.isdir(parent_dir):
                module_ki_obj_for_path = path_to_key_info.get(parent_dir)
        if not (module_ki_obj_for_path and module_ki_obj_for_path.is_directory):
            grandparent_dir = os.path.dirname(os.path.dirname(current_path_check_input))
            if os.path.isdir(grandparent_dir):
                module_ki_obj_for_path = path_to_key_info.get(grandparent_dir)
        if not module_ki_obj_for_path or not module_ki_obj_for_path.is_directory:
            logger.error(
                f"Mini Update Critical: No module dir from '{output_file_suggestion}'."
            )
            return None
        module_path_for_mini = module_ki_obj_for_path.norm_path
        if not module_path_for_mini:
            logger.error(
                f"Mini Update Critical: Module KI '{module_ki_obj_for_path.key_string}' empty norm_path."
            )
            return None
        try:
            output_file = get_mini_tracker_path(module_path_for_mini)
        except ValueError as ve:
            logger.error(
                f"Mini Update Critical: get_mini_tracker_path for '{module_path_for_mini}' fail: {ve}."
            )
            return None
        logger.debug(
            f"Mini Tracker Update Cycle: Module='{module_path_for_mini}', File='{output_file}' (Key: {module_ki_obj_for_path.key_string})"
        )

        # --- Mini-Tracker Relevance Logic (Populates relevant_key_infos_for_type) ---
        internal_kis_mini = [
            ki
            for ki in path_to_key_info.values()
            if ki.norm_path == module_path_for_mini
            or ki.parent_path == module_path_for_mini
        ]
        current_rel_paths_set: Set[str] = {ki.norm_path for ki in internal_kis_mini}
        logger.debug(
            f"  Mini '{os.path.basename(output_file)}': Initial internal paths ({len(current_rel_paths_set)})."
        )
        _existing_defs_pairs: List[Tuple[str, str]] = []
        _existing_grid_rows_tuples: List[Tuple[str, str]] = []
        _existing_grid_headers: List[str] = []
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f_mini_read:
                    _lines = f_mini_read.readlines()
                _existing_defs_pairs = read_key_definitions_from_lines(_lines)
                _existing_grid_headers, _existing_grid_rows_tuples = (
                    read_grid_from_lines(_lines)
                )
                if not (
                    len(_existing_defs_pairs) == len(_existing_grid_headers)
                    and len(_existing_defs_pairs) == len(_existing_grid_rows_tuples)
                    and all(
                        _existing_defs_pairs[i][0] == _existing_grid_rows_tuples[i][0]
                        for i in range(len(_existing_defs_pairs))
                    )
                ):
                    logger.warning(
                        f"  Mini Scan: Inconsistent structure in existing tracker '{os.path.basename(output_file)}'. Grid scan for persistence might be unreliable or skipped."
                    )
                # _existing_grid_rows_tuples = []
            except Exception as e_mini_read:
                logger.error(
                    f"  Mini Scan: Error reading existing '{os.path.basename(output_file)}': {e_mini_read}"
                )
                _existing_grid_rows_tuples = []
        if _existing_grid_rows_tuples and _existing_defs_pairs:
            logger.debug(
                f"  Mini Scan '{os.path.basename(output_file)}': Applying persistence rule: foreign FILE if linked from/to internal by non-'p/o/n' char."
            )
            for r_idx_f, (r_lbl_f, comp_row_f) in enumerate(_existing_grid_rows_tuples):
                if r_idx_f >= len(_existing_defs_pairs):
                    break
                row_path_f = _existing_defs_pairs[r_idx_f][1]
                row_ki_g_f = path_to_key_info.get(row_path_f)
                if not row_ki_g_f:
                    continue
                row_is_int_f = (
                    row_ki_g_f.norm_path == module_path_for_mini
                    or row_ki_g_f.parent_path == module_path_for_mini
                )
                try:
                    decomp_r_f = decompress(comp_row_f)
                    if len(decomp_r_f) != len(_existing_defs_pairs):
                        continue
                    for c_idx_f, dep_c_f in enumerate(decomp_r_f):
                        if dep_c_f in (
                            "n",
                            PLACEHOLDER_CHAR,
                            DIAGONAL_CHAR,
                            EMPTY_CHAR,
                        ):
                            continue
                        if c_idx_f >= len(_existing_defs_pairs):
                            break
                        col_path_f = _existing_defs_pairs[c_idx_f][1]
                        col_ki_g_f = path_to_key_info.get(col_path_f)
                        if not col_ki_g_f:
                            continue
                        col_is_int_f = (
                            col_ki_g_f.norm_path == module_path_for_mini
                            or col_ki_g_f.parent_path == module_path_for_mini
                        )
                        if not row_ki_g_f.is_directory and not col_ki_g_f.is_directory:
                            if row_is_int_f and not col_is_int_f:
                                current_rel_paths_set.add(col_path_f)
                            elif not row_is_int_f and col_is_int_f:
                                current_rel_paths_set.add(row_path_f)
                except Exception:
                    pass
        if suggestions_external:  # suggestions_external is KEY#global_instance
            excluded_abs_set = {
                normalize_path(os.path.join(project_root, p))
                for p in config.get_excluded_dirs()
            }.union(set(config.get_excluded_paths()))
            for src_gi_str, deps_gi_list in suggestions_external.items():
                src_ki_sugg = resolve_key_global_instance_to_ki(
                    src_gi_str, path_to_key_info
                )
                if not src_ki_sugg or src_ki_sugg.norm_path in excluded_abs_set:
                    continue
                src_is_internal_sugg = (
                    src_ki_sugg.norm_path == module_path_for_mini
                    or src_ki_sugg.parent_path == module_path_for_mini
                )
                for tgt_gi_str, dep_char_sugg in deps_gi_list:
                    if get_priority(dep_char_sugg) < min_positive_priority:
                        continue
                    tgt_ki_sugg = resolve_key_global_instance_to_ki(
                        tgt_gi_str, path_to_key_info
                    )
                    if not tgt_ki_sugg or tgt_ki_sugg.norm_path in excluded_abs_set:
                        continue
                    tgt_is_internal_sugg = (
                        tgt_ki_sugg.norm_path == module_path_for_mini
                        or tgt_ki_sugg.parent_path == module_path_for_mini
                    )
                    if not src_ki_sugg.is_directory and not tgt_ki_sugg.is_directory:
                        if src_is_internal_sugg and not tgt_is_internal_sugg:
                            current_rel_paths_set.add(tgt_ki_sugg.norm_path)
                        elif not src_is_internal_sugg and tgt_is_internal_sugg:
                            current_rel_paths_set.add(src_ki_sugg.norm_path)
        if keys_to_explicitly_remove:
            paths_to_remove = {
                ki.norm_path
                for ki in path_to_key_info.values()
                if ki.key_string in keys_to_explicitly_remove
            }
            current_rel_paths_set -= paths_to_remove
            logger.debug(
                f"  Mini '{os.path.basename(output_file)}': Paths after explicit removals ({len(current_rel_paths_set)})."
            )
        relevant_key_infos_for_type = [
            path_to_key_info[p] for p in current_rel_paths_set if p in path_to_key_info
        ]
        if (
            suggestions_external
        ):  # Filter already KEY#global_instance suggestions_external
            relevant_gis_mini = {
                get_key_global_instance_string(ki, path_to_key_info)
                for ki in relevant_key_infos_for_type
                if get_key_global_instance_string(ki, path_to_key_info)
            }
            for src_gi, deps_gi in suggestions_external.items():
                if src_gi in relevant_gis_mini:
                    valid_deps = [
                        (tgt_gi, c)
                        for tgt_gi, c in deps_gi
                        if tgt_gi in relevant_gis_mini
                    ]
                    if valid_deps:
                        suggestions_to_process_for_this_tracker[src_gi].extend(
                            valid_deps
                        )
        if not force_apply_suggestions and suggestions_to_process_for_this_tracker:
            logger.debug(
                f"Mini Tracker '{os.path.basename(output_file)}': Clearing {len(suggestions_to_process_for_this_tracker)} non-forced suggestions."
            )
            suggestions_to_process_for_this_tracker.clear()
        logger.debug(
            f"Mini Tracker: Populated suggestions_to_process (count: {sum(len(v) for v in suggestions_to_process_for_this_tracker.values())})"
        )
    else:
        logger.error(f"CRITICAL: Unknown tracker_type '{tracker_type}'. Aborting.")
        raise ValueError(f"Unknown tracker type: {tracker_type}")
    # --- END OF TYPE-SPECIFIC LOGIC ---

    if not output_file:
        logger.critical(
            f"CRITICAL ERROR: output_file is empty after type-specific logic. Type: '{tracker_type}', Suggestion: '{output_file_suggestion}'. Aborting."
        )
        return None

    final_key_info_list = sorted(
        relevant_key_infos_for_type,
        key=lambda ki_lambda_sort: (
            (
                get_sortable_parts_for_key(ki_lambda_sort.key_string)
                if ki_lambda_sort.key_string
                else []
            ),
            ki_lambda_sort.norm_path,
        ),
    )
    if not final_key_info_list:
        logger.warning(
            f"{tracker_type.capitalize()} tracker '{os.path.basename(output_file)}' (for module: '{module_path_for_mini if tracker_type=='mini' else 'N/A'}') has 0 relevant key-path instances for its grid. May result in an empty tracker."
        )
    else:
        logger.debug(
            f"{tracker_type.capitalize()} tracker '{os.path.basename(output_file)}': Final sorted list has {len(final_key_info_list)} key-path instances for grid."
        )

    # --- Read Existing Data, Build Migration Map, Create/Rebuild Tracker, Backup ---
    existing_key_path_pairs: List[Tuple[str, str]] = (
        []
    )  # List[(key_str_in_file, path_str_in_file)]
    existing_grid_column_headers: List[str] = []  # List[key_str_in_file] from X_LINE
    existing_grid_rows_data: List[Tuple[str, str]] = (
        []
    )  # List[(row_label_key_str_in_file, compressed_data_str)]
    current_last_key_edit: str = "Unknown"
    current_last_grid_edit: str = "Unknown"
    lines_from_old_file: List[str] = []
    tracker_exists_and_is_sound = False
    output_file_basename = os.path.basename(output_file)

    if os.path.exists(output_file):
        attempt_read_from_path = output_file
        is_reading_backup = False
        for attempt_num in range(2):  # Max 2 attempts: 0 for original, 1 for backup
            if attempt_num == 1:  # This is the backup attempt
                if not config.get_recovery_setting(
                    "auto_restore_corrupt_tracker_from_backup", True
                ):
                    logger.info(
                        f"UpdateTracker: Original file '{output_file_basename}' is corrupt/inconsistent. Auto-restore from backup is disabled. Proceeding by rebuilding."
                    )
                    break  # Do not attempt backup read if auto-restore is off

                backup_dir_rel = config.get_path("backups_dir", "cline_docs/backups")
                backup_dir_abs = normalize_path(
                    os.path.join(project_root, backup_dir_rel)
                )
                backups_for_file: List[Tuple[datetime.datetime, str]] = []
                if os.path.exists(backup_dir_abs):
                    for fname in os.listdir(backup_dir_abs):
                        if fname.startswith(
                            output_file_basename + "."
                        ) and fname.endswith(".bak"):
                            match = re.search(r"\.(\d{8}_\d{6}_\d{6})\.bak$", fname)
                            if match:
                                try:
                                    backups_for_file.append(
                                        (
                                            datetime.datetime.strptime(
                                                match.group(1), "%Y%m%d_%H%M%S_%f"
                                            ),
                                            os.path.join(backup_dir_abs, fname),
                                        )
                                    )
                                except ValueError:
                                    pass

                if not backups_for_file:
                    logger.info(
                        f"UpdateTracker: Original file '{output_file_basename}' corrupt/inconsistent. No backups found to attempt restore. Proceeding by rebuilding."
                    )
                    break

                backups_for_file.sort(key=lambda x: x[0], reverse=True)  # Newest first
                attempt_read_from_path = backups_for_file[0][1]
                is_reading_backup = True
                logger.warning(
                    f"UpdateTracker: Original file '{output_file_basename}' is corrupt/inconsistent. Attempting to read from most recent backup: '{os.path.basename(attempt_read_from_path)}'."
                )
                # Reset data structures for this new read attempt
                (
                    lines_from_old_file,
                    existing_key_path_pairs,
                    existing_grid_column_headers,
                    existing_grid_rows_data,
                ) = ([], [], [], [])
                current_last_key_edit, current_last_grid_edit = (
                    "FromBackup",
                    "FromBackup",
                )

            if not os.path.exists(attempt_read_from_path):
                if is_reading_backup:
                    logger.warning(
                        f"Backup file {attempt_read_from_path} not found during restore attempt."
                    )
                # If it was the original file and it doesn't exist, this outer 'if os.path.exists(output_file):' handles it.
                # If it was a backup attempt and backup file not found, break the attempt loop.
                if attempt_num == 0 and not os.path.exists(output_file):
                    break  # Original file doesn't exist, no need for backup attempt
                continue

            try:
                with open(attempt_read_from_path, "r", encoding="utf-8") as f_read:
                    lines_from_old_file = f_read.readlines()

                temp_defs = read_key_definitions_from_lines(lines_from_old_file)
                temp_hdrs, temp_rows_tuples = read_grid_from_lines(lines_from_old_file)

                is_sound_read = True  # Assume sound
                if (
                    not temp_defs and not temp_hdrs and not temp_rows_tuples
                ):  # Empty file is considered sound for this purpose
                    pass
                elif not (
                    len(temp_defs) == len(temp_hdrs)
                    and len(temp_defs) == len(temp_rows_tuples)
                ):
                    is_sound_read = False  # Counts mismatch
                else:  # Counts match, check if definition keys match row labels in order
                    for i in range(len(temp_defs)):
                        if (
                            temp_defs[i][0] != temp_rows_tuples[i][0]
                        ):  # Compare key string from defs with row label
                            is_sound_read = False
                            break

                if is_sound_read:
                    existing_key_path_pairs = temp_defs
                    existing_grid_column_headers = temp_hdrs
                    existing_grid_rows_data = temp_rows_tuples
                    # Extract metadata
                    _last_key_edit_line = next(
                        (
                            l
                            for l in lines_from_old_file
                            if l.strip().lower().startswith("last_key_edit:")
                        ),
                        None,
                    )
                    current_last_key_edit = (
                        _last_key_edit_line.split(":", 1)[1].strip()
                        if _last_key_edit_line
                        else ("FromBackup" if is_reading_backup else "Unknown")
                    )
                    _last_grid_edit_line = next(
                        (
                            l
                            for l in lines_from_old_file
                            if l.strip().lower().startswith("last_grid_edit:")
                        ),
                        None,
                    )
                    current_last_grid_edit = (
                        _last_grid_edit_line.split(":", 1)[1].strip()
                        if _last_grid_edit_line
                        else ("FromBackup" if is_reading_backup else "Unknown")
                    )
                    tracker_exists_and_is_sound = True

                    if is_reading_backup:  # Successfully read from backup
                        logger.info(
                            f"UpdateTracker: Successfully read sound data from backup '{os.path.basename(attempt_read_from_path)}'."
                        )
                        if (
                            config.get_recovery_setting(
                                "backup_on_restore_attempt", True
                            )
                            and output_file != attempt_read_from_path
                        ):
                            backup_corrupt_original_path = backup_tracker_file(
                                output_file
                            )
                            if backup_corrupt_original_path:
                                logger.info(
                                    f"Backed up corrupt original '{output_file_basename}' to '{os.path.basename(backup_corrupt_original_path)}' before processing with backup data."
                                )
                    break  # Successful read from current attempt_read_from_path (original or backup)

                else:  # Current attempt_read_from_path is not structurally sound
                    logger.warning(
                        f"UpdateTracker: File '{os.path.basename(attempt_read_from_path)}' (attempt {attempt_num+1}) is corrupt/inconsistent."
                    )
                    if is_reading_backup:  # Backup itself is corrupt
                        logger.error(
                            f"UpdateTracker: Backup file '{os.path.basename(attempt_read_from_path)}' is also corrupt/inconsistent. Proceeding by rebuilding."
                        )
                        break  # Stop trying backups
                    # If it was the original file (attempt_num == 0), loop will continue to attempt_num == 1 (backup) if enabled.
            except Exception as e_read_upd:
                logger.error(
                    f"UpdateTracker: Failed to read/parse {os.path.basename(attempt_read_from_path)}: {e_read_upd}.",
                    exc_info=True,
                )
                if is_reading_backup:
                    break  # Stop if backup read fails catastrophically
                # If original read fails, loop will try backup if enabled.
                (
                    lines_from_old_file,
                    existing_key_path_pairs,
                    existing_grid_column_headers,
                    existing_grid_rows_data,
                ) = ([], [], [], [])
                current_last_key_edit, current_last_grid_edit = (
                    "ErrorReading",
                    "ErrorReading",
                )
        # End of read attempts loop

    try:
        path_migration_info = build_path_migration_map(
            load_old_global_key_map() if use_old_map_for_migration else None,
            path_to_key_info,
        )
    except ValueError as ve_migmap:
        logger.critical(
            f"Path Migration Map build failed due to inconsistent global maps: {ve_migmap}. Aborting update for '{output_file_basename}'."
        )
        return None
    except Exception as e_migmap_other:
        logger.critical(
            f"Unexpected error building migration map for '{output_file_basename}': {e_migmap_other}. Aborting update.",
            exc_info=True,
        )
        return None

    if not tracker_exists_and_is_sound:  # Create new or rebuild from scratch
        logger.info(
            f"Creating new tracker (or rebuilding due to unrecoverable inconsistency): {output_file}"
        )

        # Call updated create_initial_grid from dependency_grid.py
        # It now expects List[KeyInfo] (final_key_info_list here)
        # and returns Dict[str, str] (keyed by ki.key_string)
        initial_grid_dict = create_initial_grid(final_key_info_list)

        # Convert dict to ordered list of compressed rows, matching final_key_info_list order
        initial_grid_comp_rows = [
            initial_grid_dict[ki.key_string] for ki in final_key_info_list
        ]

        relevant_new_global_keys_in_this_tracker_strs: List[str] = []
        if new_keys:
            paths_in_final_tracker_set = {ki.norm_path for ki in final_key_info_list}
            # Use get_sortable_parts_for_key for sorting
            relevant_new_global_keys_in_this_tracker_strs = sorted(
                [
                    nk.key_string
                    for nk in new_keys
                    if nk.norm_path in paths_in_final_tracker_set
                ],
                key=get_sortable_parts_for_key,
            )

        last_key_edit_msg = (
            f"Assigned keys: {', '.join(relevant_new_global_keys_in_this_tracker_strs)}"
            if relevant_new_global_keys_in_this_tracker_strs
            else (
                "Initial items: " + str(len(final_key_info_list))
                if final_key_info_list
                else "Initial creation"
            )
        )

        created_ok = False
        if tracker_type == "mini":
            if not module_path_for_mini:
                logger.critical(
                    f"CRITICAL: module_path_for_mini is empty when trying to call create_mini_tracker for new/rebuild of '{output_file}'. Aborting."
                )
                return None
            # create_mini_tracker now handles its own grid creation correctly using the updated dependency_grid.create_initial_grid
            created_ok = create_mini_tracker(
                module_path_for_mini,
                path_to_key_info,
                final_key_info_list,
                relevant_new_global_keys_in_this_tracker_strs,
            )
        else:  # Main or Doc tracker
            created_ok = write_tracker_file(
                output_file,
                final_key_info_list,
                initial_grid_comp_rows,  # Pass the correctly created list of rows
                last_key_edit_msg,
                "Initial creation",
                path_to_key_info,
            )

        if not created_ok:
            logger.error(
                f"Failed to create/rebuild tracker {output_file}. Aborting update."
            )
            return None

        _temp_global_key_counts_update = defaultdict(int)
        for _ki_global_update in path_to_key_info.values():
            _temp_global_key_counts_update[_ki_global_update.key_string] += 1

        existing_key_path_pairs = [
            (
                _get_display_key_for_tracker(
                    ki, path_to_key_info, _temp_global_key_counts_update
                ),
                ki.norm_path,
            )
            for ki in final_key_info_list
        ]
        existing_grid_column_headers = [
            _get_display_key_for_tracker(
                ki, path_to_key_info, _temp_global_key_counts_update
            )
            for ki in final_key_info_list
        ]
        existing_grid_rows_data = list(
            zip(existing_grid_column_headers, initial_grid_comp_rows)
        )
        current_last_key_edit = last_key_edit_msg
        current_last_grid_edit = "Initial creation"
        lines_from_old_file = []  # Effectively a new file content-wise
        tracker_exists_and_is_sound = True  # It's now considered sound

    # Backup the tracker file before major modifications (if it existed or was just created)
    if os.path.exists(output_file) and output_file:
        backup_tracker_file(output_file)
    elif not output_file:  # Should have been caught by earlier checks
        logger.error(
            "CRITICAL: output_file is empty before backup step post-creation. Aborting update."
        )
        return None

    # --- END OF SECTION: Read Existing Data, Build Migration Map, Create/Rebuild Tracker, Backup ---

    # --- Metadata and Grid Initialization ---
    # Determine if definitions changed compared to what was read (or implied by new creation)
    old_paths_in_tracker_file = {p_str for k_str, p_str in existing_key_path_pairs}
    new_paths_in_final_list = {ki.norm_path for ki in final_key_info_list}
    # grid_structure_changed_flag is True if the set of paths in definitions changes
    grid_structure_changed_flag = old_paths_in_tracker_file != new_paths_in_final_list

    final_last_key_edit = (
        current_last_key_edit  # Start with existing or "Initial creation"
    )
    # Determine if relevant new keys (global) were added to this tracker for metadata message
    relevant_new_global_keys_in_this_tracker_strs: List[str] = []
    if new_keys:
        paths_in_final_tracker_set = {ki.norm_path for ki in final_key_info_list}
        relevant_new_global_keys_in_this_tracker_strs = sorted(
            [
                nk.key_string
                for nk in new_keys
                if nk.norm_path in paths_in_final_tracker_set
            ],
            key=get_sortable_parts_for_key,  # MODIFIED
        )
    if relevant_new_global_keys_in_this_tracker_strs:
        final_last_key_edit = (
            f"Assigned keys: {', '.join(relevant_new_global_keys_in_this_tracker_strs)}"
        )
    elif grid_structure_changed_flag:  # No new global keys, but local defs changed
        added_paths_count = len(new_paths_in_final_list - old_paths_in_tracker_file)
        removed_paths_count = len(old_paths_in_tracker_file - new_paths_in_final_list)
        change_parts_list = []
        if added_paths_count > 0:
            change_parts_list.append(f"Added {added_paths_count} items")
        if removed_paths_count > 0:
            change_parts_list.append(f"Removed {removed_paths_count} items")
        if change_parts_list:
            final_last_key_edit = f"Definitions updated: {'; '.join(change_parts_list)}"
        elif (
            final_last_key_edit == "Initial creation"
        ):  # Structure changed from empty to something, or vice versa
            final_last_key_edit = (
                "Definitions established"
                if new_paths_in_final_list
                else "Definitions cleared"
            )

    new_grid_item_count = len(final_key_info_list)
    temp_decomp_grid_rows: List[List[str]] = [
        [PLACEHOLDER_CHAR] * new_grid_item_count for _ in range(new_grid_item_count)
    ]
    for i_diag in range(new_grid_item_count):
        if i_diag < new_grid_item_count:
            temp_decomp_grid_rows[i_diag][i_diag] = DIAGONAL_CHAR

    # This map is crucial for mapping resolved global KeyInfo paths to their local index in THIS tracker's grid
    final_path_to_new_idx = {
        ki.norm_path: i for i, ki in enumerate(final_key_info_list)
    }

    # --- Copy old values (Using Path Migration Map) ---

    logger.debug(
        f"Copying old grid values for '{os.path.basename(output_file)}' (using path migration map)..."
    )
    copied_values_count_log: int = 0
    skipped_instab_log: int = 0
    skipped_filled_log: int = 0
    row_proc_err_log: int = 0

    # Check consistency of old grid data read from file
    old_grid_consistent_for_migration: bool = True
    if existing_grid_rows_data:  # If there was an old grid successfully read
        if not (
            len(existing_key_path_pairs) == len(existing_grid_column_headers)
            and len(existing_key_path_pairs) == len(existing_grid_rows_data)
        ):
            logger.warning(
                f"Old grid structure in {os.path.basename(output_file)} is inconsistent for migration. "
                + f"Defs: {len(existing_key_path_pairs)}, Header: {len(existing_grid_column_headers)}, "
                + f"Rows: {len(existing_grid_rows_data)}. Skipping old grid data migration."
            )
            old_grid_consistent_for_migration = False
        else:
            # Further check: row labels in grid_rows_data should match key_strings in definitions_ordered at same index
            for i_check in range(len(existing_key_path_pairs)):
                if (
                    existing_key_path_pairs[i_check][0]
                    != existing_grid_rows_data[i_check][0]
                ):
                    logger.warning(
                        f"Old grid structure in {os.path.basename(output_file)}: Mismatch at def/row index {i_check}. "
                        + f"Def key: '{existing_key_path_pairs[i_check][0]}', Grid row label: '{existing_grid_rows_data[i_check][0]}'. "
                        + "Skipping old grid data migration due to potential misalignment."
                    )
                    old_grid_consistent_for_migration = False
                    break
    elif (
        tracker_exists_and_is_sound
    ):  # Tracker existed and was sound, but no grid rows (e.g. newly created but empty grid)
        old_grid_consistent_for_migration = False  # Nothing to migrate from grid
    else:  # Tracker didn't exist or wasn't sound (rebuilding)
        old_grid_consistent_for_migration = False

    if old_grid_consistent_for_migration and existing_grid_rows_data:
        logger.debug(
            f"Migrating old grid values for '{os.path.basename(output_file)}': {len(existing_grid_rows_data)} old rows to process."
        )
        # Iterate using the structure of the OLD grid (existing_grid_rows_data and existing_key_path_pairs)
        for old_row_idx, (old_row_label_in_file, compressed_row_str) in enumerate(
            existing_grid_rows_data
        ):
            # Get the path for the current old row from the old tracker's definitions
            # existing_key_path_pairs is List[Tuple[key_str_in_file, path_str_in_file]]
            old_row_path_in_tracker_def = existing_key_path_pairs[old_row_idx][1]

            # Find migration info for this path using the global path_migration_info map
            migration_info_for_row_path = path_migration_info.get(
                old_row_path_in_tracker_def
            )

            if (
                not migration_info_for_row_path
                or migration_info_for_row_path[1] is None
            ):
                # Path from old tracker def is unstable or removed globally
                skipped_instab_log += 1
                # logger.debug(f"  Skip Row Migration: Path '{old_row_path_in_tracker_def}' (from old key '{old_row_label_in_file}') is unstable/removed globally.")
                continue

            new_global_key_for_row = migration_info_for_row_path[
                1
            ]  # This is the CURRENT global key for old_row_path_in_tracker_def

            # Find the KeyInfo object in the CURRENT global map (path_to_key_info) that corresponds to this new_global_key_for_row.
            # The path of this KeyInfo might have changed if the item was renamed/moved but is logically the same.
            current_row_ki = next(
                (
                    ki
                    for ki in path_to_key_info.values()
                    if ki.key_string == new_global_key_for_row
                    and ki.norm_path == old_row_path_in_tracker_def
                ),
                None,
            )
            if (
                not current_row_ki
            ):  # Path might have changed for this key, or key was reused. Find any current path for this new_global_key.
                current_row_ki = next(
                    (
                        ki
                        for ki in path_to_key_info.values()
                        if ki.key_string == new_global_key_for_row
                    ),
                    None,
                )

            if (
                not current_row_ki
            ):  # Should not happen if new_global_key_for_row is not None
                skipped_instab_log += 1
                # logger.debug(f"  Skip Row Migration: Cannot map new global key '{new_global_key_for_row}' back to a current KeyInfo object.")
                continue

            # Get the index of this row in the NEW grid structure (based on final_key_info_list)
            new_final_row_idx = final_path_to_new_idx.get(current_row_ki.norm_path)
            if new_final_row_idx is None:
                # This item (identified by its current path) is not part of the new tracker's structure.
                # logger.debug(f"  Skip Row Migration: Path '{current_row_ki.norm_path}' (new key '{new_global_key_for_row}') not in this tracker's final structure.")
                continue

            try:
                decomp_row_values = list(decompress(compressed_row_str))
                # Decompressed row length must match the number of columns in the OLD grid (i.e., number of old defs)
                if len(decomp_row_values) != len(
                    existing_key_path_pairs
                ):  # Or use len(existing_grid_column_headers) if that's more reliable
                    logger.warning(
                        f"  Grid Copy: Row for old path '{old_row_path_in_tracker_def}' (old key '{old_row_label_in_file}') has decompressed length {len(decomp_row_values)}, expected {len(existing_key_path_pairs)}. Skipping row."
                    )
                    row_proc_err_log += 1
                    continue

                for old_col_idx, value_char in enumerate(decomp_row_values):
                    if value_char in (DIAGONAL_CHAR, PLACEHOLDER_CHAR, EMPTY_CHAR):
                        continue

                    # Get the path for the current old column from the old tracker's definitions
                    old_col_path_in_tracker_def = existing_key_path_pairs[old_col_idx][
                        1
                    ]
                    migration_info_for_col_path = path_migration_info.get(
                        old_col_path_in_tracker_def
                    )

                    if (
                        not migration_info_for_col_path
                        or migration_info_for_col_path[1] is None
                    ):
                        skipped_instab_log += 1
                        # logger.debug(f"  Skip Cell Migration: Col Path '{old_col_path_in_tracker_def}' unstable/removed globally.")
                        continue
                    new_global_key_for_col = migration_info_for_col_path[1]

                    current_col_ki = next(
                        (
                            ki
                            for ki in path_to_key_info.values()
                            if ki.key_string == new_global_key_for_col
                            and ki.norm_path == old_col_path_in_tracker_def
                        ),
                        None,
                    ) or next(
                        (
                            ki
                            for ki in path_to_key_info.values()
                            if ki.key_string == new_global_key_for_col
                        ),
                        None,
                    )
                    if not current_col_ki:
                        skipped_instab_log += 1
                        # logger.debug(f"  Skip Cell Migration: Cannot map new col global key '{new_global_key_for_col}' to current KeyInfo.")
                        continue

                    new_final_col_idx = final_path_to_new_idx.get(
                        current_col_ki.norm_path
                    )
                    if new_final_col_idx is None:
                        # logger.debug(f"  Skip Cell Migration: Col Path '{current_col_ki.norm_path}' (new key '{new_global_key_for_col}') not in this tracker's final structure.")
                        continue

                    # Check bounds for temp_decomp_grid_rows (should be new_grid_item_count x new_grid_item_count)
                    if (
                        new_final_row_idx < new_grid_item_count
                        and new_final_col_idx < new_grid_item_count
                    ):
                        if (
                            temp_decomp_grid_rows[new_final_row_idx][new_final_col_idx]
                            == PLACEHOLDER_CHAR
                        ):
                            temp_decomp_grid_rows[new_final_row_idx][
                                new_final_col_idx
                            ] = value_char
                            copied_values_count_log += 1
                        else:
                            skipped_filled_log += 1
                            # logger.debug(f"  Grid Copy: Target cell ({new_global_key_for_row}, {new_global_key_for_col}) already filled with '{temp_decomp_grid_rows[new_final_row_idx][new_final_col_idx]}'. Skipped copying '{value_char}'.")
                    else:
                        logger.warning(
                            f"  Grid Copy: Index out of bounds for new grid assignment. RowIdx: {new_final_row_idx}, ColIdx: {new_final_col_idx}. Grid Size: {new_grid_item_count}. Value '{value_char}' not copied."
                        )
                        row_proc_err_log += (
                            1  # Count as an error if we can't place it due to bounds
                        )

            except Exception as e_decompress_migrate:
                logger.warning(
                    f"  Grid Copy Error during migration for row of old path '{old_row_path_in_tracker_def}': {e_decompress_migrate}"
                )
                row_proc_err_log += 1
        logger.debug(
            f"Grid migration for '{os.path.basename(output_file)}': Copied {copied_values_count_log}, Skipped(Unstable/Path Issue): {skipped_instab_log}, Skipped(Target Filled): {skipped_filled_log}, Row Errors: {row_proc_err_log}"
        )
    else:
        logger.info(
            f"Skipping old grid value migration for '{os.path.basename(output_file)}' as old grid was not sane, non-existent, or empty."
        )

    # --- Structural Dependencies (Patched) ---
    structural_deps_applied_count = 0
    grid_content_changed_by_structural = False
    if tracker_type == "doc" or tracker_type == "mini":
        logger.debug(
            f"Calculating structural dependencies for {tracker_type} tracker '{output_file_basename}'..."
        )
        structural_deps_applied_this_run = 0
        for r_idx_s, r_ki_s in enumerate(final_key_info_list):
            if not r_ki_s.is_directory:
                continue  # Structural rules often originate from directories
            for c_idx_s, c_ki_s in enumerate(final_key_info_list):
                if r_idx_s == c_idx_s:
                    continue  # Skip diagonal

                # Parent-child ('x')
                if is_subpath(
                    c_ki_s.norm_path, r_ki_s.norm_path
                ):  # c_ki is child of r_ki (directory)
                    if temp_decomp_grid_rows[r_idx_s][c_idx_s] == PLACEHOLDER_CHAR:
                        temp_decomp_grid_rows[r_idx_s][c_idx_s] = "x"
                        grid_content_changed_by_structural = True
                        structural_deps_applied_this_run += 1
                    if (
                        temp_decomp_grid_rows[c_idx_s][r_idx_s] == PLACEHOLDER_CHAR
                    ):  # Reciprocal
                        temp_decomp_grid_rows[c_idx_s][r_idx_s] = "x"
                        grid_content_changed_by_structural = True
                        structural_deps_applied_this_run += 1

                # For Doc trackers, unrelated items can be marked 'n' if they were placeholders
                # This is a broad rule and might need more nuance based on actual doc structure.
                # It should only apply if no other rule (parent-child, suggestion, etc.) has already set a value.
                elif tracker_type == "doc":
                    # Check if not parent-child (already handled) and not already set by other means
                    if not is_subpath(
                        r_ki_s.norm_path, c_ki_s.norm_path
                    ) and not is_subpath(c_ki_s.norm_path, r_ki_s.norm_path):
                        if temp_decomp_grid_rows[r_idx_s][c_idx_s] == PLACEHOLDER_CHAR:
                            temp_decomp_grid_rows[r_idx_s][c_idx_s] = "n"
                            grid_content_changed_by_structural = True
                            structural_deps_applied_this_run += 1
                        if temp_decomp_grid_rows[c_idx_s][r_idx_s] == PLACEHOLDER_CHAR:
                            temp_decomp_grid_rows[c_idx_s][r_idx_s] = "n"
                            grid_content_changed_by_structural = True
                            structural_deps_applied_this_run += 1
        if structural_deps_applied_this_run > 0:
            logger.debug(
                f"Applied {structural_deps_applied_this_run} structural dependency cells for '{output_file_basename}'."
            )

    # --- Import Established Relationships (Mini-Trackers - Path Based - Detailed) ---
    native_foreign_import_ct = 0
    foreign_foreign_import_ct = 0
    grid_content_changed_by_imports = False  # Initialize flag

    if (
        tracker_type == "mini" and module_path_for_mini
    ):  # Ensure module_path_for_mini is correctly set
        logger.debug(
            f"Mini Tracker ({os.path.basename(module_path_for_mini)}): Importing established relationships from other trackers..."
        )

        native_ki_in_current_tracker: List[KeyInfo] = []
        foreign_ki_in_current_tracker: List[KeyInfo] = []
        # `final_path_to_new_idx` maps norm_path -> index in `final_key_info_list`
        # It should be up-to-date after any pruning.

        for ki in final_key_info_list:
            if (
                ki.parent_path == module_path_for_mini
                or ki.norm_path == module_path_for_mini
            ):
                native_ki_in_current_tracker.append(ki)
            else:
                foreign_ki_in_current_tracker.append(ki)

        logger.debug(
            f"  Import: Native items: {len(native_ki_in_current_tracker)}, Foreign items: {len(foreign_ki_in_current_tracker)}"
        )

        # Helper to get relationship char from a specified home tracker file
        # This helper needs to be robust.
        # @cached(
        #     "home_tracker_rel_char",
        #     key_func=lambda p1, p2, htf: f"htrc:{p1}:{p2}:{htf}:{(os.path.getmtime(htf) if os.path.exists(htf) else 0)}",
        # )
        def get_char_from_home_tracker_cached(
            path1_norm: str, path2_norm: str, home_tracker_file_norm: str
        ) -> Optional[str]:
            if not os.path.exists(home_tracker_file_norm):
                logger.debug(
                    f"    Home tracker {home_tracker_file_norm} not found for char lookup."
                )
                return None
            try:
                with open(home_tracker_file_norm, "r", encoding="utf-8") as hf:
                    home_lines = hf.readlines()
                home_defs_pairs = read_key_definitions_from_lines(
                    home_lines
                )  # List[(key_str, path_str)]
                _home_col_hdrs, home_grid_rows_data = read_grid_from_lines(
                    home_lines
                )  # List[(row_label_str, comp_data_str)]

                # Build path -> index map for the home tracker's definitions
                home_path_to_def_idx_map: Dict[str, int] = {}
                for i, (_k_str, p_str) in enumerate(home_defs_pairs):
                    if (
                        p_str not in home_path_to_def_idx_map
                    ):  # Take first occurrence if path duplicated in defs (should not happen)
                        home_path_to_def_idx_map[p_str] = i

                idx1_in_home_defs = home_path_to_def_idx_map.get(path1_norm)
                idx2_in_home_defs = home_path_to_def_idx_map.get(path2_norm)

                if idx1_in_home_defs is not None and idx2_in_home_defs is not None:
                    # Ensure the row we are looking for (idx1_in_home_defs) exists in the grid data read
                    # The grid rows are ordered as per definitions.
                    if idx1_in_home_defs < len(home_grid_rows_data):
                        # The label of the grid row should match the key_string from definitions if consistent
                        # home_grid_rows_data[idx1_in_home_defs] is (row_label_str, compressed_data_str)
                        _row_label, compressed_row = home_grid_rows_data[
                            idx1_in_home_defs
                        ]
                        # Optional: Check if _row_label matches home_defs_pairs[idx1_in_home_defs][0]

                        decomp_row_chars = decompress(compressed_row)

                        # The decompressed row length must match the number of items in home_defs_pairs
                        if len(decomp_row_chars) != len(home_defs_pairs):
                            logger.warning(
                                f"    Home tracker {os.path.basename(home_tracker_file_norm)}: Row for path '{path1_norm}' (def idx {idx1_in_home_defs}) has length {len(decomp_row_chars)}, expected {len(home_defs_pairs)}. Cannot get char."
                            )
                            return None

                        if idx2_in_home_defs < len(decomp_row_chars):
                            found_char = decomp_row_chars[idx2_in_home_defs]
                            logger.debug(
                                f"    Home tracker {os.path.basename(home_tracker_file_norm)}: Found '{found_char}' for {path1_norm} -> {path2_norm}"
                            )
                            return found_char
                        else:
                            logger.debug(
                                f"    Home tracker {os.path.basename(home_tracker_file_norm)}: Target path index {idx2_in_home_defs} out of bounds for decompressed row of '{path1_norm}'."
                            )
                    else:
                        logger.debug(
                            f"    Home tracker {os.path.basename(home_tracker_file_norm)}: Source path index {idx1_in_home_defs} out of bounds for grid data rows."
                        )
                else:
                    logger.debug(
                        f"    Home tracker {os.path.basename(home_tracker_file_norm)}: Path(s) not in definitions: p1({path1_norm})-{idx1_in_home_defs is not None}, p2({path2_norm})-{idx2_in_home_defs is not None}."
                    )

            except Exception as e_home_read:
                logger.warning(
                    f"    Error reading/parsing home tracker {home_tracker_file_norm} for import: {e_home_read}",
                    exc_info=False,
                )  # Reduce noise
            return None

        abs_doc_roots_set = {
            normalize_path(os.path.join(project_root, p))
            for p in config.get_doc_directories()
        }

        def is_path_in_doc_roots_local(item_path: str) -> bool:  # Local helper
            norm_item_p = normalize_path(item_path)
            return any(
                is_subpath(norm_item_p, dr) or norm_item_p == dr
                for dr in abs_doc_roots_set
            )

        # 1. Native <-> Foreign relationships
        for native_ki in native_ki_in_current_tracker:
            native_idx_in_current = final_path_to_new_idx.get(native_ki.norm_path)
            if native_idx_in_current is None:
                continue

            for foreign_ki in foreign_ki_in_current_tracker:
                foreign_idx_in_current = final_path_to_new_idx.get(foreign_ki.norm_path)
                if foreign_idx_in_current is None:
                    continue

                home_tracker_file_of_foreign: Optional[str] = None
                if is_path_in_doc_roots_local(foreign_ki.norm_path):
                    home_tracker_file_of_foreign = get_tracker_path(project_root, "doc")
                elif (
                    foreign_ki.parent_path
                    and foreign_ki.parent_path != module_path_for_mini
                ):
                    home_tracker_file_of_foreign = get_mini_tracker_path(
                        foreign_ki.parent_path
                    )

                if (
                    home_tracker_file_of_foreign
                    and home_tracker_file_of_foreign != output_file
                ):  # Not self
                    # Path N -> Path F in Foreign's Home Tracker
                    char_nf_in_home = get_char_from_home_tracker_cached(
                        native_ki.norm_path,
                        foreign_ki.norm_path,
                        home_tracker_file_of_foreign,
                    )
                    # Path F -> Path N in Foreign's Home Tracker
                    char_fn_in_home = get_char_from_home_tracker_cached(
                        foreign_ki.norm_path,
                        native_ki.norm_path,
                        home_tracker_file_of_foreign,
                    )

                    # Update current grid's N->F cell
                    if char_nf_in_home and char_nf_in_home not in (
                        PLACEHOLDER_CHAR,
                        EMPTY_CHAR,
                        DIAGONAL_CHAR,
                    ):  # Ignore 'p', '.', 'o' from home
                        current_char_nf = temp_decomp_grid_rows[native_idx_in_current][
                            foreign_idx_in_current
                        ]
                        if get_priority(char_nf_in_home) > get_priority(
                            current_char_nf
                        ) or (
                            char_nf_in_home == "n"
                            and current_char_nf in (PLACEHOLDER_CHAR, "s", "S")
                        ):
                            if (
                                temp_decomp_grid_rows[native_idx_in_current][
                                    foreign_idx_in_current
                                ]
                                != char_nf_in_home
                            ):
                                logger.debug(
                                    f"    Import N->F: Updating {native_ki.norm_path} -> {foreign_ki.norm_path} from '{current_char_nf}' to '{char_nf_in_home}' (from {os.path.basename(home_tracker_file_of_foreign)})"
                                )
                                temp_decomp_grid_rows[native_idx_in_current][
                                    foreign_idx_in_current
                                ] = char_nf_in_home
                                native_foreign_import_ct += 1
                                grid_content_changed_by_imports = True

                    # Update current grid's F->N cell
                    if char_fn_in_home and char_fn_in_home not in (
                        PLACEHOLDER_CHAR,
                        EMPTY_CHAR,
                        DIAGONAL_CHAR,
                    ):  # Ignore 'p', '.', 'o' from home
                        current_char_fn = temp_decomp_grid_rows[foreign_idx_in_current][
                            native_idx_in_current
                        ]
                        if get_priority(char_fn_in_home) > get_priority(
                            current_char_fn
                        ) or (
                            char_fn_in_home == "n"
                            and current_char_fn in (PLACEHOLDER_CHAR, "s", "S")
                        ):
                            if (
                                temp_decomp_grid_rows[foreign_idx_in_current][
                                    native_idx_in_current
                                ]
                                != char_fn_in_home
                            ):
                                logger.debug(
                                    f"    Import F->N: Updating {foreign_ki.norm_path} -> {native_ki.norm_path} from '{current_char_fn}' to '{char_fn_in_home}' (from {os.path.basename(home_tracker_file_of_foreign)})"
                                )
                                temp_decomp_grid_rows[foreign_idx_in_current][
                                    native_idx_in_current
                                ] = char_fn_in_home
                                native_foreign_import_ct += 1
                                grid_content_changed_by_imports = True

        if native_foreign_import_ct > 0:
            logger.debug(
                f"  Import: {native_foreign_import_ct} native <-> foreign relationships potentially updated."
            )

        # 2. Foreign <-> Foreign relationships (if they share a common home tracker different from current)
        for i in range(len(foreign_ki_in_current_tracker)):
            f_ki1 = foreign_ki_in_current_tracker[i]
            f_idx1_in_current = final_path_to_new_idx.get(f_ki1.norm_path)
            if f_idx1_in_current is None:
                continue

            for j in range(
                i + 1, len(foreign_ki_in_current_tracker)
            ):  # Avoid self-comparison and duplicates
                f_ki2 = foreign_ki_in_current_tracker[j]
                f_idx2_in_current = final_path_to_new_idx.get(f_ki2.norm_path)
                if f_idx2_in_current is None:
                    continue

                common_home_tracker: Optional[str] = None
                is_fki1_doc = is_path_in_doc_roots_local(f_ki1.norm_path)
                is_fki2_doc = is_path_in_doc_roots_local(f_ki2.norm_path)

                if is_fki1_doc and is_fki2_doc:  # Both are doc items
                    common_home_tracker = get_tracker_path(project_root, "doc")
                elif (
                    not is_fki1_doc
                    and not is_fki2_doc
                    and f_ki1.parent_path
                    and f_ki1.parent_path == f_ki2.parent_path
                    and f_ki1.parent_path != module_path_for_mini
                ):
                    common_home_tracker = get_mini_tracker_path(f_ki1.parent_path)

                if (
                    common_home_tracker and common_home_tracker != output_file
                ):  # Not self
                    # Path F1 -> Path F2 in their common home
                    char_f1f2_in_home = get_char_from_home_tracker_cached(
                        f_ki1.norm_path, f_ki2.norm_path, common_home_tracker
                    )
                    # Path F2 -> Path F1 in their common home
                    char_f2f1_in_home = get_char_from_home_tracker_cached(
                        f_ki2.norm_path, f_ki1.norm_path, common_home_tracker
                    )

                    # --- Apply to F1 -> F2 cell in current mini tracker ---
                    current_char_f1f2_in_mini = temp_decomp_grid_rows[
                        f_idx1_in_current
                    ][f_idx2_in_current]
                    final_char_to_set_f1f2 = current_char_f1f2_in_mini

                    if char_f1f2_in_home and char_f1f2_in_home not in (
                        PLACEHOLDER_CHAR,
                        EMPTY_CHAR,
                        DIAGONAL_CHAR,
                    ):
                        # Home has a meaningful, non-placeholder/empty/diagonal link
                        if (
                            get_priority(char_f1f2_in_home)
                            > get_priority(current_char_f1f2_in_mini)
                            or current_char_f1f2_in_mini == PLACEHOLDER_CHAR
                        ):
                            final_char_to_set_f1f2 = char_f1f2_in_home
                    elif (
                        char_f1f2_in_home is None or char_f1f2_in_home == EMPTY_CHAR
                    ) and current_char_f1f2_in_mini == PLACEHOLDER_CHAR:
                        # Home has no link or empty link, and current mini is placeholder -> default to 'n'
                        final_char_to_set_f1f2 = "n"
                    # If char_f1f2_in_home is PLACEHOLDER_CHAR, it's ignored, final_char_to_set_f1f2 remains current_char_f1f2_in_mini

                    if (
                        temp_decomp_grid_rows[f_idx1_in_current][f_idx2_in_current]
                        != final_char_to_set_f1f2
                    ):
                        logger.debug(
                            f"    Import F1->F2: Updating {f_ki1.key_string} -> {f_ki2.key_string} from '{current_char_f1f2_in_mini}' to '{final_char_to_set_f1f2}' (Home char: {char_f1f2_in_home})"
                        )
                        temp_decomp_grid_rows[f_idx1_in_current][
                            f_idx2_in_current
                        ] = final_char_to_set_f1f2
                        foreign_foreign_import_ct += 1
                        grid_content_changed_by_imports = True

                    # --- Apply to F2 -> F1 cell in current mini tracker (symmetrical logic) ---
                    current_char_f2f1_in_mini = temp_decomp_grid_rows[
                        f_idx2_in_current
                    ][f_idx1_in_current]
                    final_char_to_set_f2f1 = current_char_f2f1_in_mini

                    if char_f2f1_in_home and char_f2f1_in_home not in (
                        PLACEHOLDER_CHAR,
                        EMPTY_CHAR,
                        DIAGONAL_CHAR,
                    ):
                        if (
                            get_priority(char_f2f1_in_home)
                            > get_priority(current_char_f2f1_in_mini)
                            or current_char_f2f1_in_mini == PLACEHOLDER_CHAR
                        ):
                            final_char_to_set_f2f1 = char_f2f1_in_home
                    elif (
                        char_f2f1_in_home is None or char_f2f1_in_home == EMPTY_CHAR
                    ) and current_char_f2f1_in_mini == PLACEHOLDER_CHAR:
                        final_char_to_set_f2f1 = "n"

                    if (
                        temp_decomp_grid_rows[f_idx2_in_current][f_idx1_in_current]
                        != final_char_to_set_f2f1
                    ):
                        logger.debug(
                            f"    Import F2->F1: Updating {f_ki2.key_string} -> {f_ki1.key_string} from '{current_char_f2f1_in_mini}' to '{final_char_to_set_f2f1}' (Home char: {char_f2f1_in_home})"
                        )
                        temp_decomp_grid_rows[f_idx2_in_current][
                            f_idx1_in_current
                        ] = final_char_to_set_f2f1
                        foreign_foreign_import_ct += 1
                        grid_content_changed_by_imports = True
        if foreign_foreign_import_ct > 0:
            logger.debug(
                f"  Import: {foreign_foreign_import_ct} foreign <-> foreign relationships potentially updated."
            )

    # --- Apply Suggestions (from suggestions_to_process_for_this_tracker, which is KEY#global_instance) ---
    # suggestion_applied_flag = False
    # applied_manual_source_path: Optional[str] = None # For last_grid_edit metadata if forced
    # applied_manual_target_paths: List[Tuple[str,str]] = [] # List of (key_str, path_str) for metadata
    # applied_manual_dep_type: Optional[str] = None # For metadata

    globally_instanced_suggestions_to_apply = suggestions_to_process_for_this_tracker
    ast_overrides_applied_count = 0  # Initialize here for clarity

    if globally_instanced_suggestions_to_apply:
        logger.debug(
            f"Applying {sum(len(v) for v in globally_instanced_suggestions_to_apply.values())} globally-instanced suggestions to grid for '{os.path.basename(output_file)}' (Force Apply: {force_apply_suggestions})"
        )

        # This flag tracks if any forced suggestion was "actioned" enough to update metadata and potentially the timestamp
        any_forced_suggestion_processed_for_metadata = False

        for (
            src_key_global_instance_str,
            deps_sugg_list_global,
        ) in globally_instanced_suggestions_to_apply.items():
            source_ki_globally_resolved = resolve_key_global_instance_to_ki(
                src_key_global_instance_str, path_to_key_info
            )
            if not source_ki_globally_resolved:
                logger.warning(
                    f"ApplySugg: Could not resolve source suggestion '{src_key_global_instance_str}' globally. Skipping."
                )
                continue
            src_local_idx = final_path_to_new_idx.get(
                source_ki_globally_resolved.norm_path
            )
            if src_local_idx is None:
                logger.debug(
                    f"ApplySugg: Source '{src_key_global_instance_str}' (Path: {source_ki_globally_resolved.norm_path}) not in current tracker structure '{os.path.basename(output_file)}'. Skipping."
                )
                continue
            src_ki_in_this_tracker = final_key_info_list[src_local_idx]

            for tgt_key_global_instance_str, dep_char_sugg in deps_sugg_list_global:
                target_ki_globally_resolved = resolve_key_global_instance_to_ki(
                    tgt_key_global_instance_str, path_to_key_info
                )
                if not target_ki_globally_resolved:
                    logger.warning(
                        f"ApplySugg: Could not resolve target '{tgt_key_global_instance_str}' for source '{src_key_global_instance_str}'. Skipping."
                    )
                    continue
                tgt_local_idx = final_path_to_new_idx.get(
                    target_ki_globally_resolved.norm_path
                )
                if tgt_local_idx is None or src_local_idx == tgt_local_idx:
                    logger.debug(
                        f"ApplySugg: Target '{tgt_key_global_instance_str}' (Path: {target_ki_globally_resolved.norm_path}) not in tracker or self-link. Skipping."
                    )
                    continue
                tgt_ki_in_this_tracker = final_key_info_list[tgt_local_idx]

                existing_char_in_grid = temp_decomp_grid_rows[src_local_idx][
                    tgt_local_idx
                ]
                final_char_to_set_in_grid = dep_char_sugg

                is_upgraded_to_x_this_link = False
                if final_char_to_set_in_grid in ("<", ">"):
                    char_in_reverse_cell = temp_decomp_grid_rows[tgt_local_idx][
                        src_local_idx
                    ]
                    if (
                        final_char_to_set_in_grid == ">" and char_in_reverse_cell == "<"
                    ) or (
                        final_char_to_set_in_grid == "<" and char_in_reverse_cell == ">"
                    ):
                        logger.debug(
                            f"    Mutual dependency formed for {src_ki_in_this_tracker.norm_path} <-> {tgt_ki_in_this_tracker.norm_path}. Upgrading to 'x'."
                        )
                        final_char_to_set_in_grid = "x"
                        is_upgraded_to_x_this_link = True
                        if temp_decomp_grid_rows[tgt_local_idx][src_local_idx] != "x":
                            temp_decomp_grid_rows[tgt_local_idx][src_local_idx] = "x"
                            suggestion_applied_flag = True

                apply_this_suggestion_to_cell = False
                if force_apply_suggestions:
                    if dep_char_sugg != PLACEHOLDER_CHAR:
                        apply_this_suggestion_to_cell = True
                        any_forced_suggestion_processed_for_metadata = (
                            True  # Mark that a forced suggestion was intended
                        )
                        logger.debug(
                            f"    Force apply active for: {src_ki_in_this_tracker.key_string} -> {tgt_ki_in_this_tracker.key_string} with '{final_char_to_set_in_grid}' (current: '{existing_char_in_grid}')"
                        )
                elif existing_char_in_grid in (
                    PLACEHOLDER_CHAR,
                    EMPTY_CHAR,
                ) and final_char_to_set_in_grid not in (PLACEHOLDER_CHAR, EMPTY_CHAR):
                    apply_this_suggestion_to_cell = True
                elif existing_char_in_grid != "n":
                    try:
                        if get_priority(final_char_to_set_in_grid) > get_priority(
                            existing_char_in_grid
                        ):
                            apply_this_suggestion_to_cell = True
                            logger.info(
                                f"    Applying stronger suggestion '{final_char_to_set_in_grid}' over '{existing_char_in_grid}' for {src_ki_in_this_tracker.norm_path} -> {tgt_ki_in_this_tracker.norm_path}"
                            )
                    except KeyError:
                        logger.warning(
                            f"    Priority lookup failed for '{final_char_to_set_in_grid}' or '{existing_char_in_grid}'."
                        )

                if apply_this_suggestion_to_cell:
                    if (
                        temp_decomp_grid_rows[src_local_idx][tgt_local_idx]
                        != final_char_to_set_in_grid
                    ):
                        temp_decomp_grid_rows[src_local_idx][
                            tgt_local_idx
                        ] = final_char_to_set_in_grid
                        suggestion_applied_flag = True
                        logger.debug(
                            f"    Applied to grid (forward): {src_ki_in_this_tracker.norm_path} -> {tgt_ki_in_this_tracker.norm_path} = '{final_char_to_set_in_grid}'"
                        )

                    # Populate metadata if forced (even if char was same)
                    if force_apply_suggestions:
                        if applied_manual_source_path is None:
                            applied_manual_source_path = (
                                src_ki_in_this_tracker.norm_path
                            )
                        target_display_key_for_msg = _get_display_key_for_tracker(
                            tgt_ki_in_this_tracker,
                            path_to_key_info,
                            global_key_counts_for_update_tracker,
                        )
                        current_target_tuple_for_msg = (
                            target_display_key_for_msg,
                            tgt_ki_in_this_tracker.norm_path,
                        )
                        if (
                            current_target_tuple_for_msg
                            not in applied_manual_target_paths
                        ):
                            applied_manual_target_paths.append(
                                current_target_tuple_for_msg
                            )
                        if applied_manual_dep_type is None:
                            applied_manual_dep_type = final_char_to_set_in_grid
                        elif (
                            applied_manual_dep_type != final_char_to_set_in_grid
                            and applied_manual_dep_type != "mixed"
                        ):
                            applied_manual_dep_type = "mixed"

                    if not is_upgraded_to_x_this_link:
                        reciprocal_char_to_set: Optional[str] = None
                        if final_char_to_set_in_grid == ">":
                            reciprocal_char_to_set = "<"
                        elif final_char_to_set_in_grid == "<":
                            reciprocal_char_to_set = ">"

                        if reciprocal_char_to_set:
                            existing_char_in_reverse = temp_decomp_grid_rows[
                                tgt_local_idx
                            ][src_local_idx]
                            apply_reciprocal_to_cell = False
                            if (
                                force_apply_suggestions
                                and existing_char_in_reverse != "x"
                                and existing_char_in_reverse != reciprocal_char_to_set
                            ):
                                apply_reciprocal_to_cell = True
                            elif not force_apply_suggestions and (
                                existing_char_in_reverse == PLACEHOLDER_CHAR
                                or (
                                    existing_char_in_reverse != "n"
                                    and get_priority(reciprocal_char_to_set)
                                    > get_priority(existing_char_in_reverse)
                                )
                            ):
                                apply_reciprocal_to_cell = True

                            if apply_reciprocal_to_cell:
                                if (
                                    temp_decomp_grid_rows[tgt_local_idx][src_local_idx]
                                    != reciprocal_char_to_set
                                ):
                                    temp_decomp_grid_rows[tgt_local_idx][
                                        src_local_idx
                                    ] = reciprocal_char_to_set
                                    suggestion_applied_flag = True
                                    logger.debug(
                                        f"    Reciprocal Applied: {tgt_ki_in_this_tracker.norm_path} -> {src_ki_in_this_tracker.norm_path} = '{reciprocal_char_to_set}'"
                                    )
                                # If forced, this reciprocal action also contributes to metadata timestamp update via any_forced_suggestion_processed_for_metadata
                                # No need to update applied_manual_... vars again as the primary direction covers the intent.

                elif (
                    not force_apply_suggestions
                    and existing_char_in_grid not in (PLACEHOLDER_CHAR, DIAGONAL_CHAR)
                    and existing_char_in_grid != dep_char_sugg
                ):
                    # Log ignored suggestions
                    if existing_char_in_grid == "n" or get_priority(
                        dep_char_sugg
                    ) <= get_priority(existing_char_in_grid):
                        logger.debug(
                            f"    Suggestion Ignored: Grid '{existing_char_in_grid}' kept over sugg '{dep_char_sugg}' for {src_ki_in_this_tracker.norm_path} -> {tgt_ki_in_this_tracker.norm_path}."
                        )

        if force_apply_suggestions and any_forced_suggestion_processed_for_metadata:
            suggestion_applied_flag = True  # Ensure timestamp update if any forced suggestion was processed, even if no grid character changed

    # --- Final Grid Consolidation ---
    consolidation_changes_ct = 0
    if final_key_info_list:  # Only run if grid is not empty
        logger.debug(
            f"Consolidating grid for '{os.path.basename(output_file)}' against global highest-priority relationships..."
        )
        try:
            all_tracker_paths_for_agg = find_all_tracker_paths(config, project_root)
            if not all_tracker_paths_for_agg:
                logger.warning(
                    "Consolidation: No tracker paths found for aggregation. Skipping consolidation."
                )
            else:
                # aggregate_all_dependencies expects path_migration_info, which should be available
                # It returns Dict[Tuple[current_source_key_str, current_target_key_str], Tuple[char, Set[origin_paths]]]
                globally_aggregated_links_with_origins = aggregate_all_dependencies(
                    all_tracker_paths_for_agg,
                    path_migration_info,
                    path_to_key_info,
                    show_progress=False,
                )

                global_authoritative_rels: Dict[Tuple[str, str], str] = {
                    link_tuple: char_val
                    for link_tuple, (
                        char_val,
                        _origins,
                    ) in globally_aggregated_links_with_origins.items()
                }

                # DOC TRACKER CONSOLIDATION SCOPE
                if tracker_type == "doc":
                    logger.debug(
                        f"Doc Tracker Consolidation: Scoping relationships to doc roots."
                    )

                    key_string_to_kis = defaultdict(list)
                    for ki in path_to_key_info.values():
                        key_string_to_kis[ki.key_string].append(ki)

                    def key_in_doc_root(key_str: str) -> bool:
                        kis = key_string_to_kis.get(key_str)
                        if not kis:
                            return False
                        return any(
                            is_path_in_doc_roots(ki.norm_path, abs_doc_roots_set)
                            for ki in kis
                        )

                    scoped_rels = {
                        (src_key, tgt_key): char
                        for (
                            src_key,
                            tgt_key,
                        ), char in global_authoritative_rels.items()
                        if key_in_doc_root(src_key) and key_in_doc_root(tgt_key)
                    }

                    logger.debug(
                        f"Scoped consolidation from {len(global_authoritative_rels)} to {len(scoped_rels)} relationships for doc_tracker."
                    )
                    global_authoritative_rels = scoped_rels

                logger.debug(
                    f"Retrieved {len(global_authoritative_rels)} globally authoritative relationships for consolidation."
                )

                # Iterate through the current temp_decomp_grid_rows using final_key_info_list
                # which defines its structure. The key_strings in KeyInfo are global keys.
                for r_idx, r_ki in enumerate(final_key_info_list):
                    # r_ki.key_string is the current global key string for the row item
                    for c_idx, c_ki in enumerate(final_key_info_list):
                        if r_idx == c_idx:
                            continue  # Skip diagonal

                        # c_ki.key_string is the current global key string for the column item
                        authoritative_char = global_authoritative_rels.get(
                            (r_ki.key_string, c_ki.key_string), PLACEHOLDER_CHAR
                        )
                        current_char_in_grid = temp_decomp_grid_rows[r_idx][c_idx]

                        if (
                            authoritative_char != PLACEHOLDER_CHAR
                        ):  # Global view has a defined non-'p' relationship
                            try:
                                auth_prio = get_priority(authoritative_char)
                                curr_prio = get_priority(current_char_in_grid)

                                # Update if authoritative is strictly higher priority
                                should_update_consolidate = auth_prio > curr_prio
                                # Also update if authoritative is 'n' (verified no dep) and current is overwritable ('p','s','S','.' EMPTY)
                                if (
                                    not should_update_consolidate
                                    and authoritative_char == "n"
                                    and current_char_in_grid
                                    in (PLACEHOLDER_CHAR, "s", "S", EMPTY_CHAR)
                                ):
                                    should_update_consolidate = True
                                # Also allow authoritative 's'/'S' to overwrite placeholders or empty
                                if (
                                    not should_update_consolidate
                                    and authoritative_char in ("s", "S")
                                    and current_char_in_grid
                                    in (PLACEHOLDER_CHAR, EMPTY_CHAR)
                                ):
                                    should_update_consolidate = True

                                if should_update_consolidate:
                                    if (
                                        temp_decomp_grid_rows[r_idx][c_idx]
                                        != authoritative_char
                                    ):
                                        logger.debug(
                                            f"  Consolidating '{r_ki.key_string}' -> '{c_ki.key_string}' (Paths: '{r_ki.norm_path}' -> '{c_ki.norm_path}'): From '{current_char_in_grid}' to '{authoritative_char}' based on global view."
                                        )
                                        temp_decomp_grid_rows[r_idx][
                                            c_idx
                                        ] = authoritative_char
                                        consolidation_changes_ct += 1
                            except KeyError as e_prio_consolidate:
                                logger.warning(
                                    f"  Consolidation: Priority lookup failed for char '{str(e_prio_consolidate)}' comparing {r_ki.key_string}->{c_ki.key_string}. Skipping cell."
                                )

                if consolidation_changes_ct > 0:
                    logger.debug(
                        f"Consolidation applied {consolidation_changes_ct} updates to '{os.path.basename(output_file)}' based on global relationships."
                    )
                else:
                    logger.debug(
                        f"No consolidation changes needed for '{os.path.basename(output_file)}' based on global relationships."
                    )
        except Exception as e_consolidation:
            logger.error(
                f"Error during grid consolidation for '{os.path.basename(output_file)}': {e_consolidation}",
                exc_info=True,
            )
    else:
        logger.info(
            f"Skipping grid consolidation for '{os.path.basename(output_file)}' as its grid structure is empty."
        )

    # --- Mini Tracker Foreign Key Pruning (MOVED TO RUN LATE) ---
    if (
        tracker_type == "mini" and final_key_info_list
    ):  # Only run if mini and grid is not already empty
        # If force_apply_suggestions is true (e.g. from CLI add-dependency), we skip pruning
        # to ensure explicitly added foreign keys (even if weakly linked otherwise) are not immediately removed.
        if not force_apply_suggestions:
            logger.info(
                f"Mini Tracker ({os.path.basename(output_file)}): Performing final pruning of foreign keys on fully processed grid (force_apply_suggestions is False)..."
            )

            # Snapshot current state before potential pruning
            # final_key_info_list and temp_decomp_grid_rows reflect the state AFTER all suggestions and consolidations.
            original_final_key_info_list_before_pruning = list(final_key_info_list)
            original_temp_decomp_grid_rows_before_pruning = [
                list(row) for row in temp_decomp_grid_rows
            ]  # Deep copy

            internal_paths_for_pruning_set = {
                ki.norm_path
                for ki in original_final_key_info_list_before_pruning
                if ki.parent_path == module_path_for_mini
                or ki.norm_path == module_path_for_mini
            }
            paths_to_keep_after_pruning_set = internal_paths_for_pruning_set.copy()

            # Use the existing min_positive_priority for the pruning threshold
            pruning_priority_threshold = min_positive_priority
            logger.debug(
                f"  Pruning for '{os.path.basename(output_file)}': Using persistence priority level threshold: {pruning_priority_threshold}."
            )

            for row_idx, row_ki in enumerate(
                original_final_key_info_list_before_pruning
            ):
                row_is_internal = row_ki.norm_path in internal_paths_for_pruning_set

                # Grid rows are from original_temp_decomp_grid_rows_before_pruning
                if row_idx >= len(original_temp_decomp_grid_rows_before_pruning):
                    continue  # Should not happen
                current_decomp_row_list = original_temp_decomp_grid_rows_before_pruning[
                    row_idx
                ]
                if len(current_decomp_row_list) != len(
                    original_final_key_info_list_before_pruning
                ):
                    continue  # Consistency check

                for col_idx, dep_char in enumerate(current_decomp_row_list):
                    if row_idx == col_idx:
                        continue  # Skip diagonal
                    if col_idx >= len(original_final_key_info_list_before_pruning):
                        continue  # Boundary check

                    col_ki = original_final_key_info_list_before_pruning[col_idx]
                    col_is_internal = col_ki.norm_path in internal_paths_for_pruning_set

                    # Only consider non-placeholder, non-'n' links for persistence evaluation during pruning
                    if dep_char not in (
                        PLACEHOLDER_CHAR,
                        DIAGONAL_CHAR,
                        EMPTY_CHAR,
                        "n",
                    ):
                        try:
                            dep_priority_val = get_priority(
                                dep_char
                            )  # This is an integer

                            if dep_priority_val >= pruning_priority_threshold:
                                # Persist foreign FILES linked to/from internal FILES
                                if not row_ki.is_directory and not col_ki.is_directory:
                                    if row_is_internal and not col_is_internal:
                                        if (
                                            col_ki.norm_path
                                            not in paths_to_keep_after_pruning_set
                                        ):
                                            logger.debug(
                                                f"  PruningKeep: Foreign FILE '{col_ki.norm_path}' (Key: {col_ki.key_string}) kept due to link '{dep_char}' (Prio: {dep_priority_val}) with internal FILE '{row_ki.norm_path}' (Key: {row_ki.key_string})."
                                            )
                                            paths_to_keep_after_pruning_set.add(
                                                col_ki.norm_path
                                            )
                                    elif not row_is_internal and col_is_internal:
                                        if (
                                            row_ki.norm_path
                                            not in paths_to_keep_after_pruning_set
                                        ):
                                            logger.debug(
                                                f"  PruningKeep: Foreign FILE '{row_ki.norm_path}' (Key: {row_ki.key_string}) kept due to link '{dep_char}' (Prio: {dep_priority_val}) with internal FILE '{col_ki.norm_path}' (Key: {col_ki.key_string})."
                                            )
                                            paths_to_keep_after_pruning_set.add(
                                                row_ki.norm_path
                                            )
                        except KeyError:
                            logger.warning(
                                f"Pruning: Character '{dep_char}' has no defined priority. Link between {row_ki.key_string} and {col_ki.key_string} not evaluated for persistence during pruning."
                            )
                            pass

            if len(paths_to_keep_after_pruning_set) < len(
                original_final_key_info_list_before_pruning
            ):
                num_pruned_val = len(original_final_key_info_list_before_pruning) - len(
                    paths_to_keep_after_pruning_set
                )
                logger.info(
                    f"Mini Pruning for '{os.path.basename(output_file)}': Pruned {num_pruned_val} foreign key-path instances that no longer had significant file-to-file links with internal items satisfying priority >= {pruning_priority_threshold}."
                )

                # This flag should already reflect earlier definition changes if any.
                # If pruning changes structure, ensure it's True.
                if not grid_structure_changed_flag:
                    grid_structure_changed_flag = True

                # Rebuild final_key_info_list
                final_key_info_list = [
                    ki
                    for ki in original_final_key_info_list_before_pruning
                    if ki.norm_path in paths_to_keep_after_pruning_set
                ]
                final_key_info_list.sort(
                    key=lambda ki_sort: (
                        (
                            get_sortable_parts_for_key(ki_sort.key_string)
                            if ki_sort.key_string
                            else []
                        ),
                        ki_sort.norm_path,
                    )
                )

                new_grid_item_count = len(final_key_info_list)  # Update count

                # Rebuild temp_decomp_grid_rows for the new, smaller size
                rebuilt_temp_decomp_grid_rows = [
                    [PLACEHOLDER_CHAR] * new_grid_item_count
                    for _ in range(new_grid_item_count)
                ]
                for i_rebuild in range(new_grid_item_count):
                    rebuilt_temp_decomp_grid_rows[i_rebuild][
                        i_rebuild
                    ] = DIAGONAL_CHAR  # Set diagonals

                pruned_path_to_new_idx_map = {
                    ki.norm_path: i for i, ki in enumerate(final_key_info_list)
                }

                # Iterate through the NEW (pruned) structure to populate the rebuilt grid
                for new_r_idx, new_r_ki in enumerate(final_key_info_list):
                    # Find corresponding row index in the grid *before* pruning
                    orig_r_idx = next(
                        (
                            i
                            for i, ki_orig in enumerate(
                                original_final_key_info_list_before_pruning
                            )
                            if ki_orig.norm_path == new_r_ki.norm_path
                        ),
                        None,
                    )
                    if orig_r_idx is None:
                        continue  # Should not happen if new_r_ki came from original

                    for new_c_idx, new_c_ki in enumerate(final_key_info_list):
                        if new_r_idx == new_c_idx:
                            continue  # Diagonal already set

                        # Find corresponding col index in the grid *before* pruning
                        orig_c_idx = next(
                            (
                                i
                                for i, ki_orig in enumerate(
                                    original_final_key_info_list_before_pruning
                                )
                                if ki_orig.norm_path == new_c_ki.norm_path
                            ),
                            None,
                        )
                        if orig_c_idx is None:
                            continue  # Should not happen

                        # Copy the value from the correct cell of the grid *before* pruning
                        if orig_r_idx < len(
                            original_temp_decomp_grid_rows_before_pruning
                        ) and orig_c_idx < len(
                            original_temp_decomp_grid_rows_before_pruning[orig_r_idx]
                        ):
                            rebuilt_temp_decomp_grid_rows[new_r_idx][new_c_idx] = (
                                original_temp_decomp_grid_rows_before_pruning[
                                    orig_r_idx
                                ][orig_c_idx]
                            )

                # Update the main grid variables to the new pruned state
                temp_decomp_grid_rows = rebuilt_temp_decomp_grid_rows
                final_path_to_new_idx = pruned_path_to_new_idx_map
                # new_grid_item_count is already updated above
                logger.debug(
                    f"Mini Pruning: Grid for '{os.path.basename(output_file)}' rebuilt to {new_grid_item_count}x{new_grid_item_count}."
                )
            else:
                logger.debug(
                    f"Mini Pruning: No foreign keys pruned for '{os.path.basename(output_file)}'."
                )
        else:  # force_apply_suggestions is True for this mini_tracker update
            logger.debug(
                f"Mini Tracker ({os.path.basename(output_file)}): Skipping foreign key pruning because force_apply_suggestions is True."
            )
    elif tracker_type == "doc" and final_key_info_list:
        logger.info(
            f"Doc Tracker ({os.path.basename(output_file)}): Pruning items not in a doc root."
        )

        original_key_info_count = len(final_key_info_list)

        pruned_key_info_list = [
            ki
            for ki in final_key_info_list
            if is_path_in_doc_roots(ki.norm_path, abs_doc_roots_set)
        ]

        if len(pruned_key_info_list) < original_key_info_count:
            num_pruned = original_key_info_count - len(pruned_key_info_list)
            logger.info(
                f"Doc Tracker Pruning: Pruned {num_pruned} items that are not in a configured doc root."
            )

            grid_structure_changed_flag = True

            original_final_key_info_list_before_pruning = list(final_key_info_list)
            original_temp_decomp_grid_rows_before_pruning = [
                list(row) for row in temp_decomp_grid_rows
            ]

            final_key_info_list = pruned_key_info_list
            final_key_info_list.sort(
                key=lambda ki_sort: (
                    (
                        get_sortable_parts_for_key(ki_sort.key_string)
                        if ki_sort.key_string
                        else []
                    ),
                    ki_sort.norm_path,
                )
            )

            new_grid_item_count = len(final_key_info_list)

            rebuilt_temp_decomp_grid_rows = [
                [PLACEHOLDER_CHAR] * new_grid_item_count
                for _ in range(new_grid_item_count)
            ]
            for i_rebuild in range(new_grid_item_count):
                rebuilt_temp_decomp_grid_rows[i_rebuild][i_rebuild] = DIAGONAL_CHAR

            pruned_path_to_new_idx_map = {
                ki.norm_path: i for i, ki in enumerate(final_key_info_list)
            }

            for new_r_idx, new_r_ki in enumerate(final_key_info_list):
                orig_r_idx = next(
                    (
                        i
                        for i, ki_orig in enumerate(
                            original_final_key_info_list_before_pruning
                        )
                        if ki_orig.norm_path == new_r_ki.norm_path
                    ),
                    None,
                )
                if orig_r_idx is None:
                    continue
                for new_c_idx, new_c_ki in enumerate(final_key_info_list):
                    if new_r_idx == new_c_idx:
                        continue
                    orig_c_idx = next(
                        (
                            i
                            for i, ki_orig in enumerate(
                                original_final_key_info_list_before_pruning
                            )
                            if ki_orig.norm_path == new_c_ki.norm_path
                        ),
                        None,
                    )
                    if orig_c_idx is None:
                        continue

                    if orig_r_idx < len(
                        original_temp_decomp_grid_rows_before_pruning
                    ) and orig_c_idx < len(
                        original_temp_decomp_grid_rows_before_pruning[orig_r_idx]
                    ):
                        rebuilt_temp_decomp_grid_rows[new_r_idx][new_c_idx] = (
                            original_temp_decomp_grid_rows_before_pruning[orig_r_idx][
                                orig_c_idx
                            ]
                        )

            temp_decomp_grid_rows = rebuilt_temp_decomp_grid_rows
            final_path_to_new_idx = pruned_path_to_new_idx_map

    def _apply_ast_verified_overrides(
        temp_decomp_grid_rows: List[
            List[str]
        ],  # The current decompressed grid to modify
        final_key_info_list: List[KeyInfo],  # Defines the structure of the grid
        path_to_key_info_global: Dict[
            str, KeyInfo
        ],  # Full global map for GI resolution
        config: ConfigManager,  # For character priorities
        ast_links: List[Dict[str, str]],  # Loaded from ast_verified_links.json
    ) -> int:  # Returns the number of overrides applied
        """
        Applies AST-verified links to the grid, overriding 'n' and handling conflicts.
        Modifies temp_decomp_grid_rows in place.
        """
        if not ast_links:
            logger.debug(
                "AST Overrides: No AST-verified links provided. Skipping override step."
            )
            return 0
        if not temp_decomp_grid_rows or not final_key_info_list:
            logger.debug(
                "AST Overrides: Grid or key list is empty. Skipping override step."
            )
            return 0

        overrides_applied_count = 0
        get_priority = config.get_char_priority

        # Build a map from norm_path to its index in final_key_info_list for quick lookups
        path_to_final_idx: Dict[str, int] = {
            ki.norm_path: i for i, ki in enumerate(final_key_info_list)
        }

        # A local cache for GI string resolution for performance within this function
        # Alternatively, the module-level cache in tracker_utils.get_key_global_instance_string will be used.
        # If performance is an issue, pre-populating or passing a shared cache might be better.
        # For now, relying on the caching within get_key_global_instance_string.

        logger.debug(
            f"Applying {len(ast_links)} AST-verified links to the current grid state..."
        )

        # Build map for quick path -> index resolution within this tracker's grid scope
        path_to_final_idx: Dict[str, int] = {
            ki.norm_path: i for i, ki in enumerate(final_key_info_list)
        }

        for link_data in ast_links:
            # ast_links are authoritative path-based relations; use them directly
            source_path_from_ast = normalize_path(
                link_data.get("source_path", "") or ""
            )
            target_path_from_ast = normalize_path(
                link_data.get("target_path", "") or ""
            )
            ast_char = (link_data.get("char", "") or "").strip()
            # reason = link_data.get("reason", "UnknownReason") # optional

            if not source_path_from_ast or not target_path_from_ast or not ast_char:
                logger.warning(
                    f"AST Overrides: Skipping malformed AST link entry: {link_data}"
                )
                continue

            # Directly locate indices in this tracker’s grid by path
            row_idx = path_to_final_idx.get(source_path_from_ast)
            col_idx = path_to_final_idx.get(target_path_from_ast)

            # If either endpoint isn’t in this tracker’s grid, auto-include foreign items per user policy (all tracker types; no guardrails)
            if row_idx is None or col_idx is None:
                # Attempt to include missing endpoints into this tracker’s definitions
                # Resolve KeyInfo objects from global map using paths from AST link
                src_ki_global = path_to_key_info_global.get(source_path_from_ast)
                tgt_ki_global = path_to_key_info_global.get(target_path_from_ast)
                if not src_ki_global or not tgt_ki_global:
                    # Cannot include if not in global map; skip this link
                    continue

                # Extend final_key_info_list with any missing endpoints
                added_any = False
                if row_idx is None:
                    final_key_info_list.append(src_ki_global)
                    added_any = True
                if col_idx is None:
                    # Avoid duplicating if both refer to the same path
                    if src_ki_global.norm_path != tgt_ki_global.norm_path:
                        final_key_info_list.append(tgt_ki_global)
                        added_any = True

                if added_any:
                    # Re-sort the list deterministically (by hierarchical key then path)
                    final_key_info_list.sort(
                        key=lambda ki_sort: (
                            (
                                get_sortable_parts_for_key(ki_sort.key_string)
                                if ki_sort.key_string
                                else []
                            ),
                            ki_sort.norm_path,
                        )
                    )
                    # Rebuild path->index mapping
                    path_to_final_idx.clear()
                    path_to_final_idx.update(
                        {ki.norm_path: i for i, ki in enumerate(final_key_info_list)}
                    )

                    # Resize temp_decomp_grid_rows to new NxN, preserving existing values
                    old_n = len(temp_decomp_grid_rows)
                    new_n = len(final_key_info_list)
                    if old_n == 0:
                        # initialize fresh grid
                        temp_decomp_grid_rows[:] = [
                            [PLACEHOLDER_CHAR] * new_n for _ in range(new_n)
                        ]
                        for d in range(new_n):
                            temp_decomp_grid_rows[d][d] = DIAGONAL_CHAR
                    else:
                        # build new grid and copy old into top-left
                        new_grid = [[PLACEHOLDER_CHAR] * new_n for _ in range(new_n)]
                        for d in range(new_n):
                            new_grid[d][d] = DIAGONAL_CHAR
                        # Determine old indices of paths present before expansion
                        # We assume previous order corresponds to first old_n entries after resort; copy min-by-min
                        lim = min(old_n, new_n)
                        for r in range(lim):
                            for c in range(lim):
                                if r != c:
                                    new_grid[r][c] = temp_decomp_grid_rows[r][c]
                        # Replace
                        temp_decomp_grid_rows[:] = new_grid

                    # Update indices after expansion
                    row_idx = path_to_final_idx.get(source_path_from_ast)
                    col_idx = path_to_final_idx.get(target_path_from_ast)

                # If after expansion indices are still missing, skip
                if row_idx is None or col_idx is None:
                    continue

            if row_idx is None or col_idx is None:
                # logger.debug(f"AST Overrides: Source or target path from AST link not in current grid structure. Link: {source_path_from_ast} -> {target_path_from_ast}")
                continue

            if (
                row_idx == col_idx
            ):  # Should not happen for valid AST links, but good check
                continue

            current_char_in_grid = temp_decomp_grid_rows[row_idx][col_idx]
            char_to_set = current_char_in_grid
            changed_this_cell = False

            if current_char_in_grid == "x":
                pass  # do not demote
            elif ast_char == "x":
                if current_char_in_grid != "x":
                    char_to_set = "x"
                    changed_this_cell = True
            elif current_char_in_grid in (PLACEHOLDER_CHAR, EMPTY_CHAR, "n"):
                # Always override placeholder/empty/'n' with AST char
                char_to_set = ast_char
                changed_this_cell = True
                logger.debug(
                    f"AST_OVERRIDE: ({final_key_info_list[row_idx].norm_path} -> {final_key_info_list[col_idx].norm_path}) set to '{ast_char}' from '{current_char_in_grid}'."
                )
            else:
                # Priority comparison for other cases
                try:
                    priority_ast = get_priority(ast_char)
                    priority_current_in_grid = get_priority(current_char_in_grid)
                except KeyError:
                    priority_ast = 0
                    priority_current_in_grid = 0

                if priority_ast > priority_current_in_grid:
                    char_to_set = ast_char
                    changed_this_cell = True
                elif (
                    priority_ast == priority_current_in_grid
                    and current_char_in_grid != ast_char
                ):
                    if {current_char_in_grid, ast_char} == {"<", ">"}:
                        char_to_set = "x"
                        changed_this_cell = True
                    else:
                        char_to_set = ast_char
                        changed_this_cell = True

            if changed_this_cell:
                temp_decomp_grid_rows[row_idx][col_idx] = char_to_set
                overrides_applied_count += 1
                # Also check if this change forms an 'x' with the reverse cell
                if (
                    char_to_set == "<"
                    and temp_decomp_grid_rows[col_idx][row_idx] == ">"
                ):
                    temp_decomp_grid_rows[row_idx][col_idx] = "x"
                    temp_decomp_grid_rows[col_idx][row_idx] = "x"
                    logger.info(
                        f"AST_MUTUAL_FORMED: Grid cell ({final_key_info_list[row_idx].key_string} <-> {final_key_info_list[col_idx].key_string}) set to 'x' due to AST links."
                    )
                elif (
                    char_to_set == ">"
                    and temp_decomp_grid_rows[col_idx][row_idx] == "<"
                ):
                    temp_decomp_grid_rows[row_idx][col_idx] = "x"
                    temp_decomp_grid_rows[col_idx][row_idx] = "x"
                    logger.info(
                        f"AST_MUTUAL_FORMED: Grid cell ({final_key_info_list[row_idx].key_string} <-> {final_key_info_list[col_idx].key_string}) set to 'x' due to AST links."
                    )

        if overrides_applied_count > 0:
            logger.debug(
                f"AST Overrides: Applied/updated {overrides_applied_count} relationships in the grid based on AST-verified links."
            )

        return overrides_applied_count

    # --- AST OVERRIDES BLOCK (Conditional Execution) ---
    if apply_ast_overrides:  # <<< NEW CONDITION
        if final_key_info_list:
            loaded_ast_links = _load_ast_verified_links()
            if loaded_ast_links:
                logger.debug(
                    f"Applying AST-verified overrides to grid for tracker: {os.path.basename(output_file)}"
                )
                ast_overrides_applied_count = _apply_ast_verified_overrides(  # This function was defined in previous step
                    temp_decomp_grid_rows,
                    final_key_info_list,
                    path_to_key_info,
                    config,
                    loaded_ast_links,
                )
                # No explicit change to suggestion_applied_flag here, ast_overrides_applied_count used below
            else:
                logger.info(
                    f"No AST-verified links found or loaded. Skipping AST override step for {os.path.basename(output_file)}."
                )
        else:
            logger.info(
                f"Grid for {os.path.basename(output_file)} is empty. Skipping AST override step."
            )
    else:  # apply_ast_overrides is False
        logger.debug(
            f"Skipping AST override step as per caller request for {os.path.basename(output_file)}."
        )
    # --- END OF AST OVERRIDE CALL ---

    # --- Update Grid Edit Timestamp ---
    final_last_grid_edit = (
        current_last_grid_edit  # Start with existing or "Initial creation"
    )

    # Determine if grid content truly changed (beyond just structural placeholders)
    grid_content_truly_changed_by_ops = (
        suggestion_applied_flag
        or (consolidation_changes_ct > 0)
        or grid_content_changed_by_structural
        or grid_content_changed_by_imports
        or (ast_overrides_applied_count > 0)
    )  # ADDED AST COUNT

    timestamp_now_str = datetime.datetime.now().isoformat()

    # Check if the primary change was a forced manual suggestion
    if (
        force_apply_suggestions
        and any_forced_suggestion_processed_for_metadata
        and applied_manual_source_path
    ):  # Use any_forced_suggestion_processed_for_metadata
        src_display_label = os.path.basename(applied_manual_source_path)
        unique_target_labels_for_msg = sorted(
            list(
                set(
                    f"{key_str_msg}({os.path.basename(path_str_msg)})"
                    for key_str_msg, path_str_msg in applied_manual_target_paths
                )
            )
        )
        targets_display_str_msg = ", ".join(unique_target_labels_for_msg)
        final_last_grid_edit = f"Manual dep: {src_display_label} -> [{targets_display_str_msg}] ({applied_manual_dep_type or '?'}) ({timestamp_now_str})"
        logger.debug(f"Setting last_GRID_edit (manual forced): {final_last_grid_edit}")
    elif grid_content_truly_changed_by_ops:
        if grid_structure_changed_flag and (
            current_last_grid_edit.startswith("Grid structure updated")
            or current_last_grid_edit == "Initial creation"
            or "FromBackup" in current_last_grid_edit
            or "ErrorReading" in current_last_grid_edit
        ):
            final_last_grid_edit = (
                f"Grid structure and content updated ({timestamp_now_str})"
            )
        else:
            final_last_grid_edit = f"Grid content updated ({timestamp_now_str})"
        logger.debug(
            f"Setting last_GRID_edit (content changed by ops): {final_last_grid_edit}"
        )
    elif grid_structure_changed_flag:
        if (
            current_last_grid_edit == "Initial creation"
            or not current_last_grid_edit.startswith("Grid structure updated")
            or "FromBackup" in current_last_grid_edit
            or "ErrorReading" in current_last_grid_edit
        ):
            final_last_grid_edit = f"Grid structure updated ({timestamp_now_str})"
        # If already "Grid structure updated", timestamp is implicitly updated by rewrite. No need to change msg unless content also changed.
        logger.debug(
            f"Setting last_GRID_edit (structure changed only): {final_last_grid_edit}"
        )
    else:
        logger.debug(
            f"Keeping existing last_GRID_edit message: '{final_last_grid_edit}' (no relevant grid changes detected)."
        )

    # --- Compress final grid ---
    final_grid_comp_ordered: List[str]
    if not final_key_info_list:  # Grid is definitionally empty
        logger.info(
            f"Tracker '{os.path.basename(output_file)}' has no items in its final definition list. Grid will be empty."
        )
        final_grid_comp_ordered = []
        if temp_decomp_grid_rows:  # Should also be empty if KIs are empty after pruning
            logger.warning(
                f"Mismatch: final_key_info_list empty but temp_decomp_grid_rows not. Forcing empty grid."
            )
            temp_decomp_grid_rows = []
    elif len(temp_decomp_grid_rows) != len(final_key_info_list) or (
        temp_decomp_grid_rows
        and len(temp_decomp_grid_rows[0]) != len(final_key_info_list)
    ):
        logger.error(
            f"CRITICAL: Grid dimension mismatch before final compression for '{os.path.basename(output_file)}'. "
            f"Expected {len(final_key_info_list)}x{len(final_key_info_list)}, "
            f"got {len(temp_decomp_grid_rows)}x{len(temp_decomp_grid_rows[0]) if temp_decomp_grid_rows else 0}. Tracker write might be corrupt or empty."
        )

        # Fallback to a correctly created initial grid representation for this size
        # Uses the updated dependency_grid.create_initial_grid
        initial_grid_dict_fallback = create_initial_grid(final_key_info_list)
        final_grid_comp_ordered = [
            initial_grid_dict_fallback[ki.key_string] for ki in final_key_info_list
        ]
    else:
        final_grid_comp_ordered = [compress("".join(r)) for r in temp_decomp_grid_rows]
    # --- END OF SECTION: Compress final grid ---

    # --- Final Write ---
    logger.debug(f"Finalizing write for tracker: {output_file}")

    # Precompute global key counts ONCE before final write for efficiency
    final_global_key_counts = defaultdict(int)
    for ki_global_final_write in path_to_key_info.values():
        final_global_key_counts[ki_global_final_write.key_string] += 1

    # --- Intercept for return_update mode ---
    if return_update:
        logger.debug(f"Returning prepared data for tracker: {output_file}")
        return {
            "tracker_type": tracker_type,
            "output_file": output_file,
            "key_info_list": final_key_info_list,
            "grid_rows": final_grid_comp_ordered,
            "last_key_edit": final_last_key_edit,
            "last_grid_edit": final_last_grid_edit,
            "module_path": module_path_for_mini if tracker_type == "mini" else None,
            "existing_lines": (
                lines_from_old_file if tracker_exists_and_is_sound else []
            ),
            "tracker_exists": tracker_exists_and_is_sound,
            "path_to_key_info": path_to_key_info,
        }

    if tracker_type == "mini":
        if not module_path_for_mini:
            logger.critical(
                f"CRITICAL FINAL WRITE: module_path_for_mini is empty for mini-tracker '{output_file}'. Aborting."
            )
            return None  # Cannot format template correctly

        # Ensure lines_from_old_file is only used if tracker_exists_and_is_sound was true earlier
        # If tracker was rebuilt, lines_from_old_file would be empty.
        template_to_use = get_mini_tracker_data()["template"]
        markers_to_use = get_mini_tracker_data()["markers"]

        write_mini_tracker_with_template_preservation(
            output_file,
            (
                lines_from_old_file if tracker_exists_and_is_sound else []
            ),  # Pass empty if rebuilt
            final_key_info_list,
            final_grid_comp_ordered,
            final_last_key_edit,
            final_last_grid_edit,
            template_to_use,  # Pass raw template string
            markers_to_use,
            path_to_key_info,
            module_path_for_mini,  # Pass for template formatting {module_name}
        )
        logger.debug(
            f"Mini tracker '{os.path.basename(output_file)}' write process completed."
        )
    else:  # Main or Doc
        if not write_tracker_file(
            output_file,
            final_key_info_list,
            final_grid_comp_ordered,
            final_last_key_edit,
            final_last_grid_edit,
            path_to_key_info,
        ):
            logger.error(
                f"Write main/doc tracker {output_file} failed during final write. Review logs."
            )
            return None  # Do not invalidate caches if write failed

    logger.debug(f"Tracker update process for '{output_file}' completed successfully.")
    # --- END OF SECTION: Final Write ---

    # --- Cache Invalidation ---
    invalidate_dependent_entries(
        "tracker_data_structured",
        f"tracker_data_structured:{normalize_path(output_file)}:.*",
    )
    invalidate_dependent_entries(
        "aggregation_v2_gi", ".*"
    )  # Invalidate new GI aggregation cache
    logger.debug(f"Invalidated relevant caches for '{os.path.basename(output_file)}'.")
    return None
    # --- END OF SECTION: Cache Invalidation ---


def prepare_tracker_update(
    output_file_suggestion: str,
    path_to_key_info: Dict[str, KeyInfo],
    tracker_type: str = "main",
    suggestions_external: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    file_to_module: Optional[Dict[str, str]] = None,
    new_keys: Optional[List[KeyInfo]] = None,
    force_apply_suggestions: bool = False,
    keys_to_explicitly_remove: Optional[Set[str]] = None,
    use_old_map_for_migration: bool = True,
    apply_ast_overrides: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Prepares a tracker update in memory by calling update_tracker in return_update mode.
    This ensures that all processing logic (aggregation, migration, overrides) is identical
    between synchronous writes and batched updates.
    """
    return update_tracker(
        output_file_suggestion=output_file_suggestion,
        path_to_key_info=path_to_key_info,
        tracker_type=tracker_type,
        suggestions_external=suggestions_external,
        file_to_module=file_to_module,
        new_keys=new_keys,
        force_apply_suggestions=force_apply_suggestions,
        keys_to_explicitly_remove=keys_to_explicitly_remove,
        use_old_map_for_migration=use_old_map_for_migration,
        apply_ast_overrides=apply_ast_overrides,
        return_update=True,
    )


# --- remove_path_from_tracker (REFACTORED from remove_key_from_tracker) ---
def remove_path_from_tracker(output_file_path_str: str, path_to_remove_str: str):
    """
    Removes an item by its path from a tracker file.
    It achieves this by calling update_tracker with a modified global key map copy
    where the specified path is excluded.

    Args:
        output_file_path_str: Path to the tracker file.
        path_to_remove_str: The normalized path string of the item to remove.

    Raises:
        FileNotFoundError: If the tracker file doesn't exist.
        ValueError: If the global key map cannot be loaded.
        IOError: If reading the tracker file fails.
        Exception: For other unexpected errors during processing or update_tracker call.
    """
    output_file = normalize_path(output_file_path_str)
    path_to_remove = normalize_path(path_to_remove_str)  # Ensure path is normalized

    logger.info(
        f"Attempting removal of path '{path_to_remove}' from tracker '{output_file}'."
    )

    if not os.path.exists(output_file):
        raise FileNotFoundError(
            f"Tracker file '{output_file}' not found for path removal."
        )

    # Check if the path is even in the tracker's current definitions (optional, for user feedback)
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        current_defs = read_key_definitions_from_lines(lines)
        if not any(p == path_to_remove for _, p in current_defs):
            logger.warning(
                f"Path '{path_to_remove}' not found in current definitions of '{output_file}'. Update will proceed based on global map."
            )
    except Exception as e_read:
        logger.error(
            f"Error reading tracker {output_file} before path removal: {e_read}. Proceeding cautiously."
        )

    backup_tracker_file(output_file)  # Backup before attempting update

    # Load the current global key map
    global_path_map_full = load_global_key_map()
    if not global_path_map_full:
        # This is critical, update_tracker needs this.
        raise ValueError("Cannot load global key map. Path removal process aborted.")

    # Create a modified copy of the global map, excluding the path_to_remove
    modified_global_path_map = global_path_map_full.copy()
    if path_to_remove in modified_global_path_map:
        del modified_global_path_map[path_to_remove]
        logger.debug(
            f"Path '{path_to_remove}' removed from in-memory global map copy for update_tracker."
        )
    else:
        logger.warning(
            f"Path '{path_to_remove}' was not found in the loaded global key map. "
            "The update_tracker call will proceed with the current global map state (minus this path if it was there)."
        )

    is_mini = output_file.endswith("_module.md")
    tracker_type_val = (
        "mini" if is_mini else ("doc" if "doc_tracker.md" in output_file else "main")
    )
    f_to_m_map = {
        _info.norm_path: _info.parent_path
        for _info in (global_path_map_full).values()
        if not _info.is_directory and _info.parent_path
    }
    key_str_of_removed: Optional[str] = (
        global_path_map_full.get(
            path_to_remove, KeyInfo("", "", None, 0, False)
        ).key_string
        if path_to_remove in global_path_map_full
        else None
    )
    explicit_remove_arg = {key_str_of_removed} if key_str_of_removed else None
    try:
        update_tracker(
            output_file_suggestion=output_file,
            path_to_key_info=modified_global_path_map,  # Crucially, pass the map *without* the removed path
            tracker_type=tracker_type_val,
            file_to_module=f_to_m_map,
            suggestions_external=None,  # No suggestions being applied during a remove operation
            new_keys=None,  # No new keys being introduced globally
            force_apply_suggestions=False,
            keys_to_explicitly_remove=explicit_remove_arg,  # Hint for mini-tracker logic
            use_old_map_for_migration=True,  # Standard migration logic applies
        )
        logger.info(
            f"update_tracker completed for removal of path '{path_to_remove}' from '{output_file}'."
        )

    except Exception as e_update:
        raise Exception(
            f"update_tracker failed during path removal: {e_update}"
        ) from e_update


# --- Export Tracker (adapts to new read format) ---
def export_tracker(
    tracker_path: str, output_format: str = "json", output_path: Optional[str] = None
) -> str:
    tracker_path = normalize_path(tracker_path)
    check_file_modified(tracker_path)  # Ensures cache is fresh if used by read helpers
    logger.info(
        f"Attempting to export '{os.path.basename(tracker_path)}' to format '{output_format}'"
    )

    # Read tracker using new parsing logic
    key_info_list_for_export: List[KeyInfo] = []
    grid_key_strings_for_export: List[str] = []  # From X line
    grid_rows_compressed_for_export: List[str] = []
    # Metadata...
    if not os.path.exists(tracker_path):
        msg = f"Error: Tracker file not found for export: {tracker_path}"
        logger.error(msg)
        return msg
    try:
        with open(tracker_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Attempt to build KeyInfo list for export
        _g_map = load_global_key_map()  # Try to get global map for full KeyInfo
        temp_key_path_pairs = read_key_definitions_from_lines(lines)
        if _g_map:
            key_info_list_for_export = [
                _g_map[p_str]
                for k_str, p_str in temp_key_path_pairs
                if p_str in _g_map and _g_map[p_str].key_string == k_str
            ]
        else:  # Fallback if global map fails to load
            key_info_list_for_export = [
                KeyInfo(
                    key_string=k,
                    norm_path=p,
                    parent_path=os.path.dirname(p),
                    tier=0,
                    is_directory=os.path.isdir(p),
                )  # Tier unknown
                for k, p in temp_key_path_pairs
            ]

        temp_grid_headers, temp_grid_rows_tuples = read_grid_from_lines(lines)
        grid_key_strings_for_export = temp_grid_headers
        # Align compressed rows with key_info_list_for_export (if headers match defs order)
        if len(key_info_list_for_export) == len(temp_grid_rows_tuples) and all(
            key_info_list_for_export[i].key_string == temp_grid_rows_tuples[i][0]
            for i in range(len(key_info_list_for_export))
        ):
            grid_rows_compressed_for_export = [tpl[1] for tpl in temp_grid_rows_tuples]
        else:  # Fallback if mismatch, use as read
            grid_rows_compressed_for_export = [tpl[1] for tpl in temp_grid_rows_tuples]
            if (
                not grid_key_strings_for_export and grid_rows_compressed_for_export
            ):  # No X line, but rows exist
                # Cannot reliably determine headers for CSV/DOT if X line is missing.
                # For JSON, it's less critical as we dump the lists.
                # For CSV/DOT, this will be problematic.
                if len(key_info_list_for_export) == len(
                    grid_rows_compressed_for_export
                ):
                    grid_key_strings_for_export = [
                        ki.key_string for ki in key_info_list_for_export
                    ]
                else:
                    logger.warning(
                        f"Export: Grid header missing/mismatched in {tracker_path}. CSV/DOT export may be incomplete/incorrect."
                    )
                    # Create dummy headers if needed of same length as rows
                    if (
                        len(grid_rows_compressed_for_export) > 0
                        and not grid_key_strings_for_export
                    ):
                        grid_key_strings_for_export = [
                            f"K{i}" for i in range(len(grid_rows_compressed_for_export))
                        ]

    except Exception as e_read_export:
        msg = f"Error reading tracker for export {tracker_path}: {e_read_export}"
        logger.error(msg, exc_info=True)
        return msg

    if (
        not key_info_list_for_export and not grid_rows_compressed_for_export
    ):  # Check if anything was read
        msg = f"Error: Cannot export empty/unreadable tracker: {tracker_path}"
        logger.error(msg)
        return msg

    if output_path is None:
        base_name = os.path.splitext(tracker_path)[0]
        output_path = normalize_path(f"{base_name}_export.{output_format}")
    else:
        output_path = normalize_path(output_path)

    try:
        dirname = os.path.dirname(output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        if output_format == "md":
            shutil.copy2(tracker_path, output_path)
        elif output_format == "json":
            # Export structure should reflect List[KeyInfo] and List[compressed_rows]
            export_data = {
                "key_info_list": [ki._asdict() for ki in key_info_list_for_export],
                "grid_header_keys": grid_key_strings_for_export,  # Key strings from X line
                "grid_rows_compressed": grid_rows_compressed_for_export,  # List of compressed strings
            }
            # Add metadata if read
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

        elif output_format == "csv":
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                import csv

                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Source Key",
                        "Source Path",
                        "Target Key",
                        "Target Path",
                        "Dependency Type",
                    ]
                )

                if len(key_info_list_for_export) != len(
                    grid_rows_compressed_for_export
                ):
                    logger.warning(
                        f"CSV Export: Mismatch between definitions ({len(key_info_list_for_export)}) and grid rows ({len(grid_rows_compressed_for_export)}). CSV may be incorrect."
                    )

                # Iterate based on the number of rows we have data for.
                for row_idx, compressed_row_str in enumerate(
                    grid_rows_compressed_for_export
                ):
                    if row_idx >= len(key_info_list_for_export):
                        break  # No more source KeyInfo
                    source_ki = key_info_list_for_export[row_idx]
                    try:
                        decomp_row = decompress(compressed_row_str)
                        # Decompressed row length should match the grid header or defs count
                        expected_cols = (
                            len(grid_key_strings_for_export)
                            if grid_key_strings_for_export
                            else len(key_info_list_for_export)
                        )
                        if len(decomp_row) != expected_cols:
                            logger.warning(
                                f"CSV Export: Row {row_idx} (src path {source_ki.norm_path}) decompiled length {len(decomp_row)} != expected {expected_cols}. Skipping row."
                            )
                            continue

                        for col_idx, dep_type in enumerate(decomp_row):
                            if dep_type not in (
                                EMPTY_CHAR,
                                DIAGONAL_CHAR,
                                PLACEHOLDER_CHAR,
                            ):
                                if col_idx >= len(key_info_list_for_export):
                                    break  # No more target KeyInfo
                                target_ki = key_info_list_for_export[col_idx]
                                writer.writerow(
                                    [
                                        source_ki.key_string,
                                        source_ki.norm_path,
                                        target_ki.key_string,
                                        target_ki.norm_path,
                                        dep_type,
                                    ]
                                )
                    except Exception as e_csv:
                        logger.warning(
                            f"CSV Export: Error processing row for src path '{source_ki.norm_path}': {e_csv}"
                        )

        elif output_format == "dot":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("digraph Dependencies {\n  rankdir=LR;\n")
                f.write(
                    '  node [shape=box, style="filled", fillcolor="#EFEFEF", fontname="Arial"];\n'
                )
                f.write('  edge [fontsize=10, fontname="Arial"];\n\n')

                # Nodes: Use norm_path as unique ID for nodes to handle duplicate key_strings.
                # Label them with key_string and basename.
                for ki_node in key_info_list_for_export:
                    label_path = (
                        os.path.basename(ki_node.norm_path)
                        .replace("\\", "/")
                        .replace('"', '\\"')
                    )
                    label = f"{ki_node.key_string}\\n{label_path}"
                    # Use path as node ID to ensure uniqueness if key_strings are duplicated
                    f.write(f'  "{ki_node.norm_path}" [label="{label}"];\n')
                f.write("\n")

                if len(key_info_list_for_export) != len(
                    grid_rows_compressed_for_export
                ):
                    logger.warning(
                        f"DOT Export: Mismatch definitions/grid rows. DOT graph may be incorrect."
                    )

                for row_idx, compressed_row_str in enumerate(
                    grid_rows_compressed_for_export
                ):
                    if row_idx >= len(key_info_list_for_export):
                        break
                    source_ki = key_info_list_for_export[row_idx]
                    try:
                        decomp_row = decompress(compressed_row_str)
                        expected_cols = (
                            len(grid_key_strings_for_export)
                            if grid_key_strings_for_export
                            else len(key_info_list_for_export)
                        )
                        if len(decomp_row) != expected_cols:
                            continue  # Skip malformed row

                        for col_idx, dep_type in enumerate(decomp_row):
                            if dep_type not in (
                                EMPTY_CHAR,
                                DIAGONAL_CHAR,
                                PLACEHOLDER_CHAR,
                            ):
                                if col_idx >= len(key_info_list_for_export):
                                    break
                                target_ki = key_info_list_for_export[col_idx]
                                color = "black"
                                style = "solid"
                                arrowhead = "normal"
                                if dep_type == ">":
                                    color = "blue"
                                elif dep_type == "<":
                                    color = "green"
                                    arrowhead = "oinv"
                                elif dep_type == "x":
                                    color = "red"
                                    style = "dashed"
                                    arrowhead = "odot"
                                elif dep_type == "d":
                                    color = "orange"
                                elif dep_type == "s":
                                    color = "grey"
                                    style = "dotted"
                                elif dep_type == "S":
                                    color = "dimgrey"
                                    style = "bold"
                                # Use paths for edge definition source/target
                                f.write(
                                    f'  "{source_ki.norm_path}" -> "{target_ki.norm_path}" [label="{dep_type}", color="{color}", style="{style}", arrowhead="{arrowhead}"];\n'
                                )
                    except Exception as e_dot:
                        logger.warning(
                            f"DOT Export: Error processing row for src path '{source_ki.norm_path}': {e_dot}"
                        )
                f.write("}\n")
        else:
            msg = f"Error: Unsupported export format '{output_format}'"
            logger.error(msg)
            return msg
        logger.info(f"Successfully exported tracker to: {output_path}")
        return output_path
    except IOError as e:
        msg = f"Error exporting tracker: I/O Error - {str(e)}"
        logger.error(msg, exc_info=True)
        return msg
    except ImportError as e:
        msg = f"Error exporting tracker: Missing library for format '{output_format}' - {str(e)}"
        logger.error(msg)
        return msg
    except Exception as e:
        msg = f"Error exporting tracker: Unexpected error - {str(e)}"
        logger.exception(msg)
        return msg


# --- End of tracker_io.py ---
