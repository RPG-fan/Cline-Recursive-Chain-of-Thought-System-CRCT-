# utils/tracker_utils.py

import glob
import hashlib
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from cline_utils.dependency_system.core.dependency_grid import (
    DIAGONAL_CHAR,
    EMPTY_CHAR,
    decompress,
)
from cline_utils.dependency_system.core.key_manager import KeyInfo, validate_key

from .cache_manager import cached, normalize_path_cached as normalize_path
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

PathMigrationInfo = Dict[str, Tuple[Optional[str], Optional[str]]]


# --- GLOBAL INSTANCE RESOLUTION HELPERS (Centralized Here) ---
def resolve_key_global_instance_to_ki(
    key_hash_instance_str: str, current_global_path_to_key_info: Dict[str, KeyInfo]
) -> Optional[KeyInfo]:
    """
    Resolves a KEY or KEY#global_instance string to a specific KeyInfo object
    from the provided current_global_path_to_key_info.
    This is now a simple pass-through since the global map is authoritative.
    """
    if not key_hash_instance_str:
        return None

    for ki in current_global_path_to_key_info.values():
        if ki.key_string == key_hash_instance_str:
            return ki
    return None


# (This was moved from project_analyzer.py and made more generic)
# It's placed here because tracker_io will also need it.

# Module-level cache for get_key_global_instance_string to persist across calls within a run
_module_level_base_key_to_sorted_KIs_cache: Dict[str, List[KeyInfo]] = defaultdict(list)


def clear_global_instance_resolution_cache():  # Helper to clear if needed, e.g. for testing or between runs
    """Clears the module-level cache for GI string resolution."""
    _module_level_base_key_to_sorted_KIs_cache.clear()
    logger.debug("TrackerUtils: Cleared module-level GI resolution cache.")


def get_key_global_instance_string(
    ki_obj_to_format: KeyInfo,
    current_global_path_to_key_info: Dict[str, KeyInfo],
    # Optional cache can be passed for specific contexts, otherwise uses module-level
    base_key_to_sorted_KIs_cache: Optional[Dict[str, List[KeyInfo]]] = None,
) -> Optional[str]:
    """
    Returns the persisted KEY or KEY#GI for the given KeyInfo.
    This is now a simple pass-through since the global map is authoritative.
    """
    if not ki_obj_to_format:
        logger.warning(
            "TrackerUtils.GetGlobalInstanceString: Received None for ki_obj_to_format."
        )
        return None
    return ki_obj_to_format.key_string


def get_globally_resolved_key_info_for_cli(
    base_key_str: str,
    user_instance_num: Optional[int],
    global_map: Dict[str, KeyInfo],
    key_role: str,
) -> Optional[KeyInfo]:
    """
    Resolves a key provided via the CLI to a KeyInfo object.
    This is now a simple pass-through since the global map is authoritative.
    """
    key_to_find = (
        f"{base_key_str}#{user_instance_num}" if user_instance_num else base_key_str
    )
    for ki in global_map.values():
        if ki.key_string == key_to_find:
            return ki

    # Handle ambiguity or not found
    matching_infos = [
        info for info in global_map.values() if info.key_string.startswith(base_key_str)
    ]
    if not matching_infos:
        print(
            f"Error: Base {key_role} key '{base_key_str}' not found in global key map."
        )
        return None
    if len(matching_infos) > 1 and not user_instance_num:
        print(
            f"Error: {key_role.capitalize()} key '{base_key_str}' is globally ambiguous. Please specify which instance you mean using '#<num>':"
        )
        for i, ki in enumerate(sorted(matching_infos, key=lambda k: k.norm_path)):
            print(
                f"  [{i+1}] {ki.key_string} (Path: {ki.norm_path})  (Use as '{ki.key_string}')"
            )
        return None
    return None  # Should be found in the loop above


# --- END OF GLOBAL INSTANCE RESOLUTION HELPERS ---

# --- PARSING HELPERS (Updated for KEY#GI) ---
KEY_GI_PATTERN_PART = r"[a-zA-Z0-9]+(?:#[0-9]+)?"  # Capture KEY or KEY#num


def read_key_definitions_from_lines(lines: List[str]) -> List[Tuple[str, str]]:
    """Reads key definitions from lines. Returns a list of (key_string, path_string) tuples."""
    key_path_pairs: List[Tuple[str, str]] = []
    in_section = False
    key_def_start_pattern = re.compile(r"^---KEY_DEFINITIONS_START---$", re.IGNORECASE)
    key_def_end_pattern = re.compile(r"^---KEY_DEFINITIONS_END---$", re.IGNORECASE)
    # Regex now includes optional #instance part
    definition_pattern = re.compile(rf"^({KEY_GI_PATTERN_PART})\s*:\s*(.*)$")

    for line in lines:
        if key_def_end_pattern.match(line.strip()):
            break
        if in_section:
            line_content = line.strip()
            if not line_content or line_content.lower().startswith("key definitions:"):
                continue
            match = definition_pattern.match(line_content)  # Use updated pattern
            if match:
                k_gi, v_path = match.groups()  # k_gi is now the full KEY#GI or KEY
                # validate_key already handles KEY#GI format
                if validate_key(k_gi):
                    key_path_pairs.append((k_gi, normalize_path(v_path.strip())))
                else:  # Should be caught by regex, but as fallback
                    logger.warning(
                        f"TrackerUtils.ReadDefinitions: Skipping invalid key format '{k_gi}'."
                    )
            # else: logger.debug(f"ReadDefs: Line did not match key def pattern: '{line_content}'")
        elif key_def_start_pattern.match(line.strip()):
            in_section = True
    return key_path_pairs


def read_grid_from_lines(lines: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Reads grid from lines. Returns: (grid_column_header_key_strings, list_of_grid_rows)
    where list_of_grid_rows is List[(row_key_string_label, compressed_row_data_string)]
    """
    grid_column_header_keys_gi: List[str] = []  # Will store KEY or KEY#GI
    grid_rows_data_gi: List[Tuple[str, str]] = []  # (KEY or KEY#GI, compressed_data)
    in_section = False
    grid_start_pattern = re.compile(r"^---GRID_START---$", re.IGNORECASE)
    grid_end_pattern = re.compile(r"^---GRID_END---$", re.IGNORECASE)
    # Regex for row labels now includes optional #instance part
    row_label_pattern = re.compile(rf"^({KEY_GI_PATTERN_PART})\s*=\s*(.*)$")

    for line in lines:
        if grid_end_pattern.match(line.strip()):
            break
        if in_section:
            line_content = line.strip()
            if line_content.upper().startswith("X "):
                # Split header, keys can now be KEY or KEY#GI
                potential_keys = line_content.split()[1:]
                grid_column_header_keys_gi = [
                    k for k in potential_keys if validate_key(k)
                ]
                if len(grid_column_header_keys_gi) != len(potential_keys):
                    logger.warning(
                        f"TrackerUtils.ReadGrid: Some X-header keys are invalid and were skipped."
                    )
                continue
            if not line_content or line_content == "X":
                continue

            match = row_label_pattern.match(line_content)  # Use updated pattern
            if match:
                k_label_gi, v_data = match.groups()  # k_label_gi is KEY or KEY#GI
                if validate_key(k_label_gi):
                    grid_rows_data_gi.append((k_label_gi, v_data.strip()))
                else:  # Should be caught by regex
                    logger.warning(
                        f"TrackerUtils.ReadGrid: Skipping row with invalid key label format '{k_label_gi}'."
                    )
            # else: logger.debug(f"ReadGrid: Line did not match row data pattern: '{line_content}'")
        elif grid_start_pattern.match(line.strip()):
            in_section = True

    # Consistency check in read_tracker_file_structured will compare with definitions count
    return grid_column_header_keys_gi, grid_rows_data_gi


# --- END OF PARSING HELPERS ---


@cached(
    "tracker_data_structured",
    key_func=lambda tracker_path: f"tracker_data_structured:{normalize_path(tracker_path)}:{(os.path.getmtime(tracker_path) if os.path.exists(tracker_path) else 0)}",
    track_path_args=[0],
)
def read_tracker_file_structured(tracker_path: str) -> Dict[str, Any]:
    """
    Read a tracker file and parse its contents into list-based structures
    compatible with the new format (handles duplicate key strings).
    Args:
        tracker_path: Path to the tracker file
    Returns:
        Dictionary with "definitions_ordered": List[Tuple[str,str]],
                         "grid_headers_ordered": List[str],
                         "grid_rows_ordered": List[Tuple[str,str]], (row_label, compressed_data)
                         "last_key_edit": str, "last_grid_edit": str
        or empty structure on failure.
    """
    tracker_path = normalize_path(tracker_path)
    # Initialize with empty lists for the new structure
    empty_result = {
        "definitions_ordered": [],
        "grid_headers_ordered": [],
        "grid_rows_ordered": [],
        "last_key_edit": "",
        "last_grid_edit": "",
    }
    if not os.path.exists(tracker_path):
        logger.debug(
            f"Tracker file not found: {tracker_path}. Returning empty structured data."
        )
        return empty_result
    try:
        with open(tracker_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Use the helpers now defined in this file
        definitions = read_key_definitions_from_lines(lines)
        grid_headers, grid_rows = read_grid_from_lines(lines)
        content_str = "".join(lines)
        last_key_edit_match = re.search(
            r"^last_KEY_edit\s*:\s*(.*)$", content_str, re.MULTILINE | re.IGNORECASE
        )
        last_key_edit = (
            last_key_edit_match.group(1).strip() if last_key_edit_match else ""
        )
        last_grid_edit_match = re.search(
            r"^last_GRID_edit\s*:\s*(.*)$", content_str, re.MULTILINE | re.IGNORECASE
        )
        last_grid_edit = (
            last_grid_edit_match.group(1).strip() if last_grid_edit_match else ""
        )

        # Basic consistency check based on what was read from file directly
        if (
            definitions
            and grid_headers
            and grid_rows
            and not (len(definitions) == len(grid_headers) == len(grid_rows))
        ):
            logger.warning(
                f"ReadStructured: Inconsistent counts in '{os.path.basename(tracker_path)}'. Defs: {len(definitions)}, Headers: {len(grid_headers)}, Rows: {len(grid_rows)}. Data might be misaligned."
            )
        elif (
            definitions
            and grid_rows
            and not grid_headers
            and len(definitions) == len(grid_rows)
        ):
            logger.debug(
                f"ReadStructured: Grid headers missing but defs and rows match for '{os.path.basename(tracker_path)}'. Imputing headers from defs."
            )
            grid_headers = [d[0] for d in definitions]

        logger.debug(
            f"Read structured tracker '{os.path.basename(tracker_path)}': "
            f"{len(definitions)} defs, {len(grid_headers)} grid headers, {len(grid_rows)} grid rows."
        )

        return {
            "definitions_ordered": definitions,
            "grid_headers_ordered": grid_headers,
            "grid_rows_ordered": grid_rows,
            "last_key_edit": last_key_edit,
            "last_grid_edit": last_grid_edit,
        }
    except Exception as e:
        logger.exception(f"Error reading structured tracker file {tracker_path}: {e}")
        return empty_result


def find_all_tracker_paths(config: ConfigManager, project_root: str) -> Set[str]:
    """Finds all main, doc, and mini tracker files in the project."""
    all_tracker_paths = set()
    memory_dir_rel = config.get_path("memory_dir")
    if not memory_dir_rel:
        logger.warning("memory_dir not configured. Cannot find main/doc trackers.")
        memory_dir_abs = None
    else:
        memory_dir_abs = normalize_path(os.path.join(project_root, memory_dir_rel))
        logger.debug(
            f"Path Components: project_root='{project_root}', memory_dir_rel='{memory_dir_rel}', calculated memory_dir_abs='{memory_dir_abs}'"
        )

        # Main Tracker
        main_tracker_abs = config.get_path(
            "main_tracker_filename",
            os.path.join(memory_dir_abs, "module_relationship_tracker.md"),
        )
        logger.debug(
            f"Using main_tracker_abs from config (or default): '{main_tracker_abs}'"
        )
        if os.path.exists(main_tracker_abs):
            all_tracker_paths.add(main_tracker_abs)
        else:
            logger.debug(f"Main tracker not found at: {main_tracker_abs}")

        # Doc Tracker
        doc_tracker_abs = config.get_path(
            "doc_tracker_filename", os.path.join(memory_dir_abs, "doc_tracker.md")
        )
        logger.debug(
            f"Using doc_tracker_abs from config (or default): '{doc_tracker_abs}'"
        )
        if os.path.exists(doc_tracker_abs):
            all_tracker_paths.add(doc_tracker_abs)
        else:
            logger.debug(f"Doc tracker not found at: {doc_tracker_abs}")

    # Mini Trackers
    code_roots_rel = config.get_code_root_directories()
    if not code_roots_rel:
        logger.warning(
            "No code_root_directories configured. Cannot find mini trackers."
        )
    else:
        for code_root_rel in code_roots_rel:
            code_root_abs = normalize_path(os.path.join(project_root, code_root_rel))
            mini_tracker_pattern = os.path.join(code_root_abs, "**", "*_module.md")
            try:
                found_mini_trackers = glob.glob(mini_tracker_pattern, recursive=True)
                normalized_mini_paths = {
                    normalize_path(mt_path) for mt_path in found_mini_trackers
                }
                all_tracker_paths.update(normalized_mini_paths)
                logger.debug(
                    f"Found {len(normalized_mini_paths)} mini trackers under '{code_root_rel}'."
                )
            except Exception as e:
                logger.error(
                    f"Error during glob search for mini trackers under '{code_root_abs}': {e}"
                )
    logger.debug(f"Found {len(all_tracker_paths)} total tracker files.")
    return all_tracker_paths


def get_global_map_cache_key_part(global_map: Dict[str, Any]) -> str:
    """Creates a stable hashable key part from a dictionary."""
    # Hashing sorted keys is stable and avoids hashing unhashable values.
    # This assumes that the set of paths is the most important changing factor.
    if not global_map:
        return "empty"
    return hashlib.sha256("".join(sorted(global_map.keys())).encode()).hexdigest()


# --- MODIFIED AGGREGATION FUNCTION (Uses KEY#global_instance) ---
@cached(
    "aggregation_v2_gi",
    # MODIFIED key_func - show_progress doesn't affect cache, so it's accepted but not included in key
    key_func=lambda paths, pmi, cgptki, show_progress=True: f"agg_v2_gi:{':'.join(sorted(list(paths)))}:{hash(tuple(sorted(pmi.items())))}:{get_global_map_cache_key_part(cgptki)}",
    ttl=300,
)
def aggregate_all_dependencies(
    tracker_paths: Set[str],
    path_migration_info: PathMigrationInfo,
    current_global_path_to_key_info: Dict[str, KeyInfo],  # NEW PARAMETER
    show_progress: bool = True,
) -> Dict[
    Tuple[str, str], Tuple[str, Set[str]]
]:  # Output keys are Tuple[src_KEY#GI, tgt_KEY#GI]
    """
    Aggregates dependencies from tracker files, resolving paths to current global KeyInfo objects
    and then to their KEY#global_instance strings for instance-specific aggregation.
    """
    aggregated_links: Dict[Tuple[str, str], Tuple[str, Set[str]]] = (
        {}
    )  # Key: (src_KEY#GI, tgt_KEY#GI)
    config = ConfigManager()
    get_priority_from_char = config.get_char_priority

    logger.debug(
        f"Aggregating dependencies (outputting KEY#global_instance) from {len(tracker_paths)} trackers..."
    )

    # Parallelize per-tracker aggregation to increase throughput
    from cline_utils.dependency_system.utils.batch_processor import BatchProcessor

    def _aggregate_single_tracker(
        tracker_file_path: str,
    ) -> Dict[Tuple[str, str], Tuple[str, Set[str]]]:
        local_links: Dict[Tuple[str, str], Tuple[str, Set[str]]] = {}
        logger.debug(
            f"Aggregation: Processing tracker {os.path.basename(tracker_file_path)}"
        )
        tracker_data = read_tracker_file_structured(tracker_file_path)

        definitions_ordered_from_file = tracker_data["definitions_ordered"]
        grid_headers_from_file = tracker_data["grid_headers_ordered"]
        grid_rows_from_file = tracker_data["grid_rows_ordered"]

        if not definitions_ordered_from_file or not grid_rows_from_file:
            logger.debug(
                f"Aggregation: Skipping empty/incomplete data in: {os.path.basename(tracker_file_path)}"
            )
            return local_links

        # Build ordered KIs for this tracker
        effective_ki_list_for_this_tracker: List[Optional[KeyInfo]] = []
        for _key_in_file, path_in_file in definitions_ordered_from_file:
            mig_info = path_migration_info.get(path_in_file)
            resolved_ki_for_this_def_entry: Optional[KeyInfo] = None
            if mig_info and mig_info[1]:  # has a current global base key
                new_global_base_key = mig_info[1]
                resolved_ki_for_this_def_entry = next(
                    (
                        ki
                        for ki in current_global_path_to_key_info.values()
                        if ki.key_string == new_global_base_key
                        and ki.norm_path == path_in_file
                    ),
                    None,
                ) or next(
                    (
                        ki
                        for ki in current_global_path_to_key_info.values()
                        if ki.key_string == new_global_base_key
                    ),
                    None,
                )
            effective_ki_list_for_this_tracker.append(resolved_ki_for_this_def_entry)

        if not (
            len(effective_ki_list_for_this_tracker) == len(grid_headers_from_file)
            and len(effective_ki_list_for_this_tracker) == len(grid_rows_from_file)
        ):
            logger.warning(
                f"Aggregation: Tracker '{os.path.basename(tracker_file_path)}' has inconsistent structure after global validation. "
                f"Effective KIs: {len(effective_ki_list_for_this_tracker)}, File Headers: {len(grid_headers_from_file)}, File Rows: {len(grid_rows_from_file)}. "
                "Skipping this tracker."
            )
            return local_links

        for row_idx, (_row_label_in_file, compressed_row_str) in enumerate(
            grid_rows_from_file
        ):
            source_ki_global = effective_ki_list_for_this_tracker[row_idx]
            if not source_ki_global:
                continue

            source_key_gi_str = get_key_global_instance_string(
                source_ki_global, current_global_path_to_key_info
            )
            if not source_key_gi_str:
                logger.warning(
                    f"Aggregation: Could not get global instance for source path {source_ki_global.norm_path} from {os.path.basename(tracker_file_path)}. Skipping row."
                )
                continue

            try:
                decompressed_row_chars = decompress(compressed_row_str)
                if len(decompressed_row_chars) != len(
                    effective_ki_list_for_this_tracker
                ):
                    logger.warning(
                        f"Aggregation: Row {row_idx} (source KI: {source_key_gi_str}) in {os.path.basename(tracker_file_path)} "
                        f"has decompressed length {len(decompressed_row_chars)}, expected {len(effective_ki_list_for_this_tracker)}. Skipping row."
                    )
                    continue

                for col_idx, dep_char_val in enumerate(decompressed_row_chars):
                    if dep_char_val == DIAGONAL_CHAR or dep_char_val == EMPTY_CHAR:
                        continue

                    target_ki_global = effective_ki_list_for_this_tracker[col_idx]
                    if not target_ki_global:
                        continue

                    if source_ki_global.norm_path == target_ki_global.norm_path:
                        continue

                    target_key_gi_str = get_key_global_instance_string(
                        target_ki_global, current_global_path_to_key_info
                    )
                    if not target_key_gi_str:
                        logger.warning(
                            f"Aggregation: Could not get global instance for target path {target_ki_global.norm_path} from {os.path.basename(tracker_file_path)}. Skipping cell."
                        )
                        continue

                    link = (source_key_gi_str, target_key_gi_str)
                    existing_char, existing_origins = local_links.get(
                        link, (None, set())
                    )
                    try:
                        current_priority = get_priority_from_char(dep_char_val)
                        existing_priority = (
                            get_priority_from_char(existing_char)
                            if existing_char
                            else -1
                        )
                    except KeyError:
                        logger.warning(
                            f"Aggregation: Invalid dep char '{dep_char_val}' in {os.path.basename(tracker_file_path)}. Skipping {link}."
                        )
                        continue

                    if current_priority > existing_priority:
                        # Ensure type is Tuple[str, Set[str]]
                        local_links[link] = (str(dep_char_val), {tracker_file_path})
                    elif current_priority == existing_priority:
                        if existing_char is not None and dep_char_val == existing_char:
                            existing_origins.add(tracker_file_path)
                            local_links[link] = (
                                str(existing_char),
                                set(existing_origins),
                            )
                        elif existing_char == "n":
                            # keep existing 'n'
                            pass
                        elif dep_char_val == "n":
                            local_links[link] = ("n", {tracker_file_path})
                        else:
                            local_links[link] = (str(dep_char_val), {tracker_file_path})
                            logger.debug(
                                f"Aggregation conflict (same priority): {link} was '{existing_char}', overwritten by '{dep_char_val}' from {os.path.basename(tracker_file_path)}."
                            )
            except Exception as e_agg_row:
                logger.warning(
                    f"Aggregation: Error processing row {row_idx} for source KI {source_key_gi_str} in {os.path.basename(tracker_file_path)}: {e_agg_row}",
                    exc_info=False,
                )
        return local_links

    # Run per-tracker aggregation in parallel
    tracker_list = list(tracker_paths)
    processor = BatchProcessor(
        show_progress=show_progress, phase_name="Aggregating Dependencies"
    )
    per_tracker_results = processor.process_items(
        tracker_list, _aggregate_single_tracker
    )

    # Merge results
    for local_links in per_tracker_results:
        for link, (char_val, origins) in local_links.items():
            # Normalize existing entry types explicitly and ensure we never assign None
            existing_entry = aggregated_links.get(link)

            # Current char is always a concrete str from local_links
            current_char: str = str(char_val)

            if existing_entry is None:
                existing_char_str: str = ""  # sentinel for "no existing"
                existing_origins: Set[str] = set()
                existing_priority = -1
            else:
                e_char, e_origins = existing_entry  # e_char: str, e_origins: Set[str]
                existing_char_str = e_char
                existing_origins = set(e_origins)
                try:
                    existing_priority = get_priority_from_char(existing_char_str)
                except KeyError:
                    existing_priority = -1

            try:
                current_priority = get_priority_from_char(current_char)
            except KeyError:
                continue

            if current_priority > existing_priority:
                aggregated_links[link] = (str(current_char), set(origins))
            elif current_priority == existing_priority:
                if existing_char_str != "" and existing_char_str == current_char:
                    merged: Set[str] = existing_origins.union(set(origins))
                    aggregated_links[link] = (str(current_char), merged)
                elif existing_char_str == "n":
                    # keep existing 'n'
                    pass
                elif current_char == "n":
                    aggregated_links[link] = ("n", set(origins))
                else:
                    aggregated_links[link] = (str(current_char), set(origins))

    logger.debug(
        f"Aggregation complete. Found {len(aggregated_links)} unique KEY#global_instance directed links."
    )
    return aggregated_links


# --- End of tracker_utils.py ---
