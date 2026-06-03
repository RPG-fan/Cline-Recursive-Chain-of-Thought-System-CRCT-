# key_manager.py

"""
Core module for key management.
Handles key generation, validation, and sorting based on a hierarchical,
contextual model with tier promotion for nested subdirectories.
Persists the global key map to a file and maintains the previous version.
"""

import json  # Added for saving/loading map
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union, cast

from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import (
    get_project_root,
    normalize_path,
)

logger = logging.getLogger(__name__)

# Constants
ASCII_A_UPPER = 65  # ASCII value for 'A'
ASCII_Z_UPPER = 90  # ASCII value for 'Z'
ASCII_A_LOWER = 97  # ASCII value for 'a'
ASCII_Z_LOWER = 122  # ASCII value for 'z'

# Updated pattern to allow multi-digit tiers and file numbers,
# and structure Tier + Dir + [Subdir + [File] | File]
HIERARCHICAL_KEY_PATTERN = r"[1-9]\d*[A-Z](?:[a-z](?:[1-9]\d*)?|[1-9]\d*)?"
HIERARCHICAL_KEY_PATTERN_INSTANCE = (
    r"(?:#[1-9][0-9]*)?"  # Optional: # followed by a non-zero digit, then any digits
)
KEY_VALIDATION_PATTERN = re.compile(
    f"{HIERARCHICAL_KEY_PATTERN}({HIERARCHICAL_KEY_PATTERN_INSTANCE})$"
)

# Pattern for splitting keys into sortable parts (numbers and non-numbers)
KEY_PATTERN = r"\d+|\D+"
GLOBAL_KEY_MAP_FILENAME = "global_key_map.json"
OLD_GLOBAL_KEY_MAP_FILENAME = "global_key_map_old.json"  # <<< NEW
TRACKER_MAP_FILENAME = "tracker_map.json"
MAX_TOP_LEVEL_ROOTS = 26  # Tier-1 directory letters A–Z


def _rotate_global_key_map_atomically(
    current_map_path: str,
    old_map_path: str,
    *,
    max_retries: int = 5,
    base_delay: float = 0.05,
) -> None:
    """
    Atomically rotate the current global key map file to the old-map filename.

    Uses os.replace (not shutil.move) and retries on transient Windows file locks.
    """
    last_err: Optional[OSError] = None
    for attempt in range(max_retries):
        try:
            os.replace(current_map_path, old_map_path)
            return
        except OSError as err:
            last_err = err
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"Key map rotation retry {attempt + 1}/{max_retries} "
                    f"('{current_map_path}' -> '{old_map_path}'): {err}. "
                    f"Retrying in {delay}s."
                )
                time.sleep(delay)
    if last_err is not None:
        raise last_err


class KeyGenerationError(ValueError):
    """Custom exception for key generation failures."""

    pass


class KeyInfo(NamedTuple):
    """Stores information about a generated key and its context."""

    key_string: str  # The generated key string (e.g., "1A", "1Aa", "2Ab")
    norm_path: str  # The normalized absolute path this key represents
    parent_path: Optional[
        str
    ]  # Normalized path of the parent directory containing this item
    tier: int  # The tier number used in this key_string
    is_directory: bool  # True if the key represents a directory


# Moved from dependency_analyzer to break circular dependency (if applicable)
# Or keep it here if it's fundamental to key logic
def get_file_type_for_key(file_path: str) -> str:
    """
    Determines the file type based on its extension.
    Simplified version for key management purposes.

    Args:
        file_path: The path to the file.
    Returns:
        The file type as a string (e.g., "py", "js", "md", "generic").
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".py":
        return "py"
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        return "js"
    elif ext in (".md", ".rst"):
        return "md"
    elif ext in (".html", ".htm"):
        return "html"
    elif ext == ".css":
        return "css"
    else:
        return "generic"


def _strip_instance_suffix(key_str: str) -> str:
    """Return the base key without any #instance suffix."""
    if not key_str:
        return key_str
    return key_str.split("#", 1)[0]


def _apply_global_instance_suffixes(
    path_to_key_info: Dict[str, KeyInfo], old_map: Optional[Dict[str, KeyInfo]] = None
) -> Dict[str, KeyInfo]:
    """
    Ensure that for any base key that appears more than once globally,
    each KeyInfo.key_string is suffixed with a stable '#<n>' instance number.
    If only one occurrence exists for a base key, it remains unsuffixed.
    Instance numbers are stabilized using the old_map whenever possible.
    """
    if not path_to_key_info:
        return path_to_key_info

    # Build base_key -> [KeyInfo] buckets
    base_key_to_kis: Dict[str, List[KeyInfo]] = defaultdict(list)
    for ki in path_to_key_info.values():
        base_key = _strip_instance_suffix(ki.key_string)
        base_key_to_kis[base_key].append(ki)

    # Build previous path -> instance number map if old_map provided
    prev_instance_by_path: Dict[str, int] = {}
    if old_map:
        for old_path, old_ki in old_map.items():
            inst = _parse_instance_suffix(old_ki.key_string)
            if inst is not None and inst > 0:
                prev_instance_by_path[normalize_path(old_path)] = inst

    # We will create updated KeyInfos only where necessary
    updated_map: Dict[str, KeyInfo] = {}
    for norm_path, ki in path_to_key_info.items():
        updated_map[norm_path] = ki

    # Process each base key bucket
    for base_key, kis in base_key_to_kis.items():
        # Unique base key: ensure no suffix
        if len(kis) == 1:
            ki = kis[0]
            if "#" in ki.key_string:
                # Remove suffix if it exists
                new_ki = KeyInfo(
                    base_key, ki.norm_path, ki.parent_path, ki.tier, ki.is_directory
                )
                updated_map[ki.norm_path] = new_ki
            continue

        # Duplicated base key: assign instance numbers
        # Use stable ordering: sort by norm_path
        kis_sorted = sorted(kis, key=lambda k: k.norm_path)

        # First, try to reuse previous instance numbers
        assigned_instances: Dict[str, int] = {}  # path -> instance
        used_numbers: Set[int] = set()
        for ki in kis_sorted:
            prev_inst = prev_instance_by_path.get(ki.norm_path)
            if (
                prev_inst is not None
                and prev_inst > 0
                and prev_inst not in used_numbers
            ):
                assigned_instances[ki.norm_path] = prev_inst
                used_numbers.add(prev_inst)

        # Then assign remaining, using the next available positive integers
        next_num = 1
        for ki in kis_sorted:
            if ki.norm_path in assigned_instances:
                continue
            while next_num in used_numbers:
                next_num += 1
            assigned_instances[ki.norm_path] = next_num
            used_numbers.add(next_num)
            next_num += 1

        # Update KeyInfos with suffixes
        for ki in kis_sorted:
            inst = assigned_instances.get(ki.norm_path, None)
            if inst is None:
                continue
            desired_key = f"{base_key}#{inst}"
            if ki.key_string != desired_key:
                new_ki = KeyInfo(
                    desired_key, ki.norm_path, ki.parent_path, ki.tier, ki.is_directory
                )
                updated_map[ki.norm_path] = new_ki

    return updated_map


def _parse_instance_suffix(key_str: str) -> Optional[int]:
    """Return the instance number if present, else None."""
    if not key_str or "#" not in key_str:
        return None
    try:
        return int(key_str.split("#", 1)[1])
    except (ValueError, IndexError):
        return None


def generate_keys(
    root_paths: List[str],
    excluded_dirs: Optional[Set[str]] = None,
    excluded_extensions: Optional[Set[str]] = None,
    precomputed_excluded_paths: Optional[Set[str]] = None,
) -> Tuple[Dict[str, KeyInfo], List[KeyInfo]]:
    """
    Generate hierarchical, contextual keys for files and directories.
    Implements tier promotion (resetting dir letter to 'A') for nested subdirectories.
    Saves the resulting key map to a file alongside this script.
    Manages current and previous global key map files.
    Uses ConfigManager for default exclusions if specific arguments are not provided.

    Args:
        root_paths: List of root directory paths to process.
        excluded_dirs: Optional set of directory names to exclude. If None, uses config.
        excluded_extensions: Optional set of file extensions to exclude. If None, uses config.
        precomputed_excluded_paths: Optional set of pre-calculated absolute paths to exclude.

    Returns:
        Tuple containing:
        - Dictionary mapping normalized paths to KeyInfo objects.
        - List of newly generated KeyInfo objects (unique).

    Raises:
        FileNotFoundError: If a root path does not exist.
        KeyGenerationError: If key generation rules are violated (e.g., >26 subdirs
            or >26 top-level root paths).
    """
    if isinstance(root_paths, str):
        root_paths = [root_paths]

    # Resolve to absolute normalized paths to enable correct nesting detection
    resolved_roots = []
    for rp in root_paths:
        abs_path = os.path.abspath(rp)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"Root path '{rp}' (resolved to '{abs_path}') does not exist."
            )
        resolved_roots.append(normalize_path(abs_path))

    # Deduplicate while preserving order or sorted for consistency
    unique_roots = sorted(list(set(resolved_roots)))
    root_paths = unique_roots

    if len(root_paths) > MAX_TOP_LEVEL_ROOTS:
        raise KeyGenerationError(
            f"Key generation failed: at most {MAX_TOP_LEVEL_ROOTS} top-level root paths "
            f"are supported (letters A-Z), but {len(root_paths)} were provided."
        )

    config_manager = ConfigManager()
    excluded_dirs_names = (
        set(excluded_dirs) if excluded_dirs else config_manager.get_excluded_dirs()
    )
    excluded_extensions = (
        set(excluded_extensions)
        if excluded_extensions
        else set(config_manager.get_excluded_extensions() or [])
    )
    project_root = get_project_root()
    absolute_excluded_dirs = {
        normalize_path(os.path.join(project_root, d)) for d in excluded_dirs_names
    }

    if precomputed_excluded_paths is not None:
        exclusion_set_for_processing = precomputed_excluded_paths.union(
            absolute_excluded_dirs
        )
    else:
        calculated_excluded_paths_list = config_manager.get_excluded_paths()
        exclusion_set_for_processing = set(calculated_excluded_paths_list).union(
            absolute_excluded_dirs
        )

    path_to_key_info: Dict[str, KeyInfo] = {}  # Maps norm_path -> KeyInfo
    newly_generated_keys: List[KeyInfo] = []  # Tracks newly assigned KeyInfo objects
    top_level_dir_count = 0  # Counter for assigning 'A', 'B', ... at Tier 1

    def parse_key(
        key_string: Optional[str],
    ) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """Helper to parse a key into tier, dir, subdir components."""
        if not key_string or not validate_key(key_string):
            return None, None, None
        # Match keys with subdir first (e.g., 1Aa, 1Aa12)
        match = re.match(r"^([1-9]\d*)([A-Z])([a-z])", key_string)
        if match:
            tier_str, dir_letter, subdir_letter = match.groups()
            return int(tier_str), dir_letter, subdir_letter
        # Handle case like "1A", "1A1" (no subdir letter)
        match = re.match(r"^([1-9]\d*)([A-Z])", key_string)
        if match:
            tier_str, dir_letter = match.groups()
            return int(tier_str), dir_letter, None
        logger.warning(f"Could not parse valid key: {key_string}")
        return None, None, None

    def process_directory(
        dir_path: str, exclusion_set: Set[str], parent_info: Optional[KeyInfo]
    ):
        """Recursively processes directories and files, generating contextual keys."""
        nonlocal path_to_key_info, newly_generated_keys, top_level_dir_count

        try:
            norm_dir_path = normalize_path(dir_path)

            # 1. Skip excluded directories
            if any(norm_dir_path.startswith(ex_path) for ex_path in exclusion_set):
                # logger.debug(
                #     f"Exclusion Check 1: Skipping excluded dir path: '{norm_dir_path}'"
                # )
                return
            # else: # No need for else, debug log below covers processing
            #     logger.debug(f"Exclusion Check 1: Processing dir path: '{norm_dir_path}'")

            # --- Assign key to the current directory being processed ---
            current_dir_key_info: Optional[KeyInfo] = None
            if parent_info is None:  # This is a top-level directory from root_paths
                if top_level_dir_count > MAX_TOP_LEVEL_ROOTS - 1:
                    raise KeyGenerationError(
                        f"Key generation failed: exceeded maximum top-level directories "
                        f"({MAX_TOP_LEVEL_ROOTS}, letters A-Z)."
                    )
                dir_letter = chr(ASCII_A_UPPER + top_level_dir_count)
                key_str = f"1{dir_letter}"
                current_tier = 1
                # Parent path is None for top-level roots
                current_dir_key_info = KeyInfo(
                    key_str, norm_dir_path, None, current_tier, True
                )
                top_level_dir_count += 1
                # Store immediately so it's available if needed later in this call
                if norm_dir_path not in path_to_key_info:
                    path_to_key_info[norm_dir_path] = current_dir_key_info
                    newly_generated_keys.append(current_dir_key_info)
                    # logger.debug(
                    #     f"Assigned key '{current_dir_key_info.key_string}' to directory '{norm_dir_path}'"
                    # )
                else:  # Should not happen for top-level if processed correctly
                    logger.warning(
                        f"Top-level directory '{norm_dir_path}' seems to be processed more than once."
                    )
                    current_dir_key_info = path_to_key_info[
                        norm_dir_path
                    ]  # Use existing

            else:  # This is a subdirectory, its key was generated when processing its parent
                current_dir_key_info = path_to_key_info.get(norm_dir_path)
                if not current_dir_key_info:
                    # This indicates a potential logic flaw or race condition if multithreaded (which it isn't here)
                    logger.error(
                        f"CRITICAL LOGIC ERROR: KeyInfo not found for directory '{norm_dir_path}' which should have been generated by its parent '{parent_info.norm_path if parent_info else 'None'}'. Halting."
                    )
                    # Raising an error might be better than just returning
                    raise KeyGenerationError(
                        f"KeyInfo missing for supposedly processed directory: {norm_dir_path}"
                    )
                    # return # Cannot proceed without parent context for children

            # --- Process items within this directory ---
            try:
                items = sorted([entry.name for entry in os.scandir(dir_path)])
            except OSError as e:
                logger.error(f"Error accessing directory '{dir_path}': {e}")
                return

            # --- Initialize counters for THIS level ---
            file_counter = 1  # For files (1A1, 1Ba1, 2A1...)
            subdir_letter_ord = ASCII_A_LOWER  # For direct subdirs (1Ba, 1Bb...)
            promoted_dir_ord = (
                ASCII_A_UPPER  # <<< For promoted dirs (2A, 2B...) at this level >>>
            )

            # Determine if the current directory key represents a subdirectory level
            # This is the condition that triggers promotion for its children.
            parent_key_string = (
                current_dir_key_info.key_string if current_dir_key_info else None
            )
            # Check uses regex matching Tier + Upper + Lower pattern (no file number allowed here)
            is_parent_key_a_subdir = bool(
                parent_key_string
                and re.match(r"^[1-9]\d*[A-Z][a-z]$", parent_key_string)
            )

            # logger.debug(
            #     f"Processing items in: '{norm_dir_path}' (Key: {parent_key_string}, Is Subdir Key: {is_parent_key_a_subdir})"
            # )

            for item_name in items:
                try:
                    item_path = os.path.join(dir_path, item_name)
                    norm_item_path = normalize_path(item_path)
                    is_dir = os.path.isdir(item_path)
                    is_file = os.path.isfile(item_path)

                    # Apply standard exclusions (name, type, extension, etc.)
                    if any(
                        norm_item_path.startswith(ex_path) for ex_path in exclusion_set
                    ):  # Check again for items potentially matching deeper patterns
                        # logger.debug(
                        #     f"Exclusion Check 1b: Skipping excluded item path: '{norm_item_path}'"
                        # )
                        continue
                    if (item_name in excluded_dirs_names) or (item_name == ".gitkeep"):
                        # logger.debug(
                        #     f"Exclusion Check 3: Skipping item name '{item_name}' in '{norm_dir_path}'"
                        # )
                        continue
                    if item_name.endswith("_module.md"):
                        # logger.debug(
                        #     f"Exclusion Check 4: Skipping mini-tracker '{item_name}' in '{norm_dir_path}'"
                        # )
                        continue

                    # Skip items that are neither file nor directory
                    if not (is_dir or is_file):
                        # logger.debug(
                        #     f"Skipping item '{item_name}' (not a file or directory) in '{norm_dir_path}'"
                        # )
                        continue

                    # Check extension exclusion only for files
                    if is_file:
                        _, ext = os.path.splitext(item_name)
                        if ext in excluded_extensions:
                            # logger.debug(
                            #     f"Exclusion Check 5: Skipping file '{item_name}' with extension '{ext}' in '{norm_dir_path}'"
                            # )
                            continue

                    # --- Key Generation Logic ---
                    item_key_info: Optional[KeyInfo] = None

                    # Determine parent context
                    parent_key_string = (
                        current_dir_key_info.key_string
                        if current_dir_key_info
                        else None
                    )
                    needs_promotion = is_parent_key_a_subdir and is_dir

                    if needs_promotion:
                        # --- Tier Promotion (Triggered ONLY by Sub-Subdirectory) ---
                        # This block now only executes for directories found inside a keyed subdirectory.
                        if (
                            not parent_key_string
                        ):  # Should be impossible if needs_promotion is True
                            logger.error(
                                f"Logic Error: Promotion needed but parent key string is missing for item '{item_name}'"
                            )
                            continue  # Skip this item

                        parsed_parent_tier, _, _ = parse_key(parent_key_string)
                        if parsed_parent_tier is None:
                            logger.error(
                                f"Logic Error: Could not parse parent key '{parent_key_string}' during promotion for DIR item '{item_name}'"
                            )
                            continue  # Skip this item

                        new_tier = parsed_parent_tier + 1

                        # Check limit BEFORE assigning character (only applies to dirs here)
                        if subdir_letter_ord > ASCII_Z_LOWER:
                            error_msg = (
                                f"Key generation failed: Exceeded maximum supported subdirectories (26, 'a'-'z') requiring promotion "
                                f"within parent directory key '{parent_key_string}' (path: '{norm_dir_path}'). "
                                f"Problematic item: '{item_name}' (path: '{norm_item_path}'). "
                                f"Please reduce the number of direct subdirectories needing keys at this level "
                                f"or add problematic paths to exclusions in '.clinerules.config.json' "
                                f"(e.g., using 'cline-config --add-excluded-path \"{norm_item_path}\"')."
                            )
                            logger.critical(error_msg)
                            raise KeyGenerationError(error_msg)  # Terminate generation

                        if promoted_dir_ord > ASCII_Z_UPPER:
                            error_msg = (
                                f"Key generation failed: Exceeded maximum promoted directories (26, 'A'-'Z') at tier {new_tier} "
                                f"within parent key '{parent_key_string}' (path: '{norm_dir_path}'). Item: '{item_name}'."
                            )
                            logger.critical(error_msg)
                            raise KeyGenerationError(error_msg)

                        new_dir_letter = chr(promoted_dir_ord)
                        promoted_dir_ord += (
                            1  # Increment for the *next* promoted dir at this level
                        )

                        # <<< CORRECTED: Key for the promoted directory itself (e.g., 2A) >>>
                        key_str = f"{new_tier}{new_dir_letter}"

                        # logger.debug(
                        #     f"Promoting DIR '{item_name}': parent '{parent_key_string}' -> new key '{key_str}'"
                        # )

                        # is_dir is always True in this block now
                        item_key_info = KeyInfo(
                            key_str, norm_item_path, norm_dir_path, new_tier, True
                        )

                    else:  # No Promotion (Item is a file, OR Item is a directory whose parent is NOT a subdir)
                        # --- Standard Key Assignment ---
                        # Handles:
                        # - Files in root dirs (e.g., 1B1)
                        # - Files in subdirs (e.g., 1Ba1) <<-- This was the case causing premature promotion before
                        # - Dirs in root dirs (e.g., 1Ba)
                        if (
                            not current_dir_key_info
                        ):  # Should only happen for initial root calls
                            logger.error(
                                f"Logic Error: Missing current_dir_key_info for non-promotion case of item '{item_name}'"
                            )
                            continue  # Skip this item

                        base_key_part = (
                            current_dir_key_info.key_string
                        )  # e.g., "1A" or "1Ba"
                        current_tier = current_dir_key_info.tier

                        if is_dir:  # Assign standard subdirectory key (e.g., 1Bb, 1Bc)
                            # Check limit BEFORE assigning character
                            if subdir_letter_ord > ASCII_Z_LOWER:
                                error_msg = (
                                    f"Key generation failed: Exceeded maximum supported subdirectories (26, 'a'-'z') "
                                    f"within parent directory key '{base_key_part}' (path: '{norm_dir_path}'). "
                                    f"Problematic item: '{item_name}' (path: '{norm_item_path}'). "
                                    f"Please reduce the number of direct subdirectories needing keys at this level "
                                    f"or add problematic paths to exclusions in '.clinerules.config.json' "
                                    f"(e.g., using 'cline-config --add-excluded-path \"{norm_item_path}\"')."
                                )
                                logger.critical(error_msg)
                                raise KeyGenerationError(
                                    error_msg
                                )  # Terminate generation

                            subdir_letter = chr(subdir_letter_ord)
                            key_str = f"{base_key_part}{subdir_letter}"
                            subdir_letter_ord += 1
                            # logger.debug(
                            #     f"Assigning standard subdir key '{key_str}' for DIR item '{item_name}' under parent '{base_key_part}'"
                            # )
                        else:  # is_file: Assign standard file key (e.g., 1B1, 1Ba1, 1Ba2)
                            key_str = f"{base_key_part}{file_counter}"
                            file_counter += 1
                            # logger.debug(
                            #     f"Assigning standard file key '{key_str}' for FILE item '{item_name}' under parent '{base_key_part}'"
                            # )

                        # is_dir correctly reflects the item type here
                        item_key_info = KeyInfo(
                            key_str, norm_item_path, norm_dir_path, current_tier, is_dir
                        )

                    # --- Validate, Store Key and Recurse ---
                    if item_key_info:
                        if validate_key(item_key_info.key_string):
                            if norm_item_path in path_to_key_info:
                                # This might happen if a directory is listed in root_paths AND is also a subdirectory of another root_path
                                logger.warning(
                                    f"Path '{norm_item_path}' already has an assigned key '{path_to_key_info[norm_item_path].key_string}'. Overwriting with new key '{item_key_info.key_string}'. Check root_paths/exclusions if unexpected."
                                )
                            path_to_key_info[norm_item_path] = item_key_info
                            newly_generated_keys.append(item_key_info)
                            if is_dir:
                                # Pass the newly generated info for this item as the parent for the recursive call
                                process_directory(
                                    item_path, exclusion_set, item_key_info
                                )
                        else:
                            # This should ideally not happen if generation logic and limits are correct
                            logger.error(
                                f"Generated key '{item_key_info.key_string}' for path '{norm_item_path}' is invalid according to pattern '{HIERARCHICAL_KEY_PATTERN}'. Skipping item and its children."
                            )
                            # Consider raising error here too, as it indicates a logic flaw.
                            # raise KeyGenerationError(f"Invalid key generated: {item_key_info.key_string}")

                except Exception as item_err:
                    # Catch errors processing a specific item but continue with others in the directory
                    logger.error(
                        f"Error processing item '{item_name}' in directory '{dir_path}': {item_err}",
                        exc_info=True,
                    )
                    # Optionally re-raise if certain errors should halt everything:
                    if isinstance(item_err, KeyGenerationError):
                        raise item_err

        except KeyGenerationError:
            raise  # Propagate critical errors
        except Exception as dir_err:
            logger.error(
                f"Failed to process directory '{dir_path}': {dir_err}", exc_info=True
            )
            # Decide whether to halt or continue with other root paths
            # For now, let it propagate if not KeyGenerationError
            if not isinstance(
                dir_err, FileNotFoundError
            ):  # Allow skipping non-existent roots handled earlier
                raise dir_err

    # --- Main Loop ---
    # Sort root_paths so top-level letter assignments (A, B, C...) are
    # deterministic regardless of the order the caller supplies them.
    for root_path in sorted(root_paths):
        # Build specific exclusion set for this root path to avoid walking into other root paths
        # that are strictly nested within it.
        root_exclusions = set(exclusion_set_for_processing)
        for other_root in root_paths:
            if other_root != root_path:
                if other_root.startswith(root_path + "/"):
                    root_exclusions.add(other_root)
        process_directory(root_path, root_exclusions, parent_info=None)

    # Ensure the returned list contains unique KeyInfo objects (in case of reprocessing/overlaps)
    # Using dict.fromkeys preserves order (Python 3.7+) and ensures uniqueness based on KeyInfo equality
    # Note: KeyInfo is a tuple, so equality works as expected.
    #
    # NOTE: GI disambiguation is done once in the save block below via _apply_global_instance_suffixes.
    # A former inline first-pass block was removed because it duplicated that logic while loading
    # a stale old map (before the rename) and competed with the authoritative pass below.

    # --- Apply GLOBAL INSTANCE disambiguation and save ---
    current_map_path: str = ""
    try:
        # Get the directory where this script (key_manager.py) resides
        script_dir = os.path.dirname(os.path.abspath(__file__))
        from cline_utils.dependency_system.core import resolve_state_path

        current_map_path = normalize_path(
            resolve_state_path(GLOBAL_KEY_MAP_FILENAME, script_dir)
        )
        old_map_path = normalize_path(
            resolve_state_path(OLD_GLOBAL_KEY_MAP_FILENAME, script_dir)
        )
        os.makedirs(
            os.path.join(script_dir, "state"), exist_ok=True
        )  # Ensure directory exists

        # Step 1: Load the PREVIOUS map BEFORE renaming so we read the right file.
        # (os.replace overwrites old_map_path, so reading after the rename would
        # give us the map from two runs ago rather than the immediately prior run.)
        previous_map: Optional[Dict[str, KeyInfo]] = None
        try:
            previous_map = load_old_global_key_map()
        except Exception as _e_load_old:
            previous_map = None
            logger.warning(
                f"Could not load previous global key map for GI stabilization: {_e_load_old}"
            )

        # Step 2: Apply GI suffixes for duplicated base keys, reusing prior assignments.
        # This is the single authoritative call; the return value replaces the in-memory map.
        path_to_key_info = _apply_global_instance_suffixes(
            path_to_key_info, previous_map
        )

        # Step 3: Rotate the on-disk maps: current → old.
        if os.path.exists(current_map_path):
            try:
                _rotate_global_key_map_atomically(current_map_path, old_map_path)
                logger.info(
                    f"Renamed existing '{GLOBAL_KEY_MAP_FILENAME}' to '{OLD_GLOBAL_KEY_MAP_FILENAME}'."
                )
            except OSError as rename_err:
                logger.error(
                    f"Failed to rename '{current_map_path}' to '{old_map_path}': {rename_err}. Proceeding to save new map."
                )
        else:
            logger.info(f"No existing '{GLOBAL_KEY_MAP_FILENAME}' found to rename.")

        # Step 4: Save the newly generated map to the current filename.
        serializable_map = {
            path: info._asdict() for path, info in path_to_key_info.items()
        }
        with open(current_map_path, "w", encoding="utf-8") as f:
            json.dump(serializable_map, f, indent=2)
        logger.info(f"Successfully saved new global key map to: {current_map_path}")
    except IOError as e:
        logger.error(
            f"I/O Error saving global key map to {current_map_path}: {e}", exc_info=True
        )
        raise KeyGenerationError(f"Failed to save global key map: {e}") from e
    except Exception as e:
        logger.exception(
            f"Unexpected error saving global key map to {current_map_path}: {e}"
        )
        raise KeyGenerationError(f"Failed to save global key map: {e}") from e

    unique_new_keys = list(dict.fromkeys(newly_generated_keys).keys())
    return path_to_key_info, unique_new_keys


def load_global_key_map() -> Optional[Dict[str, KeyInfo]]:
    """
    Loads the persisted global path_to_key_info map from the JSON file
    located alongside key_manager.py.

    Returns:
        The loaded dictionary mapping normalized paths to KeyInfo objects,
        or None if the file doesn't exist or fails to load/parse.
    """
    # Predeclare for type checkers
    map_path: str = ""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        map_path = normalize_path(
            os.path.join(script_dir, "state", GLOBAL_KEY_MAP_FILENAME)
        )

        if not os.path.exists(map_path):
            logger.error(
                f"Global key map file not found at {map_path}. Run project analysis ('analyze-project') first."
            )
            return None

        with open(map_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        # Convert dictionary data back into KeyInfo objects
        path_to_key_info: Dict[str, KeyInfo] = {}
        for path, info_dict in loaded_data.items():
            try:
                path_to_key_info[path] = KeyInfo(**info_dict)
            except TypeError as te:
                logger.error(
                    f"Error converting loaded data to KeyInfo for path '{path}'. Data: {info_dict}. Error: {te}"
                )
                # Skip this entry or return None entirely? For now, skip.
                continue  # Skip this entry

        logger.debug(
            f"Successfully loaded global key map ({len(path_to_key_info)} entries) from: {map_path}"
        )
        return path_to_key_info

    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding JSON from global key map file {map_path}: {e}",
            exc_info=True,
        )
        return None
    except IOError as e:
        logger.error(
            f"I/O Error loading global key map file {map_path}: {e}", exc_info=True
        )
        return None
    except Exception as e:
        logger.exception(
            f"Unexpected error loading global key map from {map_path}: {e}"
        )
        return None


def load_old_global_key_map() -> Optional[Dict[str, KeyInfo]]:
    """Loads the persisted PREVIOUS global path_to_key_info map."""
    # Predeclare for type checkers
    map_path: str = ""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        map_path = normalize_path(
            os.path.join(script_dir, "state", OLD_GLOBAL_KEY_MAP_FILENAME)
        )  # Target old map
        if not os.path.exists(map_path):
            logger.warning(
                f"Previous global key map file not found: {map_path}. This may be the first run."
            )
            return None  # Return None gracefully if old map doesn't exist
        with open(map_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        path_to_key_info: Dict[str, KeyInfo] = {}
        for path, info_dict in loaded_data.items():
            try:
                path_to_key_info[path] = KeyInfo(**info_dict)
            except TypeError as te:
                logger.error(f"Error converting OLD KeyInfo data for '{path}': {te}")
                continue
        logger.debug(
            f"Loaded previous global key map ({len(path_to_key_info)} entries) from: {map_path}"
        )
        return path_to_key_info
    except Exception as e:
        logger.exception(
            f"Unexpected error loading previous global key map: {e} (path: {map_path})"
        )
        return None


def load_tracker_map() -> List[str]:
    """
    Loads the persisted tracker map from the JSON file
    located alongside key_manager.py.

    Returns:
        List of absolute tracker paths, or empty list if not found/failed.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        from cline_utils.dependency_system.core import resolve_state_path

        path = normalize_path(resolve_state_path(TRACKER_MAP_FILENAME, script_dir))
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    data_list = cast(List[object], data)
                    result: List[str] = []
                    for item in data_list:
                        if isinstance(item, str):
                            result.append(item)
                    return result
                logger.warning(
                    f"Tracker map data in {path} is not a list. Returning empty."
                )
    except Exception as e:
        logger.warning(f"Could not load tracker map: {e}")
    return []


def save_tracker_map(tracker_paths: List[str]):
    """
    Saves the provided list of tracker paths to the JSON file
    located alongside key_manager.py.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        from cline_utils.dependency_system.core import resolve_state_path

        path = normalize_path(resolve_state_path(TRACKER_MAP_FILENAME, script_dir))
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(set(tracker_paths))), f, indent=2)
        logger.info(f"Successfully saved tracker map to: {path}")
    except Exception as e:
        logger.error(f"Could not save tracker map: {e}")


def validate_key(key: str) -> bool:
    """
    Validate if a key follows the hierarchical key format, optionally followed by '#' and digits (for instance).
    Args:
        key: The hierarchical key to validate
    Returns:
        True if valid, False otherwise
    """
    if not key:
        return False  # Handle empty strings
    return bool(re.match(KEY_VALIDATION_PATTERN, key))


def get_path_from_key(
    key_string: str,
    path_to_key_info: Dict[str, KeyInfo],
    context_path: Optional[str] = None,
) -> Optional[str]:
    """
    Get the file/directory path corresponding to a hierarchical key string,
    potentially using context. (Requires careful use due to non-unique keys).

    Args:
        key_string: The hierarchical key string.
        path_to_key_info: Dictionary mapping normalized paths to KeyInfo objects.
        context_path: Optional normalized path of the directory where the key is being referenced.
                      Used to resolve ambiguity if multiple paths share the same key string.

    Returns:
        The file/directory path or None if key not found or ambiguous without context.
    """
    matching_infos = [
        info for info in path_to_key_info.values() if info.key_string == key_string
    ]

    if not matching_infos:
        logger.debug(f"Key string '{key_string}' not found in path_to_key_info map.")
        return None
    if len(matching_infos) == 1:
        return matching_infos[0].norm_path

    # Ambiguous key - requires context
    if context_path:
        norm_context_path = normalize_path(context_path)
        # Look for a match whose parent_path matches the context_path
        for info in matching_infos:
            # Ensure parent_path is not None before comparison
            if (
                info.parent_path
                and normalize_path(info.parent_path) == norm_context_path
            ):
                logger.debug(
                    f"Resolved ambiguous key '{key_string}' using context '{norm_context_path}' to path '{info.norm_path}'"
                )
                return info.norm_path

        # If no direct child match, could add more complex logic (e.g., check grandparents) if needed.
        logger.warning(
            f"Ambiguous key '{key_string}' found. Context path '{norm_context_path}' provided, but no direct child match found among {len(matching_infos)} possibilities: {[i.norm_path for i in matching_infos]}."
        )
        return None  # Or raise? Or return list? For now, return None.
    else:
        logger.warning(
            f"Ambiguous key '{key_string}' found. Multiple paths match: {[i.norm_path for i in matching_infos]}. Provide context_path for disambiguation."
        )
        return None  # Or raise? Or return list? For now, return None.


def get_key_from_path(path: str, path_to_key_info: Dict[str, KeyInfo]) -> Optional[str]:
    """
    Get the hierarchical key string corresponding to a file/directory path.

    Args:
        path: The file/directory path.
        path_to_key_info: Dictionary mapping normalized paths to KeyInfo objects.

    Returns:
        The hierarchical key string or None if path not found.
    """
    norm_path = normalize_path(path)
    info = path_to_key_info.get(norm_path)
    return info.key_string if info else None


def get_sortable_parts_for_key(key_str: str) -> List[Union[str, int]]:
    """
    Single source of truth for key sort ordering.

    Splits a key string into alternating int/str tokens for natural-sort
    comparison.  Because the tier is always the leading numeric token
    (e.g. ``1`` in ``1Ab3``), sorting by these tokens already sorts by
    tier first, then directory letter, then subdir letter, then file
    number — no separate tier field is ever required.

    Args:
        key_str: A key string such as ``1Ab3`` or ``2A#1``.

    Returns:
        A list of ``int`` and ``str`` tokens suitable as a ``sorted()`` key.
    """
    if not key_str:
        return []
    parts = re.findall(KEY_PATTERN, key_str)
    try:
        converted_parts: List[Union[str, int]] = [
            (int(p) if p.isdigit() else p) for p in parts
        ]
    except (ValueError, TypeError):
        logger.warning(
            f"Could not convert parts for sorting key string '{key_str}', using original string parts."
        )
        converted_parts = parts  # Fallback to string parts
    return converted_parts


def sort_key_strings_hierarchically(keys: List[str]) -> List[str]:
    """
    Sorts a list of key strings hierarchically (natural sort order).
    e.g., 1A1, 1A2, 1A10 instead of 1A1, 1A10, 1A2.

    Delegates to :func:`get_sortable_parts_for_key` — the single source of
    truth for ordering logic.  ``None`` and empty-string entries are silently
    dropped.

    Args:
        keys: A list of key strings.

    Returns:
        A new list containing the valid key strings in sorted order.
    """
    valid_keys = [k for k in keys if k]
    return sorted(valid_keys, key=get_sortable_parts_for_key)


def regenerate_keys(
    root_paths: List[str],
    excluded_dirs: Optional[Set[str]] = None,
    excluded_extensions: Optional[Set[str]] = None,
    precomputed_excluded_paths: Optional[Set[str]] = None,
) -> Tuple[Dict[str, KeyInfo], List[KeyInfo]]:
    """
    Regenerates keys for the given root paths using the new contextual logic.

    Args:
        root_paths: List of root directory paths to process.
        excluded_dirs: Optional set of directory names to exclude.
        excluded_extensions: Optional set of file extensions to exclude.
        precomputed_excluded_paths: Optional set of pre-calculated absolute paths to exclude.

    Returns:
        Tuple containing:
        - Dictionary mapping normalized paths to KeyInfo objects.
        - List of newly generated KeyInfo objects.

    Raises:
        FileNotFoundError: If a root path does not exist.
        KeyGenerationError: If key generation rules are violated (e.g., >26 subdirs).
    """
    # Simply call the main generation function
    return generate_keys(
        root_paths, excluded_dirs, excluded_extensions, precomputed_excluded_paths
    )


def build_keymap_indexes(global_map: Dict[str, KeyInfo]) -> Dict[str, Any]:
    """
    Builds derived indexes from the global key map for fast descendant and directory lookups.
    """
    children_by_parent: Dict[str, List[str]] = defaultdict(list)
    is_dir_map: Dict[str, bool] = {}
    path_meta: Dict[str, KeyInfo] = {}

    for path, info in global_map.items():
        is_dir_map[path] = info.is_directory
        path_meta[path] = info
        if info.parent_path:
            children_by_parent[info.parent_path].append(path)

    # Compute recursive file descendants
    file_descendants_by_dir: Dict[str, Tuple[str, ...]] = {}

    def get_file_descendants(dir_path: str) -> List[str]:
        if dir_path in file_descendants_by_dir:
            return list(file_descendants_by_dir[dir_path])

        descendants: List[str] = []
        for child in children_by_parent.get(dir_path, []):
            if is_dir_map.get(child):
                descendants.extend(get_file_descendants(child))
            else:
                descendants.append(child)

        res: Tuple[str, ...] = tuple(sorted(list(set(descendants))))
        file_descendants_by_dir[dir_path] = res
        return list(res)

    for path, info in global_map.items():
        if info.is_directory:
            get_file_descendants(path)

    # Compute ancestor chains
    ancestor_chain: Dict[str, Tuple[str, ...]] = {}
    for path, info in global_map.items():
        chain: List[str] = []
        curr: Optional[str] = info.parent_path
        while curr:
            chain.append(curr)
            curr_info = global_map.get(curr)
            curr = curr_info.parent_path if curr_info else None
        ancestor_chain[path] = tuple(chain)

    return {
        "is_dir_map": is_dir_map,
        "path_meta": path_meta,
        "children_by_parent": dict(children_by_parent),
        "file_descendants_by_dir": file_descendants_by_dir,
        "ancestor_chain": ancestor_chain,
    }


def get_keymap_indexes() -> Dict[str, Any]:
    """
    Returns derived indexes for the global key map.
    """
    global_map = load_global_key_map()
    if not global_map:
        return {
            "is_dir_map": {},
            "path_meta": {},
            "children_by_parent": {},
            "file_descendants_by_dir": {},
            "ancestor_chain": {},
        }
    return build_keymap_indexes(global_map)
