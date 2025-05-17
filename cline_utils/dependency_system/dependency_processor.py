# dependency_processor.py

"""
Main entry point for the dependency tracking system.
Processes command-line arguments and delegates to appropriate handlers.
"""

import argparse
from collections import defaultdict
import json
import logging
import os
import sys
import re
import glob
from typing import Dict, List, Tuple, Any, Optional, Set

# --- Core Imports ---
from cline_utils.dependency_system.core.dependency_grid import (
    PLACEHOLDER_CHAR, compress, decompress, get_char_at, set_char_at,
    add_dependency_to_grid, get_dependencies_from_grid
)
from cline_utils.dependency_system.core.key_manager import (
    KeyInfo, KeyGenerationError, load_old_global_key_map, validate_key, sort_key_strings_hierarchically,
    load_global_key_map
)

# --- IO Imports ---
from cline_utils.dependency_system.io.tracker_io import (
    PathMigrationInfo, _build_path_migration_map, remove_key_from_tracker, merge_trackers, write_tracker_file,
    export_tracker, update_tracker
)

# --- Analysis Imports ---
from cline_utils.dependency_system.analysis.project_analyzer import analyze_project
from cline_utils.dependency_system.analysis.dependency_analyzer import analyze_file

# --- Utility Imports ---
from cline_utils.dependency_system.utils.path_utils import (
    get_project_root, normalize_path
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.cache_manager import (
    clear_all_caches, file_modified, invalidate_dependent_entries
)
from cline_utils.dependency_system.utils.tracker_utils import (
    read_tracker_file, find_all_tracker_paths, aggregate_all_dependencies
)

from cline_utils.dependency_system.utils.template_generator import add_code_doc_dependency_to_checklist, _get_item_type as get_item_type_for_checklist
from cline_utils.dependency_system.utils.visualize_dependencies import generate_mermaid_diagram # Renamed for clarity

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
KEY_DEFINITIONS_START_MARKER = "---KEY_DEFINITIONS_START---"
KEY_DEFINITIONS_END_MARKER = "---KEY_DEFINITIONS_END---"

# --- Helper Functions ---
def _load_global_map_or_exit(cli_args: Optional[argparse.Namespace] = None) -> Dict[str, KeyInfo]:
    """Loads the global key map, exiting if it fails.
    Uses explicit path from cli_args if provided.
    """
    logger.info("Loading global key map...")
    key_map_path_override = None
    if cli_args and hasattr(cli_args, 'global_key_map_path') and cli_args.global_key_map_path:
        key_map_path_override = normalize_path(cli_args.global_key_map_path)
        logger.info(f"Using explicit global key map path from CLI: {key_map_path_override}")

    path_to_key_info = load_global_key_map(key_map_path_override=key_map_path_override)
    if path_to_key_info is None:
        print("Error: Global key map not found or failed to load.", file=sys.stderr)
        print("Please run 'analyze-project' first to generate the key map.", file=sys.stderr)
        logger.critical("Global key map missing or invalid. Exiting.")
        sys.exit(1)
    logger.info("Global key map loaded successfully.")
    return path_to_key_info

def is_parent_child(key1_str: str, key2_str: str, global_map: Dict[str, KeyInfo]) -> bool:
    """Checks if two keys represent a direct parent-child directory relationship."""
    info1 = next((info for path, info in global_map.items() if info.key_string == key1_str), None)
    info2 = next((info for path, info in global_map.items() if info.key_string == key2_str), None)

    if not info1 or not info2:
        logger.debug(f"is_parent_child: Could not find KeyInfo for '{key1_str if not info1 else ''}' or '{key2_str if not info2 else ''}'. Returning False.")
        return False # Cannot determine relationship if info is missing

    # Ensure paths are normalized (they should be from KeyInfo, but double-check)
    path1 = normalize_path(info1.norm_path)
    path2 = normalize_path(info2.norm_path)
    parent1 = normalize_path(info1.parent_path) if info1.parent_path else None
    parent2 = normalize_path(info2.parent_path) if info2.parent_path else None

    # Check both directions: info1 is parent of info2 OR info2 is parent of info1
    is_parent1 = parent2 == path1
    is_parent2 = parent1 == path2

    logger.debug(f"is_parent_child check: {key1_str}({path1}) vs {key2_str}({path2}). Is Parent1: {is_parent1}, Is Parent2: {is_parent2}")
    return is_parent1 or is_parent2

# --- Command Handlers ---

def command_handler_analyze_file(args):
    """Handle the analyze-file command."""
    import json
    try:
        if not os.path.exists(args.file_path): print(f"Error: File not found: {args.file_path}"); return 1
        results = analyze_file(args.file_path)
        if args.output:
            output_dir = os.path.dirname(args.output); os.makedirs(output_dir, exist_ok=True) if output_dir else None
            with open(args.output, 'w', encoding='utf-8') as f: json.dump(results, f, indent=2)
            print(f"Analysis results saved to {args.output}")
        else: print(json.dumps(results, indent=2))
        return 0
    except Exception as e: print(f"Error analyzing file: {str(e)}"); return 1

def command_handler_analyze_project(args):
    """Handle the analyze-project command."""
    import json
    try:
        if not args.project_root: args.project_root = "."; logger.info(f"Defaulting project root to CWD: {os.path.abspath(args.project_root)}")
        abs_project_root = normalize_path(os.path.abspath(args.project_root))
        if not os.path.isdir(abs_project_root): print(f"Error: Project directory not found: {abs_project_root}"); return 1
        original_cwd = os.getcwd()
        if abs_project_root != normalize_path(original_cwd):
             logger.info(f"Changing CWD to: {abs_project_root}"); os.chdir(abs_project_root)
             # ConfigManager.initialize(force=True) # Re-init if CWD matters for config finding

        logger.debug(f"Analyzing project: {abs_project_root}, force_analysis={args.force_analysis}, force_embeddings={args.force_embeddings}")
        results = analyze_project(force_analysis=args.force_analysis, force_embeddings=args.force_embeddings)
        logger.debug(f"All Suggestions before Tracker Update: {results.get('dependency_suggestion', {}).get('suggestions')}")

        if args.output:
            output_path_abs = normalize_path(os.path.abspath(args.output))
            output_dir = os.path.dirname(output_path_abs); os.makedirs(output_dir, exist_ok=True) if output_dir else None
            with open(output_path_abs, 'w', encoding='utf-8') as f: json.dump(results, f, indent=2)
            print(f"Analysis results saved to {output_path_abs}")
        else: print("Analysis complete. Results not printed (use --output to save).")
        return 0
    except Exception as e:
        logger.error(f"Error analyzing project: {str(e)}", exc_info=True); print(f"Error analyzing project: {str(e)}"); return 1
    finally:
        if 'original_cwd' in locals() and normalize_path(os.getcwd()) != normalize_path(original_cwd):
             logger.info(f"Changing CWD back to: {original_cwd}"); os.chdir(original_cwd)
             # ConfigManager.initialize(force=True) # Re-init if needed

def handle_compress(args: argparse.Namespace) -> int:
    """Handle the compress command."""
    try: result = compress(args.string); print(f"Compressed string: {result}"); return 0
    except Exception as e: logger.error(f"Error compressing: {e}"); print(f"Error: {e}"); return 1

def handle_decompress(args: argparse.Namespace) -> int:
    """Handle the decompress command."""
    try: result = decompress(args.string); print(f"Decompressed string: {result}"); return 0
    except Exception as e: logger.error(f"Error decompressing: {e}"); print(f"Error: {e}"); return 1

def handle_get_char(args: argparse.Namespace) -> int:
    """Handle the get_char command."""
    try: result = get_char_at(args.string, args.index); print(f"Character at index {args.index}: {result}"); return 0
    except IndexError: logger.error("Index out of range"); print("Error: Index out of range"); return 1
    except Exception as e: logger.error(f"Error get_char: {e}"); print(f"Error: {e}"); return 1

def handle_set_char(args: argparse.Namespace) -> int:
    """Handle the set_char command."""
    try:
        tracker_data = read_tracker_file(args.tracker_file)
        if not tracker_data or not tracker_data.get("keys"): print(f"Error: Could not read tracker: {args.tracker_file}"); return 1
        if args.key not in tracker_data["keys"]: print(f"Error: Key {args.key} not found"); return 1
        sorted_keys = sort_key_strings_hierarchically(list(tracker_data["keys"].keys())) # Use hierarchical sort
        # Check index validity against sorted list length
        if not (0 <= args.index < len(sorted_keys)): print(f"Error: Index {args.index} out of range for {len(sorted_keys)} keys."); return 1
        # Check char validity (optional, could rely on set_char_at)
        # if len(args.char) != 1: print("Error: Character must be a single character."); return 1

        row_str = tracker_data["grid"].get(args.key, "")
        if not row_str: logger.warning(f"Row for key '{args.key}' missing. Initializing."); row_str = compress('p' * len(sorted_keys))

        updated_compressed_row = set_char_at(row_str, args.index, args.char)
        tracker_data["grid"][args.key] = updated_compressed_row
        tracker_data["last_grid_edit"] = f"Set {args.key}[{args.index}] to {args.char}"
        success = write_tracker_file(args.tracker_file, tracker_data["keys"], tracker_data["grid"], tracker_data.get("last_key_edit", ""), tracker_data["last_grid_edit"])
        if success: print(f"Set char at index {args.index} to '{args.char}' for key {args.key} in {args.tracker_file}"); return 0
        else: print(f"Error: Failed write to {args.tracker_file}"); return 1
    except IndexError: print(f"Error: Index {args.index} out of range."); return 1
    except ValueError as e: print(f"Error: {e}"); return 1
    except Exception as e: logger.error(f"Error set_char: {e}", exc_info=True); print(f"Error: {e}"); return 1

def handle_remove_key(args: argparse.Namespace) -> int:
    """Handle the remove-key command."""
    try:
        # Call the updated tracker_io function
        remove_key_from_tracker(args.tracker_file, args.key)
        print(f"Removed key '{args.key}' from tracker '{args.tracker_file}'")
        # Invalidate cache for the modified tracker
        invalidate_dependent_entries('tracker_data', f"tracker_data:{normalize_path(args.tracker_file)}:.*")
        # Broader invalidation might be needed if other caches depend on grid structure
        invalidate_dependent_entries('grid_decompress', '.*'); invalidate_dependent_entries('grid_validation', '.*'); invalidate_dependent_entries('grid_dependencies', '.*')
        return 0
    except FileNotFoundError as e: print(f"Error: {e}"); return 1
    except ValueError as e: print(f"Error: {e}"); return 1 # e.g., key not found in tracker
    except Exception as e:
        logger.error(f"Failed to remove key: {str(e)}", exc_info=True); print(f"Error: {e}"); return 1

def handle_add_dependency(args: argparse.Namespace) -> int:
    """Handle the add-dependency command. Allows adding foreign keys to mini-trackers."""
    tracker_path = normalize_path(args.tracker)
    is_mini_tracker = tracker_path.endswith("_module.md")
    source_key_str = args.source_key # Renamed for clarity
    target_keys_str_list = args.target_key # Renamed for clarity
    dep_type = args.dep_type
    # Dependency type validation
    ALLOWED_DEP_TYPES = {'<', '>', 'x', 'd', 'o', 'n', 'p', 's', 'S'}
    logger.info(f"Attempting to add dependency: {source_key_str} -> {target_keys_str_list} ({dep_type}) in tracker {tracker_path}")

    # --- Basic Validation ---
    if dep_type not in ALLOWED_DEP_TYPES:
        print(f"Error: Invalid dep type '{dep_type}'. Allowed: {', '.join(sorted(list(ALLOWED_DEP_TYPES)))}")
        return 1

    # --- Load Local Tracker Data ---
    tracker_data = read_tracker_file(tracker_path)
    local_keys_map = {} # key_string -> path_string
    if tracker_data and tracker_data.get("keys"):
        local_keys_map = tracker_data.get("keys", {})
    elif not is_mini_tracker: # For main/doc, tracker must exist with keys
        print(f"Error: Could not read tracker or no keys found: {tracker_path}")
        return 1
    # If it's a mini-tracker and doesn't exist, update_tracker will handle creation.

    # --- Load global map and config (needed for item type and checklist) ---
    # Load ONCE and use this instance throughout the function.
    loaded_global_path_to_key_info = _load_global_map_or_exit(args) 
    if loaded_global_path_to_key_info is None: # Safeguard, though _load_global_map_or_exit should exit
        logger.critical("handle_add_dependency: global_path_to_key_info is None after _load_global_map_or_exit.")
        print("Critical Error: Global key map could not be loaded. Aborting add-dependency.", file=sys.stderr)
        return 1
        
    config = ConfigManager()
    project_root = get_project_root() # Needed by get_item_type_for_checklist

    # Get KeyInfo for source using the loaded map
    source_key_info = next((info for info in loaded_global_path_to_key_info.values() if info.key_string == source_key_str), None)
    if not source_key_info:
        print(f"Error: Source key '{source_key_str}' not found in global key map.")
        return 1
    source_item_type = get_item_type_for_checklist(source_key_info.norm_path, config, project_root)

    # --- Process Target Keys ---
    internal_changes = [] # List of (target_key, dep_type) for local targets
    foreign_adds = []     # List of (target_key, dep_type) for valid foreign targets (mini-trackers only)
    checklist_updates_pending = [] # Store (source_key, target_key, dep_type) for checklist

    for target_key_str in target_keys_str_list:
        if target_key_str == source_key_str:
             logger.warning(f"Skipping self-dependency: {source_key_str} -> {target_key_str}")
             continue

        # Use the 'loaded_global_path_to_key_info' which holds the actual map
        target_key_info = next((info for info in loaded_global_path_to_key_info.values() if info.key_string == target_key_str), None)
        if not target_key_info:
            print(f"Error: Target key '{target_key_str}' not found in global key map. Cannot add dependency.")
            return 1 
        
        target_item_type = get_item_type_for_checklist(target_key_info.norm_path, config, project_root)

        # Check if this is a code-doc or doc-code link
        if (source_item_type == "code" and target_item_type == "doc") or \
           (source_item_type == "doc" and target_item_type == "code"):
            checklist_updates_pending.append((source_key_str, target_key_str, dep_type))
            logger.info(f"Identified code-doc link for checklist: {source_key_str} ({source_item_type}) -> {target_key_str} ({target_item_type})")


        if target_key_str in local_keys_map:
            internal_changes.append((target_key_str, dep_type))
            logger.debug(f"Target '{target_key_str}' found locally. Queued for internal update.")
        elif is_mini_tracker:
            # No need to reload or check global_map_loaded anymore.
            # 'loaded_global_path_to_key_info' is already available and verified.
            # The target_key_info check above already confirms its global existence.
            foreign_adds.append((target_key_str, dep_type))
            logger.info(f"Target '{target_key_str}' is valid globally. Queued for foreign key addition to mini-tracker.")
        else:
            print(f"Error: Target key '{target_key_str}' not found in tracker '{tracker_path}' (and it's not a mini-tracker allowing foreign key addition).")
            return 1

    # --- Combine all changes ---
    all_targets_for_source = internal_changes + foreign_adds
    if not all_targets_for_source and not checklist_updates_pending: # Check if anything to do for tracker OR checklist
        print("No valid dependencies specified or identified to apply to tracker or checklist.")
        return 0

    suggestions_for_update = {source_key_str: all_targets_for_source} if all_targets_for_source else None # Pass None if list is empty
    
    # Generate file_to_module map using the correctly loaded global map
    file_to_module_map = {
        info.norm_path: info.parent_path
        for info in loaded_global_path_to_key_info.values() # Use the loaded map
        if not info.is_directory and info.parent_path
    }

    # --- Call update_tracker for ALL cases (mini, main, doc) ---
    # Let update_tracker handle type determination, reading, writing, content preservation
    try:
        if suggestions_for_update: # Only call update_tracker if there are tracker changes
            logger.info(f"Calling update_tracker for {tracker_path} (Force Apply: True)")
            update_tracker(
                output_file_suggestion=tracker_path,
                path_to_key_info=loaded_global_path_to_key_info, # Pass the loaded map
                tracker_type="mini" if is_mini_tracker else ("doc" if "doc_tracker.md" in tracker_path else "main"),
                suggestions=suggestions_for_update,
                file_to_module=file_to_module_map,
                new_keys=None,
                force_apply_suggestions=True 
            )
            print(f"Successfully updated tracker {tracker_path} with dependencies.")
        else:
            logger.info(f"No direct tracker updates needed for {tracker_path}, but proceeding with checklist updates if any.")

        # After successfully updating the tracker (or if no tracker update was needed but checklist is), update checklist
        if checklist_updates_pending:
            logger.info(f"Attempting to update checklist with {len(checklist_updates_pending)} code-doc dependencies.")
            all_checklist_adds_successful = True
            for src_k, tgt_k, dep_t in checklist_updates_pending:
                if not add_code_doc_dependency_to_checklist(src_k, tgt_k, dep_t):
                    all_checklist_adds_successful = False
                    logger.error(f"Failed to add {src_k} -> {tgt_k} ({dep_t}) to checklist.")
                else:
                    print(f"Added dependency {src_k} -> {tgt_k} ({dep_t}) to review checklist.")
            if not all_checklist_adds_successful:
                print("Warning: Some code-doc dependencies could not be added to the review checklist.")
                # Optionally, change exit code or overall status if this is critical
        
        return 0
    except Exception as e:
        logger.error(f"Error during add-dependency processing for {tracker_path}: {e}", exc_info=True)
        print(f"Error processing add-dependency for {tracker_path}: {e}")
        return 1

def handle_merge_trackers(args: argparse.Namespace) -> int:
    """Handle the merge-trackers command."""
    try:
        primary_tracker_path = normalize_path(args.primary_tracker_path); secondary_tracker_path = normalize_path(args.secondary_tracker_path)
        output_path = normalize_path(args.output) if args.output else primary_tracker_path
        merged_data = merge_trackers(primary_tracker_path, secondary_tracker_path, output_path)
        if merged_data: print(f"Merged trackers into {output_path}. Total keys: {len(merged_data.get('keys', {}))}"); return 0
        else: print("Error merging trackers."); return 1
    except Exception as e: logger.exception(f"Failed merge: {e}"); print(f"Error: {e}"); return 1

def handle_clear_caches(args: argparse.Namespace) -> int:
    """Handle the clear-caches command."""
    try:
        clear_all_caches()
        print("All caches cleared.")
        return 0
    except Exception as e:
        logger.exception(f"Error clearing caches: {e}")
        print(f"Error clearing caches: {e}")
        return 1

def handle_export_tracker(args: argparse.Namespace) -> int:
    """Handle the export-tracker command."""
    try:
        output_path = args.output or os.path.splitext(args.tracker_file)[0] + '.' + args.format
        result = export_tracker(args.tracker_file, args.format, output_path)
        if isinstance(result, str) and result.startswith("Error:"): # Check if export returned an error string
            print(result); return 1
        print(f"Tracker exported to {output_path}"); return 0
    except Exception as e: logger.exception(f"Error export_tracker: {e}"); print(f"Error: {e}"); return 1

def handle_update_config(args: argparse.Namespace) -> int:
    """Handle the update-config command."""
    config_manager = ConfigManager()
    try:
        # Attempt to parse value as JSON (allows lists/dicts), fall back to string
        try: value = json.loads(args.value)
        except json.JSONDecodeError: value = args.value
        success = config_manager.update_config_setting(args.key, value)
        if success: print(f"Updated config: {args.key} = {value}"); return 0
        else: print(f"Error: Failed update config (key '{args.key}' invalid?)."); return 1
    except Exception as e: logger.exception(f"Error update_config: {e}"); print(f"Error: {e}"); return 1

def handle_reset_config(args: argparse.Namespace) -> int:
    """Handle the reset-config command."""
    config_manager = ConfigManager()
    try:
        success = config_manager.reset_to_defaults()
        if success: print("Config reset to defaults."); return 0
        else: print("Error: Failed reset config."); return 1
    except Exception as e: logger.exception(f"Error reset_config: {e}"); print(f"Error: {e}"); return 1

def handle_show_dependencies(args: argparse.Namespace) -> int:
    """Handle the show-dependencies command using the contextual key system."""
    target_key_str = args.key
    logger.info(f"Showing dependencies for key string: {target_key_str}")
    
    # --- Load CURRENT Global Map ---
    current_global_map = _load_global_map_or_exit(args) # path_to_key_info
   
    config = ConfigManager()
    project_root = get_project_root()
    matching_infos = [info for info in current_global_map.values() if info.key_string == target_key_str]
    if not matching_infos:
        print(f"Error: Key string '{target_key_str}' not found in the project.")
        return 1
    
    target_info: KeyInfo
    if len(matching_infos) > 1:
        print(f"Warning: Key string '{target_key_str}' is ambiguous and matches multiple paths:")
        for i, info in enumerate(matching_infos): print(f"  [{i+1}] {info.norm_path}")
        target_info = matching_infos[0] 
        print(f"Using the first match: {target_info.norm_path}")
    else:
        target_info = matching_infos[0]
    target_norm_path = target_info.norm_path
    print(f"\n--- Dependencies for Key: {target_key_str} (Path: {target_norm_path}) ---")

    # --- Build Path Migration Map ---
    logger.debug("Building path migration map for show-dependencies...")
    old_global_map = load_old_global_key_map() # Load old map (can be None)
    path_migration_info: PathMigrationInfo
    try:
        # Call the builder function from tracker_io
        path_migration_info = _build_path_migration_map(old_global_map, current_global_map)
    except ValueError as ve:
         logger.error(f"Failed to build migration map for show-dependencies: {ve}. Results may be inaccurate.")
         # Create a dummy "current state only" migration map as a fallback
         path_migration_info = {info.norm_path: (info.key_string, info.key_string) for info in current_global_map.values()}
    except Exception as e:
         logger.error(f"Unexpected error building migration map for show-dependencies: {e}. Results may be inaccurate.", exc_info=True)
         path_migration_info = {info.norm_path: (info.key_string, info.key_string) for info in current_global_map.values()}


    # --- Use Utility Functions ---
    all_tracker_paths = find_all_tracker_paths(config, project_root)
    if not all_tracker_paths: print("Warning: No tracker files found.")
    
    # Pass the generated path_migration_info to aggregate_all_dependencies
    aggregated_links_with_origins = aggregate_all_dependencies(
        all_tracker_paths, 
        path_migration_info # MODIFIED: Pass the migration map
    )

    # --- Process Aggregated Results for Display ---
    all_dependencies_by_type = defaultdict(set) # Store sets of (dep_key_string, dep_path_string) tuples
    origin_tracker_map_display = defaultdict(lambda: defaultdict(set)) # For p/s/S origins

    logger.debug(f"Filtering aggregated links for target key display: {target_key_str}")
    for (source, target), (dep_char, origins) in aggregated_links_with_origins.items():
        dep_key_str = None
        display_char = dep_char # Character used to categorize for display
        if source == target_key_str:
            dep_key_str = target # Target is the dependency shown
        elif target == target_key_str:
            dep_key_str = source # Source is the dependency shown
            # Adjust display category based on relationship direction relative to target
            if dep_char == '>': display_char = '<' # Target depends on Source (show Source in Depends On)
            elif dep_char == '<': display_char = '>' # Source depends on Target (show Source in Depended On By)
            # 'x', 'd', 's', 'S' remain the same category regardless of direction
            else: display_char = dep_char

        if dep_key_str:
            # Find path for the dependency key
            dep_info = next((info for info in current_global_map.values() if info.key_string == dep_key_str), None)
            dep_path_str = dep_info.norm_path if dep_info else "PATH_NOT_FOUND_GLOBALLY"
            # Add to the correct category for display
            all_dependencies_by_type[display_char].add((dep_key_str, dep_path_str))
            # Track origins specifically for p/s/S if needed for display
            if display_char in ('p', 's', 'S'):
                origin_tracker_map_display[display_char][dep_key_str].update(origins)

    # --- Print results ---
    output_sections = [
        ("Mutual ('x')", 'x'), ("Documentation ('d')", 'd'), ("Semantic (Strong) ('S')", 'S'),
        ("Semantic (Weak) ('s')", 's'), ("Depends On ('<')", '<'), ("Depended On By ('>')", '>'),
        ("Placeholders ('p')", 'p')
    ]
    for section_title, dep_char_filter in output_sections: # Renamed dep_char to avoid conflict
        print(f"\n{section_title}:")
        dep_set = all_dependencies_by_type.get(dep_char_filter)
        if dep_set:
            # Define helper for hierarchical sorting
            def _hierarchical_sort_key_func(key_str_local: str): # Renamed key_str
                KEY_PATTERN = r'\d+|\D+' 
                if not key_str_local or not isinstance(key_str_local, str): return []
                parts = re.findall(KEY_PATTERN, key_str_local)
                try:
                    return [(int(p) if p.isdigit() else p) for p in parts]
                except (ValueError, TypeError): # Fallback
                    logger.warning(f"Could not convert parts for sorting display key '{key_str_local}'")
                    return parts
            sorted_deps = sorted(list(dep_set), key=lambda item: _hierarchical_sort_key_func(item[0]))
            for dep_key, dep_path in sorted_deps:
                # Check for and add origin info for p/s/S
                origin_info = ""
                if dep_char_filter in ('p', 's', 'S'):
                    origins = origin_tracker_map_display.get(dep_char_filter, {}).get(dep_key, set())
                    if origins:
                        # Format origin filenames nicely
                        origin_filenames = sorted([os.path.basename(p) for p in origins])
                        origin_info = f" (In: {', '.join(origin_filenames)})"
                # Print with optional origin info
                print(f"  - {dep_key}: {dep_path}{origin_info}")
        else: print("  None")
    print("\n------------------------------------------")
    return 0

def handle_show_keys(args: argparse.Namespace) -> int:
    """
    Handle the show-keys command.
    Displays key definitions from the specified tracker file.
    Additionally, checks if the corresponding row in the grid contains
    any 'p', 's', or 'S' characters (indicating unverified placeholders
    or suggestions) and notes which were found.
    """
    tracker_path = normalize_path(args.tracker)
    logger.info(f"Attempting to show keys and check status (p, s, S) from tracker: {tracker_path}")

    if not os.path.exists(tracker_path):
        print(f"Error: Tracker file not found: {tracker_path}", file=sys.stderr)
        logger.error(f"Tracker file not found: {tracker_path}")
        return 1
    try:
        tracker_data = read_tracker_file(tracker_path) 
        if not tracker_data:
            print(f"Error: Could not read or parse tracker file: {tracker_path}", file=sys.stderr)
            return 1

        keys_map = tracker_data.get("keys")
        grid_map = tracker_data.get("grid")

        if not keys_map:
            print(f"Error: No keys found in tracker file: {tracker_path}", file=sys.stderr)
            logger.error(f"No key definitions found in {tracker_path}")
            # Allow proceeding if grid is missing, but warn
            if not grid_map:
                logger.warning(f"Grid data also missing in {tracker_path}")
            return 1 # Considered an error state if keys are missing

        # Sort keys hierarchically for consistent output
        sorted_keys = sort_key_strings_hierarchically(list(keys_map.keys()))
        print(f"--- Keys Defined in {os.path.basename(tracker_path)} ---") # Header for clarity

        for key_string in sorted_keys:
            file_path = keys_map.get(key_string, "PATH_UNKNOWN")
            check_indicator = "" # Renamed for clarity
            # Check the grid for 'p', 's', or 'S' *only if* the grid exists
            if grid_map:
                compressed_row = grid_map.get(key_string, "")
                if compressed_row: # Check if the row string exists and is not empty
                    found_chars = []
                    # Check for each character efficiently using 'in'. This is safe
                    # as 'p', 's', 'S' won't appear in RLE counts (which are digits).
                    if 'p' in compressed_row: found_chars.append('p')
                    if 's' in compressed_row: found_chars.append('s')
                    if 'S' in compressed_row: found_chars.append('S')
                    if found_chars:
                        sorted_chars = sorted(found_chars)
                        chars_str = ", ".join(sorted_chars)
                        check_indicator = f" (checks needed: {chars_str})" # Updated indicator format
                else:
                    # Indicate if a key exists but has no corresponding grid row yet
                    check_indicator = " (grid row missing)"
                    logger.warning(f"Key '{key_string}' found in definitions but missing from grid in {tracker_path}")
            else: # Grid is missing entirely
                check_indicator = " (grid missing)"
            # Print the key, path, and the indicator (if any checks are needed or row is missing)
            print(f"{key_string}: {file_path}{check_indicator}")
        print("--- End of Key Definitions ---")
        try:
            with open(tracker_path, 'r', encoding='utf-8') as f_check:
                content = f_check.read()
                if KEY_DEFINITIONS_START_MARKER not in content:
                     logger.warning(f"Start marker '{KEY_DEFINITIONS_START_MARKER}' not found in {tracker_path}")
                if KEY_DEFINITIONS_END_MARKER not in content:
                     logger.warning(f"End marker '{KEY_DEFINITIONS_END_MARKER}' not found in {tracker_path}")
        except Exception:
             logger.warning(f"Could not perform marker check on {tracker_path}")
        return 0
    except IOError as e:
        print(f"Error reading tracker file {tracker_path}: {e}", file=sys.stderr)
        logger.error(f"IOError reading {tracker_path}: {e}", exc_info=True)
        return 1
    except Exception as e:
        print(f"An unexpected error occurred while processing {tracker_path}: {e}", file=sys.stderr)
        logger.error(f"Unexpected error processing {tracker_path}: {e}", exc_info=True)
        return 1

def handle_visualize_dependencies(args: argparse.Namespace) -> int:
    """Handles the visualize-dependencies command by calling the core generation function."""
    # 1. Get arguments
    focus_keys_list_cli = args.key if args.key is not None else [] # args.key is now a list or None
    output_format_cli = args.format.lower()
    output_path_arg_cli = args.output

    logger.info(f"CLI: visualize-dependencies called. Focus Keys: {focus_keys_list_cli or 'Project Overview'}")

    # 2. Basic validation
    if output_format_cli != "mermaid":
        print(f"Error: Only 'mermaid' format supported at this time.")
        return 1

    # 3. Load necessary data
    try:
        current_global_map_cli = _load_global_map_or_exit(args) # Renamed for clarity
        config_cli = ConfigManager()
        project_root_cli = get_project_root()
        # Use find_all_tracker_paths from tracker_io module
        all_tracker_paths_cli = find_all_tracker_paths(config_cli, project_root_cli)
        if not all_tracker_paths_cli:
            print("Warning: No tracker files found. Diagram may be empty.")

        # --- Build Path Migration Map FOR THIS COMMAND ---
        logger.debug("Building path migration map for visualize-dependencies command...")
        old_global_map_cli = load_old_global_key_map() # Load old map
        path_migration_info_cli: PathMigrationInfo
        try:
            path_migration_info_cli = _build_path_migration_map(old_global_map_cli, current_global_map_cli)
        except ValueError as ve:
            logger.error(f"Failed to build migration map for visualize-dependencies: {ve}. Visualization may be based on current state only or fail.")
            # Fallback to a "current state only" dummy migration map
            path_migration_info_cli = {info.norm_path: (info.key_string, info.key_string) for info in current_global_map_cli.values()}
        except Exception as e:
            logger.error(f"Unexpected error building migration map for visualize-dependencies: {e}. Visualization may be inaccurate.", exc_info=True)
            path_migration_info_cli = {info.norm_path: (info.key_string, info.key_string) for info in current_global_map_cli.values()}

    except Exception as e:
        logger.exception("Failed to load data required for visualization.")
        print(f"Error loading data needed for visualization: {e}", file=sys.stderr)
        return 1


    # 4. Call the core generation function from the new module
    mermaid_string_generated = generate_mermaid_diagram(
        focus_keys_list=focus_keys_list_cli,
        global_path_to_key_info_map=current_global_map_cli, # Still pass current map for node info
        path_migration_info=path_migration_info_cli,       # Pass the migration map
        all_tracker_paths_list=list(all_tracker_paths_cli),
        config_manager_instance=config_cli
    )

    # 5. Handle the result (Mermaid string or None)
    if mermaid_string_generated is None:
        print("Error: Mermaid diagram generation failed internally. Check logs.", file=sys.stderr)
        return 1
    elif "Error:" in mermaid_string_generated[:20]: # Check if the string itself contains an error message
         print(mermaid_string_generated, file=sys.stderr)
         return 1
    elif "// No relevant data" in mermaid_string_generated:
         print("Info: No relevant data found to visualize based on focus keys and filters.")
         # Still save the basic mermaid file for consistency? Or just return 0? Let's save it.
    else:
        logger.info("Mermaid code generated successfully.")


    # 6. Determine output path
    output_path_cli = output_path_arg_cli # User-specified output path
    if not output_path_cli:
        # Construct default path
        if focus_keys_list_cli:
            # Create a filename-safe string from potentially multiple keys
            safe_keys_str = "_".join(sorted(focus_keys_list_cli)).replace("/", "_").replace("\\", "_")
            # Limit filename length
            max_len = 50
            if len(safe_keys_str) > max_len:
                safe_keys_str = safe_keys_str[:max_len] + "_etc"
            default_filename = f"focus_{safe_keys_str}_dependencies.{output_format_cli}"
        else:
            default_filename = f"project_overview_dependencies.{output_format_cli}"

        # --- Define the default directory ---
        # Use memory_dir from config, defaulting to 'cline_docs' if not set
        memory_dir_rel = config_cli.get_path('memory_dir', 'cline_docs')
        default_output_dir_rel = os.path.join(memory_dir_rel, "dependency_diagrams")
        # --- End Define Default Dir ---

        # Combine relative default dir with project root and default filename
        output_path_cli = normalize_path(os.path.join(project_root_cli, default_output_dir_rel, default_filename))
        logger.info(f"No output path specified, using default: {output_path_cli}")

    # Ensure output path is normalized (if user provided relative path, make it absolute)
    elif not os.path.isabs(output_path_cli):
         output_path_cli = normalize_path(os.path.join(project_root_cli, output_path_cli))
    else:
         output_path_cli = normalize_path(output_path_cli)


    # 7. Write the output file
    try:
        output_dir_cli = os.path.dirname(output_path_cli)
        if output_dir_cli: # Only create if it's not the current directory
            os.makedirs(output_dir_cli, exist_ok=True)

        with open(output_path_cli, 'w', encoding='utf-8') as f_out:
            f_out.write(mermaid_string_generated)

        logger.info(f"Successfully wrote Mermaid visualization to: {output_path_cli}")
        print(f"\nDependency visualization saved to: {output_path_cli}")
        if "// No relevant data" not in mermaid_string_generated: # Only print view instructions if there's data
             print("You can view this file using Mermaid Live Editor (mermaid.live) or compatible Markdown viewers.")
        return 0
    except IOError as e:
        logger.error(f"Failed to write visualization file {output_path_cli}: {e}", exc_info=True)
        print(f"Error: Could not write output file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception(f"An unexpected error occurred during visualization file writing: {e}")
        print(f"Error: An unexpected error occurred while writing output: {e}", file=sys.stderr)
        return 1

def main():
    """Parse arguments and dispatch to handlers."""
    parser = argparse.ArgumentParser(description="Dependency tracking system CLI")
    parser.add_argument(
        "--global-key-map-path",
        help="Explicit path to the global_key_map.json file. Overrides config."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- Analysis Commands ---
    analyze_file_parser = subparsers.add_parser("analyze-file", help="Analyze a single file")
    analyze_file_parser.add_argument("file_path", help="Path to the file")
    analyze_file_parser.add_argument("--output", help="Save results to JSON file")
    analyze_file_parser.set_defaults(func=command_handler_analyze_file)

    analyze_project_parser = subparsers.add_parser("analyze-project", help="Analyze project, generate keys/embeddings, update trackers")
    analyze_project_parser.add_argument("project_root", nargs='?', default='.', help="Project directory path (default: CWD)")
    analyze_project_parser.add_argument("--output", help="Save analysis summary to JSON file")
    analyze_project_parser.add_argument("--force-embeddings", action="store_true", help="Force regeneration of embeddings")
    analyze_project_parser.add_argument("--force-analysis", action="store_true", help="Force re-analysis and bypass cache")
    analyze_project_parser.set_defaults(func=command_handler_analyze_project)

    # --- Grid Manipulation Commands ---
    compress_parser = subparsers.add_parser("compress", help="Compress RLE string")
    compress_parser.add_argument("string", help="String to compress")
    compress_parser.set_defaults(func=handle_compress)

    decompress_parser = subparsers.add_parser("decompress", help="Decompress RLE string")
    decompress_parser.add_argument("string", help="String to decompress")
    decompress_parser.set_defaults(func=handle_decompress)

    get_char_parser = subparsers.add_parser("get_char", help="Get char at logical index in compressed string")
    get_char_parser.add_argument("string", help="Compressed string")
    get_char_parser.add_argument("index", type=int, help="Logical index")
    get_char_parser.set_defaults(func=handle_get_char)

    set_char_parser = subparsers.add_parser("set_char", help="Set char at logical index in a tracker file")
    set_char_parser.add_argument("tracker_file", help="Path to tracker file")
    set_char_parser.add_argument("key", type=str, help="Row key")
    set_char_parser.add_argument("index", type=int, help="Logical index")
    set_char_parser.add_argument("char", type=str, help="New character")
    set_char_parser.set_defaults(func=handle_set_char)

    add_dep_parser = subparsers.add_parser("add-dependency", help="Add dependency between keys")
    add_dep_parser.add_argument("--tracker", required=True, help="Path to tracker file")
    add_dep_parser.add_argument("--source-key", required=True, help="Source key")
    add_dep_parser.add_argument("--target-key", required=True, nargs='+', help="One or more target keys (space-separated)")
    add_dep_parser.add_argument("--dep-type", default=">", help="Dependency type (e.g., '>', '<', 'x')")
    add_dep_parser.set_defaults(func=handle_add_dependency)

    # --- Tracker File Management ---
    remove_key_parser = subparsers.add_parser("remove-key", help="Remove a key and its row/column from a specific tracker")
    remove_key_parser.add_argument("tracker_file", help="Path to the tracker file (.md)")
    remove_key_parser.add_argument("key", type=str, help="The key string to remove from this tracker")
    remove_key_parser.set_defaults(func=handle_remove_key)

    merge_parser = subparsers.add_parser("merge-trackers", help="Merge two tracker files")
    merge_parser.add_argument("primary_tracker_path", help="Primary tracker")
    merge_parser.add_argument("secondary_tracker_path", help="Secondary tracker")
    merge_parser.add_argument("--output", "-o", help="Output path (defaults to primary)")
    merge_parser.set_defaults(func=handle_merge_trackers)

    export_parser = subparsers.add_parser("export-tracker", help="Export tracker data")
    export_parser.add_argument("tracker_file", help="Path to tracker file")
    export_parser.add_argument("--format", choices=["json", "csv", "dot"], default="json", help="Export format")
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.set_defaults(func=handle_export_tracker)

    # --- Utility Commands ---
    clear_caches_parser = subparsers.add_parser("clear-caches", help="Clear all internal caches")
    clear_caches_parser.set_defaults(func=handle_clear_caches)

    reset_config_parser = subparsers.add_parser("reset-config", help="Reset config to defaults")
    reset_config_parser.set_defaults(func=handle_reset_config)

    update_config_parser = subparsers.add_parser("update-config", help="Update a config setting")
    update_config_parser.add_argument("key", help="Config key path (e.g., 'paths.doc_dir')")
    update_config_parser.add_argument("value", help="New value (JSON parse attempted)")
    update_config_parser.set_defaults(func=handle_update_config)

    show_deps_parser = subparsers.add_parser("show-dependencies", help="Show aggregated dependencies for a key")
    show_deps_parser.add_argument("--key", required=True, help="Key string to show dependencies for")
    show_deps_parser.set_defaults(func=handle_show_dependencies)

    # --- Show Keys Command ---
    show_keys_parser = subparsers.add_parser("show-keys", help="Show keys from tracker, indicating if checks needed (p, s, S)")
    show_keys_parser.add_argument("--tracker", required=True, help="Path to the tracker file (.md)")
    show_keys_parser.set_defaults(func=handle_show_keys)

    visualize_parser = subparsers.add_parser("visualize-dependencies", help="Generate a visualization of dependencies")
    visualize_parser.add_argument(
        "--key",
        nargs='*', # Changed to accept zero or more keys
        default=None, # Default is None if not provided
        help="Optional: One or more key strings to focus the visualization on. If omitted, shows overview."
    )
    visualize_parser.add_argument(
        "--format",
        choices=["mermaid"],
        default="mermaid",
        help="Output format (only mermaid currently)"
    )
    visualize_parser.add_argument(
        "--output", "-o",
        help="Output file path (default: project_overview... or focus_KEY(s)...)"
    )
    visualize_parser.set_defaults(func=handle_visualize_dependencies)

    args = parser.parse_args()

    # --- Setup Logging ---
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger(); root_logger.setLevel(logging.DEBUG)
    log_file_path = 'debug.txt'; suggestions_log_path = 'suggestions.log'
    try: # File Handler
        file_handler = logging.FileHandler(log_file_path, mode='w'); file_handler.setLevel(logging.DEBUG); file_handler.setFormatter(log_formatter); root_logger.addHandler(file_handler)
    except Exception as e: print(f"Error setting up file logger {log_file_path}: {e}", file=sys.stderr)
    try: # Suggestions Handler
        suggestion_handler = logging.FileHandler(suggestions_log_path, mode='w'); suggestion_handler.setLevel(logging.DEBUG); suggestion_handler.setFormatter(log_formatter)
        class SuggestionLogFilter(logging.Filter):
            def filter(self, record): return record.name.startswith('cline_utils.dependency_system.analysis') # Broaden slightly
        suggestion_handler.addFilter(SuggestionLogFilter()); root_logger.addHandler(suggestion_handler)
    except Exception as e: print(f"Error setting up suggestions logger {suggestions_log_path}: {e}", file=sys.stderr)
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout); console_handler.setLevel(logging.INFO); console_handler.setFormatter(log_formatter); root_logger.addHandler(console_handler)

    # Execute command
    if hasattr(args, 'func'):
        exit_code = args.func(args)
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() # Call main function if script is executed
