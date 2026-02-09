# dependency_processor.py

"""
Main entry point for the dependency tracking system.
Processes command-line arguments and delegates to appropriate handlers.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from logging import LogRecord
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from cline_utils.dependency_system.analysis.dependency_analyzer import analyze_file

# --- Analysis Imports ---
from cline_utils.dependency_system.analysis.project_analyzer import analyze_project

# --- Core Imports ---
from cline_utils.dependency_system.core.dependency_grid import (
    DIAGONAL_CHAR,
    EMPTY_CHAR,
    PLACEHOLDER_CHAR,
    compress,
    decompress,
    get_char_at,
)
from cline_utils.dependency_system.core.key_manager import (
    KeyInfo,
    get_sortable_parts_for_key,
    load_global_key_map,
    load_old_global_key_map,
)

# --- IO Imports ---
from cline_utils.dependency_system.io.tracker_io import (
    PathMigrationInfo,
    build_path_migration_map,
    export_tracker,
    merge_trackers,
    remove_path_from_tracker,
    update_tracker,
)
from cline_utils.dependency_system.utils.cache_manager import clear_all_caches
from cline_utils.dependency_system.utils.cache_manager import (
    get_project_root_cached as get_project_root,
)
from cline_utils.dependency_system.utils.cache_manager import (
    normalize_path_cached as normalize_path,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager

# --- Utility Imports ---
from cline_utils.dependency_system.utils.template_generator import (
    add_code_doc_dependency_to_checklist,
)
from cline_utils.dependency_system.utils.template_generator import (
    get_item_type as get_item_type_for_checklist,
)
from cline_utils.dependency_system.utils.tracker_utils import (
    find_all_tracker_paths,
    get_globally_resolved_key_info_for_cli,
    get_key_global_instance_string,
    read_grid_from_lines,
    read_key_definitions_from_lines,
    resolve_key_global_instance_to_ki,
)
from cline_utils.dependency_system.utils.visualize_dependencies import (
    generate_mermaid_diagram,
)

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
KEY_DEFINITIONS_START_MARKER = "---KEY_DEFINITIONS_START---"
KEY_DEFINITIONS_END_MARKER = "---KEY_DEFINITIONS_END---"


# --- Helper Functions ---
def _load_global_map_or_exit() -> Dict[str, KeyInfo]:
    """Loads the global key map, exiting if it fails."""
    logger.info("Loading global key map...")
    path_to_key_info = load_global_key_map()
    if path_to_key_info is None:
        print("Error: Global key map not found or failed to load.", file=sys.stderr)
        print(
            "Please run 'analyze-project' first to generate the key map.",
            file=sys.stderr,
        )
        logger.critical("Global key map missing or invalid. Exiting.")
        sys.exit(1)
    logger.info("Global key map loaded successfully.")
    return path_to_key_info


def is_parent_child(
    key1_str: str, key2_str: str, global_map: Dict[str, KeyInfo]
) -> bool:
    """Checks if two keys represent a direct parent-child directory relationship."""
    info1 = next(
        (info for info in global_map.values() if info.key_string == key1_str), None
    )
    info2 = next(
        (info for info in global_map.values() if info.key_string == key2_str), None
    )

    if not info1 or not info2:
        logger.debug(
            f"is_parent_child: Could not find KeyInfo for '{key1_str if not info1 else ''}' or '{key2_str if not info2 else ''}'. Returning False."
        )
        return False  # Cannot determine relationship if info is missing

    # Ensure paths are normalized (they should be from KeyInfo, but double-check)
    path1 = normalize_path(info1.norm_path)
    path2 = normalize_path(info2.norm_path)
    parent1 = normalize_path(info1.parent_path) if info1.parent_path else None
    parent2 = normalize_path(info2.parent_path) if info2.parent_path else None

    # Check both directions: info1 is parent of info2 OR info2 is parent of info1
    is_parent1 = parent2 == path1
    is_parent2 = parent1 == path2

    logger.debug(
        f"is_parent_child check: {key1_str}({path1}) vs {key2_str}({path2}). Is Parent1: {is_parent1}, Is Parent2: {is_parent2}"
    )
    return is_parent1 or is_parent2


# --- Command Handlers ---


def command_handler_analyze_file(args: argparse.Namespace) -> int:
    """Handle the analyze-file command."""
    import json

    try:
        if not os.path.exists(args.file_path):
            print(f"Error: File not found: {args.file_path}")
            return 1
        results = analyze_file(args.file_path)
        if args.output:
            output_dir = os.path.dirname(args.output)
            os.makedirs(output_dir, exist_ok=True) if output_dir else None
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Analysis results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2))
        return 0
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return 1


def command_handler_analyze_project(args: argparse.Namespace) -> int:
    """Handle the analyze-project command."""
    import json

    original_cwd: Optional[str] = None  # Initialize to None
    try:
        if not args.project_root:
            args.project_root = "."
            logger.info(
                f"Defaulting project root to CWD: {os.path.abspath(args.project_root)}"
            )
        abs_project_root = normalize_path(os.path.abspath(args.project_root))
        if not os.path.isdir(abs_project_root):
            print(f"Error: Project directory not found: {abs_project_root}")
            return 1
        original_cwd = os.getcwd()  # Assign after initialization
        if abs_project_root != normalize_path(original_cwd):
            logger.info(
                f"Temporarily changing CWD from '{original_cwd}' to project root: '{abs_project_root}' for analysis."
            )
            os.chdir(abs_project_root)
            _ = ConfigManager().config

        # Now that we are in the correct directory, get the config manager and run validation
        config_manager_instance = ConfigManager()

        # Clear validation cache if --force-validate flag is set
        if getattr(args, "force_validate", False):
            try:
                from .utils.resource_validator import _get_cache_path

                cache_path = _get_cache_path()
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    logger.info("Cleared validation cache (--force-validate)")
            except Exception as e:
                logger.warning(f"Failed to clear validation cache: {e}")

        config_manager_instance.perform_resource_validation_and_adjustments()

        # --- Run Runtime Inspector ---
        try:
            # Construct path to runtime_inspector.py relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            runtime_inspector_path = os.path.join(
                current_dir, "analysis", "runtime_inspector.py"
            )

            if os.path.exists(runtime_inspector_path):
                logger.info(f"Running runtime inspector: {runtime_inspector_path}")
                # Run as a subprocess to avoid namespace pollution and handle crashes safely
                env = os.environ.copy()
                # Ensure PYTHONPATH includes the project root so cline_utils can be imported
                # We need to include the original CWD where the package is installed
                paths_to_add = [os.getcwd()]
                if original_cwd and normalize_path(original_cwd) != normalize_path(
                    os.getcwd()
                ):
                    paths_to_add.append(original_cwd)

                path_str = os.pathsep.join(paths_to_add)

                if "PYTHONPATH" in env:
                    env["PYTHONPATH"] = f"{path_str}{os.pathsep}{env['PYTHONPATH']}"
                else:
                    env["PYTHONPATH"] = path_str

                process = subprocess.run(
                    [sys.executable, runtime_inspector_path, abs_project_root],
                    capture_output=True,
                    text=True,
                    check=False,
                    env=env,
                )

                if process.returncode == 0:
                    logger.info("Runtime inspection completed successfully.")
                    logger.debug(f"Runtime inspector output: {process.stdout}")
                else:
                    logger.warning(
                        f"Runtime inspector failed with return code {process.returncode}"
                    )
                    logger.warning(f"Runtime inspector stderr: {process.stderr}")
            else:
                logger.warning(
                    f"Runtime inspector script not found at {runtime_inspector_path}"
                )
        except Exception as e:
            logger.error(f"Error running runtime inspector: {e}")

        logger.debug(
            f"Analyzing project: {abs_project_root}, force_analysis={args.force_analysis}, force_embeddings={args.force_embeddings}"
        )
        results = analyze_project(
            force_analysis=args.force_analysis, force_embeddings=args.force_embeddings
        )
        logger.debug(
            f"All Suggestions before Tracker Update: {results.get('dependency_suggestion', {}).get('suggestions')}"
        )

        # Helper function to make results JSON-serializable by removing AST objects
        def make_serializable(obj):
            """Recursively remove non-JSON-serializable objects from the results."""
            if isinstance(obj, dict):
                # Remove known non-serializable keys
                cleaned = {
                    k: make_serializable(v)
                    for k, v in obj.items()
                    if k not in ("_ast_tree", "_ts_tree")
                }
                return cleaned
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                # Convert any other non-serializable objects to string representation
                return str(obj)

        if args.output:
            output_path_abs = normalize_path(os.path.abspath(args.output))
            output_dir = os.path.dirname(output_path_abs)
            os.makedirs(output_dir, exist_ok=True) if output_dir else None
            # Clean results before JSON serialization
            serializable_results = make_serializable(results)
            with open(output_path_abs, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Analysis results saved to {output_path_abs}")
        elif results.get("status") == "success":
            print(
                "Project analysis completed successfully. Results not saved to file (use --output)."
            )

        return (
            0
            if results.get("status") == "success" or results.get("status") == "warning"
            else 1
        )
    except Exception as e:
        logger.error(f"Error analyzing project: {str(e)}", exc_info=True)
        print(f"Error analyzing project: {str(e)}")
        return 1
    finally:
        # Check if original_cwd was successfully assigned before using it
        if original_cwd is not None and normalize_path(os.getcwd()) != normalize_path(
            original_cwd
        ):
            logger.info(f"Changing CWD back to original: {original_cwd}")
            os.chdir(original_cwd)
            _ = ConfigManager().config


def handle_compress(args: argparse.Namespace) -> int:
    """Handle the compress command."""
    try:
        result = compress(args.string)
        print(f"Compressed string: {result}")
        return 0
    except Exception as e:
        logger.error(f"Error compressing: {e}")
        print(f"Error: {e}")
        return 1


def handle_decompress(args: argparse.Namespace) -> int:
    """Handle the decompress command."""
    try:
        result = decompress(args.string)
        print(f"Decompressed string: {result}")
        return 0
    except Exception as e:
        logger.error(f"Error decompressing: {e}")
        print(f"Error: {e}")
        return 1


def handle_get_char(args: argparse.Namespace) -> int:
    """Handle the get_char command."""
    try:
        result = get_char_at(args.string, args.index)
        print(f"Character at index {args.index}: {result}")
        return 0
    except IndexError:
        logger.error("Index out of range")
        print("Error: Index out of range")
        return 1
    except Exception as e:
        logger.error(f"Error get_char: {e}")
        print(f"Error: {e}")
        return 1


def handle_set_char(args: argparse.Namespace) -> int:
    # --- ADDED CRITICAL WARNING ---
    critical_message = (
        "CRITICAL WARNING: The 'set_char' command is DEPRECATED and EXTREMELY DANGEROUS "
        "with the current tracker format. It operates on an outdated understanding of grid structure "
        "and can EASILY CORRUPT tracker files. It assumes the key you provide uniquely identifies a row, "
        "and the index refers to an Nth unique key. This is no longer true. "
        "USE 'add-dependency --tracker <file> --source-key <KEY#GI> --target-key <KEY#GI> --dep-type <char>' INSTEAD "
        "for targeted changes. PROCEEDING WITH 'set_char' IS AT YOUR OWN RISK AND LIKELY TO BREAK THINGS. "
        "This command will attempt a best-effort conversion but is not guaranteed to be safe or accurate."
    )
    logger.critical(critical_message)
    print(critical_message)
    # --- END OF ADDED CRITICAL WARNING ---

    try:
        tracker_file_path = normalize_path(args.tracker_file)
        if not os.path.exists(tracker_file_path):
            print(f"Error: Tracker file not found: {tracker_file_path}")
            return 1

        with open(tracker_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Use tracker_io's parsing functions
        defs_pairs = read_key_definitions_from_lines(lines)
        _grid_hdrs, grid_rows_list = read_grid_from_lines(lines)

        # Find the first definition matching args.key to get its path and original index
        source_row_original_idx = -1
        source_path_targetted = None
        # The key from args.key is a KEY_LABEL from the tracker file (could be KEY or KEY#GI)
        for idx, (k_label_in_file, p_str) in enumerate(defs_pairs):
            if k_label_in_file == args.key:
                source_row_original_idx = idx
                source_path_targetted = p_str
                break

        if source_row_original_idx == -1 or source_path_targetted is None:
            print(
                f"Error: Source key label '{args.key}' not found in tracker definitions."
            )
            return 1

        if source_row_original_idx >= len(grid_rows_list):
            print(
                f"Error: Grid data for source key label '{args.key}' (def index {source_row_original_idx}) seems missing or tracker is corrupt."
            )
            return 1

        target_col_logical_index = (
            args.index
        )  # This index refers to the Nth definition in the *original* file
        if not (0 <= target_col_logical_index < len(defs_pairs)):
            print(
                f"Error: Target column index {target_col_logical_index} is out of range for {len(defs_pairs)} definitions."
            )
            return 1

        target_path_targetted: str = defs_pairs[target_col_logical_index][1]
        target_key_label_targetted: str = defs_pairs[target_col_logical_index][0]

        print(
            f"\n--- Attempting to set relationship for paths (via low-level 'set_char' command) ---"
        )
        print(
            f"  Source (from tracker def): '{args.key}' (Path: {source_path_targetted})"
        )
        print(
            f"  Target (from tracker def): '{target_key_label_targetted}' (Path: {target_path_targetted}) at original column index {target_col_logical_index}"
        )
        print(f"  New Char to set: '{args.char}'")
        print(
            f"-------------------------------------------------------------------------------------\n"
        )

        global_map = _load_global_map_or_exit()

        # Find the KeyInfo for source and target paths to get their current global keys
        src_ki_global = global_map.get(source_path_targetted)
        tgt_ki_global = global_map.get(target_path_targetted)

        if not src_ki_global:
            print(
                f"Error: Source path '{source_path_targetted}' (from key '{args.key}') not found in current global map. Aborting 'set_char'."
            )
            return 1
        if not tgt_ki_global:
            print(
                f"Error: Target path '{target_path_targetted}' (from target key label '{target_key_label_targetted}') not found in current global map. Aborting 'set_char'."
            )
            return 1

        # Construct KEY#GI strings for the suggestion
        source_key_for_sugg = get_key_global_instance_string(src_ki_global, global_map)
        target_key_for_sugg = get_key_global_instance_string(tgt_ki_global, global_map)

        if not source_key_for_sugg or not target_key_for_sugg:
            print(
                "Error: Could not determine KEY#GlobalInstance for source or target. Aborting 'set_char'."
            )
            return 1

        suggestions_for_set_char = {
            source_key_for_sugg: [(target_key_for_sugg, args.char)]
        }

        is_mini = tracker_file_path.endswith("_module.md")
        tracker_type_val = (
            "mini"
            if is_mini
            else (
                "doc"
                if "doc_tracker.md" in os.path.basename(tracker_file_path)
                else "main"
            )
        )
        f_to_m_map = {
            _info.norm_path: _info.parent_path
            for _info in global_map.values()
            if not _info.is_directory and _info.parent_path
        }

        update_tracker(
            output_file_suggestion=tracker_file_path,
            path_to_key_info=global_map,
            tracker_type=tracker_type_val,
            suggestions_external=suggestions_for_set_char,
            file_to_module=f_to_m_map,
            force_apply_suggestions=True,  # Force this specific change
            apply_ast_overrides=False,  # <<< MODIFIED/ADDED
        )
        print(
            f"Applied 'set_char' for source '{args.key}' targeting original column index {args.index} with char '{args.char}' "
            f"in {tracker_file_path} via forced update. VERIFY THE RESULT CAREFULLY in the tracker file."
        )
        return 0

    except Exception as e:
        logger.error(
            f"Error during 'set_char' for {args.tracker_file}: {e}", exc_info=True
        )
        print(
            f"Error during 'set_char': {e}. The tracker might be in an inconsistent state."
        )
        return 1


def handle_remove_key(args: argparse.Namespace) -> int:
    """Handle the remove-key command by resolving key to path and calling remove_path_from_tracker."""
    tracker_file_path = normalize_path(args.tracker_file)
    key_to_remove_str_arg = (
        args.key
    )  # This is the KEY_LABEL from the tracker file, could be KEY or KEY#GI

    logger.info(
        f"CLI remove-key: Attempting to remove key label '{key_to_remove_str_arg}' from tracker '{tracker_file_path}'."
    )

    if not os.path.exists(tracker_file_path):
        print(f"Error: Tracker file not found: {tracker_file_path}")
        return 1

    try:
        with open(tracker_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        definitions_in_tracker = read_key_definitions_from_lines(
            lines
        )  # List[Tuple[key_label_in_file, path_str_in_file]]
    except Exception as e_read:
        print(f"Error reading tracker file {tracker_file_path}: {e_read}")
        return 1

    # Find all paths associated with the given key_label in this tracker
    matching_paths_for_key_label: List[str] = [
        p_str
        for k_label, p_str in definitions_in_tracker
        if k_label == key_to_remove_str_arg
    ]

    if not matching_paths_for_key_label:
        print(
            f"Error: Key label '{key_to_remove_str_arg}' not found in definitions of tracker '{tracker_file_path}'."
        )
        return 1

    path_to_remove_final: str
    if len(matching_paths_for_key_label) == 1:
        path_to_remove_final = matching_paths_for_key_label[0]
        logger.info(
            f"Key label '{key_to_remove_str_arg}' uniquely maps to path '{path_to_remove_final}' in this tracker."
        )
    else:
        # This case should be rare if display keys (KEY#GI) are used correctly in trackers for duplicates.
        # If a base KEY is used as a label and it's ambiguous *within this tracker's definitions*, it's an issue.
        print(
            f"Error: Key label '{key_to_remove_str_arg}' is ambiguous within tracker '{tracker_file_path}'. It maps to multiple paths:"
        )
        for i, p_match_ambig in enumerate(matching_paths_for_key_label):
            print(f"  [{i+1}] {p_match_ambig}")
        print(
            "This indicates an inconsistency in the tracker file or the key label provided. "
        )
        print(
            "If trying to remove a globally duplicated key, ensure you are using its unique path or a unique label from the tracker."
        )
        return 1

    try:
        # remove_path_from_tracker expects the actual path string
        remove_path_from_tracker(tracker_file_path, path_to_remove_final)
        print(
            f"Successfully initiated removal of path '{path_to_remove_final}' (associated with key label '{key_to_remove_str_arg}') from tracker '{tracker_file_path}'."
        )
        return 0
    except FileNotFoundError as e_fnf_rem:
        print(f"Error during removal: {e_fnf_rem}")
        return 1
    except ValueError as e_val_rem:
        print(f"Error during removal: {e_val_rem}")
        return 1
    except Exception as e_rem_generic:
        logger.error(
            f"Failed to remove path '{path_to_remove_final}': {str(e_rem_generic)}",
            exc_info=True,
        )
        print(f"Error removing path: {e_rem_generic}")
        return 1


def handle_add_dependency(args: argparse.Namespace) -> int:
    """Handle the add-dependency command using globally-referenced key instances. Allows adding foreign keys to mini-trackers."""
    tracker_path = normalize_path(args.tracker)
    source_key_arg_raw: str = args.source_key
    target_keys_arg_raw: List[str] = args.target_key
    dep_type: str = args.dep_type

    # --- Import moved for early use ---
    from cline_utils.dependency_system.io.update_doc_tracker import doc_tracker_data

    # ---

    config = ConfigManager()
    ALLOWED_DEP_TYPES = config.get_allowed_dependency_chars() + [
        PLACEHOLDER_CHAR,
        EMPTY_CHAR,
    ]
    if dep_type not in ALLOWED_DEP_TYPES:
        print(
            f"Error: Invalid dependency type '{dep_type}'. Allowed: {ALLOWED_DEP_TYPES}"
        )
        return 1

    logger.info(
        f"CLI add-dependency (Global Instance Mode): User input: {source_key_arg_raw} -> {target_keys_arg_raw} ('{dep_type}') in {tracker_path}"
    )

    # Determine tracker type early
    is_mini_add = tracker_path.endswith("_module.md")
    tracker_type_val_add = (
        "mini"
        if is_mini_add
        else ("doc" if "doc_tracker.md" in os.path.basename(tracker_path) else "main")
    )

    # Tracker existence check (allow non-existent for mini-trackers as update_tracker can create them)
    if not os.path.exists(tracker_path) and not tracker_path.endswith("_module.md"):
        logger.error(
            f"Tracker file '{tracker_path}' does not exist and is not a mini-tracker. Cannot add dependency."
        )
        print(f"Error: Tracker file '{tracker_path}' not found.")
        return 1
    elif not os.path.exists(tracker_path):  # Mini-tracker that doesn't exist yet
        logger.warning(
            f"Tracker file '{tracker_path}' does not exist. `update_tracker` will attempt to create it if it's a mini-tracker."
        )

    global_map = _load_global_map_or_exit()  # This is path_to_key_info
    project_root = get_project_root()

    # --- Pre-filter valid paths if tracker type requires it (e.g., 'doc') ---
    valid_paths_for_tracker: Optional[Set[str]] = None
    if tracker_type_val_add == "doc":
        filtered_items_map: Dict[str, KeyInfo] = doc_tracker_data["file_inclusion"](
            project_root, global_map
        )
        valid_paths_for_tracker = set(filtered_items_map.keys())
        logger.debug(
            f"Doc tracker mode: {len(valid_paths_for_tracker)} valid doc paths identified for filtering."
        )

    # --- Resolve Source Key (Globally) ---
    src_parts = source_key_arg_raw.split("#")
    src_base_key_str = src_parts[0]
    src_user_global_instance_num: Optional[int] = None
    if len(src_parts) > 1:
        try:
            src_user_global_instance_num = int(src_parts[1])
        except ValueError:
            print(
                f"Error: Invalid instance number format in source key '{source_key_arg_raw}'. Must be '#<number>'."
            )
            return 1

    matching_source_infos = [
        info
        for info in global_map.values()
        if info.key_string.split("#")[0] == src_base_key_str
    ]
    if not matching_source_infos:
        print(
            f"Error: Base source key '{src_base_key_str}' not found in global key map."
        )
        return 1

    matching_source_infos.sort(key=lambda ki: ki.norm_path)
    resolved_source_ki: Optional[KeyInfo] = None
    if src_user_global_instance_num is not None:
        source_key_to_find = f"{src_base_key_str}#{src_user_global_instance_num}"
        found_ki = next(
            (ki for ki in matching_source_infos if ki.key_string == source_key_to_find),
            None,
        )
        if found_ki:
            resolved_source_ki = found_ki
        else:
            print(
                f"Error: Source key '{source_key_arg_raw}' specifies an invalid global instance number."
            )
            print(f"Available instances for '{src_base_key_str}':")
            for ki in matching_source_infos:
                print(f"  - {ki.key_string} (Path: {ki.norm_path})")
            return 1
    elif len(matching_source_infos) > 1:
        print(
            f"Error: Source key '{src_base_key_str}' is globally ambiguous. Please specify which instance you mean using '#<num>':"
        )
        for ki in matching_source_infos:
            print(f"  - {ki.key_string} (Path: {ki.norm_path})")
        return 1
    else:
        resolved_source_ki = matching_source_infos[0]

    if not resolved_source_ki:
        return 1

    # --- NEW: Validate source key against tracker type ---
    if (
        valid_paths_for_tracker is not None
        and resolved_source_ki.norm_path not in valid_paths_for_tracker
    ):
        print(
            f"Error: Source key '{source_key_arg_raw}' ({resolved_source_ki.norm_path}) is not a valid item for the '{tracker_type_val_add}' tracker. Aborting."
        )
        logger.error(
            f"Source path {resolved_source_ki.norm_path} rejected by '{tracker_type_val_add}' tracker filter."
        )
        return 1

    final_source_key_for_suggestion = get_key_global_instance_string(
        resolved_source_ki, global_map
    )
    if not final_source_key_for_suggestion:
        logger.error(
            f"Logic error: Could not get KEY#GI for resolved source KI: {resolved_source_ki}"
        )
        print("Internal error resolving source key instance.")
        return 1
    logger.info(
        f"Resolved source for suggestion: '{final_source_key_for_suggestion}' (Path: {resolved_source_ki.norm_path})"
    )

    # --- NEW: Initialize lists to track valid and rejected targets ---
    final_target_keys_for_suggestion_list: List[Tuple[str, str]] = []
    checklist_updates_pending: List[Tuple[str, str, str, str, str]] = []
    rejected_targets: List[Tuple[str, str]] = []  # (raw_key, reason)

    for tgt_key_arg_item_raw in target_keys_arg_raw:
        tgt_parts = tgt_key_arg_item_raw.split("#")
        tgt_base_key_str = tgt_parts[0]
        tgt_user_global_instance_num: Optional[int] = None
        if len(tgt_parts) > 1:
            try:
                tgt_user_global_instance_num = int(tgt_parts[1])
            except ValueError:
                print(
                    f"Error: Invalid instance number format in target key '{tgt_key_arg_item_raw}'. Skipping this target."
                )
                rejected_targets.append(
                    (tgt_key_arg_item_raw, "Invalid instance number format.")
                )
                continue

        matching_target_infos = [
            info
            for info in global_map.values()
            if info.key_string.split("#")[0] == tgt_base_key_str
        ]
        if not matching_target_infos:
            print(
                f"Error: Base target key '{tgt_base_key_str}' not found in global key map."
            )
            rejected_targets.append(
                (tgt_key_arg_item_raw, "Base key not found in global map.")
            )
            continue

        matching_target_infos.sort(key=lambda ki: ki.norm_path)
        resolved_target_ki: Optional[KeyInfo] = None
        if tgt_user_global_instance_num is not None:
            target_key_to_find = f"{tgt_base_key_str}#{tgt_user_global_instance_num}"
            found_ki = next(
                (
                    ki
                    for ki in matching_target_infos
                    if ki.key_string == target_key_to_find
                ),
                None,
            )
            if found_ki:
                resolved_target_ki = found_ki
            else:
                print(
                    f"Error: Target key '{tgt_key_arg_item_raw}' specifies an invalid global instance number."
                )
                print(f"Available instances for '{tgt_base_key_str}':")
                for ki in matching_target_infos:
                    print(f"  - {ki.key_string} (Path: {ki.norm_path})")
                rejected_targets.append(
                    (tgt_key_arg_item_raw, "Invalid global instance number.")
                )
                continue
        elif len(matching_target_infos) > 1:
            print(
                f"Error: Target key '{tgt_base_key_str}' is globally ambiguous. Please specify which instance you mean using '#<num>':"
            )
            for ki in matching_target_infos:
                print(f"  - {ki.key_string} (Path: {ki.norm_path})")
            rejected_targets.append((tgt_key_arg_item_raw, "Globally ambiguous key."))
            continue
        else:
            resolved_target_ki = matching_target_infos[0]

        if not resolved_target_ki:
            # This case is already covered by the ambiguity/resolution logic above, but as a safeguard:
            if (
                tgt_key_arg_item_raw,
                "Could not be resolved globally.",
            ) not in rejected_targets:
                rejected_targets.append(
                    (tgt_key_arg_item_raw, "Could not be resolved globally.")
                )
            continue

        # --- NEW: Validate target key against tracker type ---
        if (
            valid_paths_for_tracker is not None
            and resolved_target_ki.norm_path not in valid_paths_for_tracker
        ):
            reason = f"Path '{resolved_target_ki.norm_path}' is not a valid item for the '{tracker_type_val_add}' tracker."
            logger.warning(f"Rejected target '{tgt_key_arg_item_raw}': {reason}")
            rejected_targets.append((tgt_key_arg_item_raw, reason))
            continue

        final_target_key_for_suggestion = get_key_global_instance_string(
            resolved_target_ki, global_map
        )
        if not final_target_key_for_suggestion:  # Should not happen
            logger.error(
                f"Logic error: Could not get KEY#GI for resolved target KI: {resolved_target_ki}"
            )
            print(
                f"Internal error resolving target key instance for '{tgt_key_arg_item_raw}'."
            )
            rejected_targets.append(
                (tgt_key_arg_item_raw, "Internal error getting global instance string.")
            )
            continue
        logger.info(
            f"Resolved target for suggestion: '{final_target_key_for_suggestion}' (Path: {resolved_target_ki.norm_path})"
        )

        # Check for self-dependency using the resolved global paths
        if resolved_source_ki.norm_path == resolved_target_ki.norm_path:
            logger.warning(
                f"Skipping self-dependency (same global path): {final_source_key_for_suggestion} to {final_target_key_for_suggestion}"
            )
            continue

        # This target is valid, add it to the list for update_tracker
        final_target_keys_for_suggestion_list.append(
            (final_target_key_for_suggestion, dep_type)
        )

        # For checklist (using globally resolved KeyInfo objects' base keys and paths)
        src_item_type_chk = get_item_type_for_checklist(
            resolved_source_ki.norm_path, config, project_root
        )
        tgt_item_type_chk = get_item_type_for_checklist(
            resolved_target_ki.norm_path, config, project_root
        )
        if (src_item_type_chk == "code" and tgt_item_type_chk == "doc") or (
            src_item_type_chk == "doc" and tgt_item_type_chk == "code"
        ):
            checklist_updates_pending.append(
                (
                    resolved_source_ki.key_string,
                    resolved_source_ki.norm_path,
                    resolved_target_ki.key_string,
                    resolved_target_ki.norm_path,
                    dep_type,
                )
            )

    # --- After the loop, check what we have ---
    if not final_target_keys_for_suggestion_list and not checklist_updates_pending:
        print(
            "No valid dependencies resolved to apply to tracker or checklist after validation and ambiguity checks."
        )
        if rejected_targets:
            print("\nThe following targets were rejected:")
            for key, reason in rejected_targets:
                print(f"  - {key}: {reason}")
        return 0

    suggestions_for_update_tracker: Optional[Dict[str, List[Tuple[str, str]]]] = None
    if final_target_keys_for_suggestion_list:
        suggestions_for_update_tracker = {
            final_source_key_for_suggestion: final_target_keys_for_suggestion_list
        }

    file_to_module_map = {
        info.norm_path: info.parent_path
        for info in global_map.values()
        if not info.is_directory and info.parent_path
    }

    try:
        if suggestions_for_update_tracker:
            logger.info(
                f"Calling update_tracker for '{tracker_path}' with globally-instanced suggestions: {suggestions_for_update_tracker} (Force Apply: True, AST Overrides: False)"
            )
            update_tracker(
                output_file_suggestion=tracker_path,
                path_to_key_info=global_map,
                tracker_type=tracker_type_val_add,
                suggestions_external=suggestions_for_update_tracker,
                file_to_module=file_to_module_map,
                force_apply_suggestions=True,
                apply_ast_overrides=False,  # <<< MODIFIED/ADDED
            )
            # --- NEW: More informative message ---
            print(
                f"Successfully processed {len(final_target_keys_for_suggestion_list)} dependency addition(s) for tracker {tracker_path}."
            )
        else:
            logger.debug(
                f"No direct tracker updates to apply for {tracker_path} based on CLI input (possibly all targets skipped or invalid)."
            )

        if checklist_updates_pending:
            logger.debug(
                f"Attempting to update checklist with {len(checklist_updates_pending)} code-doc dependencies."
            )
            all_checklist_ok_add = True
            successful_checklist_adds = 0
            # --- MODIFIED to handle new return type from checklist function ---
            for (
                src_k_c,
                src_p_c,
                tgt_k_c,
                tgt_p_c,
                dep_t_c,
            ) in checklist_updates_pending:
                # Pass base key strings to checklist function
                result = add_code_doc_dependency_to_checklist(src_k_c, tgt_k_c, dep_t_c)
                if result is False:  # Explicit check for error
                    all_checklist_ok_add = False
                    logger.error(
                        f"Failed to add {src_k_c} ('{src_p_c}') -> {tgt_k_c} ('{tgt_p_c}') with type '{dep_t_c}' to review checklist."
                    )
                elif result is True:  # Explicit check for new addition
                    successful_checklist_adds += 1
                    logger.info(
                        f"Added dependency {src_k_c} ('{src_p_c}') -> {tgt_k_c} ('{tgt_p_c}') with type '{dep_t_c}' to review checklist."
                    )
                # If result is None (duplicate), we just log nothing, which is fine.

            # --- NEW: More informative message ---
            if successful_checklist_adds > 0:
                print(
                    f"Successfully added {successful_checklist_adds} new code-doc dependencies to the review checklist."
                )
            if not all_checklist_ok_add:
                print(
                    "Warning: Some code-doc dependencies could not be added/updated in the review checklist."
                )

        # --- NEW: Report rejected targets ---
        if rejected_targets:
            print("\nThe following targets were rejected and not processed:")
            for key, reason in rejected_targets:
                print(f"  - {key}: {reason}")
        return 0
    except Exception as e_add_dep_proc:
        logger.error(
            f"Error processing add-dependency for '{tracker_path}': {e_add_dep_proc}",
            exc_info=True,
        )
        print(f"Error processing add-dependency for '{tracker_path}': {e_add_dep_proc}")
        return 1


def handle_merge_trackers(args: argparse.Namespace) -> int:
    """Handle the merge-trackers command."""
    try:
        primary_path = normalize_path(args.primary_tracker_path)
        secondary_path = normalize_path(args.secondary_tracker_path)
        output_p = normalize_path(args.output) if args.output else primary_path

        merged_result_data = merge_trackers(primary_path, secondary_path, output_p)

        if merged_result_data:
            print(
                f"Merged trackers into {output_p}. Total items in merged definitions: {len(merged_result_data.get('key_info_list', []))}"
            )
            return 0
        else:
            print(
                f"Error merging trackers. `merge_trackers` returned: {merged_result_data}"
            )
            return 1
    except Exception as e_merge:
        logger.exception(f"Failed merge: {e_merge}")
        print(f"Error: {e_merge}")
        return 1


def handle_clear_caches(args: argparse.Namespace) -> int:
    try:
        clear_all_caches()
        print("All caches cleared.")
        return 0
    except Exception as e:
        logger.exception(f"Error clearing caches: {e}")
        print(f"Error: {e}")
        return 1


def handle_export_tracker(args: argparse.Namespace) -> int:
    """Handle the export-tracker command."""
    try:
        export_result_path_or_msg = export_tracker(
            args.tracker_file, args.format, args.output
        )
        if export_result_path_or_msg.startswith("Error:"):
            print(export_result_path_or_msg)
            return 1
        print(f"Tracker exported to {export_result_path_or_msg}")
        return 0
    except Exception as e_export:
        logger.exception(f"Error export_tracker: {e_export}")
        print(f"Error: {e_export}")
        return 1


def handle_update_config(args: argparse.Namespace) -> int:
    """Handle the update-config command."""
    config_manager = ConfigManager()
    try:
        try:
            value_parsed: Union[str, int, float, List[Any], Dict[str, Any]] = (
                json.loads(args.value)
            )
        except json.JSONDecodeError:
            value_parsed = args.value
        success = config_manager.update_config_setting(args.key, value_parsed)
        if success:
            print(f"Updated config: {args.key} = {value_parsed}")
            return 0
        else:
            print(f"Error: Failed update config (key '{args.key}' invalid?).")
            return 1
    except Exception as e:
        logger.exception(f"Error update_config: {e}")
        print(f"Error: {e}")
        return 1


def handle_reset_config(args: argparse.Namespace) -> int:
    """Handle the reset-config command."""
    config_manager = ConfigManager()
    try:
        success = config_manager.reset_to_defaults()
        if success:
            print("Config reset to defaults.")
            return 0
        else:
            print("Error: Failed reset config.")
            return 1
    except Exception as e:
        logger.exception(f"Error reset_config: {e}")
        print(f"Error: {e}")
        return 1


def handle_show_dependencies(args: argparse.Namespace) -> int:
    """
    Handle the show-dependencies command.
    Shows all relationships for a given key, directly from each tracker file where it's defined or linked.
    """
    user_provided_key_arg: str = args.key
    logger.info(
        f"ShowDependencies: User requested dependencies for '{user_provided_key_arg}'"
    )

    current_global_map = _load_global_map_or_exit()  # path_to_key_info
    config = ConfigManager()
    project_root = get_project_root()

    parts = user_provided_key_arg.split("#")
    base_key_to_show = parts[0]
    user_instance_num_to_show: Optional[int] = None
    if len(parts) > 1:
        try:
            user_instance_num_to_show = int(parts[1])
        except ValueError:
            print(
                f"Error: Invalid instance number in key '{user_provided_key_arg}'. Use format KEY#num."
            )
            return 1

    # Resolve the user-provided key to a specific KeyInfo object (target_ki_to_show)
    # This target_ki_to_show's path and global instance string will be the focus.
    matching_source_infos = [
        info
        for info in current_global_map.values()
        if info.key_string.split("#")[0] == base_key_to_show
    ]
    if not matching_source_infos:
        print(
            f"Error: Base source key '{base_key_to_show}' not found in global key map."
        )
        return 1

    matching_source_infos.sort(key=lambda ki: ki.norm_path)
    target_ki_to_show: Optional[KeyInfo] = None
    if user_instance_num_to_show is not None:
        if 0 < user_instance_num_to_show <= len(matching_source_infos):
            target_ki_to_show = matching_source_infos[user_instance_num_to_show - 1]
        else:
            print(
                f"Error: Source key '{user_provided_key_arg}' specifies an invalid global instance number. Max is {len(matching_source_infos)}."
            )
            return 1
    elif len(matching_source_infos) > 1:
        print(
            f"Error: Source key '{base_key_to_show}' is globally ambiguous. Please specify which instance you mean using '#<num>':"
        )
        for i, ki in enumerate(matching_source_infos):
            print(f"  [{i+1}] {ki.key_string} (Path: {ki.norm_path})")
        return 1
    else:
        target_ki_to_show = matching_source_infos[0]

    if not target_ki_to_show:
        return 1

    target_key_gi_str_to_show = get_key_global_instance_string(
        target_ki_to_show, current_global_map
    )
    if not target_key_gi_str_to_show:
        print(
            f"Error: Could not determine global instance string for resolved KeyInfo {target_ki_to_show}."
        )
        return 1

    print(
        f"\n--- Dependencies for: {target_key_gi_str_to_show} (Path: {target_ki_to_show.norm_path}) ---"
    )

    # Pre-calculate global counts for display formatting
    global_key_string_counts: defaultdict[str, int] = defaultdict(int)
    for ki_count in current_global_map.values():
        global_key_string_counts[ki_count.key_string] += 1

    all_tracker_paths = find_all_tracker_paths(config, project_root)

    # Structure: Dict[char_type, Dict[interacting_key_gi_str, List[origin_tracker_basename]]]
    all_deps_by_char_type_and_origin: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # Outer key: char_type (e.g. 'p', 'n', 'x')
    # Inner key: interacting_key_gi_str (e.g. '1Bc4#1')
    # Value: List of origin tracker basenames (e.g. ['database_module.md', 'config_module.md'])

    for tracker_path in all_tracker_paths:
        logger.debug(
            f"ShowDeps: Processing tracker '{os.path.basename(tracker_path)}' for key '{target_key_gi_str_to_show}'"
        )
        try:
            with open(tracker_path, "r", encoding="utf-8") as f_tracker:
                lines = f_tracker.readlines()

            defs_in_this_tracker = read_key_definitions_from_lines(lines)
            _grid_hdrs, grid_rows_in_this_tracker = read_grid_from_lines(lines)

            if not defs_in_this_tracker or not grid_rows_in_this_tracker:
                logger.debug(
                    f"  Skipping tracker {os.path.basename(tracker_path)}: no definitions or grid rows found."
                )
                continue

            # Create a mapping from path_str_in_file to its index in this tracker's definitions
            path_to_idx_in_this_tracker: Dict[str, int] = {}
            # Also, map path_str_in_file to its original key_label_in_file for reverse lookups
            path_to_key_label_in_this_tracker: Dict[str, str] = {}

            for i, (k_label, p_str) in enumerate(defs_in_this_tracker):
                if (
                    p_str not in path_to_idx_in_this_tracker
                ):  # First occurrence if path duplicated in defs
                    path_to_idx_in_this_tracker[p_str] = i
                    path_to_key_label_in_this_tracker[p_str] = k_label

            # Check if our target_ki_to_show.norm_path is defined in this tracker
            source_row_idx_in_this_tracker = path_to_idx_in_this_tracker.get(
                target_ki_to_show.norm_path
            )

            # 1. Process outgoing relationships (target_ki_to_show is the source)
            if (
                source_row_idx_in_this_tracker is not None
                and source_row_idx_in_this_tracker < len(grid_rows_in_this_tracker)
            ):
                row_label_from_grid, compressed_row_data = grid_rows_in_this_tracker[
                    source_row_idx_in_this_tracker
                ]

                # Sanity check: row_label from grid should match the key_label from definitions for this path
                expected_row_label = path_to_key_label_in_this_tracker.get(
                    target_ki_to_show.norm_path
                )
                if row_label_from_grid != expected_row_label:
                    logger.warning(
                        f"  Label mismatch in {os.path.basename(tracker_path)} for path {target_ki_to_show.norm_path}. Def label: {expected_row_label}, Grid row label: {row_label_from_grid}. Proceeding cautiously."
                    )

                decomp_row = decompress(compressed_row_data)
                if len(decomp_row) != len(defs_in_this_tracker):
                    logger.warning(
                        f"  Row length mismatch in {os.path.basename(tracker_path)} for source {row_label_from_grid}. Expected {len(defs_in_this_tracker)}, got {len(decomp_row)}. Skipping row."
                    )
                else:
                    for col_idx, char_val in enumerate(decomp_row):
                        if char_val == DIAGONAL_CHAR or char_val == EMPTY_CHAR:
                            continue

                        # Get path of the item at col_idx from this tracker's definitions
                        if col_idx < len(defs_in_this_tracker):
                            interacting_item_path_in_tracker = defs_in_this_tracker[
                                col_idx
                            ][1]
                            interacting_item_ki_global = current_global_map.get(
                                interacting_item_path_in_tracker
                            )
                            if interacting_item_ki_global:
                                interacting_item_gi_str = (
                                    get_key_global_instance_string(
                                        interacting_item_ki_global, current_global_map
                                    )
                                )
                                if interacting_item_gi_str:
                                    all_deps_by_char_type_and_origin[char_val][
                                        interacting_item_gi_str
                                    ].append(os.path.basename(tracker_path))
            # else:
            # logger.debug(f"  Key {target_key_gi_str_to_show} (path {target_ki_to_show.norm_path}) not found as a row source in {os.path.basename(tracker_path)} or grid data missing.")

        except Exception as e_tracker_proc:
            logger.error(
                f"Error processing tracker {os.path.basename(tracker_path)} for show-dependencies: {e_tracker_proc}",
                exc_info=True,
            )

    # --- Displaying the collected results ---
    output_sections_disp = [
        ("Mutual ('x')", "x"),
        ("Doc ('d')", "d"),
        ("Semantic ('S')", "S"),
        ("Semantic ('s')", "s"),
        ("Depends On ('<')", "<"),
        ("Depended On By ('>')", ">"),
        ("Placeholder ('p')", "p"),
        # "No Dependency ('n')" section is intentionally omitted from display
    ]

    for title, char_filter in output_sections_disp:
        print(f"\n{title}:")

        deps_for_this_char = all_deps_by_char_type_and_origin.get(char_filter, {})
        if not deps_for_this_char:
            print("  None")
            continue

        sorted_interacting_keys_gi = sorted(
            deps_for_this_char.keys(),
            key=lambda k_gi_str: get_sortable_parts_for_key(k_gi_str),
        )

        for interacting_key_gi in sorted_interacting_keys_gi:
            interacting_ki = resolve_key_global_instance_to_ki(
                interacting_key_gi, current_global_map
            )
            if not interacting_ki:  # Should not happen if GI string is valid
                print(
                    f"  - {interacting_key_gi}: PATH_UNKNOWN (Error resolving GI string)"
                )
                continue

            # Prepare display name for the interacting key (use base key if not globally duplicated)
            interacting_base_key = interacting_key_gi.split("#")[0]
            display_name_interacting = interacting_key_gi
            if global_key_string_counts.get(interacting_base_key, 0) <= 1:
                display_name_interacting = interacting_base_key

            origin_trackers_list = sorted(
                list(set(deps_for_this_char[interacting_key_gi]))
            )
            origins_str = (
                f" (In: {', '.join(origin_trackers_list)})"
                if origin_trackers_list
                else ""
            )

            print(
                f"  - {display_name_interacting}: {interacting_ki.norm_path}{origins_str}"
            )

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
    logger.info(
        f"Attempting to show keys and check status (p, s, S) from tracker: {tracker_path}"
    )

    if not os.path.exists(tracker_path):
        print(f"Error: Tracker file not found: {tracker_path}", file=sys.stderr)
        return 1

    global_map = load_global_key_map()
    if not global_map:
        logger.warning(
            "ShowKeys: Could not load global key map. Global instance numbers will not be shown for duplicates."
        )
        # Fallback: create an empty map so `get_key_global_instance_string` doesn't fail if called with it
        global_map_for_instance_check: Dict[str, KeyInfo] = {}
    else:
        global_map_for_instance_check = global_map

    # Pre-calculate global counts for each base key string to identify duplicates
    global_key_string_counts: defaultdict[str, int] = defaultdict(int)
    if global_map:
        for ki in global_map.values():
            global_key_string_counts[ki.key_string] += 1

    try:
        with open(tracker_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        key_def_pairs_from_file = read_key_definitions_from_lines(lines)
        _grid_headers, grid_rows_data_list = read_grid_from_lines(lines)

        if not key_def_pairs_from_file:
            print(f"No key definitions found in tracker: {tracker_path}")
            return 0

        print(
            f"--- Keys Defined in {os.path.basename(tracker_path)} (Order as in File) ---"
        )

        for idx, (key_str_in_file, path_str_in_file) in enumerate(
            key_def_pairs_from_file
        ):
            status_indicator = ""
            # Check for p, s, S in the grid row for this item
            if idx < len(grid_rows_data_list):
                _row_label_from_grid, compressed_row = grid_rows_data_list[idx]
                if compressed_row:
                    # Check for 'p', 's', 'S' in the *decompressed* row for accuracy
                    decomp_row_for_check = decompress(compressed_row)
                    found_chars = {
                        char for char in decomp_row_for_check if char in ("p", "s", "S")
                    }
                    if found_chars:
                        status_indicator += (
                            f" (Checks needed: {', '.join(sorted(list(found_chars)))})"
                        )
            else:
                status_indicator += " (Grid row data missing)"

            # Determine if this key_str_in_file is globally duplicated and add #GI
            global_instance_suffix = ""
            # key_str_in_file could be "KEY" or "KEY#GI". We need its base key for global_key_string_counts.
            base_key_from_label = key_str_in_file.split("#")[0]
            if global_map and global_key_string_counts.get(base_key_from_label, 0) > 1:
                key_info_for_this_entry = global_map.get(path_str_in_file)
                if key_info_for_this_entry:  # Check if path is in global map
                    # Get the canonical KEY#GI for this path from the global map
                    gi_str_canonical = get_key_global_instance_string(
                        key_info_for_this_entry, global_map_for_instance_check
                    )
                    if gi_str_canonical:
                        global_instance_suffix = f" (Global: {gi_str_canonical})"
                        # If the label in the file doesn't match the canonical GI, note it.
                        if (
                            key_str_in_file != gi_str_canonical
                            and key_str_in_file == base_key_from_label
                        ):  # Label was base, but has GI
                            global_instance_suffix += f" - Label in file is base key"
                        elif (
                            key_str_in_file != gi_str_canonical
                        ):  # Label was specific GI, but different from canonical
                            global_instance_suffix += f" - Label in file '{key_str_in_file}' differs from canonical"
                    else:  # Should not happen if key_info_for_this_entry is valid
                        global_instance_suffix = (
                            f" (Global: {base_key_from_label}#? - Error getting GI)"
                        )
                else:
                    global_instance_suffix = f" (Global: {base_key_from_label}#? - Path not in current global map)"

            print(
                f"{key_str_in_file}: {path_str_in_file}{global_instance_suffix}{status_indicator}"
            )

        print("--- End of Key Definitions ---")
        try:
            with open(tracker_path, "r", encoding="utf-8") as f_check:
                content = f_check.read()
                if KEY_DEFINITIONS_START_MARKER not in content:
                    logger.warning(
                        f"Start marker '{KEY_DEFINITIONS_START_MARKER}' not found in {tracker_path}"
                    )
                if KEY_DEFINITIONS_END_MARKER not in content:
                    logger.warning(
                        f"End marker '{KEY_DEFINITIONS_END_MARKER}' not found in {tracker_path}"
                    )
        except Exception:
            logger.warning(f"Could not perform marker check on {tracker_path}")
        return 0
    except IOError as e:
        print(f"Error reading tracker file {tracker_path}: {e}", file=sys.stderr)
        logger.error(f"IOError reading {tracker_path}: {e}", exc_info=True)
        return 1
    except Exception as e:
        print(
            f"An unexpected error occurred while processing {tracker_path}: {e}",
            file=sys.stderr,
        )
        logger.error(f"Unexpected error processing {tracker_path}: {e}", exc_info=True)
        return 1


def handle_show_placeholders(args: argparse.Namespace) -> int:
    """
    Handle the show-placeholders command.
    Finds and displays all unverified dependencies ('p', 's', 'S').
    """
    tracker_path = normalize_path(args.tracker)
    focus_key = args.key
    dep_char_filter = args.dep_char

    if not os.path.exists(tracker_path):
        print(f"Error: Tracker file not found: {tracker_path}", file=sys.stderr)
        return 1

    chars_to_check: Tuple[str, ...]
    if dep_char_filter:
        chars_to_check = (dep_char_filter,)
    else:
        chars_to_check = ("p", "s", "S")

    # --- Load Global Map for Path Resolution ---
    global_map = load_global_key_map()
    key_to_path_map = {}
    if global_map:
        for k_info in global_map.values():
            key_to_path_map[k_info.key_string] = k_info.norm_path

    try:
        with open(tracker_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        key_def_pairs = read_key_definitions_from_lines(lines)
        _grid_headers, grid_rows_data = read_grid_from_lines(lines)

        if not key_def_pairs or not grid_rows_data:
            print(f"No valid key definitions or grid data found in {tracker_path}.")
            return 0

        unverified_deps: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        all_row_labels = {row_label for row_label, _ in grid_rows_data}

        if focus_key and focus_key not in all_row_labels:
            print(f"Error: Key '{focus_key}' not found as a row in {tracker_path}.")
            return 1

        for row_idx, (row_label, compressed_row) in enumerate(grid_rows_data):
            if focus_key and row_label != focus_key:
                continue

            try:
                decompressed = decompress(compressed_row)
                if len(decompressed) != len(key_def_pairs):
                    logger.warning(
                        f"Row for key '{row_label}' has mismatched length. Expected {len(key_def_pairs)}, got {len(decompressed)}. Skipping row."
                    )
                    continue

                for col_idx, char in enumerate(decompressed):
                    if char in chars_to_check:
                        if col_idx < len(key_def_pairs):
                            target_label = key_def_pairs[col_idx][0]
                            unverified_deps[row_label][char].append(target_label)

            except Exception as e:
                logger.error(
                    f"Error processing row for key '{row_label}': {e}", exc_info=True
                )
                continue

        if not unverified_deps:
            print(
                f"No unverified dependencies {chars_to_check} found in {os.path.basename(tracker_path)}."
            )
            return 0

        print(
            f"Unverified dependencies {chars_to_check} in {os.path.basename(tracker_path)}:"
        )
        sorted_source_keys = sorted(
            unverified_deps.keys(), key=get_sortable_parts_for_key
        )
        for source_label in sorted_source_keys:
            source_path = key_to_path_map.get(source_label, "Path not found")
            print(f"\n--- Key: {source_label} (Path: {source_path}) ---")
            char_map = unverified_deps[source_label]
            for char_type in sorted(char_map.keys()):
                target_labels = sorted(
                    char_map[char_type], key=get_sortable_parts_for_key
                )
                print(f"  {char_type}:")
                for tgt in target_labels:
                    tgt_path = key_to_path_map.get(tgt, "Path not found")
                    print(f"    - {tgt} (Path: {tgt_path})")

        return 0

    except IOError as e:
        print(f"Error reading tracker file {tracker_path}: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(
            f"An unexpected error occurred while processing {tracker_path}: {e}",
            file=sys.stderr,
        )
        return 1


def handle_visualize_dependencies(args: argparse.Namespace) -> int:
    """Handles the visualize-dependencies command by calling the core generation function."""
    focus_keys_list_cli: List[str] = args.key if args.key is not None else []
    output_format_cli = args.format.lower()
    output_path_arg_cli = args.output

    logger.info(
        f"CLI: visualize-dependencies called. Focus Keys: {focus_keys_list_cli or 'Project Overview'}"
    )

    if output_format_cli != "mermaid":
        print(f"Error: Only 'mermaid' format supported at this time.")
        return 1

    try:
        current_global_map_cli = _load_global_map_or_exit()
        config_cli = ConfigManager()
        project_root_cli = get_project_root()
        # Use find_all_tracker_paths from tracker_utils (was tracker_io before)
        all_tracker_paths_cli = find_all_tracker_paths(config_cli, project_root_cli)
        if not all_tracker_paths_cli:
            print("Warning: No tracker files found. Diagram may be empty.")

        logger.debug(
            "Building path migration map for visualize-dependencies command..."
        )
        old_global_map_cli = load_old_global_key_map()
        path_migration_info_cli: PathMigrationInfo
        try:
            # Use build_path_migration_map from tracker_io
            path_migration_info_cli = build_path_migration_map(
                old_global_map_cli, current_global_map_cli
            )
        except ValueError as ve:
            logger.error(
                f"Failed to build migration map for visualize-dependencies: {ve}. Visualization may be based on current state only or fail."
            )
            path_migration_info_cli = {
                info.norm_path: (info.key_string, info.key_string)
                for info in current_global_map_cli.values()
            }
        except Exception as e:
            logger.error(
                f"Unexpected error building migration map for visualize-dependencies: {e}. Visualization may be inaccurate.",
                exc_info=True,
            )
            path_migration_info_cli = {
                info.norm_path: (info.key_string, info.key_string)
                for info in current_global_map_cli.values()
            }

    except Exception as e:
        logger.exception("Failed to load data required for visualization.")
        print(f"Error loading data needed for visualization: {e}", file=sys.stderr)
        return 1

    mermaid_string_generated = generate_mermaid_diagram(
        focus_keys_list_input=focus_keys_list_cli,
        global_path_to_key_info_map=current_global_map_cli,
        path_migration_info=path_migration_info_cli,
        all_tracker_paths_list=list(all_tracker_paths_cli),
        config_manager_instance=config_cli,
    )

    if mermaid_string_generated is None:
        print(
            "Error: Mermaid diagram generation failed internally. Check logs.",
            file=sys.stderr,
        )
        return 1
    elif "Error:" in mermaid_string_generated[:20]:
        print(mermaid_string_generated, file=sys.stderr)
        return 1
    elif "// No relevant data" in mermaid_string_generated:
        print(
            "Info: No relevant data found to visualize based on focus keys and filters."
        )
    else:
        logger.debug("Mermaid code generated successfully.")

    output_path_cli = output_path_arg_cli
    if not output_path_cli:
        if focus_keys_list_cli:
            # For focus keys, ensure they are resolved to KEY#GI for unique filenames if necessary
            resolved_focus_key_gis_for_filename: List[str] = []
            for fk_raw in focus_keys_list_cli:  # fk_raw is str
                fk_parts: List[str] = fk_raw.split("#")
                fk_base: str = fk_parts[0]
                fk_inst_num_user: Optional[int] = (
                    int(fk_parts[1]) if len(fk_parts) > 1 else None
                )
                fk_resolved_ki = get_globally_resolved_key_info_for_cli(
                    fk_base, fk_inst_num_user, current_global_map_cli, "filename focus"
                )
                if fk_resolved_ki:
                    fk_gi_str = get_key_global_instance_string(
                        fk_resolved_ki, current_global_map_cli
                    )
                    if fk_gi_str:
                        resolved_focus_key_gis_for_filename.append(fk_gi_str)
                else:  # Fallback to raw if resolution fails for filename part
                    resolved_focus_key_gis_for_filename.append(
                        fk_raw.replace("#", "_hash_")
                    )

            safe_keys_str = (
                "_".join(sorted(resolved_focus_key_gis_for_filename))
                .replace("/", "_")
                .replace("\\", "_")
                .replace("#", "_hash_")
            )
            max_len = 50
            if len(safe_keys_str) > max_len:
                safe_keys_str = safe_keys_str[:max_len] + "_etc"
            default_filename = f"focus_{safe_keys_str}_dependencies.{output_format_cli}"
        else:
            default_filename = f"project_overview_dependencies.{output_format_cli}"

        memory_dir_rel = config_cli.get_path("memory_dir", "cline_docs")
        default_output_dir_rel = os.path.join(memory_dir_rel, "dependency_diagrams")
        output_path_cli = normalize_path(
            os.path.join(project_root_cli, default_output_dir_rel, default_filename)
        )
        logger.debug(f"No output path specified, using default: {output_path_cli}")

    elif not os.path.isabs(output_path_cli):
        output_path_cli = normalize_path(
            os.path.join(project_root_cli, output_path_cli)
        )
    else:
        output_path_cli = normalize_path(output_path_cli)

    try:
        output_dir_cli = os.path.dirname(output_path_cli)
        if output_dir_cli:
            os.makedirs(output_dir_cli, exist_ok=True)

        with open(output_path_cli, "w", encoding="utf-8") as f_out:
            f_out.write(mermaid_string_generated)

        logger.info(f"Successfully wrote Mermaid visualization to: {output_path_cli}")
        print(f"\nDependency visualization saved to: {output_path_cli}")
        if "// No relevant data" not in mermaid_string_generated:
            print(
                "You can view this file using Mermaid Live Editor (mermaid.live) or compatible Markdown viewers."
            )
        return 0
    except IOError as e:
        logger.error(
            f"Failed to write visualization file {output_path_cli}: {e}", exc_info=True
        )
        print(f"Error: Could not write output file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred during visualization file writing: {e}"
        )
        print(
            f"Error: An unexpected error occurred while writing output: {e}",
            file=sys.stderr,
        )
        return 1


def main():
    """Parse arguments and dispatch to handlers."""
    parser = argparse.ArgumentParser(description="Dependency tracking system CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # --- Analysis Commands ---
    analyze_file_parser = subparsers.add_parser(
        "analyze-file", help="Analyze a single file"
    )
    analyze_file_parser.add_argument("file_path", help="Path to the file")
    analyze_file_parser.add_argument("--output", help="Save results to JSON file")
    analyze_file_parser.set_defaults(func=command_handler_analyze_file)

    analyze_project_parser = subparsers.add_parser(
        "analyze-project",
        help="Analyze project, generate keys/embeddings, update trackers",
    )
    analyze_project_parser.add_argument(
        "project_root",
        nargs="?",
        default=".",
        help="Project directory path (default: CWD)",
    )
    analyze_project_parser.add_argument(
        "--output", help="Save analysis summary to JSON file"
    )
    analyze_project_parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Force regeneration of embeddings",
    )
    analyze_project_parser.add_argument(
        "--force-analysis",
        action="store_true",
        help="Force re-analysis and bypass cache",
    )
    analyze_project_parser.add_argument(
        "--force-validate",
        action="store_true",
        help="Force fresh resource validation, bypassing cache",
    )
    analyze_project_parser.set_defaults(func=command_handler_analyze_project)

    # --- Grid Manipulation Commands ---
    compress_parser = subparsers.add_parser("compress", help="Compress RLE string")
    compress_parser.add_argument("string", help="String to compress")
    compress_parser.set_defaults(func=handle_compress)

    decompress_parser = subparsers.add_parser(
        "decompress", help="Decompress RLE string"
    )
    decompress_parser.add_argument("string", help="String to decompress")
    decompress_parser.set_defaults(func=handle_decompress)

    get_char_parser = subparsers.add_parser(
        "get_char", help="Get char at logical index in compressed string"
    )
    get_char_parser.add_argument("string", help="Compressed string")
    get_char_parser.add_argument("index", type=int, help="Logical index")
    get_char_parser.set_defaults(func=handle_get_char)

    set_char_parser = subparsers.add_parser(
        "set_char",
        help="DEPRECATED & UNSAFE: Set char in a tracker file. Use 'add-dependency' instead.",
    )
    set_char_parser.add_argument("tracker_file", help="Path to tracker file")
    set_char_parser.add_argument(
        "key",
        type=str,
        help="Row key label from tracker definitions (e.g., '1A1' or '1A1#2')",
    )
    set_char_parser.add_argument(
        "index",
        type=int,
        help="Logical index in row (0-based, refers to Nth definition in tracker's original order)",
    )
    set_char_parser.add_argument("char", type=str, help="New character")
    set_char_parser.set_defaults(func=handle_set_char)

    add_dep_parser = subparsers.add_parser(
        "add-dependency",
        help="Add dependency between keys (supports #instance for duplicates)",
    )
    add_dep_parser.add_argument("--tracker", required=True, help="Path to tracker file")
    add_dep_parser.add_argument(
        "--source-key", required=True, help="Source key string (e.g., '1A1' or '1A1#2')"
    )
    add_dep_parser.add_argument(
        "--target-key",
        required=True,
        nargs="+",
        help="One or more target key strings (e.g., '2Ba2' or '2Ba2#1')",
    )
    add_dep_parser.add_argument(
        "--dep-type", default=">", help="Dependency type (e.g., '>', '<', 'x')"
    )
    add_dep_parser.set_defaults(func=handle_add_dependency)

    # --- Tracker File Management ---
    remove_key_parser = subparsers.add_parser(
        "remove-key",
        help="Remove an item by its key label from a specific tracker (resolves to path)",
    )
    remove_key_parser.add_argument(
        "tracker_file", help="Path to the tracker file (.md)"
    )
    remove_key_parser.add_argument(
        "key",
        type=str,
        help="The key label (e.g., '1A1' or '1A1#2') from the tracker file to remove. If ambiguous in tracker, command will error.",
    )
    remove_key_parser.set_defaults(func=handle_remove_key)

    merge_parser = subparsers.add_parser(
        "merge-trackers", help="Merge two tracker files"
    )
    merge_parser.add_argument("primary_tracker_path", help="Primary tracker")
    merge_parser.add_argument("secondary_tracker_path", help="Secondary tracker")
    merge_parser.add_argument(
        "--output", "-o", help="Output path (defaults to overwriting primary)"
    )
    merge_parser.set_defaults(func=handle_merge_trackers)

    export_parser = subparsers.add_parser("export-tracker", help="Export tracker data")
    export_parser.add_argument("tracker_file", help="Path to tracker file")
    export_parser.add_argument(
        "--format",
        choices=["json", "csv", "dot", "md"],
        default="json",
        help="Export format",
    )
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.set_defaults(func=handle_export_tracker)

    # --- Utility Commands ---
    clear_caches_parser = subparsers.add_parser(
        "clear-caches", help="Clear all internal caches"
    )
    clear_caches_parser.set_defaults(func=handle_clear_caches)

    reset_config_parser = subparsers.add_parser(
        "reset-config", help="Reset config to defaults"
    )
    reset_config_parser.set_defaults(func=handle_reset_config)

    update_config_parser = subparsers.add_parser(
        "update-config", help="Update a config setting"
    )
    update_config_parser.add_argument(
        "key", help="Config key path (e.g., 'paths.doc_dir')"
    )
    update_config_parser.add_argument("value", help="New value (JSON parse attempted)")
    update_config_parser.set_defaults(func=handle_update_config)

    show_deps_parser = subparsers.add_parser(
        "show-dependencies", help="Show aggregated dependencies for a key"
    )
    show_deps_parser.add_argument(
        "--key",
        required=True,
        help="Key string to show dependencies for (e.g., '1A1' or '1A1#2')",
    )
    show_deps_parser.set_defaults(func=handle_show_dependencies)

    # --- Show Keys Command ---
    show_keys_parser = subparsers.add_parser(
        "show-keys",
        help="Show keys from tracker, indicating if checks needed (p, s, S)",
    )
    show_keys_parser.add_argument(
        "--tracker", required=True, help="Path to the tracker file (.md)"
    )
    show_keys_parser.set_defaults(func=handle_show_keys)

    # --- Show Placeholders Command (ENHANCED) ---
    show_placeholders_parser = subparsers.add_parser(
        "show-placeholders",
        help="Show unverified dependencies ('p', 's', 'S') in a tracker",
    )
    show_placeholders_parser.add_argument(
        "--tracker", required=True, help="Path to the tracker file (.md)"
    )
    show_placeholders_parser.add_argument(
        "--key",
        required=False,
        help="Optional: Show unverified dependencies only for this specific source key label.",
    )
    show_placeholders_parser.add_argument(
        "--dep-char",
        required=False,
        help="Optional: Show only a specific dependency character (e.g., 'p', 's'). Shows p, s, S by default.",
    )
    show_placeholders_parser.set_defaults(func=handle_show_placeholders)

    visualize_parser = subparsers.add_parser(
        "visualize-dependencies", help="Generate a visualization of dependencies"
    )
    visualize_parser.add_argument(
        "--key",
        nargs="*",
        default=None,
        help="Optional: One or more key strings to focus the visualization on (e.g., '1A1', '2B#3'). If omitted, shows overview.",
    )
    visualize_parser.add_argument(
        "--format",
        choices=["mermaid"],
        default="mermaid",
        help="Output format (only mermaid currently)",
    )
    visualize_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: project_overview... or focus_KEY(s)...)",
    )
    visualize_parser.set_defaults(func=handle_visualize_dependencies)

    args = parser.parse_args()

    # --- Setup Logging ---
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    log_file_path: Optional[str] = None
    try:
        log_file_path = normalize_path(os.path.join(get_project_root(), "debug.txt"))
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e_fh:
        if log_file_path is not None:
            print(
                f"Error setting up file logger {log_file_path}: {e_fh}", file=sys.stderr
            )
        else:
            print(
                f"Error setting up file logger (path not determined): {e_fh}",
                file=sys.stderr,
            )

    # File Handler specifically for suggestion-related logs (if desired)
    suggestions_log_path: Optional[str] = None
    try:
        suggestions_log_path = normalize_path(
            os.path.join(get_project_root(), "suggestions.log")
        )
        suggestion_handler = logging.FileHandler(suggestions_log_path, mode="w")
        suggestion_handler.setLevel(logging.DEBUG)
        suggestion_handler.setFormatter(log_formatter)

        class SuggestionLogFilter(logging.Filter):
            def filter(self, record: LogRecord) -> bool:
                return (
                    record.name.startswith(
                        "cline_utils.dependency_system.analysis.dependency_suggester"
                    )
                    or record.name.startswith(
                        "cline_utils.dependency_system.analysis.project_analyzer"
                    )
                    and "suggestion" in record.getMessage().lower()
                    or record.name.startswith(
                        "cline_utils.dependency_system.io.tracker_io"
                    )
                    and "suggestion" in record.getMessage().lower()
                )

        suggestion_handler.addFilter(SuggestionLogFilter())
        root_logger.addHandler(suggestion_handler)
    except Exception as e_sh:
        if suggestions_log_path is not None:
            print(
                f"Error setting up suggestions logger {suggestions_log_path}: {e_sh}",
                file=sys.stderr,
            )
        else:
            print(
                f"Error setting up suggestions logger (path not determined): {e_sh}",
                file=sys.stderr,
            )

    # Console Handler for user-facing messages (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger.addHandler(console_handler)

    # Execute command
    if hasattr(args, "func"):
        exit_code = args.func(args)
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
