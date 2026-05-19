# dependency_processor.py

"""
Main entry point for the dependency tracking system.
Processes command-line arguments and delegates to appropriate handlers.
"""

import argparse
import json
import logging
from cline_utils.dependency_system.io.file_io import read_file_content_safely
import os
import subprocess
import sys
from collections import defaultdict
from logging import LogRecord
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union, cast

from cline_utils.dependency_system.analysis.dependency_analyzer import analyze_file
from cline_utils.dependency_system.analysis.dependency_suggester import (
    load_project_symbol_map,
)

# Type alias for sortable parts
SortableParts = list[str]
from cline_utils.dependency_system.analysis.embedding_manager import (
    generate_symbol_essence_string,
)
from cline_utils.dependency_system.analysis.local_llm_processor import LocalLLMProcessor

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
    get_keymap_indexes,
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
from cline_utils.dependency_system.utils.tracker_batch_collector import (
    TrackerBatchCollector,
    create_doc_tracker_update,
    create_main_tracker_update,
    create_mini_tracker_update,
)
from cline_utils.dependency_system.utils.tracker_utils import (
    aggregate_all_dependencies,
    find_all_tracker_paths,
    get_globally_resolved_key_info_for_cli,
    get_key_global_instance_string,
    read_grid_from_lines,
    read_key_definitions_from_lines,
    resolve_key_global_instance_to_ki,
)
from cline_utils.dependency_system.utils.visualize_dependencies import (
    generate_dependency_diagram,
    render_mermaid_to_image,
)

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
KEY_DEFINITIONS_START_MARKER = "---KEY_DEFINITIONS_START---"
KEY_DEFINITIONS_END_MARKER = "---KEY_DEFINITIONS_END---"


# --- Helper Functions ---
def _configure_stdio_for_unicode() -> None:
    """Avoid UnicodeEncodeError on Windows terminals using legacy code pages."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="backslashreplace")
            except (OSError, ValueError):
                pass


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


def _load_token_metadata(project_root: str) -> Dict[str, Dict[str, int]]:
    """Loads token counts from metadata.json."""
    metadata_path = os.path.join(
        project_root,
        "cline_utils",
        "dependency_system",
        "analysis",
        "embeddings",
        "metadata.json",
    )
    token_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                keys = data.get("keys", {})
                for key_data in keys.values():
                    path = key_data.get("path")
                    if not path:
                        continue
                    path = normalize_path(path)

                    ses = key_data.get("ses_tokens")
                    full = key_data.get("full_tokens")

                    if ses is None and "tokens" in key_data:
                        ses = key_data["tokens"]

                    if full is None:
                        full = ses

                    if ses is not None:
                        token_map[path] = {"ses_tokens": ses, "full_tokens": full}
        except Exception as e:
            logger.warning(f"Failed to load token metadata: {e}")
    return token_map


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


def handle_determine_dependency(args: argparse.Namespace) -> int:
    """Handle the determine-dependency command."""
    global_map = _load_global_map_or_exit()
    config_manager = ConfigManager()
    project_root = get_project_root()

    # Load token metadata from embeddings/metadata.json
    embeddings_dir = config_manager.get_path(
        "embeddings_dir", "cline_utils/dependency_system/analysis/embeddings"
    )
    if not os.path.isabs(embeddings_dir):
        embeddings_dir = os.path.join(project_root, embeddings_dir)

    metadata_path = os.path.join(embeddings_dir, "metadata.json")
    token_metadata: Dict[str, Any] = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                token_metadata = data.get("keys", {})
        except Exception as e:
            logger.warning(f"Failed to load token metadata: {e}")

    # Resolve source key
    source_ki = resolve_key_global_instance_to_ki(args.source_key, global_map)
    if not source_ki:
        print(f"Error: Could not resolve source key '{args.source_key}'")
        return 1

    # Resolve target key
    target_ki = resolve_key_global_instance_to_ki(args.target_key, global_map)
    if not target_ki:
        print(f"Error: Could not resolve target key '{args.target_key}'")
        return 1

    source_path = normalize_path(source_ki.norm_path)
    target_path = normalize_path(target_ki.norm_path)

    if not os.path.exists(source_path):
        print(f"Error: Source file not found: {source_path}")
        return 1
    if not os.path.exists(target_path):
        print(f"Error: Target file not found: {target_path}")
        return 1

    # Load symbol map
    symbol_map = load_project_symbol_map()

    try:
        # Use SES for source if available
        if source_path in symbol_map:
            source_content = generate_symbol_essence_string(
                source_path, symbol_map[source_path], symbol_map=symbol_map
            )
            source_basename = f"{os.path.basename(source_path)} (SES)"
        else:
            source_content = read_file_content_safely(source_path)
            if source_content is None:
                raise Exception(f"Failed to read source file: {source_path}")
            source_basename = os.path.basename(source_path)

        # Use SES for target if available
        if target_path in symbol_map:
            target_content = generate_symbol_essence_string(
                target_path, symbol_map[target_path], symbol_map=symbol_map
            )
            target_basename = f"{os.path.basename(target_path)} (SES)"
        else:
            target_content = read_file_content_safely(target_path)
            if target_content is None:
                raise Exception(f"Failed to read target file: {target_path}")
            target_basename = os.path.basename(target_path)

    except Exception as e:
        print(f"Error reading files: {e}")
        return 1

    model_path = args.model
    if not model_path:
        model_path = os.path.join(
            project_root, "models", "Qwen3-4B-Instruct-2507-Q8_0.gguf"
        )

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return 1

    try:
        processor = LocalLLMProcessor(model_path=model_path)

        # Get token counts if available
        key_data_source = token_metadata.get(source_ki.key_string, {})
        key_data_target = token_metadata.get(target_ki.key_string, {})

        # Helper to decide token count
        def get_count_for_key(key_data: Dict[str, Any], is_ses: bool) -> Optional[int]:
            if is_ses:
                return key_data.get("ses_tokens", key_data.get("tokens"))
            return key_data.get("full_tokens", key_data.get("tokens"))

        source_is_ses = source_path in symbol_map
        target_is_ses = target_path in symbol_map

        source_tokens = get_count_for_key(key_data_source, source_is_ses)
        target_tokens = get_count_for_key(key_data_target, target_is_ses)

        char, reasoning = processor.determine_dependency(
            source_content=source_content,
            target_content=target_content,
            source_basename=source_basename,
            target_basename=target_basename,
            source_tokens=source_tokens,
            target_tokens=target_tokens,
        )

        print(f"\nDependency Result: {char}")
        print(f"Source: {source_ki.key_string} ({source_path})")
        print(f"Target: {target_ki.key_string} ({target_path})")
        print(f"\n--- LLM Reasoning ---\n{reasoning}\n---------------------")

        return 0
    except Exception as e:
        logger.error(f"Error determining dependency: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1


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
                from .utils.resource_validator import get_cache_path

                cache_path = get_cache_path()
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
        # logger.debug(
        #     f"All Suggestions before Tracker Update: {results.get('dependency_suggestion', {}).get('suggestions')}"
        # )

        # Helper function to make results JSON-serializable by removing AST objects
        def make_serializable(obj: Any) -> Any:
            """Recursively remove non-JSON-serializable objects from the results."""
            if isinstance(obj, dict):
                # Remove known non-serializable keys
                cleaned: Dict[str, Any] = {
                    k: make_serializable(v)
                    for k, v in cast(Dict[str, Any], obj).items()
                    if k not in ("_ast_tree", "_ts_tree")
                }
                return cleaned
            elif isinstance(obj, list):
                return [make_serializable(item) for item in cast(List[Any], obj)]
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
    from cline_utils.dependency_system.io.update_doc_tracker import (
        doc_file_inclusion_logic,
    )

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
        filtered_items_map: Dict[str, KeyInfo] = doc_file_inclusion_logic(
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
    else:
        if len(matching_source_infos) > 1:
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
        else:
            if len(matching_target_infos) > 1:
                print(
                    f"Error: Target key '{tgt_base_key_str}' is globally ambiguous. Please specify which instance you mean using '#<num>':"
                )
                for ki in matching_target_infos:
                    print(f"  - {ki.key_string} (Path: {ki.norm_path})")
                rejected_targets.append(
                    (tgt_key_arg_item_raw, "Globally ambiguous key.")
                )
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
        clear_all_caches(wipe=True)
        print("All caches wiped from memory and disk.")
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
    token_map = _load_token_metadata(project_root)

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

    token_count = token_map.get(target_ki_to_show.norm_path)
    token_info = f" [Tokens: {token_count}]" if token_count is not None else ""

    print(
        f"\n--- Dependencies for: {target_key_gi_str_to_show} (Path: {target_ki_to_show.norm_path}){token_info} ---"
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

            token_count = token_map.get(interacting_ki.norm_path)
            token_info = f" [Tokens: {token_count}]" if token_count is not None else ""

            print(
                f"  - {display_name_interacting}: {interacting_ki.norm_path}{token_info}{origins_str}"
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

    project_root = get_project_root()
    token_map = _load_token_metadata(project_root)

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

            token_count = token_map.get(normalize_path(path_str_in_file))
            if token_count is not None:
                print(f"    | Tokens: {token_count}")

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

    When --tracker is provided: shows detailed per-key breakdown for that tracker.
    When --tracker is omitted: shows aggregate summary across all trackers from tracker_map.json.
    """
    focus_key = args.key
    dep_char_filter = args.dep_char

    chars_to_check: Tuple[str, ...]
    if dep_char_filter:
        chars_to_check = (dep_char_filter,)
    else:
        chars_to_check = ("p", "s", "S")

    # --- Bare mode: no --tracker provided, show aggregate summary across all trackers ---
    if args.tracker is None:
        from cline_utils.dependency_system.core.key_manager import (
            load_tracker_map,
        )

        all_trackers = load_tracker_map()
        if not all_trackers:
            print(
                "Error: No tracker map found. Run 'analyze-project' first.",
                file=sys.stderr,
            )
            return 1

        results: List[Tuple[str, int, int, int]] = (
            []
        )  # (tracker_rel_path, p_count, s_count, S_count)
        total_p = total_s = total_S = 0

        for t_path in all_trackers:
            if not os.path.exists(t_path):
                continue
            try:
                with open(t_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                key_def_pairs = read_key_definitions_from_lines(lines)
                _grid_headers, grid_rows_data = read_grid_from_lines(lines)
                if not key_def_pairs or not grid_rows_data:
                    continue
            except Exception:
                continue

            p_cnt = s_cnt = S_cnt = 0
            for _, (row_label, compressed_row) in enumerate(grid_rows_data):
                if focus_key and row_label != focus_key:
                    continue
                try:
                    decompressed = decompress(compressed_row)
                    if len(decompressed) != len(key_def_pairs):
                        continue
                    for char in decompressed:
                        if char in chars_to_check:
                            if char == "p":
                                p_cnt += 1
                            elif char == "s":
                                s_cnt += 1
                            elif char == "S":
                                S_cnt += 1
                except Exception:
                    continue

            if p_cnt or s_cnt or S_cnt:
                # Use relative path for cleaner output
                rel_path = os.path.relpath(t_path, get_project_root())
                results.append((rel_path, p_cnt, s_cnt, S_cnt))
                total_p += p_cnt
                total_s += s_cnt
                total_S += S_cnt

        if not results:
            print(f"No unverified dependencies {chars_to_check} found in any tracker.")
            return 0

        for rel_path, p_cnt, s_cnt, S_cnt in results:
            print(f"{rel_path} - p:{p_cnt} | s:{s_cnt} | S:{S_cnt}")
        print("---")
        total_all = total_p + total_s + total_S
        print(f"Total Unresolved dependencies: {total_all}")
        return 0

    # --- Detailed mode: --tracker provided ---
    tracker_path = normalize_path(args.tracker)

    if not os.path.exists(tracker_path):
        print(f"Error: Tracker file not found: {tracker_path}", file=sys.stderr)
        return 1

    # --- Load Global Map for Path Resolution ---
    project_root = get_project_root()
    token_map = _load_token_metadata(project_root)
    global_map = load_global_key_map()
    key_to_path_map: Dict[str, str] = {}
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

        for _, (row_label, compressed_row) in enumerate(grid_rows_data):
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
            source_path: str = key_to_path_map.get(source_label, "Path not found")
            source_token_count = token_map.get(normalize_path(source_path))
            source_token_info = (
                f" [Tokens: {source_token_count}]"
                if source_token_count is not None
                else ""
            )
            print(
                f"\n--- Key: {source_label} (Path: {source_path}){source_token_info} ---"
            )
            char_map = unverified_deps[source_label]
            for char_type in sorted(char_map.keys()):
                target_labels = sorted(
                    char_map[char_type], key=get_sortable_parts_for_key
                )
                print(f"  {char_type}:")
                for tgt in target_labels:
                    tgt_path: str = key_to_path_map.get(tgt, "Path not found")
                    tgt_token_count = token_map.get(normalize_path(tgt_path))
                    tgt_token_info = (
                        f" [Tokens: {tgt_token_count}]"
                        if tgt_token_count is not None
                        else ""
                    )
                    print(f"    - {tgt} (Path: {tgt_path}){tgt_token_info}")

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
    backend_cli = getattr(args, "backend", None) or (
        "native" if output_format_cli == "svg" else "mermaid"
    )
    output_path_arg_cli = args.output

    logger.info(
        f"CLI: visualize-dependencies called. Focus Keys: {focus_keys_list_cli or 'Project Overview'}"
    )

    if output_format_cli == "svg":
        backend_cli = "native"
    if backend_cli not in {"mermaid", "native"}:
        print(f"Error: Unsupported visualization backend '{backend_cli}'.")
        return 1

    try:
        current_global_map_cli = _load_global_map_or_exit()  # path_to_key_info
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

    diagram_string_generated = generate_dependency_diagram(
        focus_keys_list_input=focus_keys_list_cli,
        global_path_to_key_info_map=current_global_map_cli,
        path_migration_info=path_migration_info_cli,
        all_tracker_paths_list=list(all_tracker_paths_cli),
        config_manager_instance=config_cli,
        backend=backend_cli,
        render=False,
    )

    if diagram_string_generated is None:
        print(
            "Error: Dependency diagram generation failed internally. Check logs.",
            file=sys.stderr,
        )
        return 1
    elif "Error:" in diagram_string_generated[:20]:
        print(diagram_string_generated, file=sys.stderr)
        return 1
    elif "No relevant data" in diagram_string_generated:
        print(
            "Info: No relevant data found to visualize based on focus keys and filters."
        )
    else:
        logger.debug(f"{backend_cli} visualization generated successfully.")

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
            ext = "svg" if backend_cli == "native" else output_format_cli
            default_filename = f"focus_{safe_keys_str}_dependencies.{ext}"
        else:
            ext = "svg" if backend_cli == "native" else output_format_cli
            default_filename = f"project_overview_dependencies.{ext}"

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
            f_out.write(diagram_string_generated)

        logger.info(
            f"Successfully wrote dependency visualization to: {output_path_cli}"
        )
        print(f"\nDependency visualization saved to: {output_path_cli}")
        if (
            backend_cli == "mermaid"
            and "No relevant data" not in diagram_string_generated
            and not diagram_string_generated.startswith("Error:")
        ):
            rendered_svg_path = os.path.splitext(output_path_cli)[0] + ".svg"
            render_mermaid_to_image(diagram_string_generated, rendered_svg_path)
            print(f"Rendered Mermaid SVG saved to: {rendered_svg_path}")
        if (
            "No relevant data" not in diagram_string_generated
            and backend_cli == "mermaid"
        ):
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


from cline_utils.dependency_system.utils.placeholder_resolver import PreparedPair


def _prepare_pair(
    srckey: str,
    srcpath: str,
    tgtkey: str,
    tgtpath: str,
    symbol_map: Dict[str, Any],
    token_map: Dict[str, Any],
    max_model_tokens: int,
    wrapper_overhead: int,
) -> PreparedPair:
    """
    All CPU-side work for one pair: normalize, SES, file read, token count.
    Runs in a prefetch thread while GPU handles the previous pair.
    """
    src_norm = normalize_path(srcpath)
    tgt_norm = normalize_path(tgtpath)
    src_is_ses = src_norm in symbol_map
    tgt_is_ses = tgt_norm in symbol_map

    def _get_tokens(path_norm: str, is_ses: bool) -> int:
        data = token_map.get(path_norm, {})
        if is_ses:
            val = data.get("ses_tokens", data.get("tokens"))
        else:
            val = data.get("full_tokens", data.get("tokens"))
        return val if val is not None else 0

    stokens = _get_tokens(src_norm, src_is_ses)
    ttokens = _get_tokens(tgt_norm, tgt_is_ses)

    # Token limit pre-check
    if stokens > 0 and ttokens > 0:
        total_est = stokens + ttokens + wrapper_overhead
        if total_est > max_model_tokens:
            return PreparedPair(
                srckey=srckey,
                srcpath=srcpath,
                tgtkey=tgtkey,
                tgtpath=tgtpath,
                srccontent="",
                tgtcontent="",
                srcbase="",
                tgtbase="",
                stokens=stokens,
                ttokens=ttokens,
                skip=True,
                skip_reason=f"Combined tokens {total_est} exceed limit {max_model_tokens}",
            )

    # File reads
    try:
        srccontent = read_file_content_safely(srcpath)
        if srccontent is None:
            raise Exception("Failed to read src")
        tgtcontent = read_file_content_safely(tgtpath)
        if tgtcontent is None:
            raise Exception("Failed to read tgt")
    except Exception as e:
        return PreparedPair(
            srckey=srckey,
            srcpath=srcpath,
            tgtkey=tgtkey,
            tgtpath=tgtpath,
            srccontent="",
            tgtcontent="",
            srcbase="",
            tgtbase="",
            stokens=stokens,
            ttokens=ttokens,
            skip=True,
            skip_reason=f"File read error: {e}",
        )

    srcbase = os.path.basename(srcpath)
    tgtbase = os.path.basename(tgtpath)

    # SES substitution
    if src_is_ses:
        srccontent = generate_symbol_essence_string(
            src_norm, symbol_map[src_norm], symbol_map=symbol_map
        )
        srcbase = f"{srcbase} (SES)"
    if tgt_is_ses:
        tgtcontent = generate_symbol_essence_string(
            tgt_norm, symbol_map[tgt_norm], symbol_map=symbol_map
        )
        tgtbase = f"{tgtbase} (SES)"

    return PreparedPair(
        srckey=srckey,
        srcpath=srcpath,
        tgtkey=tgtkey,
        tgtpath=tgtpath,
        srccontent=srccontent,
        tgtcontent=tgtcontent,
        srcbase=srcbase,
        tgtbase=tgtbase,
        stokens=stokens,
        ttokens=ttokens,
    )


def handle_resolve_placeholders(args: argparse.Namespace) -> int:
    """
    Resolve placeholders using Local LLM in batches.
    """

    if not hasattr(args, "_processed_pairs"):
        args._processed_pairs = set()
    if not hasattr(args, "accumulated_tracker_updates"):
        args.accumulated_tracker_updates = []

    _raw_pairs = getattr(args, "_processed_pairs")
    processed_pairs = cast(Set[Tuple[str, str, str, str]], _raw_pairs)

    limit = args.limit
    dep_char = args.dep_char
    focus_key = args.key

    if args.tracker is None:
        from cline_utils.dependency_system.core.key_manager import (
            load_tracker_map,
            save_tracker_map,
        )

        all_trackers = load_tracker_map()
        if not all_trackers:
            print(
                "Tracker map not found or empty. Scanning project to generate it...",
                file=sys.stderr,
            )
            all_trackers = list(
                find_all_tracker_paths(
                    ConfigManager(), get_project_root(), force_scan=True
                )
            )
            if all_trackers:
                save_tracker_map(all_trackers)
                print(
                    f"Generated tracker map with {len(all_trackers)} entries.",
                    file=sys.stderr,
                )
            else:
                print("Error: No trackers found in the project.", file=sys.stderr)
                return 1

        doc_trackers: List[str] = []
        mini_trackers: List[str] = []
        main_trackers: List[str] = []
        for t in all_trackers:
            basename = os.path.basename(t)
            if "doc_tracker.md" in basename:
                doc_trackers.append(t)
            elif basename.endswith("_module.md"):
                mini_trackers.append(t)
            elif "module_relationship_tracker.md" in basename:
                main_trackers.append(t)

        trackers_to_scan = doc_trackers + mini_trackers + main_trackers
        dep_chars = ("p", "S", "s") if dep_char == "p" else (dep_char,)
    else:
        tracker_path: str = str(normalize_path(args.tracker))
        if not os.path.isfile(tracker_path):
            print(
                f"Error: Tracker file not found or is a directory: {tracker_path}",
                file=sys.stderr,
            )
            return 1
        trackers_to_scan = [tracker_path]
        dep_chars = (dep_char,)

    model_path = args.model if args.model else "models/Qwen3-4B-Instruct-2507-Q8_0.gguf"

    # Load Global Map and derived indexes
    global_map = _load_global_map_or_exit()
    indexes = get_keymap_indexes()
    is_dir_map = indexes.get("is_dir_map", {})
    file_descendants_by_dir = indexes.get("file_descendants_by_dir", {})
    ancestor_chain = indexes.get("ancestor_chain", {})

    # --- PREPARATION ---
    config_mgr = ConfigManager()
    get_prio = config_mgr.get_char_priority
    all_tp = find_all_tracker_paths(config_mgr, get_project_root())
    old_global_map = load_old_global_key_map()
    path_migration_info = build_path_migration_map(old_global_map, global_map)

    # Build Global Edges Map (Aggregated from all trackers)
    agg_deps = aggregate_all_dependencies(
        tracker_paths=all_tp,
        path_migration_info=path_migration_info,
        current_global_path_to_key_info=global_map,
        show_progress=False,
    )
    edges: DefaultDict[str, Dict[str, str]] = defaultdict(dict)
    gi_to_path = {ki.key_string: p for p, ki in global_map.items()}
    for (src_gi, tgt_gi), (char, _) in agg_deps.items():
        s_path = gi_to_path.get(src_gi)
        t_path = gi_to_path.get(tgt_gi)
        if s_path and t_path:
            edges[s_path][t_path] = char

    import time

    # ---------------------------------------------------------
    # GLOBAL PHASE 1 & 2: Algorithmic Resolution across ALL trackers
    # ---------------------------------------------------------
    global_suggestions: DefaultDict[str, DefaultDict[str, List[Tuple[str, str]]]] = (
        defaultdict(lambda: defaultdict(list))
    )

    algo_processed_count = 0
    shortcut_processed_count = 0

    # Collect all algorithmic/shortcut tasks globally
    for t_path in trackers_to_scan:
        if not os.path.isfile(t_path):
            continue

        try:
            with open(t_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            key_def_pairs = read_key_definitions_from_lines(lines)
            _, grid_rows_data = read_grid_from_lines(lines)
        except Exception:
            continue

        if not key_def_pairs or not grid_rows_data:
            continue

        for row_idx, (row_label, compressed_row) in enumerate(grid_rows_data):
            if focus_key and row_label != focus_key:
                continue
            if row_idx >= len(key_def_pairs):
                continue

            _, src_path = key_def_pairs[row_idx]
            if not src_path or not os.path.exists(src_path):
                continue

            try:
                decomp = list(decompress(compressed_row))
                for col_idx, char in enumerate(decomp):
                    if char in dep_chars and col_idx < len(key_def_pairs):
                        tgt_key_label, tgt_path = key_def_pairs[col_idx]
                        if tgt_path and os.path.exists(tgt_path):
                            if (
                                t_path,
                                row_label,
                                tgt_key_label,
                                char,
                            ) in processed_pairs:
                                continue

                            sn = normalize_path(src_path)
                            tn = normalize_path(tgt_path)

                            # Check Shortcut 1A (Global Cache)
                            existing_global_char = edges.get(sn, {}).get(tn)
                            if existing_global_char and existing_global_char in (
                                "x",
                                "<",
                                ">",
                                "d",
                                "n",
                            ):
                                global_suggestions[t_path][row_label].append(
                                    (tgt_key_label, existing_global_char)
                                )
                                processed_pairs.add(
                                    (t_path, row_label, tgt_key_label, char)
                                )
                                shortcut_processed_count += 1
                                logger.info(
                                    f"Shortcut: Resolved {row_label} -> {tgt_key_label}: '{existing_global_char}' (Propagated from global cache)"
                                )
                                continue

                            # Check Shortcut 1B (Parent-Child)
                            is_pc = (
                                sn == tn
                                or sn in ancestor_chain.get(tn, ())
                                or tn in ancestor_chain.get(sn, ())
                            )
                            if is_pc:
                                global_suggestions[t_path][row_label].append(
                                    (tgt_key_label, "x")
                                )
                                processed_pairs.add(
                                    (t_path, row_label, tgt_key_label, char)
                                )
                                shortcut_processed_count += 1
                                logger.info(
                                    f"Shortcut: Resolved {row_label} -> {tgt_key_label}: 'x' (Parent-Child relation)"
                                )
                                if sn not in edges:
                                    edges[sn] = {}
                                edges[sn][tn] = "x"
                                continue

                            # Check Algorithmic Phase 2 (Directories)
                            if is_dir_map.get(
                                sn, os.path.isdir(src_path)
                            ) or is_dir_map.get(tn, os.path.isdir(tgt_path)):
                                s_children_tuple = file_descendants_by_dir.get(sn)
                                s_children = (
                                    list(s_children_tuple)
                                    if s_children_tuple is not None
                                    else (
                                        [sn]
                                        if not is_dir_map.get(
                                            sn, os.path.isdir(src_path)
                                        )
                                        else []
                                    )
                                )
                                t_children_tuple = file_descendants_by_dir.get(tn)
                                t_children = (
                                    list(t_children_tuple)
                                    if t_children_tuple is not None
                                    else (
                                        [tn]
                                        if not is_dir_map.get(
                                            tn, os.path.isdir(tgt_path)
                                        )
                                        else []
                                    )
                                )

                                best_char: str = " "
                                best_prio: int = -1
                                found_chars: Set[str] = set()

                                s_set = set(s_children)
                                t_set = set(t_children)
                                relevant_sources = s_set.intersection(edges.keys())

                                for sc in relevant_sources:
                                    sc_edges = edges[sc]
                                    relevant_targets = t_set.intersection(
                                        sc_edges.keys()
                                    )
                                    for tc in relevant_targets:
                                        echar = sc_edges[tc]
                                        if echar:
                                            prio = get_prio(echar)
                                            found_chars.add(echar)
                                            if prio > best_prio:
                                                best_prio = prio
                                                best_char = echar

                                if best_prio > -1:
                                    if {"<", ">"} <= found_chars:
                                        best_char = "x"
                                    if best_char in ("p", "S", "s"):
                                        # Not ready to resolve. Skip, but mark as processed so it's not checked again in this run
                                        processed_pairs.add(
                                            (t_path, row_label, tgt_key_label, char)
                                        )
                                        continue

                                    global_suggestions[t_path][row_label].append(
                                        (tgt_key_label, best_char)
                                    )
                                    logger.info(
                                        f"Algorithmically resolved {row_label} -> {tgt_key_label}: '{best_char}' (Rolled up)"
                                    )
                                    if sn not in edges:
                                        edges[sn] = {}
                                    edges[sn][tn] = best_char
                                else:
                                    global_suggestions[t_path][row_label].append(
                                        (tgt_key_label, "n")
                                    )
                                    logger.info(
                                        f"Algorithmically resolved {row_label} -> {tgt_key_label}: 'n' (No dependencies found)"
                                    )
                                    if sn not in edges:
                                        edges[sn] = {}
                                    edges[sn][tn] = "n"

                                processed_pairs.add(
                                    (t_path, row_label, tgt_key_label, char)
                                )
                                algo_processed_count += 1
                                continue
            except Exception:
                continue

    # Apply all algorithmic/shortcut suggestions globally
    if global_suggestions:
        print(
            f"\n--- Resolving {algo_processed_count} directory placeholders and {shortcut_processed_count} shortcuts algorithmically ---"
        )
        algo_start_time = time.time()

        global_collector = TrackerBatchCollector()
        for t_path, suggestions in global_suggestions.items():
            t_type = (
                "mini"
                if t_path.endswith("_module.md")
                else ("doc" if "doc_tracker.md" in os.path.basename(t_path) else "main")
            )

            # Use cast to ensure type checker knows suggestions is Dict[str, List[Tuple[str, str]]]
            cast_suggestions: Dict[str, List[Tuple[str, str]]] = dict(suggestions)
            update_data = update_tracker(
                output_file_suggestion=t_path,
                path_to_key_info=global_map,
                tracker_type=t_type,
                suggestions_external=cast_suggestions,
                return_update=True,
                force_apply_suggestions=True,
                apply_ast_overrides=False,
            )
            if not update_data:
                continue

            out_path = update_data.get("output_file", t_path)
            t_update = None
            if t_type == "mini":
                t_update = create_mini_tracker_update(
                    output_file=out_path,
                    key_info_list=update_data["key_info_list"],
                    grid_rows=update_data["grid_rows"],
                    last_key_edit=update_data["last_key_edit"],
                    last_grid_edit=update_data["last_grid_edit"],
                    module_path=update_data.get("module_path", ""),
                    path_to_key_info=update_data.get("path_to_key_info", global_map),
                    existing_lines=update_data.get("existing_lines", []),
                    tracker_exists=update_data.get("tracker_exists", False),
                    ast_overrides_applied_count=update_data.get(
                        "ast_overrides_applied_count", 0
                    ),
                    suggestion_applied_count=update_data.get(
                        "suggestion_applied_count", 0
                    ),
                    structural_deps_applied_count=update_data.get(
                        "structural_deps_applied_count", 0
                    ),
                    force_apply_suggestions=True,
                )
            elif t_type == "doc":
                t_update = create_doc_tracker_update(
                    output_file=out_path,
                    key_info_list=update_data["key_info_list"],
                    grid_rows=update_data["grid_rows"],
                    last_key_edit=update_data["last_key_edit"],
                    last_grid_edit=update_data["last_grid_edit"],
                    path_to_key_info=update_data.get("path_to_key_info", global_map),
                    ast_overrides_applied_count=update_data.get(
                        "ast_overrides_applied_count", 0
                    ),
                    suggestion_applied_count=update_data.get(
                        "suggestion_applied_count", 0
                    ),
                    structural_deps_applied_count=update_data.get(
                        "structural_deps_applied_count", 0
                    ),
                    force_apply_suggestions=True,
                )
            else:
                t_update = create_main_tracker_update(
                    output_file=out_path,
                    key_info_list=update_data["key_info_list"],
                    grid_rows=update_data["grid_rows"],
                    last_key_edit=update_data["last_key_edit"],
                    last_grid_edit=update_data["last_grid_edit"],
                    path_to_key_info=update_data.get("path_to_key_info", global_map),
                    ast_overrides_applied_count=update_data.get(
                        "ast_overrides_applied_count", 0
                    ),
                    suggestion_applied_count=update_data.get(
                        "suggestion_applied_count", 0
                    ),
                    structural_deps_applied_count=update_data.get(
                        "structural_deps_applied_count", 0
                    ),
                    force_apply_suggestions=True,
                )

            if t_update:
                global_collector.add(t_update)

        # Commit all algorithmic updates at once, passing accumulated_updates
        global_collector.commit_all(
            skip_populate_hook=True,
            accumulated_updates=cast(
                List[Any], getattr(args, "accumulated_tracker_updates", [])
            ),
        )
        logger.info(
            f"Global Algorithmic/Shortcut phase complete in {time.time()-algo_start_time:.2f}s"
        )

    # ---------------------------------------------------------
    # PHASE 3: LLM Resolution (Per-Tracker)
    # ---------------------------------------------------------
    selected_tracker: Optional[str] = None
    llm_tasks: List[Tuple[str, str, str, str]] = []
    tracker_type = ""

    for t_path in trackers_to_scan:
        if not os.path.isfile(t_path):
            continue

        t_type = (
            "mini"
            if t_path.endswith("_module.md")
            else ("doc" if "doc_tracker.md" in os.path.basename(t_path) else "main")
        )
        try:
            with open(t_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            key_def_pairs = read_key_definitions_from_lines(lines)
            _, grid_rows_data = read_grid_from_lines(lines)
        except Exception:
            continue

        if not key_def_pairs or not grid_rows_data:
            continue

        found_llm_tasks: List[Tuple[str, str, str, str]] = []
        for row_idx, (row_label, compressed_row) in enumerate(grid_rows_data):
            if focus_key and row_label != focus_key:
                continue
            if row_idx >= len(key_def_pairs):
                continue

            _, src_path = key_def_pairs[row_idx]
            if not src_path or not os.path.exists(src_path):
                continue

            try:
                decomp = list(decompress(compressed_row))
                for col_idx, char in enumerate(decomp):
                    if char in dep_chars and col_idx < len(key_def_pairs):
                        tgt_key_label, tgt_path = key_def_pairs[col_idx]
                        if tgt_path and os.path.exists(tgt_path):
                            if (
                                t_path,
                                row_label,
                                tgt_key_label,
                                char,
                            ) in processed_pairs:
                                continue

                            sn = normalize_path(src_path)
                            tn = normalize_path(tgt_path)
                            if is_dir_map.get(
                                sn, os.path.isdir(src_path)
                            ) or is_dir_map.get(tn, os.path.isdir(tgt_path)):
                                # This is a directory placeholder that wasn't resolved in Phase 2
                                # Do NOT send it to the LLM.
                                processed_pairs.add(
                                    (t_path, row_label, tgt_key_label, char)
                                )
                                continue

                            found_llm_tasks.append(
                                (row_label, src_path, tgt_key_label, tgt_path)
                            )
            except Exception:
                continue

        if found_llm_tasks:
            selected_tracker = t_path
            llm_tasks = found_llm_tasks
            tracker_type = t_type
            break

    if not selected_tracker or not llm_tasks:
        print("Finished resolving. No LLM tasks required.")
        return 0

    dep_chars_str = "', '".join(dep_chars)
    print(
        f"Automatically selected tracker: {selected_tracker} (scanning for: '{dep_chars_str}')"
    )
    print(f"Found {len(llm_tasks)} unresolved dependencies for ('{dep_chars_str}').")

    llm_tasks = llm_tasks[:limit]

    MAX_MODEL_TOKENS = 30000  # Leave buffer for system prompt (total ctx 32768)
    WRAPPER_OVERHEAD = 1000  # Approximate system prompt + overhead

    token_map = _load_token_metadata(get_project_root())
    symbol_map = load_project_symbol_map()

    from cline_utils.dependency_system.utils.placeholder_resolver import (
        PlaceholderResolver,
    )

    # --- Initialize processor first so its tokenizer is available for exact counting ---
    processor = LocalLLMProcessor(model_path=model_path)

    # --- Eager preparation: read files + SES substitution for every pair ---
    print(f"Preparing {len(llm_tasks)} pairs and measuring exact token requirements...")
    all_prepared: List[PreparedPair] = [
        _prepare_pair(
            sk, sp, tk, tp, symbol_map, token_map, MAX_MODEL_TOKENS, WRAPPER_OVERHEAD
        )
        for sk, sp, tk, tp in llm_tasks
    ]

    # Partition: pairs that exceed the token limit are already flagged skip=True by _prepare_pair
    valid_prepared = [p for p in all_prepared if not p.skip]
    skipped_prepared = [p for p in all_prepared if p.skip]

    if skipped_prepared:
        print(f"  {len(skipped_prepared)} pair(s) skipped (exceed token limit).")

    # --- Exact token measurement: count src+tgt content tokens for each valid pair ---
    # The instruction wrapper is constant across all pairs, so counting content tokens
    # gives an exact proportional ordering.  We use the processor's own tokenizer so
    # the counts reflect the actual model vocabulary — no heuristics involved.
    print(f"  Counting exact tokens for {len(valid_prepared)} pair(s)...")
    for prep in valid_prepared:
        exact = processor.get_token_count(prep.srccontent + prep.tgtcontent)
        # Store the exact count back into the PreparedPair fields for the sort key.
        # stokens + ttokens are used as a hint by determine_dependency but not for
        # correctness — overwriting them with the exact combined count is safe here.
        prep.stokens = exact
        prep.ttokens = 0  # absorbed into stokens; ttokens is now redundant

    # --- Deterministic sort: largest context first so n_ctx only ever shrinks ---
    valid_prepared.sort(key=lambda p: p.stokens, reverse=True)
    print(
        f"Processing batch of {len(valid_prepared)} items (sorted by descending exact token count)..."
    )

    # Re-mark processed_pairs to include all tasks (valid + skipped) before GPU work begins
    for prep in all_prepared:
        processed_pairs.add((selected_tracker, prep.srckey, prep.tgtkey, dep_chars[0]))

    resolver = PlaceholderResolver(processor)

    # Identity wrapper: pairs are already prepared; no I/O needed inside resolve_batch
    prepared_index: Dict[Tuple[str, str], PreparedPair] = {
        (p.srckey, p.tgtkey): p for p in valid_prepared
    }

    def _identity_prepare(sk: str, sp: str, tk: str, tp: str) -> PreparedPair:
        return prepared_index.get(
            (sk, tk),
            _prepare_pair(
                sk,
                sp,
                tk,
                tp,
                symbol_map,
                token_map,
                MAX_MODEL_TOKENS,
                WRAPPER_OVERHEAD,
            ),
        )

    # Pass valid_prepared as (srckey, srcpath, tgtkey, tgtpath) tuples
    sorted_tasks = [(p.srckey, p.srcpath, p.tgtkey, p.tgtpath) for p in valid_prepared]

    total_processed = resolver.resolve_batch(
        tasks=sorted_tasks,
        tracker_path=selected_tracker,
        global_map=global_map,
        tracker_type=tracker_type,
        prepare_func=_identity_prepare,
        accumulated_updates=cast(
            List[Any], getattr(args, "accumulated_tracker_updates", [])
        ),
    )

    args.limit -= total_processed
    if args.limit > 0 and total_processed > 0 and args.tracker is None:
        print(f"Continuing to next tracker. Remaining limit: {args.limit}")
        processor.close()
        return handle_resolve_placeholders(args)

    processor.close()
    return 0


def handle_reconcile_transparency(args: argparse.Namespace) -> int:
    """
    Scans files for documentation markers and reconciles them with the transparency registry.
    """
    from cline_utils.dependency_system.io.transparency_manager import (
        get_transparency_manager,
    )

    manager = get_transparency_manager()
    project_root = get_project_root()
    config = ConfigManager()
    excluded_paths = config.get_excluded_paths()

    # Identify files to scan
    all_files: List[str] = []
    scan_path = args.path if args.path else project_root

    if os.path.isfile(scan_path):
        all_files.append(scan_path)
    elif os.path.isdir(scan_path):
        for root, dirs, files in os.walk(scan_path):
            # Skip common hidden/build dirs
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in ("node_modules", "venv", "__pycache__")
                and normalize_path(os.path.join(root, d)) not in excluded_paths
            ]
            for f in files:
                file_path = normalize_path(os.path.join(root, f))
                if file_path in excluded_paths:
                    continue
                if f.endswith((".md", ".txt", ".rst")):
                    all_files.append(file_path)

    # Secondary filter for absolute paths
    all_files = [f for f in all_files if normalize_path(f) not in excluded_paths]

    if not all_files:
        logger.debug(f"No documentation files found to scan in {scan_path}.")
        return 0

    reconciled_count = 0
    for file_path in all_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Find markers and calculate shifts
            sections: Dict[str, Tuple[int, int]] = {}
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("---") and stripped.endswith("_START---"):
                    section_name = stripped[3:-9]
                    # Find end
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip() == f"---{section_name}_END---":
                            sections[section_name] = (i, j)
                            break

            # Dispatch based on transform
            if args.transform == "restore":
                if manager.restore_markers(file_path):
                    logger.info(
                        f"  [RESTORED] {os.path.relpath(file_path, project_root)}: Markers re-inserted from registry."
                    )
                    reconciled_count += 1
                else:
                    logger.debug(
                        f"  [SKIP RESTORE] {os.path.relpath(file_path, project_root)}: No registry data or already tagged."
                    )
                continue

            if args.transform == "remove":
                if manager.remove_markers(file_path):
                    logger.info(
                        f"  [REMOVED MARKERS] {os.path.relpath(file_path, project_root)}: Markers moved to registry."
                    )
                    reconciled_count += 1
                else:
                    logger.debug(
                        f"  [SKIP REMOVE] {os.path.relpath(file_path, project_root)}: No markers found or already removed."
                    )
                continue

            # For other transforms (html, register), we need to find existing markers
            if not sections:
                continue

            if args.transform == "html":
                # Convert to HTML comments
                new_lines = list(lines)
                for name, (start_idx, end_idx) in sections.items():
                    new_lines[start_idx] = f"<!-- ---{name}_START--- -->\n"
                    new_lines[end_idx] = f"<!-- ---{name}_END--- -->\n"

                content = "".join(new_lines)
                # Register with markers present (but commented out)
                registry_sections: Dict[str, Any] = {
                    name: (start_idx + 1, end_idx + 1)
                    for name, (start_idx, end_idx) in sections.items()
                }
                manager.update_file_metadata(file_path, registry_sections, content)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                logger.info(
                    f"  [HTML COMMENTS] {os.path.relpath(file_path, project_root)}: {len(sections)} sections registered."
                )

            else:
                # Just register with current markers
                content = "".join(lines)
                registry_sections: Dict[str, Any] = {
                    name: (start_idx + 1, end_idx + 1)
                    for name, (start_idx, end_idx) in sections.items()
                }
                manager.update_file_metadata(file_path, registry_sections, content)
                logger.info(
                    f"  [REGISTERED] {os.path.relpath(file_path, project_root)}: {len(sections)} sections found."
                )

            reconciled_count += 1

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            logger.error(f"Reconciliation error for {file_path}: {e}", exc_info=True)

    print(f"\nSuccessfully reconciled transparency for {reconciled_count} files.")
    return 0


def handle_normalize_docs(args: argparse.Namespace) -> int:
    """
    Two-step documentation workflow:
    1. Scan candidates and query local LLM to normalize docs with physical compliance markers.
    2. If not dry-run, immediately run transparency virtualization to relocate markers/metadata
       to transparency_registry.json, leaving the physical file completely clean.
    """
    from cline_utils.dependency_system.io.normalize_docs import (
        normalize_docs_batch,
        load_transparency_registry,
        find_normalization_candidates,
    )
    from cline_utils.dependency_system.analysis.local_llm_processor import (
        LocalLLMProcessor,
    )
    from cline_utils.dependency_system.utils.config_manager import ConfigManager

    project_root = get_project_root()
    config_mgr = ConfigManager()

    # Get doc directories dynamically
    doc_dirs = config_mgr.get_doc_directories()
    logger.info(f"Resolved doc directories from default-rules.md: {doc_dirs}")

    registered_files = load_transparency_registry(project_root)
    logger.info(
        f"Loaded {len(registered_files)} registered files from transparency registry."
    )

    # Find candidate files
    try:
        candidates = find_normalization_candidates(
            project_root=project_root,
            doc_dirs=doc_dirs,
            registered_files=registered_files,
            file_path_arg=args.file,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    total_candidates = len(candidates)
    logger.info(
        f"Found {total_candidates} candidate files that need normalization or placeholder resolution."
    )

    if total_candidates == 0:
        print("All files are fully compliant and resolved!")
        return 0

    # Initialize Local LLM
    model_path = (
        args.model
        if args.model
        else os.path.join(project_root, "models", "Qwen3-4B-Instruct-2507-Q8_0.gguf")
    )
    if not os.path.exists(model_path):
        print(f"Error: Local LLM model not found at {model_path}")
        return 1

    logger.info(f"Initializing local LLM from {model_path}...")
    processor = LocalLLMProcessor(model_path=model_path)

    # Use the reusable batch normalization function
    normalize_docs_batch(
        candidates=candidates,
        processor=processor,
        project_root=project_root,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    processor.close()
    return 0


def main():
    """Parse arguments and dispatch to handlers."""
    _configure_stdio_for_unicode()
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
        help="Show unverified dependencies ('p', 's', 'S') in a tracker. Without --tracker, shows summary across all trackers.",
    )
    show_placeholders_parser.add_argument(
        "--tracker",
        required=False,
        help="Path to the tracker file (.md). If omitted, shows aggregate summary across all trackers from tracker_map.json.",
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
        choices=["mermaid", "svg"],
        default="mermaid",
        help="Output format. Use 'svg' for the native renderer.",
    )
    visualize_parser.add_argument(
        "--backend",
        choices=["mermaid", "native"],
        default=None,
        help="Rendering backend. Defaults to native when --format svg is used.",
    )
    visualize_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: project_overview... or focus_KEY(s)...)",
    )
    visualize_parser.set_defaults(func=handle_visualize_dependencies)

    # --- Resolve Placeholders Command (Batch LLM) ---
    resolve_placeholders_parser = subparsers.add_parser(
        "resolve-placeholders",
        help="Resolve unverified 'p' dependencies using Local LLM in batches",
    )
    resolve_placeholders_parser.add_argument(
        "--tracker",
        required=False,
        help="Path to the tracker file (optional, uses tracker_map.json by default)",
    )
    resolve_placeholders_parser.add_argument(
        "--key", required=False, help="Restricts to dependencies of this source key"
    )
    resolve_placeholders_parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max dependencies to process (default: 200)",
    )
    resolve_placeholders_parser.add_argument(
        "--dep-char", default="p", help="Dependency char to resolve (default: p)"
    )
    resolve_placeholders_parser.add_argument(
        "--model", required=False, help="Path to GGUF model"
    )
    resolve_placeholders_parser.set_defaults(func=handle_resolve_placeholders)

    # --- Determine Dependency Command (LOCAL LLM) ---
    determine_dep_parser = subparsers.add_parser(
        "determine-dependency",
        help="Use local LLM to determine dependency between two keys",
    )
    determine_dep_parser.add_argument(
        "--source-key", required=True, help="Source key (e.g., '1A1' or '1A1#2')"
    )
    determine_dep_parser.add_argument(
        "--target-key", required=True, help="Target key (e.g., '2Ba2' or '2Ba2#1')"
    )
    determine_dep_parser.add_argument(
        "--model", required=False, help="Optional: Path to the GGUF model"
    )
    determine_dep_parser.set_defaults(func=handle_determine_dependency)

    # --- Reconcile Transparency Command ---
    reconcile_parser = subparsers.add_parser(
        "reconcile-transparency",
        help="Reconcile documentation markers with the transparency registry",
    )
    reconcile_parser.add_argument(
        "--path", help="Path to file or directory to scan (default: project root)"
    )
    reconcile_parser.add_argument(
        "--transform",
        choices=["html", "remove", "restore"],
        help="Transform markers: 'html' (convert to comments), 'remove' (delete), or 'restore' (re-insert)",
    )
    reconcile_parser.set_defaults(func=handle_reconcile_transparency)

    # --- Normalize Docs Command ---
    normalize_docs_parser = subparsers.add_parser(
        "normalize-docs",
        help="Normalize documentation files and relocate markers to transparency registry",
    )
    normalize_docs_parser.add_argument("--file", type=str, help="Process a single file")
    normalize_docs_parser.add_argument(
        "--limit", type=int, default=10, help="Max files to process"
    )
    normalize_docs_parser.add_argument(
        "--dry-run", action="store_true", help="Dry run mode"
    )
    normalize_docs_parser.add_argument(
        "--model", required=False, help="Path to GGUF model"
    )
    normalize_docs_parser.set_defaults(func=handle_normalize_docs)

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
        file_handler = logging.FileHandler(
            log_file_path, mode="w", encoding="utf-8", errors="backslashreplace"
        )
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
        suggestion_handler = logging.FileHandler(
            suggestions_log_path,
            mode="w",
            encoding="utf-8",
            errors="backslashreplace",
        )
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

        # If it was resolve-placeholders, run the hook at the end
        if args.func == handle_resolve_placeholders and hasattr(
            args, "accumulated_tracker_updates"
        ):
            if args.accumulated_tracker_updates:
                from cline_utils.dependency_system.utils.populate_comments import (
                    populate_comments_for_batch,
                    report_batch_results,
                )
                from cline_utils.dependency_system.io.transparency_manager import (
                    get_transparency_manager,
                )
                from cline_utils.dependency_system.analysis.dependency_suggester import (
                    load_project_symbol_map,
                )
                from cline_utils.dependency_system.utils.cache_manager import (
                    get_project_root_cached,
                )
                from pathlib import Path

                print(
                    f"Running final populate_comments_hook for {len(args.accumulated_tracker_updates)} accumulated updates..."
                )
                try:
                    results = populate_comments_for_batch(
                        project_root=Path(get_project_root_cached()),
                        updates=args.accumulated_tracker_updates,
                        symbol_map=load_project_symbol_map(),
                        dry_run=False,
                        verbose=False,
                    )
                    if results:
                        report_batch_results(results, dry_run=False)
                        manager = get_transparency_manager()
                        files_processed: Set[str] = set()
                        files_with_maps: Set[str] = set()
                        paths_to_virtualize: List[str] = []
                        for result in results:
                            if isinstance(result, dict):
                                path = result.get("path")
                                if isinstance(path, str):
                                    files_processed.add(path)
                                    maps_added = result.get("maps_added")
                                    maps_updated = result.get("maps_updated")
                                    maps_count = (
                                        maps_added if isinstance(maps_added, int) else 0
                                    ) + (
                                        maps_updated
                                        if isinstance(maps_updated, int)
                                        else 0
                                    )
                                    if maps_count > 0:
                                        files_with_maps.add(path)
                                        paths_to_virtualize.append(path)
                        virtualized = 0
                        if paths_to_virtualize:
                            virtualized = manager.bulk_virtualize_connection_maps(
                                paths_to_virtualize, clear_if_absent=False
                            )
                        if files_processed:
                            manager.bulk_prune_stale_virtual_maps(
                                files_processed, files_with_maps
                            )
                        if virtualized:
                            logger.info(
                                "Virtualized CONNECTION_MAP comments for "
                                f"{virtualized} populated file(s)"
                            )
                except Exception as e:
                    logger.error(f"Error in final populate_comments hook: {e}")
                args.accumulated_tracker_updates.clear()

        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
