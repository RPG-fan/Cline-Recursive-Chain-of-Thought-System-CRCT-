# analysis/project_analyzer.py

import fnmatch
import json
import logging
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from cline_utils.dependency_system.analysis import embedding_manager
from cline_utils.dependency_system.analysis.dependency_analyzer import analyze_file
from cline_utils.dependency_system.analysis.dependency_suggester import (
    suggest_dependencies,
)
from cline_utils.dependency_system.analysis.embedding_manager import generate_embeddings
from cline_utils.dependency_system.core import key_manager
from cline_utils.dependency_system.io import tracker_io
from cline_utils.dependency_system.utils.batch_processor import (
    BatchProcessor,
    process_items,
)
from cline_utils.dependency_system.utils.cache_manager import (
    cache_manager,
    cached,
    clear_all_caches,
    file_modified,
    get_project_root_cached as get_project_root,
    normalize_path_cached as normalize_path,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import is_subpath
from cline_utils.dependency_system.utils.phase_tracker import PhaseTracker
from cline_utils.dependency_system.utils.template_generator import (
    generate_final_review_checklist,
)
from cline_utils.dependency_system.utils.tracker_batch_collector import (
    TrackerBatchCollector,
    create_doc_tracker_update,
    create_main_tracker_update,
    create_mini_tracker_update,
)
from cline_utils.dependency_system.utils.tracker_utils import (
    aggregate_all_dependencies,
    get_key_global_instance_string,
)
from cline_utils.dependency_system.utils.visualize_dependencies import (
    generate_mermaid_diagram,
)

logger = logging.getLogger(__name__)

# Type alias from tracker_io
PathMigrationInfo = Dict[str, Tuple[Optional[str], Optional[str]]]

# --- ADDED: Constants for the new symbol map ---
PROJECT_SYMBOL_MAP_FILENAME = "project_symbol_map.json"
OLD_PROJECT_SYMBOL_MAP_FILENAME = "project_symbol_map_old.json"

# --- Constants for the AST verified links file ---
AST_VERIFIED_LINKS_FILENAME = "ast_verified_links.json"
OLD_AST_VERIFIED_LINKS_FILENAME = "ast_verified_links_old.json"


# Caching for analyze_project (Consider if key_func needs more refinement)
# @cached("project_analysis",
#        key_func=lambda force_analysis=False, force_embeddings=False, **kwargs:
#        f"analyze_project:{normalize_path(get_project_root())}:{(os.path.getmtime(ConfigManager().config_path) if os.path.exists(ConfigManager().config_path) else 0)}:{force_analysis}:{force_embeddings}")


def analyze_project(
    force_analysis: bool = False, force_embeddings: bool = False
) -> Dict[str, Any]:
    """
    Analyzes all files in a project to identify dependencies between them,
    initialize trackers, suggest dependencies using the new contextual key system,
    and generate relevant project templates like the final review checklist.
    Also auto-generates dependency diagrams if enabled.

    Args:
        force_analysis: Bypass cache and force reanalysis of files
        force_embeddings: Force regeneration of embeddings
    Returns:
        Dictionary containing project-wide analysis results and status
    """
    # --- Initial Setup ---
    embedding_manager.RERANKED_FILES = set()
    embedding_manager.RERANKING_COUNTER = 0
    config = ConfigManager()
    project_root = get_project_root()
    logger.info(f"Starting project analysis in directory: {project_root}")

    analyzer_batch_processor = BatchProcessor()
    if force_analysis:
        logger.info("Force analysis requested. Clearing all caches.")
        clear_all_caches()
    analysis_results: Dict[str, Any] = {
        "status": "success",
        "message": "Analysis initiated.",
        "warnings": [],
        "key_generation": {},
        "embedding_generation": {},
        "dependency_suggestion": {},
        "tracker_updates": {},  # MUST BE INITIALIZED AS A DICT
        "file_analysis": {},
        "template_generation": {},
        "auto_visualization": {},
        "symbol_map_generation": {},  # NEW entry for symbol map status
    }
    # --- Exclusion Setup ---
    excluded_dirs_rel = config.get_excluded_dirs()
    excluded_paths_config = config.config.get(
        "excluded_paths", []
    )  # Get raw "excluded_paths" list from config
    excluded_paths_rel = [
        p for p in excluded_paths_config if not os.path.isabs(p)
    ]  # Filter for relative
    all_excluded_paths_abs_set = set(config.get_excluded_paths())
    excluded_extensions = set(config.get_excluded_extensions())
    excluded_file_patterns_config = config.config.get("excluded_file_patterns", [])

    norm_project_root = normalize_path(project_root)
    if any(
        norm_project_root == excluded_path
        or norm_project_root.startswith(
            excluded_path + os.sep
            if not excluded_path.endswith(os.sep)
            else excluded_path
        )  # Ensure trailing sep for startswith
        for excluded_path in all_excluded_paths_abs_set
        if excluded_path
    ):  # check if excluded_path is not empty
        logger.info(f"Skipping analysis of excluded project root: {project_root}")
        analysis_results["status"] = "skipped"
        analysis_results["message"] = "Project root is excluded"
        return analysis_results

    # --- Root Directories Setup ---
    code_root_directories_rel = config.get_code_root_directories()
    doc_directories_rel = config.get_doc_directories()
    all_roots_rel = sorted(list(set(code_root_directories_rel + doc_directories_rel)))
    abs_code_roots = {
        normalize_path(os.path.join(project_root, r)) for r in code_root_directories_rel
    }
    abs_all_roots = {
        normalize_path(os.path.join(project_root, r)) for r in all_roots_rel
    }
    logger.debug(
        f"Absolute code roots for mini-tracker consideration: {abs_code_roots}"
    )

    # old_map_existed_before_gen logic block
    old_map_existed_before_gen = False
    try:
        # Determine the expected path for the old map file RELATIVE to key_manager.py
        # Use the imported key_manager module to find its location
        key_manager_dir = os.path.dirname(os.path.abspath(key_manager.__file__))
        old_map_path = normalize_path(
            os.path.join(key_manager_dir, key_manager.OLD_GLOBAL_KEY_MAP_FILENAME)
        )
        old_map_existed_before_gen = os.path.exists(old_map_path)
        if old_map_existed_before_gen:
            logger.info(
                f"Found existing '{key_manager.OLD_GLOBAL_KEY_MAP_FILENAME}' before key generation. Grid migration will prioritize it."
            )
        else:
            logger.info(
                f"'{key_manager.OLD_GLOBAL_KEY_MAP_FILENAME}' not found before key generation. Grid migration will use tracker definitions as fallback."
            )
    except Exception as path_err:
        logger.error(
            f"Error determining path or checking existence of old key map file: {path_err}. Assuming it didn't exist."
        )
        old_map_existed_before_gen = False

    # --- Key Generation ---
    logger.info("Generating/Regenerating keys...")
    path_to_key_info: Dict[str, key_manager.KeyInfo] = {}
    newly_generated_keys: List[key_manager.KeyInfo] = []
    try:
        # Call generate_keys using the module reference
        path_to_key_info, newly_generated_keys = key_manager.generate_keys(
            all_roots_rel,  # Use this variable name
            excluded_dirs=(set(excluded_dirs_rel)),
            excluded_extensions=(set(excluded_extensions)),
            precomputed_excluded_paths=all_excluded_paths_abs_set,
        )
        analysis_results["key_generation"]["count"] = len(path_to_key_info)
        analysis_results["key_generation"]["new_count"] = len(newly_generated_keys)
        logger.info(
            f"Generated {len(path_to_key_info)} keys for {len(path_to_key_info)} files/dirs."
        )
        if newly_generated_keys:
            logger.info(f"Assigned {len(newly_generated_keys)} new keys.")

        embedding_manager.TOTAL_FILES_TO_RERANK = len(
            [ki for ki in path_to_key_info.values() if not ki.is_directory]
        )
    except key_manager.KeyGenerationError as kge:
        analysis_results["status"] = "error"
        analysis_results["message"] = f"Key generation failed: {kge}"
        logger.critical(analysis_results["message"])
        return analysis_results
    except Exception as e:
        analysis_results["status"] = "error"
        analysis_results["message"] = f"Key generation failed unexpectedly: {e}"
        logger.exception(analysis_results["message"])
        return analysis_results

    # --- Build Path Migration Map (Early, after new keys are generated) ---
    logger.debug("Building path migration map for analysis and updates...")
    old_global_map = key_manager.load_old_global_key_map()  # Load old map (can be None)
    path_migration_info: PathMigrationInfo
    try:
        path_migration_info = tracker_io.build_path_migration_map(
            old_global_map, path_to_key_info
        )
    except ValueError as ve:
        logger.critical(
            f"Failed to build migration map during analysis: {ve}. Downstream functions may fail."
        )
        analysis_results["status"] = "error"
        analysis_results["message"] = f"Migration map build failed: {ve}"
        return analysis_results
    except Exception as e:
        logger.critical(
            f"Unexpected error building migration map during analysis: {e}. Downstream functions may fail.",
            exc_info=True,
        )
        analysis_results["status"] = "error"
        analysis_results["message"] = f"Migration map build error: {e}"
        return analysis_results

    # --- File Identification and Filtering ---
    logger.debug("Identifying files for analysis...")
    files_to_analyze_abs = []
    for abs_root_dir in abs_all_roots:
        if not os.path.isdir(abs_root_dir):
            logger.warning(f"Configured root directory not found: {abs_root_dir}")
            continue
        # Use os.walk - order within walk is OS-dependent but shouldn't affect analysis analysis_results
        for root, dirs, files in os.walk(abs_root_dir, topdown=True):
            norm_root = normalize_path(root)
            dirs[:] = [
                d
                for d in dirs
                if d not in excluded_dirs_rel
                and normalize_path(os.path.join(norm_root, d))
                not in all_excluded_paths_abs_set
            ]
            is_root_excluded_by_path = False
            if norm_root in all_excluded_paths_abs_set or any(
                is_subpath(norm_root, excluded)
                for excluded in all_excluded_paths_abs_set
                if os.path.isdir(excluded)
            ):  # Check against dirs in exclusion set
                is_root_excluded_by_path = True
            if is_root_excluded_by_path:
                dirs[:] = []
                continue
            for file_name in files:
                file_path_abs = normalize_path(os.path.join(norm_root, file_name))
                file_basename = os.path.basename(file_path_abs)
                _, file_ext_tuple = os.path.splitext(file_name)
                file_ext = file_ext_tuple.lower()

                is_excluded = (
                    file_path_abs in all_excluded_paths_abs_set
                    or any(
                        is_subpath(file_path_abs, excluded_path_iter)
                        for excluded_path_iter in all_excluded_paths_abs_set
                        if os.path.isdir(excluded_path_iter)
                    )
                    or file_ext in excluded_extensions
                    or any(
                        fnmatch.fnmatch(file_basename, pattern)
                        for pattern in excluded_file_patterns_config
                    )  # Use original pattern list from config
                )
                if is_excluded:
                    logger.debug(f"Skipping excluded file: {file_path_abs}")
                    continue
                if file_path_abs in path_to_key_info:  # Check against the generated map
                    files_to_analyze_abs.append(file_path_abs)
                else:
                    logger.warning(f"File found but no key generated: {file_path_abs}")
    logger.debug(f"Found {len(files_to_analyze_abs)} files to analyze.")

    # --- File Analysis ---
    logger.debug("Starting file analysis...")
    # Use process_items for potential parallelization
    # Pass force_analysis flag down to analyze_file if caching is implemented there
    analysis_results_list = process_items(
        files_to_analyze_abs, analyze_file, force=force_analysis
    )
    file_analysis_results: Dict[str, Any] = {}
    analyzed_count, skipped_count, error_count = 0, 0, 0
    for file_path_abs, analysis_result in zip(
        files_to_analyze_abs, analysis_results_list
    ):
        if analysis_result:
            if "error" in analysis_result:
                logger.warning(
                    f"Analysis error for {file_path_abs}: {analysis_result['error']}"
                )
                error_count += 1
            elif "skipped" in analysis_result:
                skipped_count += 1
            else:
                file_analysis_results[file_path_abs] = analysis_result
                analyzed_count += 1
        else:
            logger.warning(f"Analysis returned no result for {file_path_abs}")
            error_count += 1
    analysis_results["file_analysis"] = file_analysis_results
    logger.info(
        f"File analysis complete. Analyzed: {analyzed_count}, Skipped: {skipped_count}, Errors: {error_count}"
    )

    # --- NEW: Generate and Save Project Symbol Map ---
    logger.debug("Generating project symbol map...")
    project_symbol_data: Dict[str, Dict[str, Any]] = {}
    for path_abs, single_file_analysis_result in file_analysis_results.items():
        if (
            single_file_analysis_result
            and "error" not in single_file_analysis_result
            and "skipped" not in single_file_analysis_result
        ):
            symbols_for_file: Dict[str, Any] = {
                "file_type": single_file_analysis_result.get("file_type", "unknown")
            }

            # Add specific symbol lists if they exist in the analysis result
            # Expanded list to include ALL analysis data as requested
            for symbol_key in [
                "imports",
                "links",
                "functions",
                "classes",
                "calls",
                "attribute_accesses",
                "inheritance",
                "type_references",
                "globals_defined",
                "exports",
                "code_blocks",
                "scripts",
                "stylesheets",
                "images",
                "decorators_used",
                "exceptions_handled",
                "with_contexts_used",
            ]:
                if symbol_key in single_file_analysis_result:
                    value = single_file_analysis_result[symbol_key]
                    # Only add if the value is truthy (non-empty list/dict/string)
                    if value:
                        symbols_for_file[symbol_key] = value

            # Only add the file entry if it has symbols (besides file_type)
            project_symbol_data[path_abs] = symbols_for_file

    # --- MERGE RUNTIME SYMBOLS USING NEW MERGER MODULE ---
    logger.info("Merging runtime_symbols with AST analysis...")
    try:
        from cline_utils.dependency_system.analysis.symbol_map_merger import (
            load_runtime_symbols,
            merge_runtime_and_ast,
            save_merged_symbol_map,
            validate_merged_output,
        )

        # Load runtime symbols
        runtime_symbols = load_runtime_symbols(project_root)

        # Merge runtime (primary) with AST (enhancement)
        merged_symbol_map = merge_runtime_and_ast(runtime_symbols, project_symbol_data)

        # Validate merged output
        validation_results = validate_merged_output(merged_symbol_map)

        # Report categorized issues
        total_issues = sum(len(v) for v in validation_results.values())
        if total_issues > 0:
            logger.info(f"Symbol map validation found {total_issues} items to review:")

            # Missing data (may be expected)
            if validation_results["missing_runtime_data"]:
                logger.debug(
                    f"  INFO: {len(validation_results['missing_runtime_data'])} files without runtime data"
                )

            if validation_results["missing_ast_data"]:
                logger.debug(
                    f"  INFO: {len(validation_results['missing_ast_data'])} files missing AST imports"
                )

            # General info
            if validation_results["info"]:
                logger.debug(
                    f"  INFO: {len(validation_results['info'])} informational items"
                )

        # Save merged map with backup
        core_dir = os.path.dirname(os.path.abspath(key_manager.__file__))
        symbol_map_path = normalize_path(
            os.path.join(core_dir, PROJECT_SYMBOL_MAP_FILENAME)
        )
        save_merged_symbol_map(merged_symbol_map, symbol_map_path, backup_old=True)

        analysis_results["symbol_map_generation"]["status"] = "success"
        analysis_results["symbol_map_generation"]["path"] = symbol_map_path
        analysis_results["symbol_map_generation"]["count"] = len(merged_symbol_map)
        analysis_results["symbol_map_generation"]["validation_summary"] = {
            k: len(v) for k, v in validation_results.items()
        }

        logger.info(f"Symbol map merge complete: {len(merged_symbol_map)} files")
    except Exception as e:
        logger.error(f"Failed to merge symbol maps: {e}", exc_info=True)
        analysis_results["symbol_map_generation"]["status"] = "error"
        analysis_results["symbol_map_generation"]["error_message"] = str(e)
    # --- END OF SYMBOL MAP MERGE ---

    # --- Create file_to_module mapping (Adapted for path_to_key_info) ---
    # Maps normalized absolute file path -> normalized absolute parent directory path (module path)
    logger.debug("Creating file-to-module mapping...")
    file_to_module: Dict[str, str] = {}
    # Iterate through all the generated key information from path_to_key_info
    for key_info_obj in path_to_key_info.values():  # Iterate over KeyInfo objects
        # We only care about mapping files
        if not key_info_obj.is_directory:
            # Ensure the file has a parent path recorded
            if key_info_obj.parent_path:
                file_path = key_info_obj.norm_path
                module_path = key_info_obj.parent_path  # Direct parent directory path
                file_to_module[file_path] = module_path
                logger.debug(
                    f"Mapped file '{file_path}' to module '{module_path}'"
                )  # Optional debug log

            else:
                logger.warning(
                    f"File '{key_info_obj.norm_path}' (Key: {key_info_obj.key_string}) has no parent path in KeyInfo. Cannot map to a module."
                )
    logger.info(f"File-to-module mapping created with {len(file_to_module)} entries.")

    # --- Embedding generation ---
    logger.info("Starting embedding generation...")
    try:

        def get_optimal_batch_size() -> int:
            """Determine batch size based on available VRAM."""
            import torch

            if torch.cuda.is_available():
                available_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
                if available_gb >= 6:
                    return 256
                elif available_gb >= 4:
                    return 128
                else:
                    return 64
            return 32

        optimal_batch = get_optimal_batch_size()
        logger.debug(f"Using batch size {optimal_batch} based on available VRAM")
        success = generate_embeddings(
            all_roots_rel,
            path_to_key_info,
            force=force_embeddings,
            batch_size=optimal_batch,
        )
        analysis_results["embedding_generation"]["status"] = (
            "success" if success else "partial_failure"
        )
        if not success:
            analysis_results["warnings"].append(
                "Embedding generation failed for some paths."
            )
            logger.warning("Embedding generation failed or skipped for some paths.")
        else:
            logger.info("Embedding generation completed successfully.")
    except Exception as e:
        analysis_results["embedding_generation"]["status"] = "error"
        analysis_results["status"] = (
            "error"  # Upgrade status to error if embedding process fails critically
        )
        analysis_results["message"] = (
            f"Embedding generation process failed critically: {e}"
        )
        logger.exception(analysis_results["message"])
        return analysis_results

    # --- Dependency Suggestion (Adapted for Path-Based Output) ---
    logger.info("Starting dependency suggestion (path-based)...")
    analysis_results["dependency_suggestion"] = {
        "status": "pending",
        "suggestion_count": 0,
        "ast_link_count": 0,
    }  # Added ast_link_count

    # --- >>> INITIALIZE all_suggestions HERE <<< ---
    all_path_based_suggestions: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    # --- Initialize list to collect all AST-verified links from the project ---
    all_project_ast_links: List[Dict[str, str]] = []

    analyzed_file_paths = list(file_analysis_results.keys())
    # Use configured threshold for doc_similarity
    doc_similarity_threshold = config.get_threshold("doc_similarity")

    # PERFORMANCE: process suggestions in parallel using BatchProcessor to increase throughput
    # Build a lightweight wrapper to call suggest_dependencies for a single file
    def _suggest_wrapper(
        single_file_path: str,
        *,
        path_to_key_info_map,
        project_root_abs,
        file_analysis_blob,
        doc_similarity_threshold,
        shared_scan_counter=None,
    ) -> Tuple[str, List[Tuple[str, str]], List[Dict[str, str]]]:
        # Returns (source_path, suggestions, ast_links)

        try:
            suggs, ast_links = suggest_dependencies(
                single_file_path,
                path_to_key_info_map,
                project_root_abs,
                file_analysis_blob,
                doc_similarity_threshold,
                shared_scan_counter=shared_scan_counter,
            )
            return (single_file_path, suggs or [], ast_links or [])
        except Exception as e:
            logger.error(
                f"Suggestion wrapper error for {single_file_path}: {e}", exc_info=True
            )
            return (single_file_path, [], [])

    suggestion_batcher = BatchProcessor(
        show_progress=True, phase_name="Dependency Suggestion"
    )

    # Create a shared counter for global reranker limit
    import multiprocessing

    # Use multiprocessing.Value which is thread-safe and has get_lock()
    # We don't need Manager() since we are using ThreadPoolExecutor (threads share memory)
    shared_scan_counter = multiprocessing.Value("i", 0)

    # Parallel process suggestion generation
    suggestion_results = suggestion_batcher.process_items(
        analyzed_file_paths,
        _suggest_wrapper,
        path_to_key_info_map=path_to_key_info,
        project_root_abs=project_root,
        file_analysis_blob=file_analysis_results,
        doc_similarity_threshold=doc_similarity_threshold,
        shared_scan_counter=shared_scan_counter,  # Pass shared counter
    )
    # Use configured threshold for doc_similarity
    doc_similarity_threshold = config.get_threshold("doc_similarity")

    # --- Store the length of the last printed progress line ---
    _last_progress_message_length = 0
    # Aggregate results and print progress using PhaseTracker
    with PhaseTracker(
        total=len(suggestion_results), phase_name="Dependency Suggestion"
    ) as tracker:
        for i, (
            src_path_processed,
            suggestions_for_file,
            ast_links_for_file,
        ) in enumerate(suggestion_results):
            if suggestions_for_file:
                all_path_based_suggestions[src_path_processed].extend(
                    suggestions_for_file
                )
                analysis_results["dependency_suggestion"]["suggestion_count"] += len(
                    suggestions_for_file
                )
            if ast_links_for_file:
                all_project_ast_links.extend(ast_links_for_file)
                analysis_results["dependency_suggestion"]["ast_link_count"] += len(
                    ast_links_for_file
                )

            current_suggestion_count = analysis_results["dependency_suggestion"][
                "suggestion_count"
            ]
            current_ast_link_count = analysis_results["dependency_suggestion"][
                "ast_link_count"
            ]

            tracker.update(
                description=f"Found {current_suggestion_count} suggestions, {current_ast_link_count} AST links"
            )

    # --- ADDED: Unload the reranker model to free up VRAM ---
    # This is called once after all files have been processed.
    try:
        logger.debug("Unloading reranker model after suggestion phase.")
        embedding_manager.unload_reranker_model()
    except Exception as e:
        logger.warning(f"Error during reranker model unload: {e}")
    # --- END OF ADDED BLOCK ---

    # 2. Print the final summary message using the standard logger
    final_suggestion_count = analysis_results["dependency_suggestion"][
        "suggestion_count"
    ]
    final_ast_link_count = analysis_results["dependency_suggestion"][
        "ast_link_count"
    ]  # Get final count
    logger.info(
        f"Dependency suggestion complete. Generated {final_suggestion_count} total raw suggestions "
        + f"and {final_ast_link_count} AST-verified links from file analysis."
    )

    # --- NEW: Save all_project_ast_links to ast_verified_links.json ---
    analysis_results["ast_verified_links_generation"] = {}  # Initialize status dict

    # Consolidate duplicates in all_project_ast_links
    if all_project_ast_links:
        consolidated_links_map = {}
        for link in all_project_ast_links:
            key = (link["source_path"], link["target_path"])
            if key not in consolidated_links_map:
                consolidated_links_map[key] = {
                    "source_path": link["source_path"],
                    "target_path": link["target_path"],
                    "char": link.get("char", "<"),  # Default char if missing
                    "reasons": set(),
                }

            # Add reason(s) to the set
            reason = link.get("reason")
            if reason:
                consolidated_links_map[key]["reasons"].add(reason)

        # Convert back to list and format reasons
        consolidated_links = []
        for link_data in consolidated_links_map.values():
            # Sort reasons for deterministic output
            sorted_reasons = sorted(list(link_data["reasons"]))
            link_data["reason"] = ", ".join(sorted_reasons)
            del link_data["reasons"]  # Remove the set
            consolidated_links.append(link_data)

        all_project_ast_links = consolidated_links

    if all_project_ast_links:
        try:
            core_dir = os.path.dirname(
                os.path.abspath(key_manager.__file__)
            )  # Assuming key_manager is imported
            current_ast_links_path = normalize_path(
                os.path.join(core_dir, AST_VERIFIED_LINKS_FILENAME)
            )
            old_ast_links_path = normalize_path(
                os.path.join(core_dir, OLD_AST_VERIFIED_LINKS_FILENAME)
            )
            os.makedirs(core_dir, exist_ok=True)

            if os.path.exists(current_ast_links_path):
                try:
                    shutil.move(current_ast_links_path, old_ast_links_path)
                    logger.debug(
                        f"Renamed existing '{AST_VERIFIED_LINKS_FILENAME}' to '{OLD_AST_VERIFIED_LINKS_FILENAME}'."
                    )
                except OSError as rename_err:
                    logger.error(
                        f"Failed to rename current AST verified links file to old: {rename_err}"
                    )

            with open(current_ast_links_path, "w", encoding="utf-8") as f_ast_links:
                json.dump(all_project_ast_links, f_ast_links, indent=2)
            logger.info(
                f"Successfully saved AST verified links to: {current_ast_links_path} ({len(all_project_ast_links)} links)"
            )
            analysis_results["ast_verified_links_generation"]["status"] = "success"
            analysis_results["ast_verified_links_generation"][
                "path"
            ] = current_ast_links_path
            analysis_results["ast_verified_links_generation"]["count"] = len(
                all_project_ast_links
            )
        except Exception as e_ast_links_save:
            logger.error(
                f"Failed to save AST verified links: {e_ast_links_save}", exc_info=True
            )
            analysis_results["ast_verified_links_generation"]["status"] = "error"
            analysis_results["ast_verified_links_generation"]["error_message"] = str(
                e_ast_links_save
            )
    else:
        logger.info(
            "No AST-verified links collected. 'ast_verified_links.json' will not be created or will be empty if it previously existed and was moved."
        )
        analysis_results["ast_verified_links_generation"]["status"] = "no_data"
        # Optionally, ensure an empty file or remove old if no data
        core_dir = os.path.dirname(os.path.abspath(key_manager.__file__))
        current_ast_links_path = normalize_path(
            os.path.join(core_dir, AST_VERIFIED_LINKS_FILENAME)
        )
        if not os.path.exists(
            current_ast_links_path
        ):  # If it wasn't created because no data
            with open(current_ast_links_path, "w", encoding="utf-8") as f_empty:
                json.dump([], f_empty)  # Create empty

    # --- Combine suggestions within each source key using priority ---
    # This step is crucial before adding reciprocal ones
    # Import helper here to avoid potential circular dependencies at module level
    from cline_utils.dependency_system.analysis.dependency_suggester import (
        combine_suggestions_path_based_with_char_priority,
    )  # Needs import

    combined_path_suggestions_for_conversion = defaultdict(list)
    for source_path, path_sugg_list in all_path_based_suggestions.items():
        combined_path_suggestions_for_conversion[source_path] = (
            combine_suggestions_path_based_with_char_priority(
                path_sugg_list, source_path
            )
        )
    logger.debug(
        f"Combined path-based suggestions by priority. Count remains: {sum(len(v) for v in combined_path_suggestions_for_conversion.values())}"
    )

    # --- CONVERT Path-Based Suggestions to KEY#global_instance Format ---
    logger.debug(
        "Converting path-based suggestions to KEY#global_instance format for tracker updates..."
    )
    all_global_instance_suggestions: Dict[str, List[Tuple[str, str]]] = defaultdict(
        list
    )

    # --- MODIFICATION: Use a local cache for this specific conversion loop if desired,
    # or rely on the module-level cache in tracker_utils.
    # For clarity, let's show passing a local cache instance.
    # If the module-level cache in tracker_utils is sufficient, this can be omitted.
    _base_key_to_sorted_KIs_cache_for_conversion: Dict[
        str, List[key_manager.KeyInfo]
    ] = defaultdict(list)
    # ---

    # Ensure path_to_key_info is the current global map
    current_global_map = path_to_key_info

    for (
        src_path,
        path_deps_list,
    ) in (
        combined_path_suggestions_for_conversion.items()
    ):  # Assuming combined_path_suggestions_for_conversion is populated
        src_ki = current_global_map.get(src_path)
        if not src_ki:
            logger.warning(
                f"ProjectAnalyzerConversion: Source path '{src_path}' from suggestions not in global map. Skipping."
            )
            continue

        # --- MODIFICATION: Call the utility function from tracker_utils ---
        src_key_global_instance_str = get_key_global_instance_string(
            src_ki,
            current_global_map,
            _base_key_to_sorted_KIs_cache_for_conversion,  # Pass the local cache
        )
        # ---
        if not src_key_global_instance_str:
            logger.warning(
                f"ProjectAnalyzerConversion: Could not get global instance string for source KI '{src_ki.norm_path}'. Skipping suggestions from it."
            )
            continue

        processed_target_deps = []
        for tgt_path, char_val in path_deps_list:
            tgt_ki = current_global_map.get(tgt_path)
            if not tgt_ki:
                logger.warning(
                    f"ProjectAnalyzerConversion: Target path '{tgt_path}' not in global map. Skipping for source '{src_path}'."
                )
                continue

            # --- MODIFICATION: Call the utility function from tracker_utils ---
            tgt_key_global_instance_str = get_key_global_instance_string(
                tgt_ki,
                current_global_map,
                _base_key_to_sorted_KIs_cache_for_conversion,  # Pass the local cache
            )
            # ---
            if not tgt_key_global_instance_str:
                logger.warning(
                    f"ProjectAnalyzerConversion: Could not get global instance for target KI '{tgt_ki.norm_path}'. Skipping for source '{src_path}'."
                )
                continue
            processed_target_deps.append((tgt_key_global_instance_str, char_val))

        if processed_target_deps:
            all_global_instance_suggestions[src_key_global_instance_str].extend(
                processed_target_deps
            )

    logger.debug(
        f"Converted to {sum(len(v) for v in all_global_instance_suggestions.values())} KEY#global_instance formatted suggestions."
    )

    # --- Update Trackers ---
    logger.info("Updating trackers...")
    # Ensure suggestions always overwrite placeholders/empty in update phase
    # by signaling intent via force flag for suggestion-application behavior.
    # We still let pruning and other safety logic run normally inside update_tracker.
    analysis_results["tracker_updates"]["mini"] = {}
    analysis_results["tracker_updates"]["doc"] = "pending"
    analysis_results["tracker_updates"]["main"] = "pending"

    # --- Update Mini Trackers FIRST ---
    mini_tracker_paths_updated = (
        set()
    )  # Keep track of updated module paths to avoid redundant processing if paths overlap
    potential_mini_tracker_dirs: List[key_manager.KeyInfo] = []
    for ki_obj in path_to_key_info.values():
        if ki_obj.is_directory:
            if not ki_obj.norm_path or ki_obj.norm_path == ".":
                logger.warning(
                    f"Skipping KeyInfo with invalid norm_path for module consideration: {ki_obj}"
                )
                continue
            is_mod_dir = False
            # A directory is a potential module dir if it IS a code root OR if it is a SUBDIRECTORY of any code root.
            for code_r_abs in abs_code_roots:
                if ki_obj.norm_path == code_r_abs or is_subpath(
                    ki_obj.norm_path, code_r_abs
                ):
                    if ki_obj.norm_path not in all_excluded_paths_abs_set:
                        is_mod_dir = True
                        break
            if is_mod_dir:
                potential_mini_tracker_dirs.append(ki_obj)
    unique_potential_module_dirs_map: Dict[str, key_manager.KeyInfo] = {}
    for ki_dir in potential_mini_tracker_dirs:
        if ki_dir.norm_path not in unique_potential_module_dirs_map:
            unique_potential_module_dirs_map[ki_dir.norm_path] = ki_dir
    potential_mini_tracker_dirs = list(unique_potential_module_dirs_map.values())
    # Optional: Sort them by path for consistent processing order, e.g., parent dirs first.
    potential_mini_tracker_dirs.sort(key=lambda ki: ki.norm_path)
    logger.debug(
        f"Identified {len(potential_mini_tracker_dirs)} potential directories for mini-trackers:"
    )
    for ki_dir_log in potential_mini_tracker_dirs:
        logger.debug(
            f"  - Potential Module Dir: {ki_dir_log.norm_path} (Key: {ki_dir_log.key_string})"
        )

    # --- Batch Update Trackers ---
    # Using batch collector to minimize disk I/O by collecting all updates
    # in memory and writing them all at once at the end
    batch_collector = TrackerBatchCollector()

    # --- Collect Mini Tracker Updates ---
    with PhaseTracker(
        total=len(potential_mini_tracker_dirs), phase_name="Preparing Mini Trackers"
    ) as tracker:
        for module_key_info_obj in potential_mini_tracker_dirs:
            norm_module_path = module_key_info_obj.norm_path
            tracker.update(
                description=f"Preparing {os.path.basename(norm_module_path)}"
            )

            if not norm_module_path:
                logger.error(
                    f"Encountered module KeyInfo with empty norm_path: {module_key_info_obj}. Skipping mini-tracker update."
                )
                analysis_results["tracker_updates"]["mini"][
                    f"ERROR_EMPTY_MODULE_PATH_FOR_{module_key_info_obj.key_string}"
                ] = "failure_empty_path"
                continue
            if norm_module_path in mini_tracker_paths_updated:
                continue
            expected_mini_tracker_filename = os.path.basename(
                tracker_io.get_tracker_path(
                    project_root=project_root,
                    tracker_type="mini",
                    module_path=norm_module_path,
                )
            )
            if os.path.isdir(norm_module_path) and not _is_empty_dir(
                norm_module_path, expected_mini_tracker_filename
            ):
                mini_tracker_path_val = tracker_io.get_tracker_path(
                    project_root=project_root,
                    tracker_type="mini",
                    module_path=norm_module_path,
                )
                logger.debug(
                    f"Preparing mini tracker for module '{norm_module_path}' (Key: {module_key_info_obj.key_string})"
                )
                mini_tracker_paths_updated.add(norm_module_path)

                try:
                    # Prepare tracker update in memory (no disk I/O yet)
                    prepared_data = tracker_io.prepare_tracker_update(
                        output_file_suggestion=mini_tracker_path_val,
                        path_to_key_info=path_to_key_info,
                        tracker_type="mini",
                        suggestions_external=all_global_instance_suggestions,
                        file_to_module=file_to_module,
                        new_keys=newly_generated_keys,
                        force_apply_suggestions=True,
                        use_old_map_for_migration=old_map_existed_before_gen,
                    )

                    if prepared_data:
                        # Create TrackerUpdate and add to batch
                        update = create_mini_tracker_update(
                            output_file=prepared_data["output_file"],
                            key_info_list=prepared_data["key_info_list"],
                            grid_rows=prepared_data["grid_rows"],
                            last_key_edit=prepared_data["last_key_edit"],
                            last_grid_edit=prepared_data["last_grid_edit"],
                            module_path=prepared_data["module_path"]
                            or norm_module_path,
                            path_to_key_info=path_to_key_info,
                            existing_lines=prepared_data.get("existing_lines"),
                            tracker_exists=prepared_data.get("tracker_exists", False),
                        )
                        batch_collector.add(update)
                        analysis_results["tracker_updates"]["mini"][
                            norm_module_path
                        ] = "pending"
                    else:
                        analysis_results["tracker_updates"]["mini"][
                            norm_module_path
                        ] = "failure_preparation"
                        analysis_results["status"] = "warning"

                except Exception as mini_err:
                    logger.error(
                        f"Error preparing mini tracker {mini_tracker_path_val}: {mini_err}",
                        exc_info=True,
                    )
                    analysis_results["tracker_updates"]["mini"][
                        norm_module_path
                    ] = "failure"
                    analysis_results["status"] = "warning"
            elif os.path.isdir(norm_module_path):
                logger.debug(
                    f"Skipping mini-tracker update for empty module directory: {norm_module_path}"
                )

    # --- Collect Doc Tracker Update ---
    doc_directories_rel = config.get_doc_directories()
    doc_tracker_path = (
        tracker_io.get_tracker_path(project_root, tracker_type="doc")
        if doc_directories_rel
        else None
    )
    if doc_tracker_path:
        logger.info(f"Preparing doc tracker: {doc_tracker_path}")
        try:
            prepared_data = tracker_io.prepare_tracker_update(
                output_file_suggestion=doc_tracker_path,
                path_to_key_info=path_to_key_info,
                tracker_type="doc",
                suggestions_external=all_global_instance_suggestions,
                file_to_module=file_to_module,
                new_keys=newly_generated_keys,
                force_apply_suggestions=True,
                use_old_map_for_migration=old_map_existed_before_gen,
            )

            if prepared_data:
                update = create_doc_tracker_update(
                    output_file=prepared_data["output_file"],
                    key_info_list=prepared_data["key_info_list"],
                    grid_rows=prepared_data["grid_rows"],
                    last_key_edit=prepared_data["last_key_edit"],
                    last_grid_edit=prepared_data["last_grid_edit"],
                    path_to_key_info=path_to_key_info,
                )
                batch_collector.add(update)
                analysis_results["tracker_updates"]["doc"] = "pending"
            else:
                analysis_results["tracker_updates"]["doc"] = "failure_preparation"
                analysis_results["status"] = "warning"

        except Exception as doc_err:
            logger.error(
                f"Error preparing doc tracker {doc_tracker_path}: {doc_err}",
                exc_info=True,
            )
            analysis_results["tracker_updates"]["doc"] = "failure"
            analysis_results["status"] = "warning"
    else:
        logger.info(
            "Doc tracker update skipped: No doc directories configured or path determination failed."
        )
        analysis_results["tracker_updates"]["doc"] = "skipped_no_config"

    # --- Collect Main Tracker Update LAST ---
    main_tracker_path = tracker_io.get_tracker_path(project_root, tracker_type="main")
    logger.info(f"Preparing main tracker: {main_tracker_path}")
    try:
        prepared_data = tracker_io.prepare_tracker_update(
            output_file_suggestion=main_tracker_path,
            path_to_key_info=path_to_key_info,
            tracker_type="main",
            suggestions_external=all_global_instance_suggestions,
            file_to_module=file_to_module,
            new_keys=newly_generated_keys,
            force_apply_suggestions=True,
            use_old_map_for_migration=old_map_existed_before_gen,
        )

        if prepared_data:
            update = create_main_tracker_update(
                output_file=prepared_data["output_file"],
                key_info_list=prepared_data["key_info_list"],
                grid_rows=prepared_data["grid_rows"],
                last_key_edit=prepared_data["last_key_edit"],
                last_grid_edit=prepared_data["last_grid_edit"],
                path_to_key_info=path_to_key_info,
            )
            batch_collector.add(update)
            analysis_results["tracker_updates"]["main"] = "pending"
        else:
            analysis_results["tracker_updates"]["main"] = "failure_preparation"
            analysis_results["status"] = "warning"

    except Exception as main_err:
        logger.error(
            f"Error preparing main tracker {main_tracker_path}: {main_err}",
            exc_info=True,
        )
        analysis_results["tracker_updates"]["main"] = "failure"
        analysis_results["status"] = "warning"

    # --- Commit All Tracker Updates in a Single Batch ---
    if not batch_collector.is_empty():
        logger.info(f"Committing {len(batch_collector)} tracker updates in batch...")
        try:
            results = batch_collector.commit_all()

            # Update analysis results based on commit outcomes
            for file_path, success in results.items():
                # Determine tracker type from file path
                if "_module.md" in file_path:
                    tracker_cat = "mini"
                    # Find the module path for this mini tracker
                    for path, status in analysis_results["tracker_updates"][
                        "mini"
                    ].items():
                        if status == "pending":
                            analysis_results["tracker_updates"]["mini"][path] = (
                                "success" if success else "failure"
                            )
                            break
                elif "doc_tracker.md" in file_path:
                    tracker_cat = "doc"
                    analysis_results["tracker_updates"]["doc"] = (
                        "success" if success else "failure"
                    )
                else:
                    tracker_cat = "main"
                    analysis_results["tracker_updates"]["main"] = (
                        "success" if success else "failure"
                    )

                if not success:
                    logger.error(f"Failed to write {tracker_cat} tracker: {file_path}")
                    analysis_results["status"] = "warning"

            logger.info("Batch tracker update completed successfully")

        except Exception as batch_err:
            logger.error(f"Batch tracker update failed: {batch_err}", exc_info=True)
            # Mark all pending updates as failed
            for tracker_cat in ["mini", "doc", "main"]:
                if isinstance(analysis_results["tracker_updates"][tracker_cat], dict):
                    for path, status in analysis_results["tracker_updates"][
                        tracker_cat
                    ].items():
                        if status == "pending":
                            analysis_results["tracker_updates"][tracker_cat][
                                path
                            ] = "failure"
                elif analysis_results["tracker_updates"][tracker_cat] == "pending":
                    analysis_results["tracker_updates"][tracker_cat] = "failure"
            analysis_results["status"] = "error"
    else:
        logger.info("No tracker updates to commit")

    # --- Template Generation ---
    logger.info("Starting template generation (e.g., final review checklist)...")
    try:
        # Pass the pre-computed current global map and path migration map
        checklist_generated_successfully = generate_final_review_checklist(
            global_key_map_param=path_to_key_info,
            path_migration_info_param=path_migration_info,
        )
        if checklist_generated_successfully:
            analysis_results["template_generation"][
                "final_review_checklist"
            ] = "success"
            logger.info("Final review checklist generated successfully.")
        else:
            analysis_results["template_generation"][
                "final_review_checklist"
            ] = "failure"
            logger.warning(
                "Final review checklist generation failed. Check logs for details."
            )
            if analysis_results["status"] == "success":
                analysis_results["status"] = "warning"
                analysis_results[
                    "message"
                ] += " Warning: Failed to generate final review checklist."
    except Exception as template_err:
        analysis_results["template_generation"]["final_review_checklist"] = "error"
        logger.error(
            f"Critical error during template generation: {template_err}", exc_info=True
        )
        if analysis_results["status"] == "success":
            analysis_results["status"] = "warning"
        analysis_results[
            "message"
        ] += f" Critical error during template generation: {template_err}."

    # --- Auto Diagram Generation ---
    auto_generate_enabled = config.config.get("visualization", {}).get(
        "auto_generate_on_analyze", True
    )
    if auto_generate_enabled:
        logger.info("Starting automatic diagram generation...")
        analysis_results["auto_visualization"] = {"overview": "skipped", "modules": {}}
        memory_dir_rel_analyzer = config.get_path("memory_dir", "cline_docs")
        default_diagram_subdir = "dependency_diagrams"
        default_auto_diagram_dir_abs = normalize_path(
            os.path.join(project_root, memory_dir_rel_analyzer, default_diagram_subdir)
        )
        auto_diagram_output_dir_config_rel = config.config.get("visualization", {}).get(
            "auto_diagram_output_dir"
        )
        if auto_diagram_output_dir_config_rel:
            auto_diagram_output_dir_abs = normalize_path(
                os.path.join(project_root, auto_diagram_output_dir_config_rel)
            )
            logger.debug(
                f"Using configured auto diagram output directory: {auto_diagram_output_dir_abs}"
            )
        else:
            auto_diagram_output_dir_abs = default_auto_diagram_dir_abs
            logger.debug(
                f"Using default auto diagram output directory: {auto_diagram_output_dir_abs}"
            )
        try:
            if not os.path.exists(auto_diagram_output_dir_abs):
                os.makedirs(auto_diagram_output_dir_abs, exist_ok=True)

            # Find all tracker paths *again* here, as they might have just been created/updated
            current_tracker_paths = tracker_io.find_all_tracker_paths(
                config, project_root
            )

            # --- Calculate module keys to visualize (needed for progress bar total) ---
            module_keys_to_visualize = []
            for key_info_obj_analyzer in path_to_key_info.values():
                if key_info_obj_analyzer.is_directory:
                    is_top_level_module_dir_analyzer = any(
                        key_info_obj_analyzer.norm_path == acr_path_analyzer
                        or key_info_obj_analyzer.parent_path == acr_path_analyzer
                        for acr_path_analyzer in abs_code_roots
                    )
                    if is_top_level_module_dir_analyzer:
                        module_keys_to_visualize.append(
                            key_info_obj_analyzer.key_string
                        )
            module_keys_unique = sorted(list(set(module_keys_to_visualize)))
            logger.info(
                f"Identified module keys for auto-visualization: {module_keys_unique}"
            )

            # --- Generate Project Overview Diagram ---

            # Pre-calculate aggregation once for all diagrams to avoid repeated overhead
            logger.info(
                "Aggregating dependencies for visualization (once for all diagrams)..."
            )
            try:
                project_aggregated_links = aggregate_all_dependencies(
                    set(current_tracker_paths),
                    path_migration_info,
                    path_to_key_info,
                )
            except Exception as e_agg:
                logger.error(
                    f"Failed to aggregate dependencies for visualization: {e_agg}"
                )
                project_aggregated_links = {}

            total_diagrams = 1 + len(module_keys_unique)  # 1 for overview + N modules
            with PhaseTracker(
                total=total_diagrams, phase_name="Generating Diagrams"
            ) as diagram_tracker:
                diagram_tracker.update(description="Generating project overview")
                logger.debug("Generating project overview diagram...")
                overview_filename = "project_overview_dependencies.mermaid"
                overview_path = os.path.join(
                    auto_diagram_output_dir_abs, overview_filename
                )
                # Pass path_migration_info to generate_mermaid_diagram
                overview_mermaid_code = generate_mermaid_diagram(
                    focus_keys_list_input=[],
                    global_path_to_key_info_map=path_to_key_info,
                    path_migration_info=path_migration_info,
                    all_tracker_paths_list=list(current_tracker_paths),
                    config_manager_instance=config,
                    pre_aggregated_links=project_aggregated_links,
                )
                if (
                    overview_mermaid_code
                    and "// No relevant data" not in overview_mermaid_code
                    and not overview_mermaid_code.strip().startswith("Error:")
                ):
                    with open(overview_path, "w", encoding="utf-8") as f:
                        f.write(overview_mermaid_code)
                    logger.debug(f"Project overview diagram saved to {overview_path}")
                    analysis_results["auto_visualization"]["overview"] = "success"
                else:
                    logger.warning(
                        f"Skipping save for project overview diagram (no data or failed generation). Error: {overview_mermaid_code if overview_mermaid_code and 'Error:' in overview_mermaid_code[:20] else 'No data'}"
                    )
                    analysis_results["auto_visualization"][
                        "overview"
                    ] = "nodata_or_failed"
                diagram_tracker.update()

                # --- Generate Per-Module Diagrams ---
            module_keys_to_visualize = []
            for key_info_obj_analyzer in path_to_key_info.values():
                if key_info_obj_analyzer.is_directory:
                    is_top_level_module_dir_analyzer = any(
                        key_info_obj_analyzer.norm_path == acr_path_analyzer
                        or key_info_obj_analyzer.parent_path == acr_path_analyzer
                        for acr_path_analyzer in abs_code_roots
                    )
                    if is_top_level_module_dir_analyzer:
                        module_keys_to_visualize.append(
                            key_info_obj_analyzer.key_string
                        )
            module_keys_unique = sorted(list(set(module_keys_to_visualize)))
            logger.debug(
                f"Identified module keys for auto-visualization: {module_keys_unique}"
            )
            analysis_results["auto_visualization"]["modules"] = {
                mk: "skipped" for mk in module_keys_unique
            }
            for module_key_str in module_keys_unique:
                diagram_tracker.update(
                    description=f"Generating module {module_key_str}"
                )
                logger.debug(f"Generating diagram for module: {module_key_str}...")
                module_diagram_filename = (
                    f"module_{module_key_str}_dependencies.mermaid".replace(
                        "/", "_"
                    ).replace("\\", "_")
                )
                module_diagram_path = os.path.join(
                    auto_diagram_output_dir_abs, module_diagram_filename
                )
                module_mermaid_code = generate_mermaid_diagram(
                    focus_keys_list_input=[module_key_str],
                    global_path_to_key_info_map=path_to_key_info,
                    path_migration_info=path_migration_info,
                    all_tracker_paths_list=list(current_tracker_paths),
                    config_manager_instance=config,
                    pre_aggregated_links=project_aggregated_links,
                )
                if (
                    module_mermaid_code
                    and "// No relevant data" not in module_mermaid_code
                    and "Error:" not in module_mermaid_code[:20]
                ):
                    with open(module_diagram_path, "w", encoding="utf-8") as f:
                        f.write(module_mermaid_code)
                    logger.debug(
                        f"Module {module_key_str} diagram saved to {module_diagram_path}"
                    )
                    analysis_results["auto_visualization"]["modules"][
                        module_key_str
                    ] = "success"
                else:
                    logger.warning(
                        f"Skipping save for module {module_key_str} diagram (no data or failed generation). Error: {module_mermaid_code if module_mermaid_code and 'Error:' in module_mermaid_code[:20] else 'No data'}"
                    )
                    analysis_results["auto_visualization"]["modules"][
                        module_key_str
                    ] = "nodata_or_failed"
        except Exception as viz_err:
            logger.error(
                f"Error during automatic diagram generation: {viz_err}", exc_info=True
            )
            analysis_results["auto_visualization"]["status"] = "error"
            if analysis_results["status"] == "success":
                analysis_results["status"] = "warning"
            analysis_results[
                "message"
            ] += f" Warning: Automatic diagram generation failed: {viz_err}."
    else:
        logger.info("Automatic diagram generation is disabled in config.")
        analysis_results["auto_visualization"]["status"] = "disabled"

    if analysis_results["status"] == "success" and not analysis_results["warnings"]:
        final_message = "Project analysis completed successfully."
    elif analysis_results["status"] == "success" and analysis_results["warnings"]:
        final_message = f"Project analysis completed with warnings: {'; '.join(analysis_results['warnings'])}. Check logs."
        analysis_results["status"] = "warning"
    elif analysis_results["status"] == "warning":
        final_message = f"Project analysis completed with warnings: {analysis_results.get('message', '')} {'; '.join(analysis_results['warnings'])}. Check logs."
    else:
        final_message = f"Project analysis failed: {analysis_results.get('message', '')}. Check logs."
    analysis_results["message"] = final_message
    logger.info(final_message)

    # --- MODIFICATION: Clear the dedicated AST cache at the end of the project analysis ---
    try:
        ast_cache_instance = cache_manager.get_cache("ast_cache")
        ast_cache_instance.data.clear()  # More direct way to clear
        logger.info("Cleared in-memory AST cache ('ast_cache') after project analysis.")
    except Exception as e_clear_ast:
        # Catch any exception during cache clearing to prevent analyze_project from failing here
        logger.warning(
            f"Could not explicitly clear 'ast_cache' at the end of project analysis: {e_clear_ast}"
        )
    # --- END OF MODIFICATION ---

    return analysis_results


def _is_empty_dir(
    dir_path: str, tracker_filename_to_ignore: Optional[str] = None
) -> bool:
    """
    Checks if a directory is effectively empty for tracker creation purposes.
    It's empty if it contains no files or subdirectories, OR if it only contains
    the tracker file that would be associated with it (if provided).
    """
    try:
        items_in_dir = os.listdir(dir_path)
        if not items_in_dir:
            return True
        if tracker_filename_to_ignore:
            # Check if only the tracker file itself is present
            if (
                len(items_in_dir) == 1
                and items_in_dir[0].lower() == tracker_filename_to_ignore.lower()
            ):
                logger.debug(
                    f"Directory '{dir_path}' considered empty for tracker creation as it only contains '{tracker_filename_to_ignore}'."
                )
                return True
            # Check if it contains more than just the tracker file
            elif (
                tracker_filename_to_ignore.lower()
                in [item.lower() for item in items_in_dir]
                and len(items_in_dir) > 1
            ):
                return False
            elif (
                tracker_filename_to_ignore.lower()
                not in [item.lower() for item in items_in_dir]
                and items_in_dir
            ):
                return False
        return False if items_in_dir else True
    except FileNotFoundError:
        logger.debug(
            f"Directory not found '{dir_path}' while checking if empty (for tracker creation). Treating as empty."
        )
        return True
    except NotADirectoryError:
        logger.debug(
            f"Path '{dir_path}' is not a directory (for tracker creation). Treating as empty."
        )
        return True
    except OSError as e:
        logger.error(
            f"OS error checking dir '{dir_path}' (for tracker creation): {e}. Assuming not empty for safety."
        )
        return False  # Assume not empty on permission error etc. to be safe


# --- End of project_analyzer.py ---
