# cline_utils/dependency_system/utils/template_generator.py
import datetime
import logging
import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from cline_utils.dependency_system.core.key_manager import KeyInfo, load_global_key_map
from cline_utils.dependency_system.io.tracker_io import PathMigrationInfo
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import (
    get_project_root,
    is_subpath,
    normalize_path,
)
from cline_utils.dependency_system.utils.tracker_utils import (
    aggregate_all_dependencies,
    find_all_tracker_paths,
    get_key_global_instance_string,
)

logger = logging.getLogger(__name__)

# --- Configuration ---
CHECKLIST_FILENAME = "final_review_checklist.md"
ARCHIVE_DIR_RELATIVE = "cline_docs/Archive"
SYSTEM_MANIFEST_RELATIVE = "cline_docs/system_manifest.md"
POSITIVE_DOC_DEPENDENCY_CHARS = {"x", "<", ">", "d", "s", "S"}

ADDED_DEPS_TABLE_START_MARKER = "<!-- ADDED_DEPENDENCIES_TABLE_START -->"
ADDED_DEPS_TABLE_END_MARKER = "<!-- ADDED_DEPENDENCIES_TABLE_END -->"

CHECKLIST_TEMPLATE_CONTENT = f"""
# Code-Documentation Cross-Reference Checklist

## Review Information
- **Project**: {{project_name}}
- **Cycle**: {{cycle_number}}
- **Date Created**: {{date_created}}

## 1. Review Scope
- Code Directories: {{code_root_dirs}}
- Documentation Directories: {{doc_dirs}}
- Total Code Files: {{code_file_count}}
- Total Documentation Files: {{doc_file_count}}

## 2. Documentation Coverage Analysis
| Module        | Code Files without Doc Dependencies | Action Taken |
|---------------|-----------------------------------|--------------|
{{coverage_analysis_rows}}

## 3. Added Dependencies
| Source Key | Target Key | Dependency Type | Justification |
|------------|------------|-----------------|---------------|
{ADDED_DEPS_TABLE_START_MARKER}
| [ITEM_1_KEY] | [ITEM_2_KEY]  | [DEP_TYPE]      | [JUSTIFICATION] |
{ADDED_DEPS_TABLE_END_MARKER}

## 4. Documentation Coverage Metrics
- **Initial Coverage**: {{initial_coverage_percentage}}% of code files had doc dependencies
- **Final Coverage**: [FINAL_PERCENTAGE]% of code files have doc dependencies
- **Coverage Delta**: [COVERAGE_DELTA]% improvement

## 5. Review Completion
- [ ] All code files reviewed against documentation corpus
- [ ] All identified gaps addressed
- [ ] Coverage metrics calculated and recorded
- [ ] All trackers verified (no remaining placeholders)

## 6. Sign-off
- **Completion Date**: [COMPLETION_DATE]
- **Notes**: [COMPLETION_NOTES]

## 7. Future Recommendations
- [RECOMMENDATIONS_FOR_NEXT_CYCLE]
"""


# --- Helper to determine if a path is code or doc ---
def get_item_type(
    item_path: str, config: ConfigManager, project_root: str
) -> Optional[str]:
    """Determines if a given path is a 'code' or 'doc' item based on configuration."""
    norm_item_path = normalize_path(item_path)
    code_root_dirs_abs = [
        normalize_path(os.path.join(project_root, cr))
        for cr in config.get_code_root_directories()
    ]
    doc_dirs_abs = [
        normalize_path(os.path.join(project_root, dd))
        for dd in config.get_doc_directories()
    ]
    for code_root in code_root_dirs_abs:
        if is_subpath(norm_item_path, code_root) or norm_item_path == code_root:
            return "code"
    for doc_root in doc_dirs_abs:
        if is_subpath(norm_item_path, doc_root) or norm_item_path == doc_root:
            return "doc"
    return None


# --- Functions for generating the main checklist ---


def _get_project_name(project_root: str) -> str:
    """Extracts project name from system_manifest.md."""
    manifest_path = normalize_path(os.path.join(project_root, SYSTEM_MANIFEST_RELATIVE))
    project_name = "Unknown Project"
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            match = re.match(r"#\s*System\s*:\s*(.+)", first_line, re.IGNORECASE)
            if match:
                project_name = match.group(1).strip()
            else:
                logger.warning(
                    f"Could not parse project name from first line of {manifest_path}: '{first_line}'"
                )
        except Exception as e:
            logger.warning(f"Error reading project name from {manifest_path}: {e}")
    else:
        logger.warning(
            f"System manifest file not found at {manifest_path}. Project name will be '{project_name}'."
        )
    return project_name


def _archive_and_get_cycle_number(
    project_root: str,
    checklist_filename_in_root: str,
    archive_dir_abs: str,
    config: ConfigManager,
    global_key_map: Dict[str, KeyInfo],  # Current global map
    path_migration_info: PathMigrationInfo,  # Path migration map
) -> int:
    """
    Archives existing checklist, calculating and updating final coverage metrics before archiving.
    Determines the new cycle number.
    Uses path_migration_info for calling aggregate_all_dependencies.
    """
    current_checklist_path = normalize_path(
        os.path.join(project_root, checklist_filename_in_root)
    )
    last_cycle_number = 0

    if os.path.exists(current_checklist_path):
        try:
            with open(current_checklist_path, "r", encoding="utf-8") as f:
                content = f.read()

            match_cycle = re.search(r"-\s*\*\*Cycle\*\*:\s*(\d+)", content)
            if match_cycle:
                last_cycle_number = int(match_cycle.group(1))
            else:
                logger.warning(
                    f"Could not parse cycle number from existing {checklist_filename_in_root}. Assuming it was cycle 0 for archiving."
                )

            # --- Calculate Final Coverage and Delta ---
            initial_coverage_val = 0.0
            match_initial_cov = re.search(
                r"-\s*\*\*Initial Coverage\*\*:\s*([\d\.]+)\s*%", content
            )
            if match_initial_cov:
                try:
                    initial_coverage_val = float(match_initial_cov.group(1))
                except ValueError:
                    logger.warning(
                        "Could not parse Initial Coverage value from checklist."
                    )
            # Recalculate current coverage (this is the "Final Coverage" for the cycle being archived)
            code_files, doc_files = _get_code_and_doc_files(
                global_key_map, config, project_root
            )
            # Pass path_migration_info to _calculate_initial_coverage_and_gaps
            final_coverage_val, _ = _calculate_initial_coverage_and_gaps(
                global_key_map,
                code_files,
                doc_files,
                config,
                project_root,
                path_migration_info,
            )
            coverage_delta_val = final_coverage_val - initial_coverage_val

            logger.info(
                f"Checklist to be archived: Initial Coverage: {initial_coverage_val:.2f}%, Final Coverage: {final_coverage_val:.2f}%, Delta: {coverage_delta_val:.2f}%"
            )

            def final_coverage_repl(match_obj):
                # match_obj.group(1) is "(- \*\*Final Coverage\*\*: )"
                # match_obj.group(2) is " % of code files have doc dependencies)"
                return (
                    f"{match_obj.group(1)}{final_coverage_val:.2f}{match_obj.group(2)}"
                )

            content = re.sub(
                r"(-\s*\*\*Final Coverage\*\*:\s*)\[FINAL_PERCENTAGE\](\s*% of code files have doc dependencies)",
                final_coverage_repl,
                content,
            )

            def delta_coverage_repl(match_obj):
                # match_obj.group(1) is "(- \*\*Coverage Delta\*\*: )"
                # match_obj.group(2) is " % improvement)"
                return (
                    f"{match_obj.group(1)}{coverage_delta_val:.2f}{match_obj.group(2)}"
                )

            content = re.sub(
                r"(-\s*\*\*Coverage Delta\*\*:\s*)\[COVERAGE_DELTA\](\s*% improvement)",
                delta_coverage_repl,
                content,
            )
            # Write updated content back to the file before moving
            with open(current_checklist_path, "w", encoding="utf-8") as f_update:
                f_update.write(content)
            logger.debug(
                f"Updated final coverage metrics in {checklist_filename_in_root} before archiving."
            )

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checklist_filename_base = os.path.splitext(checklist_filename_in_root)[0]
            archive_filename = f"{checklist_filename_base}_{timestamp}_cycle{last_cycle_number if last_cycle_number > 0 else 'unknown'}.md"
            archive_filepath = normalize_path(
                os.path.join(archive_dir_abs, archive_filename)
            )
            os.makedirs(archive_dir_abs, exist_ok=True)
            shutil.move(current_checklist_path, archive_filepath)
            logger.info(f"Archived updated checklist to: {archive_filepath}")
            return last_cycle_number + 1
        except Exception as e:
            logger.error(
                f"Error archiving/updating checklist {current_checklist_path}: {e}. Starting new cycle 1.",
                exc_info=True,
            )
            if os.path.exists(current_checklist_path):
                try:
                    os.remove(current_checklist_path)
                except OSError as os_err:
                    logger.warning(
                        f"Could not remove conflicting checklist after archive error: {os_err}"
                    )
            return 1
    else:
        return 1


def _get_code_and_doc_files(
    global_key_map: Dict[str, KeyInfo], config: ConfigManager, project_root: str
) -> Tuple[List[KeyInfo], List[KeyInfo]]:
    """Filters global key map for code and documentation files, respecting exclusions."""
    code_files: List[KeyInfo] = []
    doc_files: List[KeyInfo] = []
    resolved_excluded_paths_abs = config.get_excluded_paths()
    excluded_extensions_config = set(config.get_excluded_extensions())
    excluded_dirs_config_rel = config.get_excluded_dirs()
    excluded_dirs_abs_set = {
        normalize_path(os.path.join(project_root, p)) for p in excluded_dirs_config_rel
    }
    # Combine all absolute exclusion paths for efficient checking
    all_specific_excluded_paths_abs = set(resolved_excluded_paths_abs)

    for key_info in global_key_map.values():
        if key_info.is_directory:
            continue
        norm_item_path = key_info.norm_path
        item_basename = os.path.basename(norm_item_path)
        _, item_ext = os.path.splitext(item_basename)
        item_ext = item_ext.lower()
        if item_ext in excluded_extensions_config:
            # logger.debug(f"Excluding by extension: {norm_item_path}")
            continue
        if norm_item_path in all_specific_excluded_paths_abs:
            continue
        is_in_excluded_dir_name = False
        for (
            excluded_dir_abs_path
        ) in (
            excluded_dirs_abs_set
        ):  # Check against dir names/paths from "excluded_dirs"
            if (
                is_subpath(norm_item_path, excluded_dir_abs_path)
                or norm_item_path == excluded_dir_abs_path
            ):
                is_in_excluded_dir_name = True
                break
        if is_in_excluded_dir_name:
            continue
        # Use get_item_type to classify
        item_type = get_item_type(norm_item_path, config, project_root)
        if item_type == "code":
            code_files.append(key_info)
        elif item_type == "doc":
            doc_files.append(key_info)
    logger.debug(
        f"Identified {len(code_files)} code files and {len(doc_files)} doc files for checklist."
    )
    return code_files, doc_files


def _calculate_initial_coverage_and_gaps(
    global_key_map: Dict[str, KeyInfo],  # Current global map
    code_files: List[KeyInfo],
    doc_files: List[KeyInfo],
    config: ConfigManager,
    project_root: str,
    path_migration_info: PathMigrationInfo,  # Pass the migration map
) -> Tuple[float, Dict[str, List[str]]]:
    """
    Calculates initial documentation coverage percentage and identifies code files lacking doc dependencies.
    Uses path_migration_info for calling aggregate_all_dependencies.
    """
    code_files_with_doc_deps_count = 0
    coverage_gaps = defaultdict(list)  # module_display_name -> list of code_file_keys

    path_to_keyinfo_map = {ki.norm_path: ki for ki in global_key_map.values()}
    doc_file_keys = {df.key_string for df in doc_files}  # Uses CURRENT keys
    if not code_files:
        logger.info("No code files found to calculate coverage.")
        return 100.0 if not doc_files else 0.0, {}  # Or handle as appropriate

    if not doc_files:
        logger.warning(
            "No documentation files found. Initial coverage is 0%. All code files listed as gaps."
        )
        for cf_info in code_files:
            module_display_name = "Unknown Module"
            if cf_info.parent_path:
                parent_dir_info = path_to_keyinfo_map.get(cf_info.parent_path)
                if parent_dir_info and parent_dir_info.is_directory:
                    module_display_name = f"{parent_dir_info.key_string} ({os.path.basename(parent_dir_info.norm_path)})"
                else:  # Fallback if parent_path isn't in map or isn't a dir
                    module_display_name = os.path.basename(cf_info.parent_path)
            else:  # File directly under a root that isn't itself a module directory
                module_display_name = "Top-Level Files"  # Or derive from root path
            coverage_gaps[module_display_name].append(cf_info.key_string)
        return 0.0, dict(sorted(coverage_gaps.items()))

    all_tracker_paths = find_all_tracker_paths(config, project_root)
    if not all_tracker_paths:
        logger.warning(
            "No tracker files found. Cannot determine dependencies for coverage calculation."
        )
        # Treat as 0% coverage, all code files are gaps
        for cf_info in code_files:
            module_display_name = "Unknown Module"
            if cf_info.parent_path:
                parent_dir_info = path_to_keyinfo_map.get(cf_info.parent_path)
                if parent_dir_info and parent_dir_info.is_directory:
                    module_display_name = f"{parent_dir_info.key_string} ({os.path.basename(parent_dir_info.norm_path)})"
                else:
                    module_display_name = os.path.basename(cf_info.parent_path)
            else:
                module_display_name = "Top-Level Files"
            coverage_gaps[module_display_name].append(cf_info.key_string)
        return 0.0, dict(sorted(coverage_gaps.items()))

    # Call aggregate_all_dependencies with path_migration_info
    # The keys in aggregated_links will be CURRENT keys.
    try:
        aggregated_links = aggregate_all_dependencies(
            all_tracker_paths, path_migration_info, global_key_map
        )
    except ValueError as ve:  # Catch potential errors from aggregation
        logger.error(
            f"Coverage calculation failed: Error during dependency aggregation: {ve}"
        )
        return 0.0, {"Error": [f"Aggregation failed: {ve}"]}  # Return error indication
    except Exception as e:
        logger.error(
            f"Coverage calculation failed: Unexpected error during dependency aggregation: {e}",
            exc_info=True,
        )
        return 0.0, {"Error": [f"Unexpected aggregation error: {e}"]}

    # We need a way to get KeyInfo from KEY#GI. tracker_utils.resolve_key_global_instance_to_ki does this.
    from cline_utils.dependency_system.utils.tracker_utils import (
        resolve_key_global_instance_to_ki,
    )

    doc_file_global_instance_strs = {
        gis
        for df_ki in doc_files
        if (gis := get_key_global_instance_string(df_ki, global_key_map))
        is not None  # from tracker_utils
    }

    for code_file_info in code_files:
        code_file_gi_str = get_key_global_instance_string(
            code_file_info, global_key_map
        )  # from tracker_utils
        if not code_file_gi_str:
            logger.warning(
                f"Coverage: Could not get GI string for code file {code_file_info.norm_path}. Skipping."
            )
            # Add to gaps directly if GI string cannot be formed
            # ... (add to coverage_gaps logic) ...
            continue

        has_doc_dep = False
        for (source_key_gi, target_key_gi), (dep_char, _) in aggregated_links.items():
            if source_key_gi == code_file_gi_str:  # Code file is the source
                # Check if target_key_gi corresponds to a known doc file's global instance string
                if (
                    target_key_gi in doc_file_global_instance_strs
                    and dep_char in POSITIVE_DOC_DEPENDENCY_CHARS
                ):
                    has_doc_dep = True
                    break
        if has_doc_dep:
            code_files_with_doc_deps_count += 1
        else:
            module_display_name = "Unknown Module"
            if code_file_info.parent_path:
                parent_dir_info = path_to_keyinfo_map.get(code_file_info.parent_path)
                if parent_dir_info and parent_dir_info.is_directory:
                    module_display_name = f"{parent_dir_info.key_string} ({os.path.basename(parent_dir_info.norm_path)})"
                else:
                    module_display_name = os.path.basename(
                        code_file_info.parent_path
                    )  # Fallback to parent dir name
            else:  # File directly under a root
                module_display_name = (
                    "Top-Level Files"  # Or could try to find root name
                )
            coverage_gaps[module_display_name].append(code_file_info.key_string)

    total_code_files = len(code_files)
    percentage = (
        (code_files_with_doc_deps_count / total_code_files) * 100
        if total_code_files > 0
        else 0.0
    )
    return percentage, dict(sorted(coverage_gaps.items()))


def generate_final_review_checklist(
    global_key_map_param: Optional[
        Dict[str, KeyInfo]
    ] = None,  # For direct call flexibility
    path_migration_info_param: Optional[
        PathMigrationInfo
    ] = None,  # For direct call flexibility
) -> bool:
    """
    Generates the Code-Documentation Cross-Reference Checklist.
    Archives any existing checklist and populates the new one.
    Accepts global_key_map and path_migration_info as parameters to allow
    being called by analyze_project with pre-computed maps.

    Returns True on success, False on failure.
    """
    logger.info("Starting generation of final review checklist...")
    try:
        project_root = get_project_root()
        config = ConfigManager()
    except Exception as e:
        logger.error(
            f"Error initializing project context (project_root or ConfigManager): {e}",
            exc_info=True,
        )
        return False

    # Use provided maps or load/build them
    current_global_key_map: Dict[str, KeyInfo]
    if global_key_map_param is not None:
        current_global_key_map = global_key_map_param
    else:
        loaded_map = load_global_key_map()
        if not loaded_map:
            logger.error(
                "Global key map not found or failed to load. Cannot generate/update checklist."
            )
            return False
        current_global_key_map = loaded_map

    path_migration_info_to_use: PathMigrationInfo
    if path_migration_info_param is not None:
        path_migration_info_to_use = path_migration_info_param
    else:
        from cline_utils.dependency_system.core.key_manager import (
            load_old_global_key_map,
        )
        from cline_utils.dependency_system.io.tracker_io import (
            build_path_migration_map,
        )  # Local import

        old_global_map = load_old_global_key_map()
        try:
            path_migration_info_to_use = build_path_migration_map(
                old_global_map, current_global_key_map
            )
        except ValueError as ve:
            logger.error(
                f"Failed to build path migration map for checklist generation: {ve}"
            )
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error building path migration map for checklist: {e}",
                exc_info=True,
            )
            return False

    checklist_path_abs = normalize_path(os.path.join(project_root, CHECKLIST_FILENAME))
    archive_dir_abs = normalize_path(os.path.join(project_root, ARCHIVE_DIR_RELATIVE))

    # Pass current_global_key_map and path_migration_info_to_use
    cycle_number = _archive_and_get_cycle_number(
        project_root,
        CHECKLIST_FILENAME,
        archive_dir_abs,
        config,
        current_global_key_map,
        path_migration_info_to_use,
    )

    project_name = _get_project_name(project_root)
    date_created = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    code_roots_list = config.get_code_root_directories()
    doc_dirs_list = config.get_doc_directories()
    code_root_dirs_str = (
        ", ".join(sorted([normalize_path(cr) for cr in code_roots_list]))
        if code_roots_list
        else "N/A"
    )
    doc_dirs_str = (
        ", ".join(sorted([normalize_path(dd) for dd in doc_dirs_list]))
        if doc_dirs_list
        else "N/A"
    )
    code_files, doc_files = _get_code_and_doc_files(
        current_global_key_map, config, project_root
    )
    code_file_count = len(code_files)
    doc_file_count = len(doc_files)

    # Pass path_migration_info_to_use for coverage calculation
    initial_coverage_percentage_for_new_checklist, coverage_gaps_dict = (
        _calculate_initial_coverage_and_gaps(
            current_global_key_map,
            code_files,
            doc_files,
            config,
            project_root,
            path_migration_info_to_use,
        )
    )
    coverage_analysis_rows_list = []
    if not coverage_gaps_dict:
        coverage_analysis_rows_list.append(
            "| N/A | All identified code files appear to have positive document dependencies. | Review to confirm. |"
        )
    else:
        for module_name, files_without_deps_keys in coverage_gaps_dict.items():
            # Sort keys hierarchically for consistent display
            sorted_files_keys_str = ", ".join(sorted(files_without_deps_keys))
            coverage_analysis_rows_list.append(
                f"| {module_name} | {sorted_files_keys_str} | [TODO: Describe action] |"
            )
    coverage_analysis_rows_str = (
        "\n".join(coverage_analysis_rows_list)
        if coverage_analysis_rows_list
        else "| All good! | No coverage gaps identified. | Review to confirm. |"
    )

    template_content_to_use = CHECKLIST_TEMPLATE_CONTENT
    try:
        formatted_checklist = template_content_to_use.format(
            project_name=project_name,
            cycle_number=cycle_number,
            date_created=date_created,
            code_root_dirs=code_root_dirs_str,
            doc_dirs=doc_dirs_str,
            code_file_count=code_file_count,
            doc_file_count=doc_file_count,
            coverage_analysis_rows=coverage_analysis_rows_str,
            initial_coverage_percentage=f"{initial_coverage_percentage_for_new_checklist:.2f}",
        )
    except KeyError as ke:
        logger.error(
            f"Missing key in checklist template formatting: {ke}", exc_info=True
        )
        return False
    try:
        with open(checklist_path_abs, "w", encoding="utf-8") as f:
            f.write(formatted_checklist)
        logger.info(
            f"Successfully generated new checklist for cycle {cycle_number}: {checklist_path_abs}"
        )

        # Track reranker performance for this cycle
        try:
            from cline_utils.dependency_system.analysis.reranker_history_tracker import (
                track_reranker_performance,
            )

            track_reranker_performance(cycle_number, project_root)
            logger.debug(f"Reranker performance tracked for cycle {cycle_number}")
        except Exception as e:
            logger.warning(f"Failed to track reranker performance: {e}")
        return True
    except IOError as e:
        logger.error(
            f"Error writing checklist file {checklist_path_abs}: {e}", exc_info=True
        )
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error writing checklist file {checklist_path_abs}: {e}",
            exc_info=True,
        )
        return False


# add dependency to checklist table
def add_code_doc_dependency_to_checklist(
    source_key_str: str, target_key_str: str, dep_type_char: str
) -> bool:
    """
    Adds a new row to the 'Added Dependencies' table in the final_review_checklist.md.
    This is intended for code-doc or doc-code relationships.
    """
    project_root = get_project_root()
    checklist_path_abs = normalize_path(os.path.join(project_root, CHECKLIST_FILENAME))

    if not os.path.exists(checklist_path_abs):
        logger.warning(
            f"Checklist file {checklist_path_abs} not found. Generating a new one before adding dependency entry."
        )
        if not generate_final_review_checklist():  # Try to generate it
            logger.error(
                "Failed to generate new checklist. Cannot add dependency entry."
            )
            return False
        # If generation was successful, the file now exists.

    # Construct the core part of the new row for duplicate checking (excluding justification)
    # Normalize spacing for consistent duplicate checks
    new_row_check_str = f"| {source_key_str} | {target_key_str} | {dep_type_char} |"
    new_row_to_insert = f"| {source_key_str.ljust(10)} | {target_key_str.ljust(10)} | {dep_type_char.center(15)} | [JUSTIFICATION] |"

    try:
        with open(checklist_path_abs, "r+", encoding="utf-8") as f:
            content = f.read()
            start_marker_idx = content.find(ADDED_DEPS_TABLE_START_MARKER)
            end_marker_idx = content.find(ADDED_DEPS_TABLE_END_MARKER)

            if (
                start_marker_idx == -1
                or end_marker_idx == -1
                or start_marker_idx >= end_marker_idx
            ):
                logger.error(
                    f"Could not find or invalid markers '{ADDED_DEPS_TABLE_START_MARKER}' / '{ADDED_DEPS_TABLE_END_MARKER}' in checklist. Cannot add entry."
                )
                return False

            before_table_data = content[
                : start_marker_idx + len(ADDED_DEPS_TABLE_START_MARKER) + 1
            ]
            after_table_data = content[end_marker_idx:]
            table_data_str = content[
                start_marker_idx
                + len(ADDED_DEPS_TABLE_START_MARKER)
                + 1 : end_marker_idx
            ].strip()
            existing_rows = [
                row.strip()
                for row in table_data_str.split("\n")
                if row.strip().startswith("|")
            ]

            # Duplicate check:
            for row in existing_rows:
                # Normalize existing row for comparison (take first 3 columns essentially)
                cols = [c.strip() for c in row.strip("|").split("|")]
                if len(cols) >= 3:
                    existing_row_check_str = f"| {cols[0]} | {cols[1]} | {cols[2]} |"
                    if new_row_check_str == existing_row_check_str:
                        logger.info(
                            f"Duplicate dependency entry found in checklist, not adding: {new_row_to_insert}"
                        )
                        return True

            placeholder_row_template_part = "| [ITEM_1_KEY]"
            if (
                len(existing_rows) == 1
                and placeholder_row_template_part in existing_rows[0]
            ):
                updated_table_rows_str = new_row_to_insert
                logger.debug(
                    f"Replacing placeholder row in checklist with: {new_row_to_insert}"
                )
            elif not existing_rows:
                updated_table_rows_str = new_row_to_insert
                logger.debug(
                    f"Inserting first data row into empty table: {new_row_to_insert}"
                )
            else:
                updated_table_rows_str = table_data_str + "\n" + new_row_to_insert
                logger.debug(
                    f"Appending new dependency row to checklist: {new_row_to_insert}"
                )

            final_content = (
                before_table_data
                + updated_table_rows_str.strip()
                + "\n"
                + after_table_data
            )
            f.seek(0)
            f.write(final_content)
            f.truncate()

        logger.info(
            f"Successfully added dependency ({source_key_str} -> {target_key_str}): {new_row_to_insert}"
        )
        return True
    except IOError as e:
        logger.error(
            f"IOError updating checklist {checklist_path_abs}: {e}", exc_info=True
        )
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error updating checklist {checklist_path_abs}: {e}",
            exc_info=True,
        )
        return False
