"""
Tracker Batch Collector Module

Provides functionality to collect tracker updates in memory and write them
to disk in a single batch operation, reducing disk I/O overhead.
"""

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from cline_utils.dependency_system.core.dependency_grid import (
    DIAGONAL_CHAR,
    EMPTY_CHAR,
    PLACEHOLDER_CHAR,
    compress,
    decompress,
)
from cline_utils.dependency_system.core.key_manager import (
    KeyInfo,
    get_sortable_parts_for_key,
)
from cline_utils.dependency_system.io.tracker_io import is_path_in_doc_roots
from cline_utils.dependency_system.utils.cache_manager import (
    get_project_root_cached as get_project_root,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import is_subpath, normalize_path

logger = logging.getLogger(__name__)

DIRECTIONAL_RECIPROCAL_CHARS = {">": "<", "<": ">", "x": "x"}


def get_reciprocal_dependency_char(dep_char: str) -> Optional[str]:
    return DIRECTIONAL_RECIPROCAL_CHARS.get(dep_char)


def build_dependency_suggestions_with_reciprocals(
    suggestions: Dict[str, List[Tuple[str, str]]],
) -> Dict[str, List[Tuple[str, str]]]:
    suggestions_with_reciprocals = {
        source_key: list(deps) for source_key, deps in suggestions.items()
    }

    for source_key, deps in suggestions.items():
        for target_key, dep_char in deps:
            reciprocal_char = get_reciprocal_dependency_char(dep_char)
            if reciprocal_char is None:
                continue
            suggestions_with_reciprocals.setdefault(target_key, []).append(
                (source_key, reciprocal_char)
            )

    return suggestions_with_reciprocals


def _should_apply_reciprocal_dependency(
    current_reverse: str,
    reciprocal_char: str,
    get_priority: Callable[[str], int],
    force_apply_suggestions: bool,
) -> bool:
    if force_apply_suggestions:
        return current_reverse != "x" and current_reverse != reciprocal_char

    return current_reverse == PLACEHOLDER_CHAR or (
        current_reverse != "n"
        and get_priority(reciprocal_char) > get_priority(current_reverse)
    )


def apply_reciprocal_dependencies_to_grid(
    grid_rows: List[List[str]],
    force_apply_suggestions: bool,
    get_priority: Callable[[str], int],
) -> int:
    changes = 0

    for row_idx, row in enumerate(grid_rows):
        for col_idx in range(len(row)):
            if row_idx == col_idx:
                continue
            if col_idx >= len(grid_rows):
                continue
            if row_idx >= len(grid_rows[col_idx]):
                continue

            forward_char = row[col_idx]
            reciprocal_char = get_reciprocal_dependency_char(forward_char)
            if reciprocal_char is None:
                continue

            current_reverse = grid_rows[col_idx][row_idx]
            if (
                current_reverse != reciprocal_char
                and _should_apply_reciprocal_dependency(
                    current_reverse,
                    reciprocal_char,
                    get_priority,
                    force_apply_suggestions,
                )
            ):
                grid_rows[col_idx][row_idx] = reciprocal_char
                changes += 1

    return changes


@dataclass
class TrackerUpdate:
    """
    Holds all data needed for a single tracker update.

    This dataclass stores the processed tracker data in memory
    until it's ready to be written to disk as part of a batch.
    """

    tracker_type: str  # "mini", "doc", or "main"
    output_file: str  # Target file path
    key_info_list: List[KeyInfo]  # List[KeyInfo] - key definitions
    grid_rows: List[str]  # Compressed grid data
    last_key_edit: str
    last_grid_edit: str
    module_path: Optional[str] = None  # For mini trackers
    manual_foreign_pins: Optional[List[str]] = None  # For mini trackers
    template_data: Optional[Dict[str, Any]] = None  # For mini trackers
    existing_lines: Optional[List[str]] = None  # Preserved content
    tracker_exists: bool = False
    # Additional metadata for processing
    path_to_key_info: Optional[Dict[str, Any]] = None
    # Change counters for reporting
    ast_overrides_applied_count: int = 0
    suggestion_applied_count: int = 0
    structural_deps_applied_count: int = 0
    force_apply_suggestions: bool = False

    def __post_init__(self):
        """Validate the update data."""
        if self.tracker_type not in ("mini", "doc", "main"):
            raise ValueError(f"Invalid tracker_type: {self.tracker_type}")
        if not self.output_file:
            raise ValueError("output_file cannot be empty")
        if len(self.key_info_list) != len(self.grid_rows) and self.key_info_list:
            raise ValueError(
                f"Grid size mismatch: {len(self.key_info_list)} keys vs "
                f"{len(self.grid_rows)} grid rows"
            )


class TrackerBatchCollector:
    """
    Collects tracker updates in memory and performs batch writes.

    This class reduces disk I/O by collecting all tracker updates during
    project analysis and writing them all at once at the end.

    Usage:
        collector = TrackerBatchCollector()

        # Collect updates during processing
        for module in modules:
            update = prepare_tracker_update(...)
            collector.add(update)

        # Commit all at once
        results = collector.commit_all()

        # Check results
        for file_path, success in results.items():
            if success:
                print(f"Successfully wrote {file_path}")
    """

    def __init__(self):
        self.pending_updates: List[TrackerUpdate] = []
        self._temp_backup_dir: Optional[str] = None
        self._backed_up_files: Set[str] = set()
        self._backup_mappings: Dict[str, str] = {}
        self.highest_dependency_cache: Dict[Tuple[str, str], Tuple[str, Set[str]]] = {}
        self.key_to_paths: Dict[str, Set[str]] = {}
        super().__init__()

    def add(self, update: TrackerUpdate) -> None:
        """
        Add a tracker update to the batch.

        If an update for the same output_file already exists, the newer update
        replaces the older one. This prevents double-writes to the same file
        (e.g., when two paths with invisible-character differences normalize
        to the same target), avoiding corruption from overlapping writes.

        Args:
            update: A TrackerUpdate object containing all data needed for the update

        Raises:
            ValueError: If the update is invalid
        """
        if not update:
            raise ValueError("Cannot add None update to batch")

        # Deduplicate by output_file: if an update for the same file already
        # exists, replace it with the newer one (last-write-wins, which is
        # correct for the batch collector's final state semantics).
        for i, existing in enumerate(self.pending_updates):
            if existing.output_file == update.output_file:
                logger.debug(
                    f"Replacing existing update for {os.path.basename(update.output_file)} "
                    f"(type: {existing.tracker_type}) with newer update (type: {update.tracker_type})"
                )
                self.pending_updates[i] = update
                break
        else:
            self.pending_updates.append(update)
        try:
            config = ConfigManager()
            get_priority = config.get_char_priority

            # Track all keys and their paths for scoping
            if update.key_info_list:
                for ki in update.key_info_list:
                    if ki.key_string not in self.key_to_paths:
                        self.key_to_paths[ki.key_string] = set()
                    self.key_to_paths[ki.key_string].add(ki.norm_path)

            if update.key_info_list and update.grid_rows:
                for r_idx, r_ki in enumerate(update.key_info_list):
                    try:
                        decomp = list(decompress(update.grid_rows[r_idx]))
                    except:
                        continue
                    for c_idx, c_ki in enumerate(update.key_info_list):
                        if r_idx == c_idx or c_idx >= len(decomp):
                            continue
                        c_val = decomp[c_idx]
                        if c_val not in (PLACEHOLDER_CHAR, EMPTY_CHAR):
                            kp = (r_ki.key_string, c_ki.key_string)
                            ex = self.highest_dependency_cache.get(
                                kp, (PLACEHOLDER_CHAR, set())
                            )

                            # Use priority logic to decide if we should update the cache
                            new_prio = get_priority(c_val)
                            old_prio = get_priority(ex[0])

                            should_update = new_prio > old_prio
                            if (
                                not should_update
                                and c_val == "n"
                                and ex[0] in (PLACEHOLDER_CHAR, "s", "S", EMPTY_CHAR)
                            ):
                                should_update = True
                            if (
                                not should_update
                                and c_val in ("s", "S")
                                and ex[0] in (PLACEHOLDER_CHAR, EMPTY_CHAR)
                            ):
                                should_update = True

                            if should_update:
                                self.highest_dependency_cache[kp] = (
                                    c_val,
                                    {update.output_file},
                                )
                            elif new_prio == old_prio and c_val == ex[0]:
                                ex[1].add(update.output_file)
        except Exception:
            pass
        logger.debug(
            f"Added {update.tracker_type} tracker update for {os.path.basename(update.output_file)}"
        )

    def __len__(self) -> int:
        """Return the number of pending updates."""
        return len(self.pending_updates)

    def is_empty(self) -> bool:
        """Check if there are any pending updates."""
        return len(self.pending_updates) == 0

    def validate_all(self) -> Tuple[bool, List[str]]:
        """
        Validate all pending updates before writing.

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors: List[str] = []
        output_files: Set[str] = set()

        for i, update in enumerate(self.pending_updates):
            # Check for duplicate output files
            if update.output_file in output_files:
                errors.append(f"Update {i}: Duplicate output file {update.output_file}")
            output_files.add(update.output_file)

            # Check output directory exists or can be created
            output_dir = os.path.dirname(update.output_file)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except OSError as e:
                    errors.append(
                        f"Update {i}: Cannot create directory {output_dir}: {e}"
                    )

        return len(errors) == 0, errors

    def commit_all(
        self,
        skip_populate_hook: bool = False,
        accumulated_updates: Optional[List[TrackerUpdate]] = None,
    ) -> Dict[str, bool]:
        """
        Write all pending updates to disk in a batch.

        This method:
        1. Creates backups of existing tracker files
        2. Writes all tracker updates to disk
        3. Invalidates caches once after all writes
        4. Cleans up backups on success

        Returns:
            Dictionary mapping output_file -> success_status (True/False)

        Raises:
            RuntimeError: If the batch write fails and rollback is unsuccessful
        """
        if not self.pending_updates:
            logger.debug("No pending updates to commit")
            return {}

        # Validate before writing
        is_valid, errors = self.validate_all()
        if not is_valid:
            error_msg = "Validation failed:\n" + "\n".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        results: Dict[str, bool] = {}

        try:
            self._consolidate_grids()
            # Create backups for rollback capability
            self._create_backups()

            # Create permanent backups to cline_docs/backups (like original update_tracker)
            self._create_permanent_backups()

            # Write all trackers
            for update in self.pending_updates:
                try:
                    success = self._write_single_tracker(update)
                    results[update.output_file] = success

                    if not success:
                        logger.warning(
                            f"Failed to write {update.output_file}, "
                            "will attempt rollback"
                        )

                except Exception as e:
                    logger.error(
                        f"Error writing {update.output_file}: {e}", exc_info=True
                    )
                    results[update.output_file] = False
                    raise  # Re-raise to trigger rollback

            # Populate runtime aggregation cache before invalidating
            from cline_utils.dependency_system.utils.tracker_utils import (
                set_runtime_aggregation_cache,
            )

            set_runtime_aggregation_cache(self.highest_dependency_cache)

            # Trigger populate_comments hook
            if not skip_populate_hook:
                self._run_populate_comments_hook()
            elif accumulated_updates is not None:
                accumulated_updates.extend(self.pending_updates)

            # Invalidate caches once after all successful writes
            self._invalidate_caches()

            # Clean up backups on success
            self._cleanup_backups()

            logger.info(
                f"Successfully committed {len(self.pending_updates)} tracker updates"
            )

        except Exception as e:
            logger.error(f"Batch write failed: {e}. Initiating rollback...")
            self.rollback()
            raise RuntimeError(f"Batch write failed and was rolled back: {e}") from e

        finally:
            # Clear pending updates
            self.pending_updates = []

        return results

    def _run_populate_comments_hook(self) -> None:
        """
        Trigger comment population on files affected by the batch.
        Runs after trackers are written and aggregation cache is hot,
        but before cache invalidation.
        """
        try:
            from pathlib import Path
            from cline_utils.dependency_system.utils.populate_comments import (
                populate_comments_for_batch,
                report_batch_results,
            )
            from cline_utils.dependency_system.io.transparency_manager import (
                get_transparency_manager,
            )
            from cline_utils.dependency_system.utils.cache_manager import (
                get_project_root_cached,
            )
            from cline_utils.dependency_system.analysis.dependency_suggester import (
                load_project_symbol_map,
            )

            project_root = Path(get_project_root_cached())
            symbol_map = load_project_symbol_map()

            results = populate_comments_for_batch(
                project_root=project_root,
                updates=self.pending_updates,
                symbol_map=symbol_map,
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
                    if result is not None:
                        path = result.get("path")
                        if isinstance(path, str):
                            files_processed.add(path)
                            paths_to_virtualize.append(path)
                            maps_added = result.get("maps_added")
                            maps_updated = result.get("maps_updated")
                            maps_count = (
                                maps_added if isinstance(maps_added, int) else 0
                            ) + (maps_updated if isinstance(maps_updated, int) else 0)
                            if maps_count > 0:
                                files_with_maps.add(path)
                virtualized = 0
                if paths_to_virtualize:
                    virtualized = manager.bulk_virtualize_connection_maps(
                        paths_to_virtualize, clear_if_absent=True
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

        except ImportError as e:
            logger.debug(f"populate_comments hook skipped: {e}")
        except Exception as e:
            logger.error(f"Error in populate_comments hook: {e}")

    def rollback(self) -> None:
        """
        Restore from backups if batch write fails.

        This method restores all tracker files from their backups
        to ensure data consistency after a failed batch write.
        """
        if not self._temp_backup_dir:
            logger.debug("No backups to restore")
            return

        logger.warning("Rolling back tracker files from backups...")

        restored_count = 0
        failed_count = 0

        for update in self.pending_updates:
            original_path = update.output_file
            backup_path = self._backup_mappings.get(original_path)

            try:
                if backup_path and os.path.exists(backup_path):
                    shutil.copy2(backup_path, original_path)
                    restored_count += 1
                    logger.debug(f"Restored {original_path} from backup")
                else:
                    # If backup doesn't exist, original might be a new file
                    # Remove the partially written file
                    if os.path.exists(original_path):
                        os.remove(original_path)
                        logger.debug(f"Removed partially written file {original_path}")

            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to restore {original_path}: {e}")

        # Clean up backup directory
        self._cleanup_backups()

        logger.info(
            f"Rollback complete: {restored_count} restored, " f"{failed_count} failed"
        )

    def _consolidate_grids(self) -> None:
        if not getattr(self, "highest_dependency_cache", None):
            return
        logger.info(
            f"Starting batch consolidation for {len(self.pending_updates)} trackers..."
        )
        try:
            config = ConfigManager()
            get_priority = config.get_char_priority
            pr = get_project_root()
            d_roots = {
                normalize_path(os.path.join(pr, p))
                for p in config.get_doc_directories()
            }

            # --- PHASE 0: Global Context Prep ---
            ast_links = self._load_ast_links()
            self._import_external_relationships(config, pr, d_roots)

            def key_in_doc_root(key_str: str) -> bool:
                # Optimization: replace any() generator expressions with explicitly unrolled for loops with early return to avoid generator overhead
                paths = self.key_to_paths.get(key_str, set())
                for p in paths:
                    norm_p = normalize_path(p)
                    if norm_p in d_roots:
                        return True
                    for d in d_roots:
                        if d and norm_p.startswith(d + "/"):
                            return True
                return False

            total_changes = 0
            total_suggestions = 0
            total_structural = 0
            total_ast = 0

            for u in self.pending_updates:
                total_suggestions += getattr(u, "suggestion_applied_count", 0)
                total_structural += getattr(u, "structural_deps_applied_count", 0)
                total_ast += getattr(u, "ast_overrides_applied_count", 0)

                if not u.grid_rows or not u.key_info_list:
                    continue

                # Apply AST overrides centrally if available
                if ast_links:
                    u.ast_overrides_applied_count += (
                        self._apply_ast_overrides_to_update(u, ast_links, config)
                    )
                    total_ast += u.ast_overrides_applied_count

                new_rows: List[str] = []
                tracker_changes = 0
                for ri, rki in enumerate(u.key_info_list):
                    try:
                        dr = list(decompress(u.grid_rows[ri]))
                    except:
                        new_rows.append(u.grid_rows[ri])
                        continue
                    rc = False
                    for ci, cki in enumerate(u.key_info_list):
                        if ri == ci or ci >= len(dr):
                            continue
                        kp = (rki.key_string, cki.key_string)
                        ac_entry = self.highest_dependency_cache.get(
                            kp, (PLACEHOLDER_CHAR, set())
                        )
                        ac = ac_entry[0]

                        # Doc Tracker Scoping: only consolidate if both keys have a doc presence
                        if u.tracker_type == "doc":
                            if not (
                                key_in_doc_root(rki.key_string)
                                and key_in_doc_root(cki.key_string)
                            ):
                                ac = PLACEHOLDER_CHAR

                        cc = dr[ci]
                        if ac != PLACEHOLDER_CHAR and ac != cc:
                            try:
                                ap, cp = get_priority(ac), get_priority(cc)
                                su = ap > cp
                                if (
                                    not su
                                    and ac == "n"
                                    and cc in (PLACEHOLDER_CHAR, "s", "S", EMPTY_CHAR)
                                ):
                                    su = True
                                if (
                                    not su
                                    and ac in ("s", "S")
                                    and cc in (PLACEHOLDER_CHAR, EMPTY_CHAR)
                                ):
                                    su = True

                                if su:
                                    dr[ci] = ac
                                    rc = True
                                    tracker_changes += 1
                            except:
                                pass
                    new_rows.append(compress("".join(dr)) if rc else u.grid_rows[ri])
                u.grid_rows = new_rows
                total_changes += tracker_changes
                if tracker_changes > 0:
                    logger.debug(
                        f"Applied {tracker_changes} consolidation changes to {os.path.basename(u.output_file)}"
                    )

            # --- PHASE 1B: Batch Reciprocal Application ---
            # Instead of applying reciprocals per-suggestion inside update_tracker
            # (which would do up to 2246 individual set operations), we apply them
            # here in a single pass after all forward dependencies have been
            # consolidated from the master highest_dependency_cache.
            #
            # This is correct because the reciprocal is purely a function of the
            # forward-direction character already set in the grid.
            reciprocal_changes = 0
            for u in self.pending_updates:
                if not u.grid_rows or not u.key_info_list:
                    continue

                tracker_recip_changes = 0
                n = len(u.key_info_list)

                # Decompress all rows up-front into a mutable matrix so that
                # both forward reads and reverse writes operate on the same
                # in-memory representation.  This avoids the ordering hazard
                # where `u.grid_rows[ci]` is mutated *after* the outer loop
                # already appended the old compressed value to `new_rows`.
                decomp: List[List[str]] = []
                decomp_ok: List[bool] = []
                for ri in range(n):
                    try:
                        decomp.append(list(decompress(u.grid_rows[ri])))
                        decomp_ok.append(True)
                    except Exception:
                        decomp.append([])
                        decomp_ok.append(False)

                tracker_recip_changes = apply_reciprocal_dependencies_to_grid(
                    decomp, u.force_apply_suggestions, get_priority
                )

                # Re-compress all rows from the fully-mutated matrix
                if tracker_recip_changes > 0:
                    u.grid_rows = [
                        (
                            compress("".join(decomp[ri]))
                            if decomp_ok[ri]
                            else u.grid_rows[ri]
                        )
                        for ri in range(n)
                    ]
                    reciprocal_changes += tracker_recip_changes
                    logger.debug(
                        f"Applied {tracker_recip_changes} reciprocal changes to {os.path.basename(u.output_file)}"
                    )

            summary_parts: List[str] = []
            if total_changes > 0:
                summary_parts.append(f"{total_changes} consolidation changes")
            if reciprocal_changes > 0:
                summary_parts.append(f"{reciprocal_changes} reciprocal changes")
            if total_suggestions > 0:
                summary_parts.append(f"{total_suggestions} suggestions applied")
            if total_structural > 0:
                summary_parts.append(f"{total_structural} structural changes")
            if total_ast > 0:
                summary_parts.append(f"{total_ast} AST overrides")

            summary_msg = "Batch consolidation complete. " + (
                ", ".join(summary_parts) if summary_parts else "Total changes: 0"
            )
            logger.info(summary_msg)

            # --- PHASE 2: PRUNING ---
            logger.info("Starting batch pruning for trackers...")
            total_pruned_count = 0
            for u in self.pending_updates:
                if not u.grid_rows or not u.key_info_list:
                    continue

                if u.tracker_type == "mini":
                    # --- Mini Tracker Foreign Key Pruning ---
                    if not u.force_apply_suggestions and u.module_path:
                        logger.debug(
                            f"Pruning Mini Tracker: {os.path.basename(u.output_file)}"
                        )

                        original_ki_list = list(u.key_info_list)
                        # Identify internal vs foreign
                        internal_paths = {
                            ki.norm_path
                            for ki in original_ki_list
                            if ki.parent_path == u.module_path
                            or ki.norm_path == u.module_path
                        }

                        manual_pins = set(u.manual_foreign_pins or [])
                        paths_to_keep = internal_paths.union(manual_pins)

                        # Decompress all rows to check for links
                        decomp_rows: List[List[str]] = []
                        for row_data in u.grid_rows:
                            try:
                                decomp_rows.append(list(decompress(row_data)))
                            except:
                                decomp_rows.append([])

                        # Analyze links for foreign files
                        for ri, rki in enumerate(original_ki_list):
                            if ri >= len(decomp_rows):
                                continue
                            row_is_int = rki.norm_path in internal_paths

                            for ci, cki in enumerate(original_ki_list):
                                if ri == ci or ci >= len(decomp_rows[ri]):
                                    continue
                                col_is_int = cki.norm_path in internal_paths
                                val = decomp_rows[ri][ci]

                                if val not in (
                                    PLACEHOLDER_CHAR,
                                    DIAGONAL_CHAR,
                                    EMPTY_CHAR,
                                    "n",
                                ):
                                    try:
                                        prio = get_priority(val)
                                        # Threshold logic (matches tracker_io)
                                        if prio >= 1:
                                            if (
                                                not rki.is_directory
                                                and not cki.is_directory
                                            ):
                                                if row_is_int and not col_is_int:
                                                    paths_to_keep.add(cki.norm_path)
                                                elif not row_is_int and col_is_int:
                                                    paths_to_keep.add(rki.norm_path)
                                    except:
                                        pass

                        if len(paths_to_keep) < len(original_ki_list):
                            pruned_ki = [
                                ki
                                for ki in original_ki_list
                                if ki.norm_path in paths_to_keep
                            ]
                            # Sort matches tracker_io
                            pruned_ki.sort(
                                key=lambda x: (
                                    get_sortable_parts_for_key(x.key_string),
                                    x.norm_path,
                                )
                            )

                            # Rebuild grid
                            new_size = len(pruned_ki)
                            new_grid = [
                                [PLACEHOLDER_CHAR] * new_size for _ in range(new_size)
                            ]
                            for i in range(new_size):
                                new_grid[i][i] = DIAGONAL_CHAR

                            orig_path_to_idx = {
                                ki.norm_path: i for i, ki in enumerate(original_ki_list)
                            }

                            for ni, nki in enumerate(pruned_ki):
                                oi = orig_path_to_idx.get(nki.norm_path)
                                if oi is None or oi >= len(decomp_rows):
                                    continue

                                for nj, nkj in enumerate(pruned_ki):
                                    if ni == nj:
                                        continue
                                    oj = orig_path_to_idx.get(nkj.norm_path)
                                    if oj is not None and oj < len(decomp_rows[oi]):
                                        new_grid[ni][nj] = decomp_rows[oi][oj]

                            u.key_info_list = pruned_ki
                            u.grid_rows = [compress("".join(r)) for r in new_grid]
                            total_pruned_count += len(original_ki_list) - new_size
                            logger.debug(
                                f"  Pruned {len(original_ki_list) - new_size} keys from {os.path.basename(u.output_file)}"
                            )

                elif u.tracker_type == "doc":
                    # --- Doc Tracker Pruning ---
                    logger.debug(
                        f"Pruning Doc Tracker: {os.path.basename(u.output_file)}"
                    )
                    original_ki_list = list(u.key_info_list)

                    pruned_ki = [
                        ki
                        for ki in original_ki_list
                        if is_path_in_doc_roots(ki.norm_path, d_roots)
                    ]

                    if len(pruned_ki) < len(original_ki_list):
                        # Sort
                        pruned_ki.sort(
                            key=lambda x: (
                                get_sortable_parts_for_key(x.key_string),
                                x.norm_path,
                            )
                        )

                        # Rebuild grid (simplified since we don't care about links for docs)
                        new_size = len(pruned_ki)
                        new_grid = [
                            [PLACEHOLDER_CHAR] * new_size for _ in range(new_size)
                        ]
                        for i in range(new_size):
                            new_grid[i][i] = DIAGONAL_CHAR

                        decomp_rows: List[List[str]] = []
                        for row_data in u.grid_rows:
                            try:
                                decomp_rows.append(list(decompress(row_data)))
                            except:
                                decomp_rows.append([])

                        orig_path_to_idx = {
                            ki.norm_path: i for i, ki in enumerate(original_ki_list)
                        }

                        for ni, nki in enumerate(pruned_ki):
                            oi = orig_path_to_idx.get(nki.norm_path)
                            if oi is None or oi >= len(decomp_rows):
                                continue
                            for nj, nkj in enumerate(pruned_ki):
                                if ni == nj:
                                    continue
                                oj = orig_path_to_idx.get(nkj.norm_path)
                                if oj is not None and oj < len(decomp_rows[oi]):
                                    new_grid[ni][nj] = decomp_rows[oi][oj]

                        u.key_info_list = pruned_ki
                        u.grid_rows = [compress("".join(r)) for r in new_grid]
                        total_pruned_count += len(original_ki_list) - new_size
                        logger.debug(
                            f"  Pruned {len(original_ki_list) - new_size} keys from doc tracker"
                        )

            if total_pruned_count > 0:
                logger.info(
                    f"Batch pruning complete. Total keys pruned: {total_pruned_count}"
                )
            else:
                logger.info("Batch pruning complete. No keys pruned.")
        except Exception as e:
            logger.error(f"Batch consolidation/pruning error: {e}", exc_info=True)

    def _load_ast_links(self) -> List[Dict[str, str]]:
        """Loads AST-verified links from the central JSON file once per batch."""

        ast_links_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "analysis",
            "ast_verified_links.json",
        )
        if os.path.exists(ast_links_path):
            try:
                import json

                with open(ast_links_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("links", [])
            except Exception as e:
                logger.error(f"Error loading AST links: {e}")
        return []

    def _apply_ast_overrides_to_update(
        self, u: TrackerUpdate, ast_links: List[Dict[str, str]], config: ConfigManager
    ) -> int:
        """
        Applies AST-verified links to a single tracker update.
        Note: This is a simplified version of the logic in tracker_io.
        It focuses on character overrides within the existing grid.
        """
        if not u.grid_rows or not u.key_info_list:
            return 0

        applied_count = 0
        get_priority = config.get_char_priority

        # Decompress rows for modification
        decomp_rows: List[List[str]] = []
        for row in u.grid_rows:
            try:
                decomp_rows.append(list(decompress(row)))
            except:
                decomp_rows.append([])

        path_to_idx = {ki.norm_path: i for i, ki in enumerate(u.key_info_list)}

        for link in ast_links:
            src = normalize_path(link.get("source_path", ""))
            tgt = normalize_path(link.get("target_path", ""))
            char = link.get("char", "")

            if not src or not tgt or not char:
                continue

            ri = path_to_idx.get(src)
            ci = path_to_idx.get(tgt)

            if ri is not None and ci is not None:
                if ri < len(decomp_rows) and ci < len(decomp_rows[ri]):
                    current = decomp_rows[ri][ci]
                    if current != char:
                        # AST overrides handle conflict via priority or 'n' logic
                        if get_priority(char) >= get_priority(current) or char == "n":
                            decomp_rows[ri][ci] = char
                            applied_count += 1

        if applied_count > 0:
            u.grid_rows = [compress("".join(r)) for r in decomp_rows]

        return applied_count

    def _import_external_relationships(
        self, config: ConfigManager, project_root: str, d_roots: Set[str]
    ) -> None:
        """
        Imports established relationships from trackers NOT in the current batch.
        Populates highest_dependency_cache to be used in consolidation.
        """
        from cline_utils.dependency_system.io.tracker_io import (
            get_mini_tracker_path,
            get_tracker_path,
        )
        from cline_utils.dependency_system.utils.tracker_utils import (
            read_tracker_file_structured,
        )

        needed_home_trackers: Set[str] = set()
        for u in self.pending_updates:
            if u.tracker_type != "mini" or not u.module_path:
                continue
            for ki in u.key_info_list:
                # If foreign to THIS tracker
                if ki.parent_path != u.module_path and ki.norm_path != u.module_path:
                    home_path = None
                    if is_path_in_doc_roots(ki.norm_path, d_roots):
                        home_path = get_tracker_path(project_root, "doc")
                    elif ki.parent_path:
                        home_path = get_mini_tracker_path(ki.parent_path)

                    if home_path and os.path.exists(home_path):
                        needed_home_trackers.add(home_path)

        if not needed_home_trackers:
            return

        logger.debug(
            f"Batch Collector: Importing relationships from {len(needed_home_trackers)} external home trackers..."
        )

        get_priority = config.get_char_priority
        batch_output_files = {u.output_file for u in self.pending_updates}

        for ht_path in needed_home_trackers:
            if ht_path in batch_output_files:
                continue  # Already have this data in cache from add()

            try:
                structured = read_tracker_file_structured(ht_path)
                defs = structured.get("definitions_ordered", [])
                rows = structured.get("grid_rows_ordered", [])
                if not defs or not rows:
                    continue

                # KeyInfo objects from definitions
                # (read_tracker_file_structured returns List[Tuple[key_str, path_str]])
                # We need to resolve these to key_strings
                for ri, (r_key, _) in enumerate(defs):
                    if ri >= len(rows):
                        break
                    try:
                        row_chars = list(decompress(rows[ri][1]))
                    except:
                        continue

                    for ci, (c_key, _) in enumerate(defs):
                        if ri == ci or ci >= len(row_chars):
                            continue
                        char = row_chars[ci]
                        if char not in (PLACEHOLDER_CHAR, EMPTY_CHAR, DIAGONAL_CHAR):
                            kp = (r_key, c_key)
                            prio = get_priority(char)
                            existing = self.highest_dependency_cache.get(kp)
                            if existing is None or prio > get_priority(existing[0]):
                                self.highest_dependency_cache[kp] = (char, {ht_path})
                            elif prio == get_priority(existing[0]):
                                existing[1].add(ht_path)
            except Exception as e:
                logger.warning(f"Error reading home tracker {ht_path}: {e}")

    def _create_backups(self) -> None:
        """Create temporary backups of existing tracker files."""
        # Create temp directory for backups
        self._temp_backup_dir = tempfile.mkdtemp(prefix="tracker_batch_backup_")

        for update in self.pending_updates:
            if os.path.exists(update.output_file):
                try:
                    backup_path = os.path.join(
                        self._temp_backup_dir, os.path.basename(update.output_file)
                    )
                    # Handle potential name collisions
                    counter = 1
                    original_backup_path = backup_path
                    while os.path.exists(backup_path):
                        name, ext = os.path.splitext(original_backup_path)
                        backup_path = f"{name}_{counter}{ext}"
                        counter += 1

                    shutil.copy2(update.output_file, backup_path)
                    self._backed_up_files.add(update.output_file)
                    self._backup_mappings[update.output_file] = backup_path
                    logger.debug(
                        f"Created backup for {update.output_file} at {backup_path}"
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to create backup for {update.output_file}: {e}"
                    )

    def _create_permanent_backups(self) -> None:
        """
        Create permanent backups of existing tracker files to cline_docs/backups.

        This replicates the backup behavior of the original update_tracker function,
        ensuring all tracker files are backed up before modification.
        """
        try:
            # Import here to avoid circular imports
            from cline_utils.dependency_system.io.tracker_io import backup_tracker_file

            for update in self.pending_updates:
                if os.path.exists(update.output_file):
                    try:
                        backup_path = backup_tracker_file(update.output_file)
                        if backup_path:
                            logger.debug(
                                f"Created permanent backup for {update.output_file} at {backup_path}"
                            )
                    except Exception as e:
                        # Log but don't fail - temp backups still exist for rollback
                        logger.warning(
                            f"Failed to create permanent backup for {update.output_file}: {e}"
                        )
        except Exception as e:
            # If import fails or other error, log but continue
            logger.warning(f"Could not create permanent backups: {e}")

    def _write_single_tracker(self, update: TrackerUpdate) -> bool:
        """
        Write a single tracker update to disk.

        This method imports the necessary write functions from tracker_io
        to avoid circular dependencies.
        """
        try:
            # Import here to avoid circular imports
            from cline_utils.dependency_system.io.tracker_io import (
                get_mini_tracker_data,
                write_mini_tracker_with_template_preservation,
                write_tracker_file,
            )

            if update.tracker_type == "mini":
                # Write mini tracker with template preservation
                template_data = update.template_data or get_mini_tracker_data()

                write_mini_tracker_with_template_preservation(
                    output_file=update.output_file,
                    lines_from_old_file=update.existing_lines or [],
                    key_info_list_to_write=update.key_info_list,
                    grid_compressed_rows_to_write=update.grid_rows,
                    last_key_edit_msg=update.last_key_edit,
                    last_grid_edit_msg=update.last_grid_edit,
                    template_string=template_data.get("template", ""),
                    template_markers=template_data.get(
                        "markers",
                        ("---mini_tracker_start---", "---mini_tracker_end---"),
                    ),
                    current_global_map=update.path_to_key_info or {},
                    module_path_for_template=update.module_path or "",
                    manual_foreign_pins=set(update.manual_foreign_pins or []),
                )
            else:
                # Write main or doc tracker
                from collections import defaultdict

                # Precompute global key counts
                global_key_counts: Dict[str, int] = defaultdict(int)
                if update.path_to_key_info:
                    for ki in update.path_to_key_info.values():
                        global_key_counts[ki.key_string] += 1

                success = write_tracker_file(
                    tracker_path=update.output_file,
                    key_info_to_write=update.key_info_list,
                    grid_rows_ordered=update.grid_rows,
                    last_key_edit=update.last_key_edit,
                    last_grid_edit=update.last_grid_edit,
                    current_global_map=update.path_to_key_info or {},
                )

                if not success:
                    return False

            logger.debug(f"Successfully wrote {update.output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to write {update.output_file}: {e}", exc_info=True)
            return False

    def _invalidate_caches(self) -> None:
        """Invalidate all relevant caches after batch write."""
        try:
            from cline_utils.dependency_system.utils.cache_manager import (
                invalidate_dependent_entries,
            )
            from cline_utils.dependency_system.utils.path_utils import normalize_path

            # Invalidate tracker data structured cache for all updated files
            for update in self.pending_updates:
                invalidate_dependent_entries(
                    "tracker_data_structured",
                    f"tracker_data_structured:{normalize_path(update.output_file)}:.*",
                )

            # Invalidate aggregation cache once for all
            invalidate_dependent_entries("aggregation_v2_gi", ".*")

            logger.debug("Invalidated relevant caches after batch write")

        except Exception as e:
            logger.warning(f"Failed to invalidate caches: {e}")

    def _cleanup_backups(self) -> None:
        """Remove temporary backup files and directory."""
        if self._temp_backup_dir and os.path.exists(self._temp_backup_dir):
            try:
                shutil.rmtree(self._temp_backup_dir)
                logger.debug(f"Cleaned up backup directory: {self._temp_backup_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up backup directory: {e}")
            finally:
                self._temp_backup_dir = None
                self._backed_up_files.clear()
                self._backup_mappings.clear()


# Convenience functions for creating TrackerUpdate objects


def create_mini_tracker_update(
    output_file: str,
    key_info_list: List[Any],
    grid_rows: List[str],
    last_key_edit: str,
    last_grid_edit: str,
    module_path: str,
    path_to_key_info: Dict[str, Any],
    existing_lines: Optional[List[str]] = None,
    tracker_exists: bool = False,
    manual_foreign_pins: Optional[List[str]] = None,
    ast_overrides_applied_count: int = 0,
    suggestion_applied_count: int = 0,
    structural_deps_applied_count: int = 0,
    force_apply_suggestions: bool = False,
) -> TrackerUpdate:
    """
    Create a TrackerUpdate for a mini tracker.

    Args:
        output_file: Path to the mini tracker file
        key_info_list: List of KeyInfo objects for this tracker
        grid_rows: Compressed grid data rows
        last_key_edit: Last key edit timestamp message
        last_grid_edit: Last grid edit timestamp message
        module_path: The module path this tracker belongs to
        path_to_key_info: Global path to KeyInfo mapping
        existing_lines: Existing file content lines (for template preservation)
        tracker_exists: Whether the tracker file already exists
        manual_foreign_pins: Persisted manual foreign pins for mini-pruning
        ast_overrides_applied_count: Count of AST overrides applied
        suggestion_applied_count: Count of suggestions applied
        structural_deps_applied_count: Count of structural dependencies applied
        force_apply_suggestions: Whether to skip foreign key pruning

    Returns:
        TrackerUpdate object ready for batch collection
    """
    from cline_utils.dependency_system.io.tracker_io import get_mini_tracker_data

    return TrackerUpdate(
        tracker_type="mini",
        output_file=output_file,
        key_info_list=key_info_list,
        grid_rows=grid_rows,
        last_key_edit=last_key_edit,
        last_grid_edit=last_grid_edit,
        module_path=module_path,
        template_data=get_mini_tracker_data(),
        existing_lines=existing_lines,
        tracker_exists=tracker_exists,
        path_to_key_info=path_to_key_info,
        manual_foreign_pins=manual_foreign_pins,
        ast_overrides_applied_count=ast_overrides_applied_count,
        suggestion_applied_count=suggestion_applied_count,
        structural_deps_applied_count=structural_deps_applied_count,
        force_apply_suggestions=force_apply_suggestions,
    )


def create_main_tracker_update(
    output_file: str,
    key_info_list: List[Any],
    grid_rows: List[str],
    last_key_edit: str,
    last_grid_edit: str,
    path_to_key_info: Dict[str, Any],
    ast_overrides_applied_count: int = 0,
    suggestion_applied_count: int = 0,
    structural_deps_applied_count: int = 0,
    force_apply_suggestions: bool = False,
) -> TrackerUpdate:
    """
    Create a TrackerUpdate for the main tracker.

    Args:
        output_file: Path to the main tracker file
        key_info_list: List of KeyInfo objects for this tracker
        grid_rows: Compressed grid data rows
        last_key_edit: Last key edit timestamp message
        last_grid_edit: Last grid edit timestamp message
        path_to_key_info: Global path to KeyInfo mapping

    Returns:
        TrackerUpdate object ready for batch collection
    """
    return TrackerUpdate(
        tracker_type="main",
        output_file=output_file,
        key_info_list=key_info_list,
        grid_rows=grid_rows,
        last_key_edit=last_key_edit,
        last_grid_edit=last_grid_edit,
        path_to_key_info=path_to_key_info,
        ast_overrides_applied_count=ast_overrides_applied_count,
        suggestion_applied_count=suggestion_applied_count,
        structural_deps_applied_count=structural_deps_applied_count,
        force_apply_suggestions=force_apply_suggestions,
    )


def create_doc_tracker_update(
    output_file: str,
    key_info_list: List[Any],
    grid_rows: List[str],
    last_key_edit: str,
    last_grid_edit: str,
    path_to_key_info: Dict[str, Any],
    ast_overrides_applied_count: int = 0,
    suggestion_applied_count: int = 0,
    structural_deps_applied_count: int = 0,
    force_apply_suggestions: bool = False,
) -> TrackerUpdate:
    """
    Create a TrackerUpdate for the doc tracker.

    Args:
        output_file: Path to the doc tracker file
        key_info_list: List of KeyInfo objects for this tracker
        grid_rows: Compressed grid data rows
        last_key_edit: Last key edit timestamp message
        last_grid_edit: Last grid edit timestamp message
        path_to_key_info: Global path to KeyInfo mapping

    Returns:
        TrackerUpdate object ready for batch collection
    """
    return TrackerUpdate(
        tracker_type="doc",
        output_file=output_file,
        key_info_list=key_info_list,
        grid_rows=grid_rows,
        last_key_edit=last_key_edit,
        last_grid_edit=last_grid_edit,
        path_to_key_info=path_to_key_info,
        ast_overrides_applied_count=ast_overrides_applied_count,
        suggestion_applied_count=suggestion_applied_count,
        structural_deps_applied_count=structural_deps_applied_count,
        force_apply_suggestions=force_apply_suggestions,
    )
