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
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrackerUpdate:
    """
    Holds all data needed for a single tracker update.

    This dataclass stores the processed tracker data in memory
    until it's ready to be written to disk as part of a batch.
    """

    tracker_type: str  # "mini", "doc", or "main"
    output_file: str  # Target file path
    key_info_list: List[Any]  # List[KeyInfo] - key definitions
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
        super().__init__()

    def add(self, update: TrackerUpdate) -> None:
        """
        Add a tracker update to the batch.

        Args:
            update: A TrackerUpdate object containing all data needed for the update

        Raises:
            ValueError: If the update is invalid
        """
        if not update:
            raise ValueError("Cannot add None update to batch")

        self.pending_updates.append(update)
        logger.debug(
            f"Added {update.tracker_type} tracker update for "
            f"{os.path.basename(update.output_file)}"
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
        errors = []
        output_files = set()

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

    def commit_all(self) -> Dict[str, bool]:
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

        results = {}

        try:
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

    def rollback(self) -> None:
        """
        Restore from backups if batch write fails.

        This method restores all tracker files from their backups
        to ensure data consistency after a failed batch write.
        """
        if not self._temp_backup_dir or not self._backed_up_files:
            logger.debug("No backups to restore")
            return

        logger.warning("Rolling back tracker files from backups...")

        restored_count = 0
        failed_count = 0

        for original_path in self._backed_up_files:
            backup_path = os.path.join(
                self._temp_backup_dir, os.path.basename(original_path)
            )

            try:
                if os.path.exists(backup_path):
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
                global_key_counts = defaultdict(int)
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
    )


def create_main_tracker_update(
    output_file: str,
    key_info_list: List[Any],
    grid_rows: List[str],
    last_key_edit: str,
    last_grid_edit: str,
    path_to_key_info: Dict[str, Any],
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
    )


def create_doc_tracker_update(
    output_file: str,
    key_info_list: List[Any],
    grid_rows: List[str],
    last_key_edit: str,
    last_grid_edit: str,
    path_to_key_info: Dict[str, Any],
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
    )
