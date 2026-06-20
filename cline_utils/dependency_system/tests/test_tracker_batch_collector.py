import pytest
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock

from cline_utils.dependency_system.utils.tracker_batch_collector import (
    TrackerBatchCollector,
    TrackerUpdate,
)


class TestTrackerBatchCollectorRollback:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_dummy_update(self, output_file: str) -> TrackerUpdate:
        return TrackerUpdate(
            tracker_type="main",
            output_file=output_file,
            key_info_list=[],
            grid_rows=[],
            last_key_edit="test",
            last_grid_edit="test",
        )

    def test_rollback_no_backup_dir(self):
        collector = TrackerBatchCollector()
        collector.rollback()
        assert collector._temp_backup_dir is None

    def test_rollback_restores_existing_files(self):
        collector = TrackerBatchCollector()

        # Create an existing file
        existing_file = os.path.join(self.temp_dir, "existing.txt")
        with open(existing_file, "w") as f:
            f.write("original content")

        # Add to updates
        update1 = self.create_dummy_update(existing_file)
        collector.add(update1)

        # Create backups
        collector._create_backups()
        assert existing_file in collector._backed_up_files

        # Modify the existing file (simulate a partial write)
        with open(existing_file, "w") as f:
            f.write("modified content")

        # Perform rollback
        collector.rollback()

        # Verify content was restored
        with open(existing_file, "r") as f:
            content = f.read()
        assert content == "original content"

        # Verify backup dir was cleaned up
        assert collector._temp_backup_dir is None
        assert not collector._backed_up_files

    def test_rollback_removes_new_files(self):
        collector = TrackerBatchCollector()

        # A new file that didn't exist before
        new_file = os.path.join(self.temp_dir, "new.txt")

        # Add to updates
        update = self.create_dummy_update(new_file)
        collector.add(update)

        # Create backups (should not back up the non-existent file)
        collector._create_backups()
        assert new_file not in collector._backed_up_files

        # Manually write the new file to simulate it being partially written
        with open(new_file, "w") as f:
            f.write("partial content")

        # Perform rollback
        collector.rollback()

        # Verify the new file was removed
        assert not os.path.exists(new_file)

    def test_commit_all_triggers_rollback_on_failure(self):
        collector = TrackerBatchCollector()

        # Create an existing file
        existing_file = os.path.join(self.temp_dir, "existing.txt")
        with open(existing_file, "w") as f:
            f.write("original content")

        # New file path
        new_file = os.path.join(self.temp_dir, "new.txt")

        collector.add(self.create_dummy_update(existing_file))
        collector.add(self.create_dummy_update(new_file))

        # Mock _write_single_tracker to write the first file but fail on the second
        def mock_write(update):
            if update.output_file == new_file:
                # Let's simulate a partially written file before failing
                with open(new_file, "w") as f:
                    f.write("partial content")
                raise Exception("Simulated write failure")
            else:
                # Modify the existing file to simulate it being overwritten
                with open(update.output_file, "w") as f:
                    f.write("modified content")
                return True

        with patch.object(collector, "_write_single_tracker", side_effect=mock_write):
            with patch.object(collector, "_consolidate_grids"), patch.object(
                collector, "_create_permanent_backups"
            ):

                with pytest.raises(RuntimeError) as exc_info:
                    collector.commit_all()

                assert "Batch write failed and was rolled back" in str(exc_info.value)

        # Now verify the state
        with open(existing_file, "r") as f:
            content = f.read()
        assert content == "original content", "Existing file was not restored!"

        assert not os.path.exists(
            new_file
        ), "New partially written file was not removed!"

    def test_rollback_with_basename_collisions(self):
        collector = TrackerBatchCollector()

        # Create two subdirectories
        dir_a = os.path.join(self.temp_dir, "dir_a")
        dir_b = os.path.join(self.temp_dir, "dir_b")
        os.makedirs(dir_a, exist_ok=True)
        os.makedirs(dir_b, exist_ok=True)

        # Create files with identical basenames but different content
        file_a = os.path.join(dir_a, "tracker.txt")
        file_b = os.path.join(dir_b, "tracker.txt")

        with open(file_a, "w") as f:
            f.write("content A")
        with open(file_b, "w") as f:
            f.write("content B")

        # Add updates
        collector.add(self.create_dummy_update(file_a))
        collector.add(self.create_dummy_update(file_b))

        # Create backups
        collector._create_backups()
        assert file_a in collector._backed_up_files
        assert file_b in collector._backed_up_files

        # Verify backups have unique backup paths in mappings
        backup_path_a = collector._backup_mappings.get(file_a)
        backup_path_b = collector._backup_mappings.get(file_b)
        assert backup_path_a != backup_path_b

        # Simulate partial/corrupted write
        with open(file_a, "w") as f:
            f.write("corrupted A")
        with open(file_b, "w") as f:
            f.write("corrupted B")

        # Rollback
        collector.rollback()

        # Verify both were restored correctly to their respective contents
        with open(file_a, "r") as f:
            assert f.read() == "content A"
        with open(file_b, "r") as f:
            assert f.read() == "content B"

