"""Unit tests for atomic tracker backups and retrying deletion rotation."""

import datetime
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from cline_utils.dependency_system.io.tracker_io import backup_tracker_file


def test_backup_tracker_file_creates_backup(tmp_path: Path) -> None:
    # Set up mock project structure and config
    tracker = tmp_path / "main_tracker.md"
    tracker.write_text("dummy key definition and grid content", encoding="utf-8")

    # Mock ConfigManager and get_project_root
    with patch("cline_utils.dependency_system.io.tracker_io.ConfigManager") as mock_config_class, \
         patch("cline_utils.dependency_system.io.tracker_io.get_project_root", return_value=str(tmp_path)):
        
        mock_config = mock_config_class.return_value
        mock_config.get_path.return_value = "cline_docs/backups"

        backup_path = backup_tracker_file(str(tracker))

        assert backup_path != ""
        assert os.path.exists(backup_path)
        assert backup_path.endswith(".bak")
        with open(backup_path, "r", encoding="utf-8") as f:
            assert f.read() == "dummy key definition and grid content"


def test_backup_tracker_file_retries_on_lock(tmp_path: Path) -> None:
    tracker = tmp_path / "main_tracker.md"
    tracker.write_text("dummy content", encoding="utf-8")

    calls = {"count": 0}
    real_replace = os.replace

    def flaky_replace(src: str, dest: str) -> None:
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError("sharing violation / file locked")
        real_replace(src, dest)

    with patch("cline_utils.dependency_system.io.tracker_io.ConfigManager") as mock_config_class, \
         patch("cline_utils.dependency_system.io.tracker_io.get_project_root", return_value=str(tmp_path)), \
         patch("cline_utils.dependency_system.io.tracker_io.os.replace", side_effect=flaky_replace), \
         patch("cline_utils.dependency_system.io.tracker_io.time.sleep") as mock_sleep:

        mock_config = mock_config_class.return_value
        mock_config.get_path.return_value = "cline_docs/backups"

        backup_path = backup_tracker_file(str(tracker))

        assert backup_path != ""
        assert os.path.exists(backup_path)
        assert calls["count"] == 3
        assert mock_sleep.call_count == 2


def test_backup_tracker_file_cleanup_retries_on_lock(tmp_path: Path) -> None:
    tracker = tmp_path / "main_tracker.md"
    tracker.write_text("dummy content", encoding="utf-8")

    # Pre-create old backup files to trigger cleanup
    # Needs to match the base_name.YYYYMMDD_HHMMSS_ffffff.bak format
    backup_dir = tmp_path / "cline_docs" / "backups"
    os.makedirs(backup_dir, exist_ok=True)
    
    # We will create 3 old backups. Since we keep 2, and the new backup makes 4, the 2 oldest will be deleted.
    old_backup_1 = backup_dir / "main_tracker.md.20260519_120000_000000.bak"
    old_backup_2 = backup_dir / "main_tracker.md.20260519_130000_000000.bak"
    old_backup_3 = backup_dir / "main_tracker.md.20260519_140000_000000.bak"
    
    old_backup_1.write_text("old 1", encoding="utf-8")
    old_backup_2.write_text("old 2", encoding="utf-8")
    old_backup_3.write_text("old 3", encoding="utf-8")

    calls = {"count": 0}
    real_remove = os.remove

    def flaky_remove(path: str) -> None:
        if "20260519_120000_000000" in path or "20260519_130000_000000" in path:
            calls["count"] += 1
            if calls["count"] < 3:
                raise PermissionError("file locked")
        real_remove(path)

    with patch("cline_utils.dependency_system.io.tracker_io.ConfigManager") as mock_config_class, \
         patch("cline_utils.dependency_system.io.tracker_io.get_project_root", return_value=str(tmp_path)), \
         patch("cline_utils.dependency_system.io.tracker_io.os.remove", side_effect=flaky_remove), \
         patch("cline_utils.dependency_system.io.tracker_io.time.sleep") as mock_sleep:

        mock_config = mock_config_class.return_value
        mock_config.get_path.return_value = "cline_docs/backups"

        backup_path = backup_tracker_file(str(tracker))

        assert backup_path != ""
        # The newest old backup (old_backup_3) + the new backup must survive
        assert os.path.exists(old_backup_3)
        assert os.path.exists(backup_path)
        
        # The 2 oldest backups (old_backup_1 and old_backup_2) should be successfully removed
        assert not os.path.exists(old_backup_1)
        assert not os.path.exists(old_backup_2)
        
        # flaky_remove will trigger retries for both deleted files
        assert calls["count"] >= 3
        assert mock_sleep.call_count >= 2
