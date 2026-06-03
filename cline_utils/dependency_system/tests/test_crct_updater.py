import os
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from cline_utils.dependency_system.utils import crct_updater

@pytest.fixture
def mock_repo_root(tmp_path):
    """Creates a mock project directory structure with managed directories."""
    # Create managed directories
    (tmp_path / "cline_docs" / "CRCT_Documentation").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cline_docs" / "templates").mkdir(parents=True, exist_ok=True)
    (tmp_path / "code_analysis").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cline_utils").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".agent").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".clinerules").mkdir(parents=True, exist_ok=True)
    
    # Create project_root.cfg to mark root
    (tmp_path / "project_root.cfg").write_text("root=true", encoding="utf-8")
    
    return tmp_path


def test_project_root_detection(mock_repo_root):
    """Verify that _get_project_root correctly locates the mock repository root."""
    dummy_file = mock_repo_root / "cline_utils" / "dependency_system" / "utils" / "dummy.py"
    dummy_file.parent.mkdir(parents=True, exist_ok=True)
    dummy_file.write_text("print('test')", encoding="utf-8")
    
    # Since _get_project_root resolves __file__ inside crct_updater.py, we patch __file__ in crct_updater module
    with patch("cline_utils.dependency_system.utils.crct_updater.__file__", str(dummy_file)):
        detected_root = crct_updater._get_project_root()
        assert detected_root.resolve() == mock_repo_root.resolve()


def test_managed_and_excluded_paths():
    """Verify path categorization rules including exclusions."""
    assert crct_updater._is_managed("cline_docs/CRCT_Documentation/guide.md")
    assert crct_updater._is_managed("cline_utils/dependency_system/dependency_processor.py")
    assert crct_updater._is_managed(".agent/workflows/crct-core.md")
    assert crct_updater._is_managed(".clinerules/execution_plugin.md")
    
    # Excluded files
    assert crct_updater._is_excluded(".clinerules/default-rules.md")
    assert not crct_updater._is_excluded(".clinerules/execution_plugin.md")
    
    # Unmanaged files
    assert not crct_updater._is_managed("src/main.py")
    assert not crct_updater._is_managed("package.json")


def test_load_save_state(mock_repo_root):
    """Verify updater state is correctly loaded and saved to the relocated state path."""
    state = crct_updater._load_state(mock_repo_root)
    # Check default structure
    assert state[crct_updater.CFG_AUTO_UPDATE] is False
    assert state[crct_updater.CFG_LAST_RUN] == 0
    assert state[crct_updater.CFG_KNOWN_SHAS] == {}
    
    # Modify and save
    state[crct_updater.CFG_AUTO_UPDATE] = True
    state[crct_updater.CFG_KNOWN_SHAS] = {"test.py": "abc123sha"}
    crct_updater._save_state(mock_repo_root, state)
    
    # Reload and check
    reloaded = crct_updater._load_state(mock_repo_root)
    assert reloaded[crct_updater.CFG_AUTO_UPDATE] is True
    assert reloaded[crct_updater.CFG_KNOWN_SHAS]["test.py"] == "abc123sha"


@patch("cline_utils.dependency_system.utils.crct_updater._resolve_branch_commit")
@patch("cline_utils.dependency_system.utils.crct_updater._fetch_tree")
@patch("cline_utils.dependency_system.utils.crct_updater._make_request")
def test_check_for_updates_basic(mock_make_req, mock_fetch_tree, mock_resolve_branch, mock_repo_root):
    """Verify updates are identified and processed for missing or modified files."""
    # 1. Setup mock remote state
    commit_sha = "remote_commit_sha_xyz"
    commit_ts = 1000000.0  # mock timestamp
    mock_resolve_branch.return_value = (commit_sha, commit_ts)
    
    mock_fetch_tree.return_value = [
        {"path": "cline_utils/helper.py", "type": "blob", "sha": "remote_sha_1"},
        {"path": "cline_docs/CRCT_Documentation/readme.md", "type": "blob", "sha": "remote_sha_2"},
        {"path": ".clinerules/default-rules.md", "type": "blob", "sha": "excluded_sha"}, # excluded
        {"path": "src/main.py", "type": "blob", "sha": "unmanaged_sha"} # unmanaged
    ]
    
    # Mock download file contents
    mock_make_req.side_effect = [
        b"content of helper.py",
        b"content of readme.md"
    ]
    
    state = crct_updater._load_state(mock_repo_root)
    
    # Run update check
    updated = crct_updater.check_for_updates(mock_repo_root, state, dry_run=False)
    
    # Verify both managed files were downloaded and updated
    assert len(updated) == 2
    assert "cline_utils/helper.py" in updated
    assert "cline_docs/CRCT_Documentation/readme.md" in updated
    
    # Verify files exist in the mock project root
    assert (mock_repo_root / "cline_utils" / "helper.py").exists()
    assert (mock_repo_root / "cline_utils" / "helper.py").read_text(encoding="utf-8") == "content of helper.py"
    
    # Check that mtime was set to remote commit timestamp
    assert (mock_repo_root / "cline_utils" / "helper.py").stat().st_mtime == commit_ts
    
    # Verify state record was updated with the remote SHAs
    assert state[crct_updater.CFG_KNOWN_SHAS]["cline_utils/helper.py"] == "remote_sha_1"
    assert state[crct_updater.CFG_KNOWN_SHAS]["cline_docs/CRCT_Documentation/readme.md"] == "remote_sha_2"


@patch("cline_utils.dependency_system.utils.crct_updater._resolve_branch_commit")
@patch("cline_utils.dependency_system.utils.crct_updater._fetch_tree")
@patch("cline_utils.dependency_system.utils.crct_updater._make_request")
def test_check_for_updates_mtime_gates(mock_make_req, mock_fetch_tree, mock_resolve_branch, mock_repo_root):
    """Verify that locally-modified files with newer mtimes are preserved, while older files are updated."""
    commit_sha = "new_remote_commit"
    commit_ts = 2000000.0  # Remote commit time is 2,000,000
    mock_resolve_branch.return_value = (commit_sha, commit_ts)
    
    mock_fetch_tree.return_value = [
        {"path": "cline_utils/local_edit.py", "type": "blob", "sha": "remote_sha_new"},
        {"path": "cline_utils/outdated.py", "type": "blob", "sha": "remote_sha_new_outdated"}
    ]
    
    # Create the files locally
    local_edit_path = mock_repo_root / "cline_utils" / "local_edit.py"
    local_edit_path.write_text("local changes", encoding="utf-8")
    
    outdated_path = mock_repo_root / "cline_utils" / "outdated.py"
    outdated_path.write_text("old version", encoding="utf-8")
    
    # Set local modification times:
    # local_edit is NEWER than remote commit (local modifications)
    os.utime(local_edit_path, (2500000.0, 2500000.0))
    # outdated is OLDER than remote commit
    os.utime(outdated_path, (1500000.0, 1500000.0))
    
    mock_make_req.return_value = b"new server version content"
    
    state = crct_updater._load_state(mock_repo_root)
    state[crct_updater.CFG_KNOWN_SHAS] = {
        "cline_utils/local_edit.py": "old_sha_1",
        "cline_utils/outdated.py": "old_sha_2"
    }
    
    # Run update check
    updated = crct_updater.check_for_updates(mock_repo_root, state, dry_run=False)
    
    # local_edit should NOT be updated (preserved because local mtime 2500000 >= commit 2000000)
    # outdated SHOULD be updated (local mtime 1500000 < commit 2000000)
    assert len(updated) == 1
    assert "cline_utils/outdated.py" in updated
    assert "cline_utils/local_edit.py" not in updated
    
    # Verify content
    assert local_edit_path.read_text(encoding="utf-8") == "local changes"
    assert outdated_path.read_text(encoding="utf-8") == "new server version content"


@patch("cline_utils.dependency_system.utils.crct_updater.check_for_updates")
def test_auto_update_cooldown_gating(mock_check_updates, mock_repo_root):
    """Verify that auto_update_check respects the opt-in flag and cooldown period."""
    state = crct_updater._load_state(mock_repo_root)
    
    # Scenario 1: Auto-update is disabled (opted out). Should return False immediately.
    state[crct_updater.CFG_AUTO_UPDATE] = False
    crct_updater._save_state(mock_repo_root, state)
    
    res = crct_updater.auto_update_check(mock_repo_root)
    assert res is False
    mock_check_updates.assert_not_called()
    
    # Scenario 2: Enable auto-update. First run should trigger check_for_updates.
    state[crct_updater.CFG_AUTO_UPDATE] = True
    crct_updater._save_state(mock_repo_root, state)
    
    mock_check_updates.return_value = ["file1.py"]
    res = crct_updater.auto_update_check(mock_repo_root)
    assert res is True
    assert mock_check_updates.call_count == 1
    
    # Scenario 3: Call again immediately. Cooldown should gate it (should return False).
    mock_check_updates.reset_mock()
    res = crct_updater.auto_update_check(mock_repo_root)
    assert res is False
    mock_check_updates.assert_not_called()
    
    # Scenario 4: Force check should bypass cooldown.
    res = crct_updater.auto_update_check(mock_repo_root, force=True)
    assert res is True
    assert mock_check_updates.call_count == 1
