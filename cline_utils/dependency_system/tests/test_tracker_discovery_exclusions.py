"""find_all_tracker_paths must not register mini trackers under excluded directories."""

import json
import os
from pathlib import Path
from typing import Set

import pytest

from cline_utils.dependency_system.utils import tracker_utils
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import normalize_path


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# tracker\n", encoding="utf-8")
    return normalize_path(str(path))


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Tmp project whose single code root is '.', with explicit exclusion settings."""
    root = tmp_path / "proj"
    (root / ".clinerules").mkdir(parents=True)
    (root / "project_root.cfg").write_text("", encoding="utf-8")
    (root / ".clinerules" / "default-rules.md").write_text(
        "[CODE_ROOT_DIRECTORIES]\n- .\n\n[DOC_DIRECTORIES]\n- docs\n", encoding="utf-8"
    )
    (root / ".clinerules.config.json").write_text(
        json.dumps(
            {
                "excluded_dirs": ["node_modules", "cline_docs", ".agents", "dist"],
                "excluded_paths": ["generated/cache"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    ConfigManager._instance = None  # pyright: ignore[reportPrivateUsage]
    return root


def _found(project: Path, force_scan: bool = True) -> Set[str]:
    config = ConfigManager()
    return tracker_utils.find_all_tracker_paths(
        config, normalize_path(str(project)), force_scan=force_scan
    )


def test_mini_trackers_under_excluded_dirs_are_skipped(project: Path):
    real = _touch(project / "services" / "auth" / "auth_module.md")
    excluded = [
        _touch(project / "cline_docs" / "snapshot" / "snapshot_module.md"),
        _touch(project / "node_modules" / "pkg" / "pkg_module.md"),
        _touch(project / ".agents" / "worker" / "worker_module.md"),
        _touch(project / "generated" / "cache" / "cache_module.md"),
        # Bare names in excluded_dirs are pruned at any depth by the project walker.
        _touch(project / "services" / "dist" / "dist_module.md"),
        _touch(project / "services" / "auth" / "node_modules" / "dep" / "dep_module.md"),
    ]

    found = _found(project)

    assert real in found
    assert not found.intersection(excluded)


def test_bare_dir_names_match_exactly_like_the_walker(project: Path):
    """The project walker prunes with `d not in excluded_dirs`: case-sensitive."""
    differently_cased = _touch(project / "services" / "Dist" / "Dist_module.md")

    assert differently_cased in _found(project)


def test_main_and_doc_trackers_in_excluded_memory_dir_are_kept(project: Path):
    main_tracker = _touch(project / "cline_docs" / "module_relationship_tracker.md")
    doc_tracker = _touch(project / "cline_docs" / "doc_tracker.md")
    real = _touch(project / "services" / "auth" / "auth_module.md")
    _touch(project / "cline_docs" / "snapshot" / "snapshot_module.md")

    assert _found(project) == {main_tracker, doc_tracker, real}


def test_cached_tracker_map_is_filtered_too(
    project: Path, monkeypatch: pytest.MonkeyPatch
):
    """A stale tracker_map.json must not keep resurrecting excluded mini trackers."""
    main_tracker = _touch(project / "cline_docs" / "module_relationship_tracker.md")
    real = _touch(project / "services" / "auth" / "auth_module.md")
    stale = _touch(project / "cline_docs" / "snapshot" / "snapshot_module.md")
    monkeypatch.setattr(
        tracker_utils, "load_tracker_map", lambda: [main_tracker, real, stale]
    )

    assert _found(project, force_scan=False) == {main_tracker, real}


def test_paths_outside_project_root_are_not_filtered(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    outside = _touch(tmp_path / "elsewhere" / "node_modules" / "x" / "x_module.md")
    monkeypatch.setattr(tracker_utils, "load_tracker_map", lambda: [outside])

    assert _found(project, force_scan=False) == {outside}
    assert os.path.exists(outside)
