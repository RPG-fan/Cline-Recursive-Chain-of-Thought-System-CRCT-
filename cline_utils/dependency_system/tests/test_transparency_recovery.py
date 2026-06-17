from pathlib import Path

from cline_utils.dependency_system.io.transparency_manager import (
    TransparencyManager,
    read_file_transparently,
)
from cline_utils.dependency_system.utils.path_utils import normalize_path


def _write_sample_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.py"
    path.write_text("def alpha():\n    pass\n", encoding="utf-8")
    return path


def test_restore_drifted_metadata_handles_malformed_list_section(
    tmp_path: Path,
) -> None:
    path = _write_sample_file(tmp_path)
    manager = TransparencyManager(str(tmp_path / "transparency_registry.json"))

    manager.update_file_metadata(
        file_path=str(path),
        sections={"FUNC": [1, 2], "BAD": [1]},
        content=path.read_text(encoding="utf-8"),
    )

    norm_path = normalize_path(str(path))
    manager._registry["files"][norm_path]["sections"]["BAD"] = [1]

    result = manager.recover_alignment(str(path), path.read_text(encoding="utf-8"))

    assert result is None
    assert manager.get_raw_file_metadata(str(path))["locked"] is True


def test_restore_drifted_metadata_handles_malformed_anchor_list(
    tmp_path: Path,
) -> None:
    path = _write_sample_file(tmp_path)
    manager = TransparencyManager(str(tmp_path / "transparency_registry.json"))

    manager.update_file_metadata(
        file_path=str(path),
        sections={
            "FUNC": {
                "range": [1, 2],
                "anchors": ["def alpha():"],
            }
        },
        content=path.read_text(encoding="utf-8"),
    )

    norm_path = normalize_path(str(path))
    manager._registry["files"][norm_path]["sections"]["FUNC"]["anchors"] = [
        "def alpha():"
    ]

    result = manager.recover_alignment(str(path), path.read_text(encoding="utf-8"))

    assert result is not None


def test_recover_alignment_handles_malformed_dict_range(
    tmp_path: Path,
) -> None:
    path = _write_sample_file(tmp_path)
    manager = TransparencyManager(str(tmp_path / "transparency_registry.json"))

    manager.update_file_metadata(
        file_path=str(path),
        sections={"FUNC": {"range": [1, 2], "anchors": ["def alpha():", "    pass"]}},
        content=path.read_text(encoding="utf-8"),
    )

    norm_path = normalize_path(str(path))
    manager._registry["files"][norm_path]["sections"]["FUNC"]["range"] = "bad"
    manager._save()

    result = manager.recover_alignment(str(path), path.read_text(encoding="utf-8"))

    assert result is None
    assert manager.get_raw_file_metadata(str(path))["locked"] is True


def test_recover_alignment_handles_malformed_start_line(
    tmp_path: Path,
) -> None:
    path = _write_sample_file(tmp_path)
    manager = TransparencyManager(str(tmp_path / "transparency_registry.json"))

    manager.update_file_metadata(
        file_path=str(path),
        sections={"TAGS": {"start_line": 1, "content": "tags"}},
        content=path.read_text(encoding="utf-8"),
    )

    norm_path = normalize_path(str(path))
    manager._registry["files"][norm_path]["sections"]["TAGS"]["start_line"] = "bad"
    manager._save()

    result = manager.recover_alignment(str(path), path.read_text(encoding="utf-8"))

    assert result is None
    assert manager.get_raw_file_metadata(str(path))["locked"] is True


def test_recover_alignment_derives_total_lines_from_sections(
    tmp_path: Path,
) -> None:
    path = _write_sample_file(tmp_path)
    manager = TransparencyManager(str(tmp_path / "transparency_registry.json"))

    manager.update_file_metadata(
        file_path=str(path),
        sections={"FUNC": [1, 2]},
        content=path.read_text(encoding="utf-8"),
    )

    norm_path = normalize_path(str(path))
    manager._registry["files"][norm_path]["total_lines"] = "bad"
    manager._save()

    result = manager.recover_alignment(str(path), path.read_text(encoding="utf-8"))

    assert result is not None
    assert manager.get_raw_file_metadata(str(path))["total_lines"] == 2


def test_recover_alignment_ignores_malformed_connection_metadata(
    tmp_path: Path,
) -> None:
    path = _write_sample_file(tmp_path)
    manager = TransparencyManager(str(tmp_path / "transparency_registry.json"))

    manager.update_file_metadata(
        file_path=str(path),
        sections={"FUNC": [1, 2]},
        content=path.read_text(encoding="utf-8"),
    )

    norm_path = normalize_path(str(path))
    manager._registry["files"][norm_path]["connection_maps"] = "bad"
    manager._registry["files"][norm_path]["connection_map_lines"] = "bad"
    manager._save()

    result = manager.recover_alignment(str(path), path.read_text(encoding="utf-8"))

    assert result is not None
    assert manager.get_raw_file_metadata(str(path))["connection_maps"] == []
    assert manager.get_raw_file_metadata(str(path))["connection_map_lines"] == []


def test_read_file_transparently_locks_when_recovery_raises(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = _write_sample_file(tmp_path)
    manager = TransparencyManager(str(tmp_path / "transparency_registry.json"))
    drifted_content = path.read_text(encoding="utf-8") + "\ndef beta():\n    pass\n"

    manager.update_file_metadata(
        file_path=str(path),
        sections={"FUNC": [1, 2]},
        content=path.read_text(encoding="utf-8"),
    )
    path.write_text(drifted_content, encoding="utf-8")
    monkeypatch.setattr(
        "cline_utils.dependency_system.io.transparency_manager.get_transparency_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        manager,
        "recover_alignment",
        lambda file_path, content: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    content, metadata = read_file_transparently(str(path))

    assert content == drifted_content
    assert metadata is None
    assert manager.get_raw_file_metadata(str(path))["locked"] is True
