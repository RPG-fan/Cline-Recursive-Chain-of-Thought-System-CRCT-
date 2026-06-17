from pathlib import Path

from cline_utils.dependency_system.core.dependency_grid import compress, decompress
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.io import update_main_tracker
from cline_utils.dependency_system.io.tracker_io import update_tracker
from cline_utils.dependency_system.utils.tracker_batch_collector import (
    TrackerBatchCollector,
    apply_reciprocal_dependencies_to_grid,
    build_dependency_suggestions_with_reciprocals,
    create_main_tracker_update,
)
from cline_utils.dependency_system.utils.tracker_utils import read_grid_from_lines


def test_build_dependency_suggestions_with_reciprocals() -> None:
    suggestions = build_dependency_suggestions_with_reciprocals(
        {
            "A": [("B", ">"), ("C", "x"), ("D", "n")],
            "E": [("F", "<")],
        }
    )

    assert suggestions == {
        "A": [("B", ">"), ("C", "x"), ("D", "n")],
        "B": [("A", "<")],
        "C": [("A", "x")],
        "E": [("F", "<")],
        "F": [("E", ">")],
    }


def test_apply_reciprocal_dependencies_to_grid() -> None:
    grid_rows = [list("n>"), list("nn")]

    changes = apply_reciprocal_dependencies_to_grid(grid_rows, True, lambda char: 1)

    assert changes == 1
    assert grid_rows == [list("n>"), list("<n")]


def _key(key: str, path: Path, *, directory: bool = False) -> KeyInfo:
    return KeyInfo(
        key_string=key,
        norm_path=str(path),
        parent_path=str(path.parent),
        tier=1,
        is_directory=directory,
    )


def test_batch_collector_applies_reciprocal_after_consolidation(tmp_path: Path) -> None:
    source = _key("A", tmp_path / "a.py")
    target = _key("B", tmp_path / "b.py")
    tracker_path = tmp_path / "main_tracker.md"

    update = create_main_tracker_update(
        output_file=str(tracker_path),
        key_info_list=[source, target],
        grid_rows=[compress("n>"), compress("nn")],
        last_key_edit="Initial creation",
        last_grid_edit="Initial creation",
        path_to_key_info={source.norm_path: source, target.norm_path: target},
        suggestion_applied_count=1,
        force_apply_suggestions=True,
    )

    collector = TrackerBatchCollector()
    collector.highest_dependency_cache = {("A", "B"): (">", {str(tracker_path)})}
    collector.add(update)

    collector._consolidate_grids()

    assert len(collector.pending_updates) == 1
    rows = [decompress(row) for row in collector.pending_updates[0].grid_rows]
    assert rows == ["n>", "<n"]


def test_update_tracker_direct_reciprocal_suggestions_write_both_directions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _key("1A", tmp_path / "a", directory=True)
    target = _key("1B", tmp_path / "b", directory=True)
    tracker_path = tmp_path / "main_tracker.md"
    monkeypatch.setitem(
        update_main_tracker.main_tracker_data,
        "get_tracker_path",
        lambda project_root: str(tracker_path),
    )
    monkeypatch.setitem(
        update_main_tracker.main_tracker_data,
        "key_filter",
        lambda project_root, path_to_key_info: path_to_key_info,
    )

    update_tracker(
        output_file_suggestion=str(tracker_path),
        path_to_key_info={source.norm_path: source, target.norm_path: target},
        tracker_type="main",
        suggestions_external=build_dependency_suggestions_with_reciprocals(
            {"1A": [("1B", ">")]}
        ),
        force_apply_suggestions=True,
        apply_ast_overrides=False,
    )

    headers, rows = read_grid_from_lines(
        tracker_path.read_text(encoding="utf-8").splitlines()
    )
    row_map = dict(rows)

    assert headers == ["1A", "1B"]
    assert row_map["1A"] == compress("ox")
    assert row_map["1B"] == compress("xo")
