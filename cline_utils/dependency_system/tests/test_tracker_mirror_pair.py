"""Mirrored '<'/'>' cells are one directional edge; only the same arrow in both cells is mutual ('x')."""

from pathlib import Path

import pytest

from cline_utils.dependency_system.core.dependency_grid import decompress
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.io import update_main_tracker
from cline_utils.dependency_system.io.tracker_io import update_tracker
from cline_utils.dependency_system.utils.tracker_batch_collector import (
    build_dependency_suggestions_with_reciprocals,
)
from cline_utils.dependency_system.utils.tracker_utils import read_grid_from_lines


def _key(key: str, path: Path) -> KeyInfo:
    return KeyInfo(key_string=key, norm_path=str(path), parent_path=str(path.parent),
                   tier=1, is_directory=True)


def _grid(tmp_path: Path, monkeypatch, suggestions, force: bool, *, then=()) -> dict:
    """Apply ``suggestions`` to a fresh main tracker, then each ``(suggestions, force)`` in ``then``."""
    a, b = _key("1A", tmp_path / "a"), _key("1B", tmp_path / "b")
    tracker_path = tmp_path / "main_tracker.md"
    monkeypatch.setitem(update_main_tracker.main_tracker_data, "get_tracker_path",
                        lambda project_root: str(tracker_path))
    monkeypatch.setitem(update_main_tracker.main_tracker_data, "key_filter",
                        lambda project_root, path_to_key_info: path_to_key_info)
    for sugg, frc in [(suggestions, force), *then]:
        update_tracker(
            output_file_suggestion=str(tracker_path),
            path_to_key_info={a.norm_path: a, b.norm_path: b},
            tracker_type="main",
            suggestions_external=sugg,
            force_apply_suggestions=frc,
            apply_ast_overrides=False,
        )
    _, rows = read_grid_from_lines(tracker_path.read_text(encoding="utf-8").splitlines())
    return {k: decompress(v) for k, v in rows}


@pytest.mark.parametrize("force", [False, True])
def test_mirrored_pair_stays_directional(tmp_path, monkeypatch, force):
    # A imports B ('<'), and the suggester/reciprocal builder also emits the mirror (B, A, '>').
    rows = _grid(tmp_path, monkeypatch,
                 {"1A": [("1B", "<")], "1B": [("1A", ">")]}, force)
    assert rows == {"1A": "o<", "1B": ">o"}


def test_reciprocal_builder_output_stays_directional(tmp_path, monkeypatch):
    rows = _grid(tmp_path, monkeypatch,
                 build_dependency_suggestions_with_reciprocals({"1A": [("1B", ">")]}), True)
    assert rows == {"1A": "o>", "1B": "<o"}


def test_genuine_mutual_still_becomes_x(tmp_path, monkeypatch):
    # A depends on B AND B depends on A: both rows carry '<'.
    rows = _grid(tmp_path, monkeypatch,
                 {"1A": [("1B", "<")], "1B": [("1A", "<")]}, False)
    assert rows == {"1A": "ox", "1B": "xo"}


def test_forward_only_after_on_disk_mirror_stays_directional(tmp_path, monkeypatch):
    # Run 1 left the consistent pair on disk (batch reciprocal pass): [B][A] = '>'.
    # Run 2 brings only the forward suggestion A -> B '<' (no mirror suggestion at all).
    rows = _grid(tmp_path, monkeypatch, {"1B": [("1A", ">")]}, False,
                 then=[({"1A": [("1B", "<")]}, False)])
    assert rows == {"1A": "o<", "1B": ">o"}


@pytest.mark.parametrize(
    "initial,followup",
    [
        ({"1B": [("1A", "<")]}, {"1A": [("1B", "<")]}),
        ({"1B": [("1A", "<")]}, {"1B": [("1A", ">")]}),
        ({"1A": [("1B", ">")]}, {"1A": [("1B", "<")]}),
        ({"1A": [("1B", ">")]}, {"1B": [("1A", ">")]}),
    ],
)
def test_force_add_dependency_over_existing_reverse_edge_is_mutual(
    tmp_path, monkeypatch, initial, followup
):
    # B already depends on A; user force-adds "A depends on B" with reciprocals.
    rows = _grid(
        tmp_path,
        monkeypatch,
        initial,
        False,
        then=[(build_dependency_suggestions_with_reciprocals(followup), True)],
    )
    assert rows == {"1A": "ox", "1B": "xo"}  # the mirror must not overwrite the promoted 'x'


def test_unforced_mutual_does_not_clobber_manual_n_override(tmp_path, monkeypatch):
    # [1A][1B] is manually pinned to 'n' while [1B][1A] has '<'.
    # An unforced '<' suggestion on 1A -> 1B must not promote [1B][1A] to 'x'.
    rows = _grid(
        tmp_path,
        monkeypatch,
        {"1A": [("1B", "n")], "1B": [("1A", "<")]},
        True,
        then=[({"1A": [("1B", "<")]}, False)],
    )
    assert rows == {"1A": "on", "1B": "<o"}

