"""Mirrored '<'/'>' cells are one directional edge; only the same arrow in both cells is mutual ('x')."""

from pathlib import Path

import pytest

from cline_utils.dependency_system.core.dependency_grid import decompress
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.io import update_main_tracker
from cline_utils.dependency_system.io.tracker_io import update_tracker
from cline_utils.dependency_system.utils.config_manager import ConfigManager
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
    # backup_tracker_file() writes to config 'backups_dir', which resolves to an ABSOLUTE
    # path inside the real repo, so patching get_project_root alone is not enough.
    real_get_path = ConfigManager.get_path

    def _sandboxed_get_path(self, path_type, default_path=None):
        if path_type == "backups_dir":
            return str(tmp_path / "backups")
        return real_get_path(self, path_type, default_path)

    monkeypatch.setattr(ConfigManager, "get_path", _sandboxed_get_path)
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
        # B depends on A on record; force "A depends on B" from A's row ...
        ({"1B": [("1A", "<")]}, {"1A": [("1B", "<")]}),
        # ... or from B's row (the opposite arrow written into the same cell).
        ({"1B": [("1A", "<")]}, {"1B": [("1A", ">")]}),
        # A depends on B on record (stated from A's row as '>'); the same two ways.
        ({"1A": [("1B", ">")]}, {"1A": [("1B", "<")]}),
        ({"1A": [("1B", ">")]}, {"1B": [("1A", ">")]}),
    ],
)
def test_force_add_dependency_over_existing_reverse_edge_is_mutual(
    tmp_path, monkeypatch, initial, followup
):
    # Upstream aa85dd1: a forced edit adding the other direction makes the pair 'x',
    # whether it lands in the reverse cell or as the opposite arrow in the same cell.
    rows = _grid(
        tmp_path,
        monkeypatch,
        initial,
        False,
        then=[(build_dependency_suggestions_with_reciprocals(followup), True)],
    )
    assert rows == {"1A": "ox", "1B": "xo"}  # the mirror must not overwrite the promoted 'x'


def test_unforced_opposite_arrow_over_existing_arrow_is_not_mutual(tmp_path, monkeypatch):
    # Without force, an arrow already on disk is not evidence from this pass (it may be
    # stale or inverted), so the forced-only rule above must not fire.
    rows = _grid(tmp_path, monkeypatch, {"1B": [("1A", "<")]}, False,
                 then=[({"1B": [("1A", ">")]}, False)])
    assert "x" not in rows["1A"] + rows["1B"]


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
