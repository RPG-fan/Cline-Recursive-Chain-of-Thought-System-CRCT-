"""One-time 'x' -> 'p' reset (utils/reset_mutual.py + reset-mutual-dependencies command)."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from cline_utils.dependency_system import dependency_processor
from cline_utils.dependency_system.core import key_manager
from cline_utils.dependency_system.core.dependency_grid import compress, decompress
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.io import tracker_io
from cline_utils.dependency_system.io.tracker_io import (
    write_mini_tracker_with_template_preservation,
    write_tracker_file,
)
from cline_utils.dependency_system.io.update_mini_tracker import get_mini_tracker_data
from cline_utils.dependency_system.utils import reset_mutual
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import normalize_path
from cline_utils.dependency_system.utils.reset_mutual import (
    reset_mutual_dependencies,
)
from cline_utils.dependency_system.utils.tracker_utils import (
    read_tracker_file_structured,
)

KEYS = ["1A1", "1A2", "1A3", "1A4", "1A5", "1A6"]

# Upper-triangle edits; the second char of each pair is the mirror cell.
#   A-B x/x   A-C x/<(asymmetric)   A-D n/n   B-C </>   B-D d/d   B-E s/S
#   C-D p/p   E-F x/x
EDITS: Dict[Tuple[int, int], Tuple[str, str]] = {
    (0, 1): ("x", "x"),
    (0, 2): ("x", "<"),
    (0, 3): ("n", "n"),
    (1, 2): ("<", ">"),
    (1, 3): ("d", "d"),
    (1, 4): ("s", "S"),
    (4, 5): ("x", "x"),
}


@pytest.fixture(autouse=True)
def sandboxed_backups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # backup_tracker_file() resolves 'backups_dir' to an ABSOLUTE path inside the real repo
    # (see test_tracker_mirror_pair.py), so redirect it for every test in this module.
    backups = tmp_path / "backups"
    real_get_path = ConfigManager.get_path

    def _sandboxed_get_path(self, path_type, default_path=None):  # type: ignore[no-untyped-def]
        if path_type == "backups_dir":
            return str(backups)
        return real_get_path(self, path_type, default_path)

    monkeypatch.setattr(ConfigManager, "get_path", _sandboxed_get_path)
    return backups


def _infos(base: Path) -> Tuple[List[KeyInfo], Dict[str, KeyInfo]]:
    infos = [
        KeyInfo(
            key_string=k,
            norm_path=normalize_path(str(base / f"f{i}.py")),
            parent_path=normalize_path(str(base)),
            tier=1,
            is_directory=False,
        )
        for i, k in enumerate(KEYS)
    ]
    return infos, {ki.norm_path: ki for ki in infos}


def _rows(n: int = len(KEYS)) -> List[List[str]]:
    grid = [["p"] * n for _ in range(n)]
    for i in range(n):
        grid[i][i] = "o"
    for (i, j), (a, b) in EDITS.items():
        grid[i][j], grid[j][i] = a, b
    return grid


def _compressed(grid: List[List[str]]) -> List[str]:
    return [compress("".join(r)) for r in grid]


def _make_main(tmp_path: Path) -> Tuple[Path, Dict[str, KeyInfo]]:
    infos, gmap = _infos(tmp_path)
    path = tmp_path / "module_relationship_tracker.md"
    assert write_tracker_file(str(path), infos, _compressed(_rows()), "k", "g", gmap)
    return path, gmap


def _read_grid(path: Path) -> List[List[str]]:
    data = read_tracker_file_structured(str(path))
    return [list(decompress(c)) for _, c in data["grid_rows_ordered"]]


def _expected_after() -> List[List[str]]:
    return [
        ["p" if (c == "x" and i != j) else c for j, c in enumerate(row)]
        for i, row in enumerate(_rows())
    ]


def _backup_files(backups: Path) -> List[Path]:
    return sorted(p for p in backups.rglob("*") if p.is_file()) if backups.exists() else []


def test_dry_run_changes_nothing_and_reports_counts(tmp_path: Path, sandboxed_backups: Path):
    path, gmap = _make_main(tmp_path)
    before = path.read_bytes()

    report = reset_mutual_dependencies(
        [str(path)], gmap, doc_roots=[], apply=False, project_root=str(tmp_path)
    )

    assert path.read_bytes() == before
    assert _backup_files(sandboxed_backups) == []
    (res,) = report.results
    assert res.status == "would_reset"
    # A-B (2 cells), A-C (only the x cell), E-F (2 cells)
    assert res.x_cells == 5
    assert res.asymmetric_cells == 1
    assert len(res.pair_keys) == 3
    assert report.total_cells == 5 and len(report.unique_pairs) == 3
    assert len(report.pairs_by_category()["code-code"]) == 3
    assert "DRY RUN" in reset_mutual.format_report(report)


def test_apply_flips_only_x_cells_in_both_directions(tmp_path: Path, sandboxed_backups: Path):
    path, gmap = _make_main(tmp_path)

    report = reset_mutual_dependencies(
        [str(path)], gmap, apply=True, project_root=str(tmp_path)
    )

    (res,) = report.results
    assert res.status == "reset"
    assert _read_grid(path) == _expected_after()
    grid = _read_grid(path)
    # untouched characters survive, including the mirrored '<' left by the asymmetric x
    assert grid[2][0] == "<" and grid[1][2] == "<" and grid[2][1] == ">"
    assert grid[0][3] == "n" and grid[1][3] == "d" and grid[1][4] == "s" and grid[4][1] == "S"
    assert all(grid[i][i] == "o" for i in range(len(KEYS)))
    # definitions / headers are unchanged
    data = read_tracker_file_structured(str(path))
    assert [k for k, _ in data["definitions_ordered"]] == KEYS
    assert data["grid_headers_ordered"] == KEYS
    assert data["last_key_edit"] == "k"
    assert "one-time migration" in data["last_grid_edit"]


def test_backups_land_only_in_sandbox(tmp_path: Path, sandboxed_backups: Path):
    path, gmap = _make_main(tmp_path)
    original = path.read_bytes()

    report = reset_mutual_dependencies(
        [str(path)], gmap, apply=True, project_root=str(tmp_path)
    )

    files = _backup_files(sandboxed_backups)
    # one rotating backup + one run-scoped snapshot, both inside the sandbox
    assert len(files) == 2
    assert all(sandboxed_backups in f.parents for f in files)
    # CRCT globs *_module.md / *.md under the repo: nothing we leave behind may look like a tracker
    assert all(f.suffix == ".bak" and not f.name.endswith(".md") for f in files)
    assert {f.read_bytes() for f in files} == {original}
    (res,) = report.results
    assert Path(res.backup_path).parent == sandboxed_backups
    assert Path(res.backup_path).read_bytes() == original
    assert sandboxed_backups in Path(res.snapshot_path).parents


def test_snapshot_of_mini_tracker_is_not_named_like_a_tracker(tmp_path: Path, sandboxed_backups: Path):
    moddir = tmp_path / "mod"
    moddir.mkdir()
    infos, gmap = _infos(moddir)
    mini = moddir / "mod_module.md"
    data = get_mini_tracker_data()
    write_mini_tracker_with_template_preservation(
        str(mini), [], infos, _compressed(_rows()), "k", "g",
        data["template"], data["markers"], gmap, normalize_path(str(moddir)), set(),
    )

    reset_mutual_dependencies([str(mini)], gmap, apply=True, project_root=str(tmp_path))

    leftovers = _backup_files(sandboxed_backups)
    assert leftovers and not any(f.name.endswith("_module.md") or f.suffix == ".md" for f in leftovers)
    assert list(sandboxed_backups.rglob("*_module.md")) == []


def test_second_apply_is_a_no_op(tmp_path: Path, sandboxed_backups: Path):
    path, gmap = _make_main(tmp_path)
    reset_mutual_dependencies([str(path)], gmap, apply=True, project_root=str(tmp_path))
    after_first = path.read_bytes()
    backups_first = _backup_files(sandboxed_backups)

    report = reset_mutual_dependencies(
        [str(path)], gmap, apply=True, project_root=str(tmp_path)
    )

    assert report.results[0].status == "clean"
    assert report.total_cells == 0
    assert path.read_bytes() == after_first
    assert _backup_files(sandboxed_backups) == backups_first


def test_mini_tracker_header_footer_and_pins_survive(tmp_path: Path, sandboxed_backups: Path):
    moddir = tmp_path / "mod"
    moddir.mkdir()
    infos, gmap = _infos(moddir)
    mini = moddir / "mod_module.md"
    data = get_mini_tracker_data()
    pin = normalize_path(str(tmp_path / "elsewhere.py"))
    write_mini_tracker_with_template_preservation(
        str(mini), [], infos, _compressed(_rows()), "k", "g",
        data["template"], data["markers"], gmap, normalize_path(str(moddir)), {pin},
    )
    text = mini.read_text(encoding="utf-8").replace(
        "## Purpose & Responsibility", "## Purpose CUSTOM-HEADER"
    )
    end_marker = data["markers"][1]
    text = text.replace(end_marker, end_marker + "\nCUSTOM-FOOTER", 1)
    mini.write_text(text, encoding="utf-8", newline="\n")

    report = reset_mutual_dependencies(
        [str(mini)], gmap, apply=True, project_root=str(tmp_path)
    )

    assert report.results[0].status == "reset"
    new_text = mini.read_text(encoding="utf-8")
    assert "CUSTOM-HEADER" in new_text and "CUSTOM-FOOTER" in new_text
    assert pin in new_text  # manual_foreign_pins preserved
    assert _read_grid(mini) == _expected_after()


def test_tracker_stale_against_global_map_keeps_its_labels(tmp_path: Path, sandboxed_backups: Path):
    path, _ = _make_main(tmp_path)

    report = reset_mutual_dependencies(
        [str(path)], {}, apply=True, project_root=str(tmp_path)
    )

    assert report.results[0].status == "reset"
    data = read_tracker_file_structured(str(path))
    assert [k for k, _ in data["definitions_ordered"]] == KEYS
    assert _read_grid(path) == _expected_after()


def test_failed_verification_restores_the_tracker(
    tmp_path: Path, sandboxed_backups: Path, monkeypatch: pytest.MonkeyPatch
):
    path, gmap = _make_main(tmp_path)
    original = path.read_bytes()
    real_write = tracker_io.write_tracker_file

    def _bad_write(tracker_path, infos, rows, kedit, gedit, gm):  # type: ignore[no-untyped-def]
        # "Reset" everything to 'p' instead of just the x cells.
        n = len(infos)
        bad = [compress("".join("o" if i == j else "p" for j in range(n))) for i in range(n)]
        return real_write(tracker_path, infos, bad, kedit, gedit, gm)

    # TrackerBatchCollector._write_single_tracker imports the writer from tracker_io at call time.
    monkeypatch.setattr(tracker_io, "write_tracker_file", _bad_write)

    report = reset_mutual_dependencies(
        [str(path)], gmap, apply=True, project_root=str(tmp_path)
    )

    assert report.results[0].status == "error"
    assert "restored" in report.results[0].reason
    assert path.read_bytes() == original


def test_batch_failure_rolls_back_every_tracker(
    tmp_path: Path, sandboxed_backups: Path, monkeypatch: pytest.MonkeyPatch
):
    first, gmap = _make_main(tmp_path)
    second = tmp_path / "doc_tracker.md"
    infos, _ = _infos(tmp_path)
    assert write_tracker_file(str(second), infos, _compressed(_rows()), "k", "g", gmap)
    originals = {p: p.read_bytes() for p in (first, second)}
    real_write = tracker_io.write_tracker_file
    calls = {"n": 0}

    def _fail_on_second(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(tracker_io, "write_tracker_file", _fail_on_second)

    report = reset_mutual_dependencies(
        [str(first), str(second)], gmap, apply=True, project_root=str(tmp_path)
    )

    assert {r.status for r in report.results} == {"error"}
    assert {p: p.read_bytes() for p in originals} == originals  # nothing half-written
    assert all(f.suffix == ".bak" for f in _backup_files(sandboxed_backups))


def test_apply_leaves_no_in_process_aggregation_cache(tmp_path: Path):
    from cline_utils.dependency_system.utils import tracker_utils

    path, gmap = _make_main(tmp_path)
    reset_mutual_dependencies([str(path)], gmap, apply=True, project_root=str(tmp_path))
    assert tracker_utils.runtime_aggregation_cache is None


def test_unreadable_module_md_is_ignored_not_a_failure(tmp_path: Path):
    notes = tmp_path / "notes_module.md"
    notes.write_text("# just a doc\n", encoding="utf-8")

    report = reset_mutual_dependencies(
        [str(notes)], {}, apply=True, project_root=str(tmp_path)
    )

    assert report.results[0].status == "ignored"
    assert not report.has_failures
    assert notes.read_text(encoding="utf-8") == "# just a doc\n"


def test_doc_pairs_are_classified_by_doc_roots(tmp_path: Path):
    docs = tmp_path / "docs"
    infos = [
        KeyInfo(key_string=k, norm_path=normalize_path(str(p)), parent_path=None, tier=1, is_directory=False)
        for k, p in (("1A1", docs / "a.md"), ("1A2", docs / "b.md"), ("1A3", tmp_path / "c.py"))
    ]
    gmap = {ki.norm_path: ki for ki in infos}
    grid = [list("oxx"), list("xox"), list("xxo")]
    path = tmp_path / "doc_tracker.md"
    assert write_tracker_file(str(path), infos, _compressed(grid), "k", "g", gmap)

    report = reset_mutual_dependencies(
        [str(path)], gmap, doc_roots=[str(docs)], apply=False, project_root=str(tmp_path)
    )

    cats = report.pairs_by_category()
    assert (len(cats["doc-doc"]), len(cats["doc-code"]), len(cats["code-code"])) == (1, 2, 0)


def test_command_handler_defaults_to_dry_run_and_apply_writes(
    tmp_path: Path, sandboxed_backups: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    path, gmap = _make_main(tmp_path)
    original = path.read_bytes()
    monkeypatch.setattr(key_manager, "load_tracker_map", lambda: [str(path)])
    monkeypatch.setattr(dependency_processor, "load_global_key_map", lambda: gmap)
    monkeypatch.setattr(dependency_processor, "get_project_root", lambda: str(tmp_path))
    out_file = tmp_path / "report.md"

    rc = dependency_processor.handle_reset_mutual_dependencies(
        argparse.Namespace(apply=False, dry_run=False, output=str(out_file))
    )
    assert rc == 0
    assert path.read_bytes() == original
    assert "DRY RUN" in out_file.read_text(encoding="utf-8")

    rc = dependency_processor.handle_reset_mutual_dependencies(
        argparse.Namespace(apply=True, dry_run=False, output=None)
    )
    assert rc == 0
    assert _read_grid(path) == _expected_after()
    assert "APPLIED" in capsys.readouterr().out
    assert len(_backup_files(sandboxed_backups)) == 2
