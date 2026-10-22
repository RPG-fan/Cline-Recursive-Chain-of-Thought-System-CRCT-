"""tracker_map.json is the single authoritative tracker set.

An existing tracker the map does not list (e.g. ``src/.hidden/x/x_module.md``: discovery's
glob skips dot-directories) must be neither updated by analyze-project nor imported by the
batch collector, so its stale 'x' cells cannot leak into the tracked trackers.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Generator, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cline_utils.dependency_system.core import key_manager
from cline_utils.dependency_system.core.dependency_grid import compress, decompress
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.dependency_processor import (
    command_handler_analyze_project,
)
from cline_utils.dependency_system.io.update_main_tracker import (
    aggregate_dependencies_contextual,
)
from cline_utils.dependency_system.io.tracker_io import (
    get_mini_tracker_path,
    write_tracker_file,
)
from cline_utils.dependency_system.utils import tracker_batch_collector as tbc
from cline_utils.dependency_system.utils.cache_manager import clear_all_caches
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import normalize_path
from cline_utils.dependency_system.utils.tracker_batch_collector import (
    TrackerBatchCollector,
    TrackerUpdate,
)
from cline_utils.dependency_system.utils.tracker_utils import (
    collect_tracker_map_paths,
    is_discoverable_mini_tracker_path,
    read_tracker_file_structured,
)

# --- Fixtures ----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_embedding_manager() -> Generator[None, None, None]:
    """Embedding model mocked as in test_e2e_workflow.py (no model download)."""
    with patch(
        "cline_utils.dependency_system.analysis.embedding_manager._load_model"
    ) as mock_load, patch(
        "cline_utils.dependency_system.analysis.embedding_manager._model_instance",
        new_callable=MagicMock,
    ) as mock_instance, patch(
        "cline_utils.dependency_system.analysis.embedding_manager._selected_model_config",
        {"type": "sentence-transformer", "name": "mock-model", "embedding_dim": 384},
    ):

        def side_effect_load(*args: Any, **kwargs: Any) -> MagicMock:
            import cline_utils.dependency_system.analysis.embedding_manager as em

            em._model_instance = mock_instance  # pyright: ignore[reportPrivateUsage]
            return mock_instance

        mock_load.return_value = mock_instance
        mock_load.side_effect = side_effect_load
        mock_instance.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_instance.embed.return_value = np.random.rand(384).tolist()
        yield


@pytest.fixture(autouse=True)
def sandboxed_backups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """backup_tracker_file() resolves 'backups_dir' inside the real repo; redirect it."""
    backups = tmp_path / "backups"
    real_get_path = ConfigManager.get_path

    def _sandboxed_get_path(self: ConfigManager, path_type: str, default_path: Any = None):
        if path_type == "backups_dir":
            return str(backups)
        return real_get_path(self, path_type, default_path)

    monkeypatch.setattr(ConfigManager, "get_path", _sandboxed_get_path)
    return backups


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Project with a tracked module ``src/pkg`` and a dot-directory module ``src/.hidden/x``."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".clinerules").write_text(
        "[CODE_ROOT_DIRECTORIES]\n- src\n\n[DOC_DIRECTORIES]\n- docs"
    )
    (root / ".clinerules.config.json").write_text(
        json.dumps(
            {
                "paths": {
                    "doc_dir": "docs",
                    "memory_dir": "cline_docs",
                    "cache_dir": "cache",
                    "embeddings_dir": "cline_docs/embeddings",
                },
                "excluded_dirs": [".git", "venv"],
                "embedding": {"model_selection": "mpnet"},
            }
        )
    )
    pkg = root / "src" / "pkg"
    pkg.mkdir(parents=True)
    hidden = root / "src" / ".hidden" / "x"
    hidden.mkdir(parents=True)
    (hidden / "h1.ts").write_text("export function h1(): number { return 1; }\n")
    (hidden / "h2.ts").write_text("export function h2(): number { return 2; }\n")
    # pkg depends on both hidden files, so they are foreign keys in pkg's tracker too.
    (pkg / "a.ts").write_text(
        "import { h1 } from '../.hidden/x/h1';\n"
        "import { h2 } from '../.hidden/x/h2';\n"
        "export const total = (): number => h1() + h2();\n"
    )
    (root / "docs").mkdir()
    (root / "docs" / "readme.md").write_text("# Readme\nDescribes a.py")
    (root / "cline_docs").mkdir()
    return root


def _analyze(root: Path) -> int:
    clear_all_caches()
    args = MagicMock(spec=argparse.Namespace)
    args.project_root = str(root)
    args.output = None
    args.force_analysis = True
    args.force_embeddings = False
    args.force_validate = False
    return command_handler_analyze_project(args)


def _mapped() -> List[str]:
    return [normalize_path(p) for p in key_manager.load_tracker_map()]


def _grid(path: Path) -> Dict[Tuple[str, str], str]:
    """{(row_path, col_path): char} of a tracker file."""
    data = read_tracker_file_structured(str(path))
    defs = [normalize_path(p) for _, p in data["definitions_ordered"]]
    cells: Dict[Tuple[str, str], str] = {}
    for ri, (_key, comp) in enumerate(data["grid_rows_ordered"]):
        for ci, ch in enumerate(decompress(comp)):
            if ri != ci:
                cells[(defs[ri], defs[ci])] = ch
    return cells


def _hidden_pair(root: Path) -> Tuple[str, str]:
    return tuple(  # type: ignore[return-value]
        normalize_path(str(root / "src" / ".hidden" / "x" / n)) for n in ("h1.ts", "h2.ts")
    )


def _defined_paths(tracker: Path) -> set:
    return {normalize_path(p) for _, p in read_tracker_file_structured(str(tracker))["definitions_ordered"]}


def _write_hidden_tracker_with_x(root: Path) -> Path:
    """Write ``src/.hidden/x/x_module.md`` for its own files h1.ts <-> h2.ts = 'x'."""
    gmap = key_manager.load_global_key_map() or {}
    infos = [gmap[p] for p in _hidden_pair(root)]
    tracker = Path(get_mini_tracker_path(str(root / "src" / ".hidden" / "x")))
    assert tracker.name == "x_module.md"
    assert write_tracker_file(str(tracker), infos, [compress("ox"), compress("xo")], "k", "g", gmap)
    assert sorted(_grid(tracker).values()) == ["x", "x"]
    return tracker


# --- analyze-project level --------------------------------------------------------------


def test_analyze_project_ignores_unmapped_hidden_tracker(project: Path):
    pkg_tracker = Path(get_mini_tracker_path(str(project / "src" / "pkg")))
    hidden_dir = project / "src" / ".hidden" / "x"

    # Run 1: a NEW module is created if discovery will find it; a new dot-dir module is not.
    assert _analyze(project) == 0
    h1, h2 = _hidden_pair(project)
    assert pkg_tracker.exists()
    assert normalize_path(str(pkg_tracker)) in _mapped()
    assert not (hidden_dir / "x_module.md").exists()

    # A stale tracker in a dot-directory, outside tracker_map.json, full of 'x' cells.
    hidden_tracker = _write_hidden_tracker_with_x(project)
    before = hidden_tracker.read_bytes()
    # Precondition: pkg's tracker holds the hidden pair as a (non-'x') file<->file cell.
    assert _grid(pkg_tracker)[(h1, h2)] != "x"

    # Run 2: neither updated nor imported.
    assert _analyze(project) == 0
    assert hidden_tracker.read_bytes() == before
    assert normalize_path(str(hidden_tracker)) not in _mapped()
    pkg_cells = _grid(pkg_tracker)
    assert pkg_cells[(h1, h2)] != "x" and pkg_cells[(h2, h1)] != "x"
    # Module keys without an approved tracker are removed from the main tracker (and from
    # other modules' trackers) completely, like remove-key; file keys stay.
    hidden_dir = normalize_path(str(hidden_dir))
    assert hidden_dir not in _defined_paths(project / "cline_docs" / "module_relationship_tracker.md")
    pkg_defs = _defined_paths(pkg_tracker)
    assert {h1, h2} <= pkg_defs and hidden_dir not in pkg_defs
    assert normalize_path(str(project / "src" / "pkg")) in _defined_paths(
        project / "cline_docs" / "module_relationship_tracker.md"
    )


def test_owner_can_opt_a_dot_dir_tracker_in_via_the_map(project: Path):
    pkg_tracker = Path(get_mini_tracker_path(str(project / "src" / "pkg")))
    assert _analyze(project) == 0
    hidden_tracker = _write_hidden_tracker_with_x(project)
    h1, h2 = _hidden_pair(project)

    key_manager.save_tracker_map(_mapped() + [normalize_path(str(hidden_tracker))])
    assert _analyze(project) == 0

    # The end-of-run refresh keeps the owner's entry (discovery alone would drop it) ...
    assert normalize_path(str(hidden_tracker)) in _mapped()
    # ... and a mapped tracker is read like any other: its mutual 'x' reaches pkg.
    assert _grid(pkg_tracker)[(h1, h2)] == "x"
    # Its module key is approved as well.
    assert normalize_path(str(hidden_tracker.parent)) in _defined_paths(
        project / "cline_docs" / "module_relationship_tracker.md"
    )


def test_owner_can_have_a_missing_dot_dir_tracker_created_by_listing_it(project: Path):
    assert _analyze(project) == 0
    hidden_tracker = Path(get_mini_tracker_path(str(project / "src" / ".hidden" / "x")))
    assert not hidden_tracker.exists()

    key_manager.save_tracker_map(_mapped() + [normalize_path(str(hidden_tracker))])
    assert _analyze(project) == 0

    assert hidden_tracker.exists()
    assert normalize_path(str(hidden_tracker)) in _mapped()


# --- batch collector level --------------------------------------------------------------


def _ki(key: str, path: Path, parent: Path) -> KeyInfo:
    return KeyInfo(
        key_string=key,
        norm_path=normalize_path(str(path)),
        parent_path=normalize_path(str(parent)),
        tier=1,
        is_directory=False,
    )


def _collector_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, map_hidden: bool
) -> Tuple[TrackerBatchCollector, TrackerUpdate, KeyInfo, KeyInfo]:
    root = tmp_path / "cproj"
    pkg, hidden = root / "src" / "pkg", root / "src" / ".hidden" / "x"
    pkg.mkdir(parents=True)
    hidden.mkdir(parents=True)
    monkeypatch.setattr(tbc, "get_project_root", lambda: str(root))

    own = _ki("1A1", pkg / "own.py", pkg)
    foreign1 = _ki("2A1", hidden / "f1.py", hidden)
    foreign2 = _ki("2A2", hidden / "f2.py", hidden)
    gmap = {k.norm_path: k for k in (own, foreign1, foreign2)}

    # Home tracker of the foreign keys: unmapped dot-dir tracker with f1 <-> f2 = 'x'.
    hidden_tracker = Path(get_mini_tracker_path(str(hidden)))
    assert write_tracker_file(
        str(hidden_tracker), [foreign1, foreign2], [compress("ox"), compress("xo")], "k", "g", gmap
    )

    pkg_tracker = Path(get_mini_tracker_path(str(pkg)))
    infos = [own, foreign1, foreign2]
    # own depends on both foreign files, so pruning keeps them (and the batch cache is non-empty).
    grid = ["o<<", "pop", "ppo"]
    update = TrackerUpdate(
        output_file=str(pkg_tracker),
        tracker_type="mini",
        key_info_list=infos,
        grid_rows=[compress(r) for r in grid],
        last_key_edit="k",
        last_grid_edit="g",
        module_path=normalize_path(str(pkg)),
        path_to_key_info=gmap,
    )

    mapped = [normalize_path(str(pkg_tracker))]
    if map_hidden:
        mapped.append(normalize_path(str(hidden_tracker)))
    key_manager.save_tracker_map(mapped)
    pkg_tracker.write_text("placeholder", encoding="utf-8")  # existing + mapped

    collector = TrackerBatchCollector()
    collector.add(update)
    return collector, update, foreign1, foreign2


def _cell(update: TrackerUpdate, src: KeyInfo, tgt: KeyInfo) -> str:
    """Cell (src, tgt) of an in-memory update ('' if consolidation pruned either key)."""
    paths = [ki.norm_path for ki in update.key_info_list]
    if src.norm_path not in paths or tgt.norm_path not in paths:
        return ""
    return decompress(update.grid_rows[paths.index(src.norm_path)])[paths.index(tgt.norm_path)]


def test_collector_does_not_import_from_unmapped_home_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector, update, f1, f2 = _collector_setup(tmp_path, monkeypatch, map_hidden=False)

    collector._consolidate_grids()  # pyright: ignore[reportPrivateUsage]

    assert (f1.norm_path, f2.norm_path) not in collector.highest_dependency_cache
    assert _cell(update, f1, f2) == "p" and _cell(update, f2, f1) == "p"


def test_collector_imports_from_mapped_home_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Control for the test above: once the owner lists the tracker, its 'x' is consolidated.
    collector, update, f1, f2 = _collector_setup(tmp_path, monkeypatch, map_hidden=True)

    collector._consolidate_grids()  # pyright: ignore[reportPrivateUsage]

    assert collector.highest_dependency_cache[(f1.norm_path, f2.norm_path)][0] == "x"
    assert _cell(update, f1, f2) == "x"


def test_collector_ignores_update_for_existing_unmapped_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector, update, f1, f2 = _collector_setup(tmp_path, monkeypatch, map_hidden=False)
    hidden_tracker = Path(get_mini_tracker_path(str(Path(f1.norm_path).parent)))
    unmapped = TrackerUpdate(
        output_file=str(hidden_tracker),
        tracker_type="mini",
        key_info_list=[f1, f2],
        grid_rows=[compress("ox"), compress("xo")],
        last_key_edit="k",
        last_grid_edit="g",
        module_path=normalize_path(str(hidden_tracker.parent)),
        path_to_key_info={},
    )

    collector.add(unmapped)

    assert [u.output_file for u in collector.pending_updates] == [update.output_file]
    assert (f1.norm_path, f2.norm_path) not in collector.highest_dependency_cache


# --- main tracker aggregation ------------------------------------------------------------


def _main_aggregation(tmp_path: Path, map_hidden: bool) -> Dict[str, List[Tuple[str, str]]]:
    """m1 (mapped) a.py -> c.py '<'; .hid/m2 (hidden) b.py -> c.py 'x'; m3 (mapped) owns c.py."""
    root = tmp_path / "mproj"
    dirs = {"m1": root / "src" / "m1", "m2": root / "src" / ".hid" / "m2", "m3": root / "src" / "m3"}
    files = {n: d / f"{n}.py" for n, d in dirs.items()}
    kis = {n: _ki(f"1A{i}", files[n], dirs[n]) for i, n in enumerate(dirs, 1)}
    gmap = {k.norm_path: k for k in kis.values()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    def write(module: str, own: KeyInfo, grid: List[str]) -> Path:
        tracker = Path(get_mini_tracker_path(str(dirs[module])))
        assert write_tracker_file(
            str(tracker), [own, kis["m3"]], [compress(r) for r in grid], "k", "g", gmap
        )
        return tracker

    t1 = write("m1", kis["m1"], ["o<", "po"])
    # Hidden module: its own file relates to m3's file with a stale 'x'.
    t2 = write("m2", kis["m2"], ["ox", "xo"])
    key_manager.save_tracker_map(
        [normalize_path(str(t1))] + ([normalize_path(str(t2))] if map_hidden else [])
    )
    module_kis = {
        normalize_path(str(d)): KeyInfo(
            key_string=f"2A{i}", norm_path=normalize_path(str(d)), parent_path=normalize_path(str(d.parent)),
            tier=2, is_directory=True,
        )
        for i, d in enumerate(dirs.values(), 1)
    }
    file_to_module = {kis[n].norm_path: normalize_path(str(dirs[n])) for n in dirs}
    return aggregate_dependencies_contextual(
        normalize_path(str(root)), {**gmap, **module_kis}, module_kis, file_to_module
    )


def test_main_tracker_aggregation_ignores_unmapped_mini_tracker(tmp_path: Path):
    result = _main_aggregation(tmp_path, map_hidden=False)

    m1, m2, m3 = (normalize_path(str(tmp_path / "mproj" / "src" / p)) for p in ("m1", ".hid/m2", "m3"))
    assert (m3, "<") in result[m1]
    assert not result.get(m2)
    assert all(tgt != m2 for deps in result.values() for tgt, _ in deps)


def test_main_tracker_aggregation_reads_mapped_mini_tracker(tmp_path: Path):
    # Control: once the owner lists the hidden module's tracker, its relation is aggregated.
    result = _main_aggregation(tmp_path, map_hidden=True)

    m2, m3 = (normalize_path(str(tmp_path / "mproj" / "src" / p)) for p in (".hid/m2", "m3"))
    assert (m3, "x") in result[m2]


def test_verified_n_is_not_overwritten_by_ast_links_or_consolidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "nproj"
    pkg = root / "src" / "pkg"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(tbc, "get_project_root", lambda: str(root))
    a, b, c = (_ki(f"1A{i}", pkg / f"{n}.py", pkg) for i, n in enumerate("abc", 1))

    def update(force: bool) -> TrackerUpdate:
        # a -> b is a verified 'n'; a -> c is '<'.
        return TrackerUpdate(
            output_file=str(pkg / "pkg_module.md"), tracker_type="mini",
            key_info_list=[a, b, c], grid_rows=[compress("on<"), compress("pop"), compress("ppo")],
            last_key_edit="k", last_grid_edit="g", module_path=normalize_path(str(pkg)),
            path_to_key_info={k.norm_path: k for k in (a, b, c)}, force_apply_suggestions=force,
        )

    link = {"source_path": a.norm_path, "target_path": b.norm_path, "char": "<"}
    (_state_dir() / "ast_verified_links.json").write_text(json.dumps([link]), encoding="utf-8")

    for force, expected in ((False, "n"), (True, "<")):
        u = update(force)
        collector = TrackerBatchCollector()
        collector.add(u)
        # Another tracker in the batch claims a -> b is '<' (equal priority to 'n').
        collector.highest_dependency_cache[(a.norm_path, b.norm_path)] = ("<", {"other"})
        collector._consolidate_grids()  # pyright: ignore[reportPrivateUsage]
        assert _cell(u, a, b) == expected, f"force={force}"


# --- AST links --------------------------------------------------------------------------


def _state_dir() -> Path:
    return Path(os.path.dirname(os.path.abspath(key_manager.__file__))) / "state"


LINK = {
    "source_path": "H:/proj/src/a.py",
    "target_path": "H:/proj/src/b.py",
    "char": "<",
    "reason": "ExplicitImport/b",
}


def test_load_ast_links_reads_core_state_file():
    (_state_dir() / "ast_verified_links.json").write_text(json.dumps([LINK]), encoding="utf-8")

    assert TrackerBatchCollector()._load_ast_links() == [LINK]  # pyright: ignore[reportPrivateUsage]


def test_load_ast_links_accepts_legacy_links_object_and_missing_file():
    collector = TrackerBatchCollector()
    assert collector._load_ast_links() == []  # pyright: ignore[reportPrivateUsage]
    (_state_dir() / "ast_verified_links.json").write_text(
        json.dumps({"links": [LINK]}), encoding="utf-8"
    )
    assert collector._load_ast_links() == [LINK]  # pyright: ignore[reportPrivateUsage]


def test_central_ast_override_applies_during_consolidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "aproj"
    pkg = root / "src" / "pkg"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(tbc, "get_project_root", lambda: str(root))
    a, b, c = (_ki(f"1A{i}", pkg / f"{n}.py", pkg) for i, n in enumerate("abc", 1))
    tracker = Path(get_mini_tracker_path(str(pkg)))
    update = TrackerUpdate(
        output_file=str(tracker),
        tracker_type="mini",
        key_info_list=[a, b, c],
        # a -> c is known ('<') so the batch cache is non-empty; a -> b is still a placeholder.
        grid_rows=[compress("op<"), compress("pop"), compress("ppo")],
        last_key_edit="k",
        last_grid_edit="g",
        module_path=normalize_path(str(pkg)),
        path_to_key_info={k.norm_path: k for k in (a, b, c)},
    )
    link = {"source_path": a.norm_path, "target_path": b.norm_path, "char": "<"}
    (_state_dir() / "ast_verified_links.json").write_text(json.dumps([link]), encoding="utf-8")
    collector = TrackerBatchCollector()
    collector.add(update)

    collector._consolidate_grids()  # pyright: ignore[reportPrivateUsage]

    assert _cell(update, a, b) == "<"
    assert update.ast_overrides_applied_count == 1


# --- tracker_utils helpers --------------------------------------------------------------


def _fake_config(excluded_dirs: List[str]) -> Any:
    return SimpleNamespace(
        get_code_root_directories=lambda: ["src"],
        get_excluded_dirs=lambda: excluded_dirs,
        config={},
    )


def test_new_tracker_is_only_discoverable_outside_dot_dirs_and_exclusions(tmp_path: Path):
    root = str(tmp_path)
    cfg = _fake_config(["node_modules"])

    def path(*parts: str) -> str:
        return normalize_path(os.path.join(root, *parts))

    assert is_discoverable_mini_tracker_path(path("src", "pkg", "pkg_module.md"), cfg, root)
    assert not is_discoverable_mini_tracker_path(path("src", ".hidden", "x", "x_module.md"), cfg, root)
    assert not is_discoverable_mini_tracker_path(path("src", "pkg", ".pkg_module.md"), cfg, root)
    assert not is_discoverable_mini_tracker_path(path("src", "node_modules", "m", "m_module.md"), cfg, root)
    assert not is_discoverable_mini_tracker_path(path("other", "o", "o_module.md"), cfg, root)


def test_map_refresh_keeps_owner_opt_ins_and_drops_vanished_entries(tmp_path: Path):
    root = tmp_path / "rproj"
    for rel in ("src/pkg/pkg_module.md", "src/.hidden/x/x_module.md"):
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("tracker", encoding="utf-8")
    pkg = normalize_path(str(root / "src/pkg/pkg_module.md"))
    hidden = normalize_path(str(root / "src/.hidden/x/x_module.md"))
    gone = normalize_path(str(root / "src/gone/gone_module.md"))
    key_manager.save_tracker_map([hidden, gone])
    cfg = SimpleNamespace(
        get_code_root_directories=lambda: ["src"],
        get_excluded_dirs=lambda: [],
        config={},
        get_path=lambda name, default=None: None,
    )

    refreshed = collect_tracker_map_paths(cfg, str(root))  # type: ignore[arg-type]

    assert refreshed == sorted([pkg, hidden])
