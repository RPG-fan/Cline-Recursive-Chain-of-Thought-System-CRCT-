"""
Upfront JS/TS passes: prepare_js_structural_index() resolves every import specifier
and indexes every module's exports (pass 1), then verifies every file's structural
links (pass 2) before the parallel workers start. A worker only looks its result up:
no resolution and no export verification during the per-file loop. Logging is one
summary line per file. Results are identical to the on-demand computation.
"""

import logging
import os
from typing import Any, Dict

import pytest

from cline_utils.dependency_system.analysis import dependency_suggester as ds
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.cache_manager import clear_all_caches
from cline_utils.dependency_system.utils.path_utils import normalize_path


@pytest.fixture
def project(tmp_path):
    ds.clear_caches()
    root = normalize_path(str(tmp_path))
    names = ["types/agentMode.ts", "lib/util.ts", "lib/index.ts"] + [f"app/f{i}.ts" for i in range(4)]
    paths = {n: normalize_path(os.path.join(root, n)) for n in names}
    for p in paths.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()
    infos = {
        p: KeyInfo(f"1A{i + 1}", p, normalize_path(os.path.dirname(p)), 1, False)
        for i, p in enumerate(paths.values())
    }
    symbol_map = {
        paths["types/agentMode.ts"]: {"exports": [{"name": "AgentMode"}, {"name": "x", "alias": "Renamed"}]},
        paths["lib/util.ts"]: {"exports": [{"name": "helper"}, {"name": "default"}]},
        paths["lib/index.ts"]: {"exports": [{"from": "./util"}]},
    }

    def app_file() -> Dict[str, Any]:
        return {
            "imports": [
                {"path": "../types/agentMode", "symbols": [
                    {"type": "named", "name": "AgentMode"}, {"type": "named", "name": "Renamed"},
                    {"type": "named", "name": "Missing"}]},
                {"path": "../lib/util", "symbols": [{"type": "namespace", "name": "*", "alias": "U"},
                                                    {"type": "default", "name": "util"}]},
                {"path": "react", "symbols": [{"type": "named", "name": "useState"}]},
            ],
            "calls": [{"target_name": n} for n in ("AgentMode", "Missing", "U.helper", "U.nope", "util", "useState")] * 3,
            "type_references": [{"type_name_str": "Renamed"}],
            "exports": [],
        }

    analyses: Dict[str, Any] = {paths[f"app/f{i}.ts"]: app_file() for i in range(4)}
    analyses[paths["lib/index.ts"]] = {"imports": [], "calls": [], "type_references": [],
                                       "exports": [{"from": "./util"}]}
    analyses[paths["types/agentMode.ts"]] = {"imports": [], "calls": [], "type_references": [], "exports": []}
    yield root, paths, infos, symbol_map, analyses
    ds.clear_caches()
    clear_all_caches()


def _expected(root, infos, symbol_map, analyses):
    """On-demand results, computed before anything is prepared."""
    ds.clear_caches()
    out = {
        p: ds._compute_js_structural_dependencies(  # pyright: ignore[reportPrivateUsage]
            p, a, infos, root, symbol_map, ds._tsconfig_for_file(p, root), set(infos)  # pyright: ignore[reportPrivateUsage]
        )
        for p, a in analyses.items()
    }
    ds.clear_caches()
    return out


def test_workers_only_look_up_precomputed_results(project, monkeypatch) -> None:
    root, paths, infos, symbol_map, analyses = project
    expected = _expected(root, infos, symbol_map, analyses)

    counts = ds.prepare_js_structural_index(analyses, infos, root, symbol_map, set(infos))
    assert counts["files"] == len(analyses)

    def boom(*a, **k):
        raise AssertionError("worker re-resolved or re-verified after the upfront passes")

    monkeypatch.setattr(ds, "_resolve_js_import_path", boom)
    monkeypatch.setattr(ds, "_js_verified_export", boom)

    for p, a in analyses.items():
        suggestions, links = ds._identify_javascript_structural_dependencies(  # pyright: ignore[reportPrivateUsage]
            p, a, infos, root, symbol_map, None, tracked_paths_globally=set(infos)
        )
        assert sorted(suggestions) == sorted(expected[p][0])
        assert links == expected[p][1]

    app = analyses[paths["app/f0.ts"]]
    reasons = {l["reason"] for l in ds._identify_javascript_structural_dependencies(  # pyright: ignore[reportPrivateUsage]
        paths["app/f0.ts"], app, infos, root, symbol_map, None)[1]}
    assert reasons == {"JSCall/AgentMode", "JSCall/U.helper", "JSCall/util", "JSTypeRef/Renamed"}
    assert ds._identify_javascript_structural_dependencies(  # pyright: ignore[reportPrivateUsage]
        paths["lib/index.ts"], analyses[paths["lib/index.ts"]], infos, root, symbol_map, None
    )[0] == [(paths["lib/util.ts"], "<")]


def test_pass_one_fills_resolution_and_export_tables(project) -> None:
    root, paths, infos, symbol_map, analyses = project
    ds.prepare_js_structural_index(analyses, infos, root, symbol_map, set(infos))

    resolved = {k[0]: v for k, v in ds._js_import_resolution_ledger.items()}  # pyright: ignore[reportPrivateUsage]
    assert resolved["../types/agentMode"] == paths["types/agentMode.ts"]
    assert resolved["../lib/util"] == paths["lib/util.ts"]
    assert resolved["./util"] == paths["lib/util.ts"]
    assert resolved["react"] is None  # packages stay unresolved, and the miss is stored
    assert set(ds._js_export_names) == set(symbol_map)  # pyright: ignore[reportPrivateUsage]


def test_a_different_analysis_object_is_computed_on_demand(project) -> None:
    root, paths, infos, symbol_map, analyses = project
    ds.prepare_js_structural_index(analyses, infos, root, symbol_map, set(infos))
    fresh = dict(analyses[paths["app/f0.ts"]], calls=[], type_references=[])

    suggestions, links = ds._identify_javascript_structural_dependencies(  # pyright: ignore[reportPrivateUsage]
        paths["app/f0.ts"], fresh, infos, root, symbol_map, None, set(infos)
    )
    assert suggestions == [] and links == []


def test_no_per_item_or_per_file_lines_from_the_js_passes(project, caplog) -> None:
    """The JS passes log one timing line each; per-file results go into the unified
    'Suggest <file> [...]' line written by suggest_dependencies."""
    root, paths, infos, symbol_map, analyses = project
    with caplog.at_level(logging.DEBUG, logger=ds.logger.name):
        ds.prepare_js_structural_index(analyses, infos, root, symbol_map, set(infos))
        for p, a in analyses.items():
            ds._identify_javascript_structural_dependencies(p, a, infos, root, symbol_map, None)  # pyright: ignore[reportPrivateUsage]

    messages = [r.getMessage() for r in caplog.records if r.name == ds.logger.name]
    assert not [m for m in messages if m.startswith(("JSVerify", "JSStructural"))]
    assert [m for m in messages if m.startswith("JS pass ")] and len(
        [m for m in messages if m.startswith("JS pass ")]
    ) == 2
