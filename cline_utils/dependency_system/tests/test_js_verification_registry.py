"""
JS/TS symbol verification is resolved once per (origin module, exported name) per run:
later files that import the same symbol reuse the registry instead of re-scanning the
target module's exports.
"""

from unittest.mock import patch

from cline_utils.dependency_system.analysis import dependency_suggester as ds
from cline_utils.dependency_system.utils.path_utils import normalize_path


def _analysis(names):
    return {
        "imports": [{"path": "./agentMode", "symbols": [{"type": "named", "name": n} for n in names]}],
        "calls": [{"target_name": n} for n in names] * 3,
        "type_references": [{"type_name_str": n} for n in names],
        "exports": [],
    }


def test_verification_runs_once_per_origin_and_name():
    ds._reset_js_export_registry()  # pyright: ignore[reportPrivateUsage]
    origin = normalize_path("/proj/types/agentMode.ts")
    symbol_map = {origin: {"exports": [{"name": "AgentMode"}, {"name": "x", "alias": "Renamed"}]}}

    with patch.object(ds, "_resolve_js_import_path", return_value=origin):
        built = []
        real = ds._js_verified_export  # pyright: ignore[reportPrivateUsage]

        def spy(sm, path, name):
            built.append((path, name))
            return real(sm, path, name)

        with patch.object(ds, "_js_verified_export", side_effect=spy):
            for i in range(5):
                src = normalize_path(f"/proj/file{i}.ts")
                _, links = ds._identify_javascript_structural_dependencies(  # pyright: ignore[reportPrivateUsage]
                    src, _analysis(["AgentMode", "Renamed", "Missing"]), {}, normalize_path("/proj"),
                    symbol_map, None, tracked_paths_globally={origin},
                )
                assert {l["reason"].split("/", 1)[1] for l in links} == {"AgentMode", "Renamed"}

    # Within a file each distinct name is checked once (calls repeat 3x + a type ref).
    assert len(built) == 5 * 3
    # Across files the registry holds exactly one entry per (origin, name).
    assert ds._js_export_verified == {  # pyright: ignore[reportPrivateUsage]
        (origin, "AgentMode"): True, (origin, "Renamed"): True, (origin, "Missing"): False,
    }


def test_registry_resets_for_a_different_symbol_map():
    ds._reset_js_export_registry()  # pyright: ignore[reportPrivateUsage]
    origin = "/p/a.ts"
    assert ds._js_verified_export({origin: {"exports": [{"name": "A"}]}}, origin, "A") is True  # pyright: ignore[reportPrivateUsage]
    assert ds._js_verified_export({origin: {"exports": []}}, origin, "A") is False  # pyright: ignore[reportPrivateUsage]
