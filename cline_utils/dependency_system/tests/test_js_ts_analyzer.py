"""
Regression tests for the JS/TS/TSX tree-sitter analyzers.

P0: file-type dispatch, grammar-correct queries with per-query/per-section isolation,
distinct call records, re-export handling in the suggester, JSON-safe `analyze-file`.
P1: import symbols and edge kinds, module-scope functions (arrow consts), class
members, heritage, JSX refs, CommonJS / `export =` / `as default` exports, and the
suggester's symbol verification built on them.

Fixtures live in tests/fixtures/jsts/. CRCT excludes `cline_utils/` and `tests/` from
analysis, so each test copies the fixture into tmp_path before analyzing it.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest

from cline_utils.dependency_system.analysis import dependency_analyzer as da
from cline_utils.dependency_system.analysis import dependency_suggester as ds
from cline_utils.dependency_system.utils.path_utils import get_file_type, normalize_path

FIXTURES = Path(__file__).parent / "fixtures" / "jsts"


def _analyze_fixture(tmp_path: Path, name: str) -> Dict[str, Any]:
    target = tmp_path / name
    shutil.copyfile(FIXTURES / name, target)
    return da.analyze_file(str(target))


def _analyze_source(tmp_path: Path, name: str, source: str) -> Dict[str, Any]:
    target = tmp_path / name
    target.write_text(source, encoding="utf-8")
    return da.analyze_file(str(target))


def _extract(name: str, source: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs the JS-family analyzer directly. analyze_file() costs ~10 s per call here (its
    project-wide exclusion scan reruns after conftest resets ConfigManager), so only the
    dispatch, consolidation and CLI tests go through it.
    """
    content = source if source is not None else (FIXTURES / name).read_text(encoding="utf-8")
    kind = {".ts": "ts", ".tsx": "tsx", ".js": "js"}[Path(name).suffix]
    result: Dict[str, Any] = {"file_path": name}
    da._analyze_js_family_file_ts(name, content, result, kind)  # pyright: ignore[reportPrivateUsage]
    return result


def _export_names(result: Dict[str, Any]) -> Set[str]:
    return {e.get("alias") or e["name"] for e in result["exports"] if e.get("name") and e["name"] != "default"}


@pytest.mark.parametrize(
    "path,expected",
    [
        ("a.ts", "ts"), ("a.mts", "ts"), ("a.cts", "ts"), ("a.tsx", "tsx"),
        ("a.js", "js"), ("a.jsx", "js"), ("a.mjs", "js"), ("a.cjs", "js"),
    ],
)
def test_js_family_file_types(path: str, expected: str) -> None:
    assert get_file_type(path) == expected


def test_typescript_file_uses_typescript_analyzer(tmp_path: Path) -> None:
    result = _analyze_fixture(tmp_path, "probe.ts")

    assert result["file_type"] == "ts"
    assert result["analysis_kind"] == "ts"
    assert "error" not in result
    assert not result.get("analysis_warnings")
    kinds = {c["name"]: c["kind"] for c in result["classes"]}
    assert kinds == {
        "Shape": "interface",
        "Id": "type_alias",
        "Color": "enum",
        "Base": "class",  # abstract
        "Derived": "class",
        "DefaultCls": "class",
        "NS": "namespace",
    }
    assert {i["path"] for i in result["imports"]} >= {"./types", "./bar", "./ns.js", "./side-effect"}
    assert {t["type_name_str"] for t in result["type_references"]} >= {"Foo", "Id"}


def test_typescript_exports(tmp_path: Path) -> None:
    result = _extract("probe.ts")
    names = _export_names(result)

    assert names >= {"Shape", "Id", "Color", "ARROW", "CONST_VAL", "plain", "overload", "Base", "Derived", "a", "c", "T1"}
    # Enum members, the un-aliased name of `b as c`, and exports nested in
    # `namespace` / `declare module` blocks are not module exports.
    assert not names & {"Red", "Green", "b", "inner", "x"}
    assert {"name": "default", "alias": "DefaultCls"} in [
        {k: e[k] for k in ("name", "alias") if k in e} for e in result["exports"]
    ]
    reexports = {e["from"]: (e["kind"], e.get("name")) for e in result["exports"] if "from" in e}
    assert reexports == {
        "./reexp": ("star", "*"),
        "./reexp2": ("namespace", "ns2"),
        "./reexp3": ("named", None),
        "./types": ("named", None),
    }


def test_tsx_file_uses_tsx_analyzer(tmp_path: Path) -> None:
    result = _analyze_fixture(tmp_path, "probe.tsx")

    assert result["file_type"] == "tsx"
    assert result["analysis_kind"] == "tsx"
    assert "error" not in result
    assert {f["name"] for f in result["functions"]} == {"Comp", "Page"}
    assert _export_names(result) == {"Comp"}


def test_javascript_file(tmp_path: Path) -> None:
    result = _analyze_fixture(tmp_path, "probe.js")

    assert result["file_type"] == "js"
    assert "error" not in result
    assert [(c["name"], c["kind"]) for c in result["classes"]] == [("K", "class")]
    assert {i["path"] for i in result["imports"]} >= {"fs", "./localmod"}
    # require() is an import, not a call.
    assert "require" not in {c["target_name"] for c in result["calls"]}


def test_invalid_query_is_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A pattern that is impossible for the TS grammar (classes are named by type_identifier).
    monkeypatch.setattr(da, "_JS_EXPORTS_QUERY", "(class_declaration name: (identifier) @export.decl)")
    result = _extract("probe.ts")

    assert "error" not in result
    assert result["analysis_warnings"] and "Query skipped" in result["analysis_warnings"][0]
    # Every other section still runs.
    assert result["imports"] and result["classes"] and result["type_references"]
    assert "Shape" in _export_names(result)  # from the separate TS-only exports query


def test_failing_section_is_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(da, "_collect_js_module_declarations", boom)
    result = _extract("probe.ts")

    assert "error" not in result
    assert any("declarations section failed" in w for w in result["analysis_warnings"])
    assert result["functions"] == [] and result["imports"] and result["exports"]


def test_calls_are_distinct_records(tmp_path: Path) -> None:
    source = (
        "const m = require('./m');\n"
        "foo();\n"
        "bar();\n"
        "obj.baz();\n"
        "foo();\n"
        "make().chain();\n"
    )
    result = _analyze_source(tmp_path, "calls.ts", source)
    calls = {c["target_name"]: c for c in result["calls"]}

    assert set(calls) == {"foo", "bar", "obj.baz", "make", "chain"}
    assert calls["foo"]["line"] == [2, 5]
    assert calls["obj.baz"]["potential_source"] == "obj"
    assert calls["chain"]["potential_source"] is None


def test_named_reexport_without_name_does_not_crash_suggester(monkeypatch: pytest.MonkeyPatch) -> None:
    source = normalize_path("/proj/src/index.ts")
    target = normalize_path("/proj/src/x.ts")
    monkeypatch.setattr(ds, "_resolve_js_import_path", lambda *args, **kwargs: target)
    analysis = {"exports": [{"from": "./x", "kind": "named", "line": 1}], "imports": [], "calls": [], "type_references": []}

    suggestions, links = ds._identify_javascript_structural_dependencies(  # pyright: ignore[reportPrivateUsage]
        source, analysis, {}, normalize_path("/proj"), {}, None, tracked_paths_globally={target}
    )

    assert suggestions == [(target, "<")]
    assert links[0]["reason"] == "JSReExport/./x"


def test_analyze_file_command_prints_json_for_typescript(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cline_utils.dependency_system.dependency_processor import command_handler_analyze_file

    target = tmp_path / "probe.ts"
    shutil.copyfile(FIXTURES / "probe.ts", target)

    assert command_handler_analyze_file(argparse.Namespace(file_path=str(target), output=None)) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["file_type"] == "ts"
    assert "_ts_tree" not in printed


# --------------------------------------------------------------------------- P1 extraction


def _symbols_by_path(result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for imp in result["imports"]:
        out.setdefault(imp["path"], []).extend(imp["symbols"])
    return out


def test_typescript_import_symbols_and_kinds(tmp_path: Path) -> None:
    result = _extract("probe.ts")
    kinds = {(i["path"], i["kind"]) for i in result["imports"]}
    symbols = _symbols_by_path(result)

    assert kinds == {
        ("./types", "type"), ("./bar", "static"), ("./ns.js", "static"),
        ("./side-effect", "side-effect"), ("./reexp", "reexport"), ("./reexp2", "reexport"),
        ("./reexp3", "reexport"), ("./types", "reexport"), ("./lazy", "dynamic"),
    }
    assert symbols["./bar"] == [
        {"type": "default", "name": "Bar"},
        {"type": "named", "name": "baz", "alias": "qux"},
        {"type": "named", "name": "Quux", "type_only": True},
    ]
    assert symbols["./ns.js"] == [{"type": "namespace", "name": "ns"}]
    assert symbols["./types"] == [{"type": "named", "name": "Foo", "type_only": True}]
    # Re-export and dynamic edges bind no local names.
    assert symbols["./reexp3"] == [] and symbols["./lazy"] == []


def test_require_bindings_and_type_position_imports(tmp_path: Path) -> None:
    source = (
        "const fs = require('fs');\n"
        "const { a, b: c } = require('./m');\n"
        "function f() { return require('./lazy'); }\n"
        "let t: import('./types').T | null = null;\n"
        "import x = require('./legacy');\n"
    )
    result = _extract("req.ts", source)
    by_path = {i["path"]: i for i in result["imports"]}

    assert by_path["fs"]["symbols"] == [{"type": "namespace", "name": "fs"}]
    assert by_path["./m"]["symbols"] == [
        {"type": "named", "name": "a"},
        {"type": "named", "name": "b", "alias": "c"},
    ]
    assert by_path["./lazy"]["kind"] == "require" and by_path["./lazy"]["symbols"] == []
    assert by_path["./types"]["kind"] == "type"
    assert by_path["./legacy"]["symbols"] == [{"type": "namespace", "name": "x"}]


def test_module_scope_functions_only(tmp_path: Path) -> None:
    result = _extract("probe.ts")
    # Arrow consts count; the namespace member `inner` and the plain constant do not.
    assert {f["name"] for f in result["functions"]} == {"ARROW", "plain", "overload", "local", "useAll"}


def test_class_members_and_heritage(tmp_path: Path) -> None:
    result = _extract("probe.ts")
    classes = {c["name"]: c for c in result["classes"]}
    base = classes["Base"]

    assert base["abstract"] is True
    assert {m["name"]: m.get("modifiers") for m in base["methods"]} == {
        "area": ["abstract"], "helper": ["protected"], "make": ["static"], "name": ["get"],
    }
    assert base["properties"] == [{"name": "field", "line": 24, "end_line": 24, "modifiers": ["private"]}]
    assert [m["name"] for m in classes["Derived"]["methods"]] == ["area", "load"]
    assert {(i["class_name"], i["base_class_name"], i["kind"]) for i in result["inheritance"]} == {
        ("Base", "Shape", "implements"), ("Derived", "Base", "extends"),
    }
    assert {(t["type_name_str"], t["context"]) for t in result["type_references"]} >= {
        ("Shape", "implements"), ("Base", "extends"),
    }


def test_constructor_and_private_fields(tmp_path: Path) -> None:
    source = "class A { #secret = 1; constructor() {} run() {} }\n"
    result = _extract("priv.js", source)
    cls = result["classes"][0]

    assert [m["name"] for m in cls["methods"]] == ["run"]
    assert [p["name"] for p in cls["properties"]] == ["#secret"]


def test_jsx_refs_are_kept_apart_from_calls(tmp_path: Path) -> None:
    source = (
        "import { Btn } from './btn';\n"
        "export const Page = () => <Ctx.Provider><Btn/><div/></Ctx.Provider>;\n"
        "Btn();\n"
    )
    result = _analyze_source(tmp_path, "jsx.tsx", source)
    records = {(c["target_name"], c.get("kind")) for c in result["calls"]}

    assert records == {("Btn", "jsx"), ("Ctx.Provider", "jsx"), ("Btn", None)}


def test_commonjs_and_special_exports(tmp_path: Path) -> None:
    js = _extract("probe.js")
    assert _export_names(js) == {"alpha", "beta", "gamma"}
    assert any(e["name"] == "default" for e in js["exports"])

    aliased = _extract("alias.js", "const a = 1, b = 2;\nexport { a as default, b };\n")
    assert {"name": "a", "alias": "default"} in [
        {k: e[k] for k in ("name", "alias") if k in e} for e in aliased["exports"]
    ]

    assign = _extract("assign.ts", "class Foo {}\nexport = Foo;\n")
    assert [(e["name"], e.get("alias")) for e in assign["exports"]] == [("default", "Foo")]


def test_records_carry_no_scope_data_for_populate_comments(tmp_path: Path) -> None:
    """populate_comments writes CONNECTION_MAPs into source files only for symbols with
    per-symbol scope data; JS/TS records must not carry any."""
    from cline_utils.dependency_system.utils.populate_comments import (
        _collect_symbol_references,  # pyright: ignore[reportPrivateUsage]
    )

    for name in ("probe.ts", "probe.tsx", "probe.js"):
        result = _extract(name)
        for func in result["functions"]:
            assert _collect_symbol_references(func, "function") is None
        for cls in result["classes"]:
            assert _collect_symbol_references(cls, "class") is None
            for method in cls.get("methods", []):
                assert _collect_symbol_references(method, "function") is None


def test_suggester_verifies_imported_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib = _extract("lib.ts", "export function helper() {}\nexport const Widget = () => null;\n")
    app = _extract(
        "app.tsx",
        "import { helper, Widget as W } from './lib';\nimport * as libns from './lib';\n"
        "helper();\nlibns.helper();\nexport const App = () => <W/>;\n",
    )
    lib_path = normalize_path(str(tmp_path / "lib.ts"))
    app_path = normalize_path(str(tmp_path / "app.tsx"))
    monkeypatch.setattr(ds, "_resolve_js_import_path", lambda *args, **kwargs: lib_path)

    _, links = ds._identify_javascript_structural_dependencies(  # pyright: ignore[reportPrivateUsage]
        app_path, app, {}, normalize_path(str(tmp_path)), {lib_path: lib}, None,
        tracked_paths_globally={lib_path, app_path},
    )

    assert {l["reason"] for l in links} >= {"JSCall/helper", "JSCall/libns.helper", "JSCall/W"}
    assert all(l["target_path"] == lib_path for l in links)
