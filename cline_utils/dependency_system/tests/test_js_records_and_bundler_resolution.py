"""
JS/TS gaps closed after Phase 4:
- records carry `end_line`, `signature` (parameters + return type, as SES prints it
  after the name), the JSDoc first line as `docstring`, per-symbol `decorators` and the
  file-level `decorators_used`; bare decorators verify as JSDecorator links;
- bundler and workspace resolution (analysis.js_package_resolution): Vite
  `resolve.alias` read statically, SvelteKit `$lib` / `kit.alias`, workspace packages
  through package.json `exports` / `main`;
- Svelte <script> imports use the same resolver, tsconfig included.
The fixture tree (fixtures/jsts_bundler) is copied to tmp_path and used as the project
root. No model is loaded.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Set

import pytest

from cline_utils.dependency_system.analysis import dependency_analyzer as da
from cline_utils.dependency_system.analysis import dependency_suggester as ds
from cline_utils.dependency_system.analysis import js_package_resolution as jpr
from cline_utils.dependency_system.analysis import structural_engine as se
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.path_utils import normalize_path as N

FIXTURE_TREE = Path(__file__).parent / "fixtures" / "jsts_bundler"


def _analyze(source: str, kind: str = "ts") -> Dict[str, Any]:
    result: Dict[str, Any] = {"file_path": f"x.{kind}"}
    da._analyze_js_family_file_ts(f"x.{kind}", source, result, kind)  # pyright: ignore[reportPrivateUsage]
    result.pop("_ts_tree", None)
    assert not result.get("analysis_warnings")
    return result


# --- records ---------------------------------------------------------------------------

_DECORATED = '''
/** Makes a thing.
 * @param a the a
 */
@Component({ selector: "x" })
export class Foo extends Base {
  /** The name. */
  @Input() name: string;
  /** Loads it. */
  @HostListener("click")
  async load(a: number, b?: string): Promise<void> {}
  @ns.dec
  static s() {}
}

/** Plain fn */
export function plain<T>(a: T): Id {
  return a.id;
}
/** @internal */
export const arrow = async x => x;
// not JSDoc
@dec export class Bar {}
export @dec2 class Baz {}

/** An interface. */
export interface Shape { area(): number }
'''


def test_records_carry_end_line_signature_doc_and_decorators() -> None:
    r = _analyze(_DECORATED)
    fns = {f["name"]: f for f in r["functions"]}
    assert fns["plain"] == {
        "name": "plain", "line": 17, "end_line": 19, "signature": "<T>(a: T): Id", "docstring": "Plain fn",
    }
    assert fns["arrow"]["signature"] == "(x)" and "docstring" not in fns["arrow"]  # tags only

    classes = {c["name"]: c for c in r["classes"]}
    foo = classes["Foo"]
    assert (foo["line"], foo["end_line"], foo["docstring"], foo["decorators"]) == (
        6, 14, "Makes a thing.", ["Component"]
    )
    methods = {m["name"]: m for m in foo["methods"]}
    assert methods["load"]["signature"] == "(a: number, b?: string): Promise<void>"
    assert methods["load"]["docstring"] == "Loads it."
    assert methods["load"]["decorators"] == ["HostListener"]
    assert methods["s"]["decorators"] == ["ns.dec"] and "docstring" not in methods["s"]
    assert foo["properties"][0]["decorators"] == ["Input"]
    assert foo["properties"][0]["docstring"] == "The name."
    assert classes["Bar"]["decorators"] == ["dec"] and "docstring" not in classes["Bar"]
    assert classes["Baz"]["decorators"] == ["dec2"]
    assert classes["Shape"]["docstring"] == "An interface."

    assert [(d["name"], d["target_type"], d["target_name"]) for d in r["decorators_used"]] == [
        ("Component", "class", "Foo"),
        ("Input", "property", "name"),
        ("HostListener", "method", "load"),
        ("ns.dec", "method", "s"),
        ("dec", "class", "Bar"),
        ("dec2", "class", "Baz"),
    ]


def test_plain_javascript_decorators_and_jsdoc() -> None:
    r = _analyze("/** JS class. */\n@dec\nexport class A {\n  /** m doc */\n  @log\n  m(a, b = 1) {}\n}\n", "js")
    cls = r["classes"][0]
    assert (cls["docstring"], cls["decorators"]) == ("JS class.", ["dec"])
    assert cls["methods"][0] == {
        "name": "m", "line": 6, "end_line": 6, "signature": "(a, b = 1)", "docstring": "m doc", "decorators": ["log"],
    }


def test_a_comment_separated_by_a_blank_line_is_not_jsdoc() -> None:
    r = _analyze("/** Stray. */\n\nexport function f() {}\n")
    assert "docstring" not in r["functions"][0]


def test_new_record_fields_add_no_populate_comments_scope_data() -> None:
    from cline_utils.dependency_system.utils.populate_comments import (
        _collect_symbol_references,  # pyright: ignore[reportPrivateUsage]
    )

    r = _analyze(_DECORATED)
    assert all(_collect_symbol_references(f, "function") is None for f in r["functions"])
    assert all(
        _collect_symbol_references(c, "class") is None
        for c in r["classes"]
        if c["name"] != "Foo"  # heritage-free classes
    )


def test_schema_version_invalidates_cached_analyses() -> None:
    assert da.ANALYSIS_SCHEMA_VERSION >= 3


def test_bare_decorators_verify_as_links_and_called_ones_are_not_duplicated(tmp_path) -> None:
    root = N(str(tmp_path))
    deco, comp, src = (N(os.path.join(root, n)) for n in ("deco.ts", "comp.ts", "app.ts"))
    for p in (deco, comp, src):
        open(p, "w").close()
    analysis = _analyze(
        'import { Injectable } from "./deco";\nimport { Component } from "./comp";\n'
        "@Injectable\n@Component()\nexport class Svc {}\n"
    )
    symbol_map = {deco: {"exports": [{"name": "Injectable"}]}, comp: {"exports": [{"name": "Component"}]}}
    infos = {p: KeyInfo(f"1A{i}", p, root, 1, False) for i, p in enumerate((deco, comp, src))}
    ds._reset_js_export_registry()  # pyright: ignore[reportPrivateUsage]
    ds._js_import_resolution_ledger.clear()  # pyright: ignore[reportPrivateUsage]
    _, links = ds._compute_js_structural_dependencies(  # pyright: ignore[reportPrivateUsage]
        src, analysis, infos, root, symbol_map, None, set(infos)
    )
    assert sorted(l["reason"] for l in links) == ["JSCall/Component", "JSDecorator/Injectable"]


# --- bundler and workspace resolution ---------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> str:
    root = tmp_path / "proj"
    shutil.copytree(FIXTURE_TREE, root)
    ds._js_import_resolution_ledger.clear()  # pyright: ignore[reportPrivateUsage]
    jpr.reset()
    return N(str(root))


def _tracked(root: str) -> Set[str]:
    return {N(os.path.join(d, f)) for d, _, files in os.walk(root) for f in files}


def _resolve(root: str, rel_file: str, spec: str) -> Optional[str]:
    file_path = N(os.path.join(root, rel_file))
    configs = ds._tsconfig_for_file(file_path, root)  # pyright: ignore[reportPrivateUsage]
    target = ds._resolve_js_import_path(  # pyright: ignore[reportPrivateUsage]
        spec, os.path.dirname(file_path), root, {}, configs, _tracked(root)
    )
    return os.path.relpath(target, root).replace("\\", "/") if target else None


def test_vite_aliases_are_read_statically_from_a_function_config(project) -> None:
    assert _resolve(project, "app/main.ts", "@/components/Button") == "app/src/components/Button.ts"
    assert _resolve(project, "app/main.ts", "~lib/math") == "app/lib/math.ts"  # fileURLToPath(new URL())
    assert _resolve(project, "app/main.ts", "#root/thing") == "app/rootrel/thing.ts"  # root-relative
    # "@" only matches "@" or "@/…": a scoped package is never captured by it.
    assert _resolve(project, "app/main.ts", "@org/pkg") is None
    # A computed replacement (someHelper(...)) is skipped, not guessed.
    assert jpr.alias_bases("computed/x", os.path.join(project, "app"), project) == []


def test_sveltekit_lib_and_kit_alias(project) -> None:
    assert _resolve(project, "kit/src/routes/Page.svelte", "$lib/util") == "kit/src/lib/util.ts"
    assert _resolve(project, "kit/src/routes/Page.svelte", "$shared/fmt") == "kit/src/shared/fmt.ts"


def test_svelte_script_imports_go_through_the_bundler_aliases(project) -> None:
    page = N(os.path.join(project, "kit/src/routes/Page.svelte"))
    tracked = _tracked(project)
    p2k = {p: KeyInfo(f"1A{i}", p, N(os.path.dirname(p)), 1, False) for i, p in enumerate(sorted(tracked))}
    analysis = {"imports": [{"path": "$lib/util", "line": 2}, {"path": "$shared/fmt", "line": 3}]}
    ctx = se.EdgeContext(p2k, project, {page: analysis}, {}, tracked)
    edges, _ = se._svelte_edges(ctx, page, analysis)  # pyright: ignore[reportPrivateUsage]
    assert sorted(edges) == [
        (N(os.path.join(project, "kit/src/lib/util.ts")), "<"),
        (N(os.path.join(project, "kit/src/shared/fmt.ts")), "<"),
    ]


def test_workspace_packages_resolve_through_exports_and_main(project) -> None:
    app = "ws/apps/web/src/app.ts"
    assert _resolve(project, app, "@acme/ui") == "ws/packages/ui/src/index.ts"  # "types" before "import"
    assert _resolve(project, app, "@acme/ui/button") == "ws/packages/ui/src/button.ts"
    assert _resolve(project, app, "@acme/ui/icons/star") == "ws/packages/ui/src/icons/star.ts"
    assert _resolve(project, app, "@acme/ui/internal") is None  # null target blocks it
    assert _resolve(project, app, "@acme/ui/unlisted") is None  # not in exports
    assert _resolve(project, app, "core") == "ws/packages/core/src/main.ts"
    assert _resolve(project, app, "react") is None  # not a workspace package


def test_an_untracked_exports_target_is_not_guessed(project) -> None:
    tracked = _tracked(project) - {N(os.path.join(project, "ws/packages/ui/src/index.ts"))}
    file_path = N(os.path.join(project, "ws/apps/web/src/app.ts"))
    target = ds._resolve_js_import_path(  # pyright: ignore[reportPrivateUsage]
        "@acme/ui", os.path.dirname(file_path), project, {}, [], tracked
    )
    # Falls to the next condition ("import" -> dist), which is a tracked file here.
    assert target == N(os.path.join(project, "ws/packages/ui/dist/index.js"))
    tracked.discard(target)
    ds._js_import_resolution_ledger.clear()  # pyright: ignore[reportPrivateUsage]
    assert (
        ds._resolve_js_import_path(  # pyright: ignore[reportPrivateUsage]
            "@acme/ui", os.path.dirname(file_path), project, {}, [], tracked
        )
        is None
    )


def test_config_edits_are_picked_up_without_clearing_caches(project) -> None:
    pkg = os.path.join(project, "ws/packages/core/package.json")
    src_dir = os.path.join(project, "ws/apps/web/src")
    assert jpr.package_bases("core", src_dir, project)[0] == N(os.path.join(project, "ws/packages/core/src/main.ts"))
    with open(pkg, "w", encoding="utf-8") as fh:
        fh.write('{ "name": "core", "main": "./lib/entry.ts" }')
    os.utime(pkg, (1, 2_000_000_000))
    assert jpr.package_bases("core", src_dir, project)[0] == N(os.path.join(project, "ws/packages/core/lib/entry.ts"))


def test_package_specifier_split_and_pnpm_workspace_parsing() -> None:
    split = jpr._split_package_specifier  # pyright: ignore[reportPrivateUsage]
    assert split("@a/b/c/d") == ("@a/b", "./c/d")
    assert split("pkg") == ("pkg", ".")
    assert split("pkg/sub") == ("pkg", "./sub")
    assert split("@scope") is None
    parse = jpr._parse_pnpm_workspace  # pyright: ignore[reportPrivateUsage]
    text = "packages:\n  - 'packages/*'\n  - \"apps/*\"  # apps\n  - '!**/test/**'\ncatalog:\n  - x\n"
    assert parse(text, "") == ["packages/*", "apps/*", "!**/test/**"]
    assert jpr.resolution_bases("node:fs", ".", ".") == []
    assert jpr.resolution_bases("./rel", ".", ".") == []
