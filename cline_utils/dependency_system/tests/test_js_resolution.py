"""
JS/TS import resolution (Phase 4): tsconfig `extends` / `references` / sibling configs,
per-file config selection, TypeScript `paths` matching, and TypeScript-style candidate
files (`.js` -> `.ts` source preference, `.mjs` -> `.mts`, index barrels, `.d.ts`).

The expected targets below were confirmed against TypeScript's own resolver
(cline_docs/crct_jsts/harness/gt_resolve.js) on the same fixture tree; these tests do
not need Node. The tree is copied to tmp_path and used as the project root, so no
directory walks up into the real repository's tsconfig.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import pytest

from cline_utils.dependency_system.analysis import dependency_analyzer as da
from cline_utils.dependency_system.analysis import dependency_suggester as ds
from cline_utils.dependency_system.analysis import js_tsconfig
from cline_utils.dependency_system.utils.path_utils import normalize_path

FIXTURE_TREE = Path(__file__).parent / "fixtures" / "jsts_resolution"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    shutil.copytree(FIXTURE_TREE, root)
    ds._js_import_resolution_ledger.clear()  # pyright: ignore[reportPrivateUsage]
    return root


def _tracked(root: Path) -> Set[str]:
    return {normalize_path(os.path.join(d, f)) for d, _, files in os.walk(root) for f in files}


def _resolve_all(root: Path, rel_file: str) -> Dict[str, Optional[str]]:
    """{specifier: target relative to root} for every import in the file."""
    file_path = normalize_path(str(root / rel_file))
    kind = {".ts": "ts", ".js": "js"}[Path(rel_file).suffix]
    analysis: Dict[str, object] = {"file_path": file_path}
    da._analyze_js_family_file_ts(  # pyright: ignore[reportPrivateUsage]
        file_path, (root / rel_file).read_text(encoding="utf-8"), analysis, kind
    )
    configs = ds._tsconfig_for_file(file_path, normalize_path(str(root)))  # pyright: ignore[reportPrivateUsage]
    tracked = _tracked(root)
    out: Dict[str, Optional[str]] = {}
    for imp in analysis["imports"]:  # type: ignore[union-attr]
        target = ds._resolve_js_import_path(  # pyright: ignore[reportPrivateUsage]
            imp["path"], os.path.dirname(file_path), normalize_path(str(root)), {}, configs, tracked
        )
        out[imp["path"]] = os.path.relpath(target, root).replace("\\", "/") if target else None
    return out


def _selected(root: Path, rel_file: str) -> List[str]:
    configs = ds._tsconfig_for_file(  # pyright: ignore[reportPrivateUsage]
        normalize_path(str(root / rel_file)), normalize_path(str(root))
    )
    return [os.path.relpath(p, root).replace("\\", "/") for p, _ in configs]


def test_relative_and_alias_resolution_matches_typescript(project: Path) -> None:
    assert _resolve_all(project, "src/app.ts") == {
        "./util.js": "src/util.ts",  # source preferred over the compiled util.js beside it
        "./esm.mjs": "src/esm.mts",
        "./comp": "src/comp.tsx",
        "./barrel": "src/barrel/index.ts",
        "@lib/thing": "lib/thing.ts",  # alias defined only in the extended base config
        "@lib/special": "lib/special-impl.ts",  # exact paths key beats the @lib/* wildcard
        "lib/thing": "lib/thing.ts",  # baseUrl (declared in the base, relative to it)
        "./x.browser": "src/x.browser.ts",  # dotted basename is not an extension
        "./data.json": "src/data.json",
        "react": None,
        "./types-only": "src/types-only.d.ts",
    }


def test_config_selection_follows_include_references_and_siblings(project: Path) -> None:
    assert _selected(project, "src/app.ts") == ["tsconfig.json"]
    # tsconfig.json does not include scripts/; its reference comes before the other sibling.
    assert _selected(project, "scripts/run.js") == ["tsconfig.z-ref.json", "tsconfig.scripts.json"]
    # The nearest directory with a config wins.
    assert _selected(project, "tools/t.ts") == ["tools/tsconfig.json"]
    # Included by no config: the first candidate still provides the context.
    assert _selected(project, "outside/o.ts") == ["tsconfig.json"]


def test_applicable_configs_are_tried_in_order(project: Path) -> None:
    assert _resolve_all(project, "scripts/run.js") == {
        "@scripts/helper": "scripts/helper.js",  # only tsconfig.scripts.json defines it
        "@dup/x": "dupa/x.ts",  # both define @dup/*; the referenced config comes first
    }
    assert _resolve_all(project, "tools/t.ts") == {"@tools/u": "tools/u.ts"}
    assert _resolve_all(project, "outside/o.ts") == {"@lib/thing": "lib/thing.ts"}


def test_solution_style_root_config(tmp_path: Path) -> None:
    """Vite react-ts layout: tsconfig.json has `"files": []` and only references the real
    projects. Also: once a `paths` pattern matches (`*`), baseUrl is not used as a fallback."""
    root = tmp_path / "solution"
    shutil.copytree(Path(__file__).parent / "fixtures" / "jsts_resolution_solution", root)
    ds._js_import_resolution_ledger.clear()  # pyright: ignore[reportPrivateUsage]

    assert _selected(root, "src/main.ts") == ["tsconfig.app.json"]
    assert _selected(root, "vite.config.ts") == ["tsconfig.node.json"]
    assert _resolve_all(root, "src/main.ts") == {
        "@/util": "src/util.ts",
        "lib/thing": None,  # `*` -> types/* matched and missed; TypeScript does not try baseUrl
        "decl": "types/decl.d.ts",
    }
    assert _resolve_all(root, "vite.config.ts") == {"./src/util": "src/util.ts"}


def test_editing_an_extended_base_config_is_picked_up(project: Path) -> None:
    base = project / "configs" / "base.json"
    data = json.loads(base.read_text(encoding="utf-8"))
    data["compilerOptions"]["paths"]["@lib/*"] = ["src/*"]
    time.sleep(0.05)
    base.write_text(json.dumps(data), encoding="utf-8")
    os.utime(base, None)
    ds._js_import_resolution_ledger.clear()  # pyright: ignore[reportPrivateUsage]

    assert _resolve_all(project, "outside/o.ts") == {"@lib/thing": None}  # src/thing.* does not exist


def test_parse_jsonc_handles_comments_and_trailing_commas() -> None:
    text = '{\n  // c\n  "a": [1, 2,], /* b */ "s": "x // not, a comment,]",\n  "o": {"k": 1,},\n}'
    assert js_tsconfig.parse_jsonc(text) == {"a": [1, 2], "s": "x // not, a comment,]", "o": {"k": 1}}


def test_typescript_include_defaults(tmp_path: Path) -> None:
    def config(raw: dict, name: str = "tsconfig.json") -> js_tsconfig.TsConfig:
        path = tmp_path / name
        path.write_text(json.dumps(raw), encoding="utf-8")
        loaded = js_tsconfig.load_tsconfig(str(path))
        assert loaded is not None
        return loaded

    def applies(cfg: js_tsconfig.TsConfig, rel: str) -> bool:
        return js_tsconfig.tsconfig_applies_to(cfg, str(tmp_path / rel))

    everything = config({"compilerOptions": {"outDir": "build"}})
    assert applies(everything, "src/a.ts")
    assert not applies(everything, "node_modules/x/index.ts")  # default exclude
    assert not applies(everything, "build/a.ts")  # outDir is excluded by default
    assert not applies(everything, "src/a.js")  # JS needs allowJs

    files_only = config({"files": ["src/main.ts"]}, "tsconfig.files.json")
    assert applies(files_only, "src/main.ts") and not applies(files_only, "src/other.ts")

    explicit = config(
        {"compilerOptions": {"allowJs": True}, "include": ["lib", "*.ts"], "exclude": ["lib/**/*.test.ts"]},
        "tsconfig.explicit.json",
    )
    assert applies(explicit, "lib/deep/a.js")  # directory include, allowJs
    assert applies(explicit, "top.ts") and not applies(explicit, "src/top.ts")  # `*.ts` is one level
    assert not applies(explicit, "lib/a.test.ts")


def test_legacy_single_tsconfig_tuple_still_resolves(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "thing.ts"
    target.write_text("export const t = 1;", encoding="utf-8")
    raw = {"compilerOptions": {"baseUrl": ".", "paths": {"~/*": ["src/*"]}}}
    legacy = (normalize_path(str(tmp_path / "tsconfig.json")), raw)
    ds._js_import_resolution_ledger.clear()  # pyright: ignore[reportPrivateUsage]

    resolved = ds._resolve_js_import_path(  # pyright: ignore[reportPrivateUsage]
        "~/thing", normalize_path(str(tmp_path)), normalize_path(str(tmp_path)), {}, legacy,
        {normalize_path(str(target))},
    )
    assert resolved == normalize_path(str(target))
