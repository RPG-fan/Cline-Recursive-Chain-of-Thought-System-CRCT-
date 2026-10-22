"""
Generated JavaScript is kept out of key generation by default (`exclude_generated_files`):
compiled twins beside their TypeScript source, minified files, and JS / declarations
whose last line is a sourceMappingURL comment. Hand-written JS and `.d.ts` stay keyed.
"""

from pathlib import Path

import pytest

from cline_utils.dependency_system.core.key_manager import generate_keys
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.generated_files import generated_reason


def _reason(directory: Path, name: str):
    return generated_reason(name, str(directory), {p.name.lower() for p in directory.iterdir()})


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    files = {
        "app.ts": "export const a = 1;\n",
        "app.js": "export const a = 1;\n",  # twin of app.ts
        "view.tsx": "export const V = () => null;\n",
        "view.js": "export const V = () => null;\n",  # twin of view.tsx
        "view.jsx": "export const V = () => null;\n",  # twin of view.tsx
        "esm.mts": "export {};\n",
        "esm.mjs": "export {};\n",  # twin of esm.mts
        "app.d.ts": "export declare const a = 1;\n",  # declaration twin of app.ts
        "ambient.d.ts": "declare module 'x';\n",  # hand-written: kept
        "vendor.min.js": "!function(){}();\n",
        "bundle.js": "console.log(1);\n//# sourceMappingURL=bundle.js.map\n",
        "types.d.ts": "export {};\n//# sourceMappingURL=types.d.ts.map\n",
        "util.js": "export function util() {}\n// sourceMappingURL is only a word here\n",
        "Notes.md": "app.js\n",
    }
    for name, text in files.items():
        (src / name).write_text(text, encoding="utf-8")
    return src


def test_classification(tree: Path) -> None:
    got = {p.name: _reason(tree, p.name) for p in tree.iterdir()}
    assert got == {
        "app.ts": None,
        "app.js": "twin",
        "view.tsx": None,
        "view.js": "twin",
        "view.jsx": "twin",
        "esm.mts": None,
        "esm.mjs": "twin",
        "app.d.ts": "twin",
        "ambient.d.ts": None,
        "vendor.min.js": "minified",
        "bundle.js": "sourcemap",
        "types.d.ts": "sourcemap",
        "util.js": None,
        "Notes.md": None,
    }


def test_twin_detection_is_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "Widget.TS").write_text("", encoding="utf-8")
    (tmp_path / "Widget.js").write_text("", encoding="utf-8")
    assert _reason(tmp_path, "Widget.js") == "twin"


def _keyed_names(root: Path) -> set:
    path_to_key_info, _ = generate_keys(
        [str(root)], excluded_dirs=set(), excluded_extensions={".pyc"}, precomputed_excluded_paths=set()
    )
    return {Path(p).name for p, info in path_to_key_info.items() if not info.is_directory}


def test_key_generation_skips_generated_files_by_default(tree: Path, caplog) -> None:
    with caplog.at_level("INFO"):
        names = _keyed_names(tree)
    assert names == {"app.ts", "view.tsx", "esm.mts", "ambient.d.ts", "util.js", "Notes.md"}
    assert any(
        "skipped 8 generated file(s) (minified 1, sourcemap 2, twin 5)" in r.getMessage() for r in caplog.records
    )


def test_the_flag_turns_it_off(tree: Path, monkeypatch) -> None:
    monkeypatch.setattr(ConfigManager, "get_exclude_generated_files", lambda self: False)
    assert len(_keyed_names(tree)) == 14
