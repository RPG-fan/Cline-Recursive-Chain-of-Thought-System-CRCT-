"""
One suggestion workflow for every language (structural_engine + per-language adapters).
Python names in scope now come from the analyzer's `import_bindings`, built on every
run. They used to come from an import map @cached by the source path alone, persisted
and never invalidated, so edited imports kept resolving through the old map.
Model-free.
"""

import ast
import os
from typing import Dict

import pytest

from cline_utils.dependency_system.analysis import dependency_analyzer as da
from cline_utils.dependency_system.analysis import dependency_suggester as ds
from cline_utils.dependency_system.analysis import structural_engine as se
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.path_utils import normalize_path as N


@pytest.fixture(autouse=True)
def _fresh():
    ds.clear_caches()
    yield
    ds.clear_caches()


def test_every_language_dispatches_to_one_adapter() -> None:
    expected = {
        "a.py": ("python", True), "a.ts": ("js", True), "a.tsx": ("js", True),
        "a.mjs": ("js", True), "a.md": ("markdown", True), "a.rst": ("markdown", True),
        "a.html": ("html", False), "a.css": ("css", False), "a.json": ("json", True),
        "a.svelte": ("svelte", True), "a.sql": ("sql", False), "a.csv": ("generic", True),
    }
    for name, (adapter, semantic) in expected.items():
        got = se.adapter_for(name)
        assert (got.name, got.semantic) == (adapter, semantic), name


@pytest.fixture
def pyproj(tmp_path):
    root = N(str(tmp_path))
    files = {
        "pkg/__init__.py": "",
        "pkg/a.py": "def alpha():\n    pass\n",
        "pkg/b.py": "def beta():\n    pass\n",
        "main.py": "",
    }
    for rel, text in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(text)
    p2k: Dict[str, KeyInfo] = {
        N(os.path.join(root, rel)): KeyInfo(f"1A{i}", N(os.path.join(root, rel)), root, 1, False)
        for i, rel in enumerate(files)
    }
    symbols = {
        N(os.path.join(root, "pkg/a.py")): {"functions": [{"name": "alpha"}]},
        N(os.path.join(root, "pkg/b.py")): {"functions": [{"name": "beta"}]},
    }
    return root, p2k, symbols, N(os.path.join(root, "main.py"))


def _analysis(code: str) -> Dict:
    tree = ast.parse(code)
    return {"_ast_tree": tree, "import_bindings": da._python_import_bindings(tree)}  # pyright: ignore[reportPrivateUsage]


def test_import_map_follows_the_files_current_imports(pyproj) -> None:
    root, p2k, symbols, main = pyproj
    m1 = ds._python_import_map(main, _analysis("from pkg.a import alpha\n"), root, p2k, symbols, set(p2k))  # pyright: ignore[reportPrivateUsage]
    m2 = ds._python_import_map(main, _analysis("from pkg.b import beta\n"), root, p2k, symbols, set(p2k))  # pyright: ignore[reportPrivateUsage]
    assert {k: os.path.basename(v) for k, v in m1.items()} == {"alpha": "a.py"}
    assert {k: os.path.basename(v) for k, v in m2.items()} == {"beta": "b.py"}  # not the old map


def test_bindings_include_nested_imports_and_match_the_ast_fallback(pyproj) -> None:
    root, p2k, symbols, main = pyproj
    code = (
        "import pkg.a as pa\n"
        "def f():\n"
        "    from pkg.b import beta as bb\n"
        "try:\n"
        "    import pkg\n"
        "except ImportError:\n"
        "    pass\n"
    )
    with_bindings = _analysis(code)
    ast_only = {"_ast_tree": with_bindings["_ast_tree"]}  # analysis cached before v2
    m1 = ds._python_import_map(main, with_bindings, root, p2k, symbols, set(p2k))  # pyright: ignore[reportPrivateUsage]
    m2 = ds._python_import_map(main, ast_only, root, p2k, symbols, set(p2k))  # pyright: ignore[reportPrivateUsage]
    assert m1 == m2
    assert set(m1) == {"pa", "bb", "pkg"}  # nested and try-block imports are in scope
    assert ds._python_import_map(main, {}, root, p2k, symbols, set(p2k)) == {}  # pyright: ignore[reportPrivateUsage]


def test_structural_call_edge_uses_current_bindings(pyproj) -> None:
    root, p2k, symbols, main = pyproj
    analysis = _analysis("from pkg.b import beta\nbeta()\n")
    analysis["calls"] = [{"target_name": "beta", "potential_source": None}]
    edges, links = ds._identify_structural_dependencies(main, analysis, p2k, root, symbols, set(p2k))  # pyright: ignore[reportPrivateUsage]
    assert edges == [(N(os.path.join(root, "pkg/b.py")), "<")]
    assert [l["reason"] for l in links] == ["Call/beta"]


def test_analysis_cache_key_carries_the_schema_version(tmp_path) -> None:
    f = tmp_path / "x.py"
    f.write_text("import os\n")
    key = da._analyze_file_key(str(f))  # pyright: ignore[reportPrivateUsage]
    assert key.startswith(f"analyze_file:v{da.ANALYSIS_SCHEMA_VERSION}:")
    assert da.ANALYSIS_SCHEMA_VERSION >= 2


def test_prepared_results_equal_on_demand_results(pyproj) -> None:
    root, p2k, symbols, main = pyproj
    analysis = _analysis("from pkg.a import alpha\nalpha()\n")
    analysis["imports"] = [{"path": "pkg.a"}]
    analysis["calls"] = [{"target_name": "alpha", "potential_source": None}]
    analyses = {main: analysis}
    on_demand = se.edges_for(main, analysis, p2k, root, analyses, symbols, set(p2k))
    se.prepare_structural_index(analyses, p2k, root, symbols, set(p2k))
    prepared = se.edges_for(main, analysis, p2k, root, analyses, symbols, set(p2k))
    assert prepared == on_demand
    assert sorted(set(prepared[0])) == [(N(os.path.join(root, "pkg/a.py")), "<")]
    assert {l["reason"] for l in prepared[1]} == {"ExplicitImport/pkg.a", "Call/alpha"}
    # A different analysis object for the same path is computed, not looked up.
    other = _analysis("")
    assert se.edges_for(main, other, p2k, root, analyses, symbols, set(p2k)) == ([], [])


def test_sql_inverse_edges_follow_symbol_map_order() -> None:
    symbol_map = {
        "/p/z.sql": {"tables_defined": [], "tables_referenced": ["t"]},
        "/p/a.sql": {"tables_defined": ["t"], "tables_referenced": []},
        "/p/m.sql": {"tables_defined": [], "tables_referenced": ["t"]},
        "/p/self.sql": {"tables_defined": ["t"], "tables_referenced": ["t"]},  # self-reference
    }
    ctx = se.EdgeContext({}, "/p", {}, symbol_map, set())
    edges, _ = se._sql_edges(ctx, "/p/a.sql", symbol_map["/p/a.sql"])  # pyright: ignore[reportPrivateUsage]
    assert edges == [("/p/z.sql", ">"), ("/p/m.sql", ">")]


# --- Fixes adopted after the equivalence proof (2026-10-21) ------------------------


def test_relative_import_resolves_from_the_files_own_package(pyproj) -> None:
    """`from . import x` / `from .a import alpha` in pkg/mod.py resolve inside pkg, as in
    Python and the explicit-import step. The structural map used to climb one directory
    too high, so structural edges through relative imports were missed."""
    root, p2k, symbols, _main = pyproj
    mod = N(os.path.join(root, "pkg", "mod.py"))
    open(mod, "w").write("")
    p2k = dict(p2k)
    p2k[mod] = KeyInfo("1A9", mod, N(os.path.join(root, "pkg")), 1, False)
    analysis = _analysis("from .a import alpha\nalpha()\n")
    analysis["calls"] = [{"target_name": "alpha", "potential_source": None}]
    imap = ds._python_import_map(mod, analysis, root, p2k, symbols, set(p2k))  # pyright: ignore[reportPrivateUsage]
    assert {k: os.path.basename(v) for k, v in imap.items()} == {"alpha": "a.py"}
    edges, links = ds._identify_structural_dependencies(mod, analysis, p2k, root, symbols, set(p2k))  # pyright: ignore[reportPrivateUsage]
    assert edges == [(N(os.path.join(root, "pkg/a.py")), "<")]
    assert [l["reason"] for l in links] == ["Call/alpha"]


def test_svelte_script_imports_resolve_from_the_normalized_path_key(tmp_path) -> None:
    """_normalize_imports stores the specifier under 'path'; the Svelte step used to read
    only 'source' and resolved nothing."""
    root = N(str(tmp_path))
    for rel in ("src/ui/App.svelte", "src/ui/styles/main.css", "src/routes/page.svelte"):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()
    paths = {rel: N(os.path.join(root, rel)) for rel in ("src/ui/App.svelte", "src/ui/styles/main.css", "src/routes/page.svelte")}
    p2k = {p: KeyInfo(f"1A{i}", p, N(os.path.dirname(p)), 1, False) for i, p in enumerate(paths.values())}
    page = paths["src/routes/page.svelte"]
    analysis = {"imports": [{"path": "../ui/App.svelte", "line": 3}, {"path": "../ui/styles/main.css", "line": 2}]}
    ctx = se.EdgeContext(p2k, root, {page: analysis}, {}, set(p2k))
    edges, _ = se._svelte_edges(ctx, page, analysis)  # pyright: ignore[reportPrivateUsage]
    assert sorted(edges) == sorted([(paths["src/ui/App.svelte"], "<"), (paths["src/ui/styles/main.css"], "<")])


def test_one_structured_summary_line_per_file(pyproj, caplog, monkeypatch) -> None:
    """suggest_dependencies logs one 'Suggest <file> [adapter]: ...' line with edge counts
    by reason, semantic counts by character, and what the tracker skip filter dropped,
    and no per-item lines."""
    import logging

    root, p2k, symbols, main = pyproj
    analysis = _analysis("from pkg.a import alpha\nalpha()\n")
    analysis["imports"] = [{"path": "pkg.a"}]
    analysis["calls"] = [{"target_name": "alpha", "potential_source": None}]
    analyses = {main: analysis}
    monkeypatch.setattr(ds, "load_project_symbol_map", lambda: symbols)
    monkeypatch.setattr(
        ds, "suggest_semantic_dependencies_path_based",
        lambda *a, **k: [(N(os.path.join(root, "pkg/b.py")), "S")],
    )
    existing = {(main, N(os.path.join(root, "pkg/b.py"))): ("n", set())}
    with caplog.at_level(logging.DEBUG, logger=ds.logger.name):
        sugg, _ = ds.suggest_dependencies(main, p2k, root, analyses, existing_state=existing)
    lines = [r.getMessage() for r in caplog.records if r.name == ds.logger.name]
    assert sugg == [(N(os.path.join(root, "pkg/a.py")), "<")]
    assert lines == [
        "Suggest main.py [python]: edges 2 (Call 1, ExplicitImport 1) + semantic 1 (S 1) "
        "-> 1 kept, 1 skipped (tracker x/n)"
    ]
