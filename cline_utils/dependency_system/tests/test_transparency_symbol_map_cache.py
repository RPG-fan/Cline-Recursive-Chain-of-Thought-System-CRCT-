"""
transparency_manager's symbol map is parsed once per version of the file. It used to
look in cache_manager.get_cache(...), which returns a Cache object and never a dict,
so every virtualize_connection_maps call re-parsed project_symbol_map.json (12 MB, about
160ms). For 702 files in bulk_virtualize_connection_maps that was about 115s per run.
"""

import json
import os

import pytest

from cline_utils.dependency_system.io import transparency_manager as tm


@pytest.fixture
def symbol_map_file(tmp_path, monkeypatch):
    path = tmp_path / "project_symbol_map.json"
    path.write_text(json.dumps({"/p/a.ts": {"functions": [{"name": "f"}]}}))
    monkeypatch.setattr(tm, "resolve_state_path", lambda name, _d: str(path))
    tm._symbol_map_cache.clear()  # pyright: ignore[reportPrivateUsage]
    yield path
    tm._symbol_map_cache.clear()  # pyright: ignore[reportPrivateUsage]


def test_parsed_once_per_file_version(symbol_map_file, monkeypatch) -> None:
    loads = []
    real_load = json.load
    monkeypatch.setattr(tm.json, "load", lambda f: loads.append(1) or real_load(f))

    first = tm._load_symbol_map_for_transparency()  # pyright: ignore[reportPrivateUsage]
    for _ in range(700):
        assert tm._load_symbol_map_for_transparency() is first  # pyright: ignore[reportPrivateUsage]
    assert loads == [1]
    assert first == {"/p/a.ts": {"functions": [{"name": "f"}]}}

    symbol_map_file.write_text(json.dumps({"/p/b.ts": {}, "/p/c.ts": {}}))
    st = os.stat(symbol_map_file)
    os.utime(symbol_map_file, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert set(tm._load_symbol_map_for_transparency()) == {"/p/b.ts", "/p/c.ts"}  # pyright: ignore[reportPrivateUsage]
    assert loads == [1, 1]


def test_missing_or_invalid_file_gives_empty_map(symbol_map_file) -> None:
    symbol_map_file.write_text("[1, 2, 3]")
    assert tm._load_symbol_map_for_transparency() == {}  # pyright: ignore[reportPrivateUsage]
    os.remove(symbol_map_file)
    assert tm._load_symbol_map_for_transparency() == {}  # pyright: ignore[reportPrivateUsage]
