"""
Upfront indexes for the semantic step. Each must give exactly what the per-file code
gave:
- tracker relations: built once from every tracker. Previously each file
  re-discovered and re-read all trackers. The first char for a target wins unless a
  later one is '<', '>', 'x' or 'd', and a source whose grid read fails gets nothing;
- skip pairs: should_skip_suggestion's answer for every pair, by path or by key, in
  either orientation;
- token metadata: parsed once per version of metadata.json.
No model is loaded; trackers are fakes.
"""

import json
import os
from typing import Dict, List

import pytest

from cline_utils.dependency_system.analysis import dependency_suggester as ds
from cline_utils.dependency_system.analysis import embedding_manager as em
from cline_utils.dependency_system.core import dependency_grid
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils import tracker_utils
from cline_utils.dependency_system.utils.path_utils import normalize_path as N


def _ki(key: str, path: str) -> KeyInfo:
    return KeyInfo(key, path, "/p", 1, False)


P2K = {N(f"/p/{n}.ts"): _ki(k, N(f"/p/{n}.ts")) for k, n in [("1A", "a"), ("1B", "b"), ("1C", "c"), ("1D", "d")]}

# tracker -> source key -> {char: [target keys]}; "BOOM" makes the grid read raise.
TRACKERS: Dict[str, Dict[str, object]] = {
    "t1": {"1A": {"s": ["1B"], "S": ["1C"]}, "1B": {"<": ["1A"]}, "1D": "BOOM"},
    "t2": {"1A": {">": ["1B"], "s": ["1C"], "p": ["1D"]}, "1D": {"<": ["1A"]}},
    "t3": {"1C": {"x": ["1A"], "n": ["1B"]}, "1A": {"S": ["1B"]}},
}


@pytest.fixture
def fake_trackers(monkeypatch):
    ds._reset_semantic_indexes()  # pyright: ignore[reportPrivateUsage]
    discovered: List[int] = []

    def read(t_path):
        rows = [(src, "") for src in TRACKERS[t_path]]
        return {"grid_rows_ordered": rows, "grid_headers_ordered": ["1A", "1B", "1C", "1D"]}

    def deps(grid, source, kis):
        spec = TRACKERS[current[0]][source]
        if spec == "BOOM":
            raise ValueError("bad grid")
        return {c: list(t) for c, t in spec.items()}  # type: ignore[union-attr]

    current = [""]

    def read_tracking(t_path):
        current[0] = t_path
        return read(t_path)

    monkeypatch.setattr(tracker_utils, "read_tracker_file_structured", read_tracking)
    monkeypatch.setattr(dependency_grid, "get_dependencies_from_grid", deps)
    monkeypatch.setattr(
        tracker_utils, "find_all_tracker_paths",
        lambda *a, **k: discovered.append(1) or ["t1", "t2", "t3"],
    )
    yield discovered
    ds._reset_semantic_indexes()  # pyright: ignore[reportPrivateUsage]


def _per_file_reference(source: str) -> Dict[str, str]:
    """The pre-index per-file loop, verbatim in behaviour."""
    key_to_ki = {ki.key_string: ki for ki in P2K.values()}
    out: Dict[str, str] = {}
    try:
        for t in ["t1", "t2", "t3"]:
            if source not in TRACKERS[t]:
                continue
            spec = TRACKERS[t][source]
            if spec == "BOOM":
                raise ValueError("bad grid")
            for char, targets in spec.items():  # type: ignore[union-attr]
                if char in ["<", ">", "x", "d", "S", "s"]:
                    for tgt in targets:
                        if tgt in key_to_ki:
                            norm = key_to_ki[tgt].norm_path
                            if norm not in out or char in ["<", ">", "x", "d"]:
                                out[norm] = char
    except ValueError:
        return {}  # the per-file loop discarded everything on error
    return out


def test_index_equals_per_file_collection_for_every_source(fake_trackers) -> None:
    ds.prepare_tracker_relations_index(None, "/p", P2K)  # type: ignore[arg-type]
    for ki in P2K.values():
        indexed = ds._tracker_relations_for(ki.key_string, None, "/p", P2K)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]
        assert indexed == _per_file_reference(ki.key_string), ki.key_string
    assert ds._tracker_relations_for("1A", None, "/p", P2K) == {  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]
        N("/p/b.ts"): ">",  # '>' (t2) overrides 's' (t1); a later 'S' (t3) does not override '>'
        N("/p/c.ts"): "S",  # 'S' (t1) is kept: a later 's' (t2) is not a strong char
    }
    assert ds._tracker_relations_for("1D", None, "/p", P2K) == {}  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]


def test_prepared_index_never_rereads_trackers(fake_trackers) -> None:
    ds.prepare_tracker_relations_index(None, "/p", P2K)  # type: ignore[arg-type]
    assert len(fake_trackers) == 1
    for ki in P2K.values():
        ds._tracker_relations_for(ki.key_string, None, "/p", P2K)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]
    assert len(fake_trackers) == 1


def test_unprepared_key_map_falls_back_to_the_same_result(fake_trackers) -> None:
    other = dict(P2K)  # a different key-map object: nothing prepared for it
    ds.prepare_tracker_relations_index(None, "/p", P2K)  # type: ignore[arg-type]
    for ki in other.values():
        assert ds._tracker_relations_for(ki.key_string, None, "/p", other) == _per_file_reference(ki.key_string)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]


def test_skip_pairs_match_should_skip_suggestion_for_every_pair() -> None:
    ds._reset_semantic_indexes()  # pyright: ignore[reportPrivateUsage]
    existing = {
        (N("/p/a.ts"), N("/p/b.ts")): ("x", set()),
        (N("/p/c.ts"), N("/p/a.ts")): ("n", set()),
        ("1B", "1D"): ("x", set()),  # key-string entry
        (N("/p/b.ts"), N("/p/c.ts")): ("<", set()),
        ("1C", "1D"): ("S", set()),
    }
    skip = ds._skip_pairs_for(existing)  # pyright: ignore[reportPrivateUsage]
    for s in P2K.values():
        for t in P2K.values():
            fast = (s.norm_path, t.norm_path) in skip or (s.key_string, t.key_string) in skip
            assert fast == ds.should_skip_suggestion(s.norm_path, t.norm_path, existing, P2K), (s, t)
    assert ds._skip_pairs_for(existing) is skip  # built once per state object  # pyright: ignore[reportPrivateUsage]
    assert ds._skip_pairs_for(None) == frozenset()  # pyright: ignore[reportPrivateUsage]


def test_token_metadata_is_parsed_once_per_file_version(tmp_path, monkeypatch) -> None:
    meta_dir = tmp_path / "cline_utils" / "dependency_system" / "analysis" / "embeddings"
    meta_dir.mkdir(parents=True)
    meta = meta_dir / "metadata.json"
    meta.write_text(json.dumps({"keys": {"1A": {"path": N("/p/a.ts"), "ses_tokens": 3, "full_tokens": 9}}}))
    parses = []
    real = em._parse_token_metadata  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(em, "_parse_token_metadata", lambda p: parses.append(p) or real(p))
    em._token_metadata_cache.clear()  # pyright: ignore[reportPrivateUsage]

    first = em._load_token_metadata(str(tmp_path))  # pyright: ignore[reportPrivateUsage]
    for _ in range(5):
        assert em._load_token_metadata(str(tmp_path)) is first  # pyright: ignore[reportPrivateUsage]
    assert len(parses) == 1
    assert list(first.values()) == [{"ses_tokens": 3, "full_tokens": 9}]

    meta.write_text(json.dumps({"keys": {"1A": {"path": N("/p/a.ts"), "ses_tokens": 4, "full_tokens": 10, "x": 1}}}))
    os.utime(meta, ns=(os.stat(meta).st_atime_ns, os.stat(meta).st_mtime_ns + 10**9))
    assert list(em._load_token_metadata(str(tmp_path)).values()) == [{"ses_tokens": 4, "full_tokens": 10}]  # pyright: ignore[reportPrivateUsage]
    assert len(parses) == 2
    em._token_metadata_cache.clear()  # pyright: ignore[reportPrivateUsage]
