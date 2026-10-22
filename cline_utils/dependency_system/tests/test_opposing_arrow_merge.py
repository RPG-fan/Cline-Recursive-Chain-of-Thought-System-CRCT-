"""Opposed '<' / '>' are one relationship seen from two sides, not a mutual 'x'.

'<' in [A][B] ("A depends on B") and '>' in [B][A] ("column requires row") say the
same thing. 'x' (mutual) needs BOTH files to report the same character from their
own side, in the current pass:
  * one file's own '<' and '>' for the same target do not merge (combine, _build_ast_map);
  * a stale arrow left in the reverse cell does not count (tracker_io mirror rule);
  * a source must not emit the reverse of its own edge (SQL inverse);
  * existing tracker relations are carried into the file's suggestions (by design).
Embeddings/reranker are mocked: no model is ever loaded.
"""

import json
import os

from typing import Dict, List

import pytest

from cline_utils.dependency_system.analysis import (
    dependency_suggester,
    embedding_manager,
)
from cline_utils.dependency_system.core.dependency_grid import compress
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils import tracker_utils

from .test_tracker_mirror_pair import _grid

norm = dependency_suggester.normalize_path


def _ki(key: str, path: str) -> KeyInfo:
    return KeyInfo(
        key_string=key,
        norm_path=path,
        parent_path=norm(os.path.dirname(path)),
        tier=1,
        is_directory=False,
    )


@pytest.fixture
def project(tmp_path):
    """Create a tmp project with an (empty) embeddings dir, return helpers."""
    from cline_utils.dependency_system.utils.config_manager import ConfigManager

    root = norm(str(tmp_path))
    embeddings_rel = ConfigManager().get_path(
        "embeddings_dir", "cline_utils/dependency_system/analysis/embeddings"
    )
    os.makedirs(os.path.join(root, embeddings_rel), exist_ok=True)

    def build(names: List[str]) -> Dict[str, KeyInfo]:
        infos: Dict[str, KeyInfo] = {}
        for i, name in enumerate(names):
            full = norm(os.path.join(root, name))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write("content")
            infos[full] = _ki(f"1A{i + 1}", full)
        return infos

    return root, build


@pytest.fixture
def mocked_semantics(monkeypatch):
    """Mock similarity, reranker, history and tracker discovery. Returns a config dict."""
    state = {
        "similarity": 0.95,
        "rerank_scores": None,  # list of scores aligned with candidate order, or None
        "ast_links": [],
        "rerank_calls": 0,
    }

    monkeypatch.setattr(
        embedding_manager,
        "calculate_similarity",
        lambda *a, **k: state["similarity"],
    )

    class _SameScoreForEveryTarget(dict):
        def get(self, key, default=None):
            return state["similarity"]

    # The semantic step reads one similarity-index row per file.
    monkeypatch.setattr(
        dependency_suggester.similarity_index,
        "get_scores",
        lambda *a, **k: _SameScoreForEveryTarget(),
    )
    monkeypatch.setattr(
        embedding_manager, "get_instruction_for_relation_type", lambda *a, **k: ""
    )
    monkeypatch.setattr(
        embedding_manager, "preprocess_doc_structure", lambda content, *a, **k: content
    )

    def fake_rerank(query, candidate_texts, top_k=10, **_kwargs):
        state["rerank_calls"] += 1
        scores = state["rerank_scores"]
        return [(i, scores[i]) for i in range(len(candidate_texts))]

    monkeypatch.setattr(embedding_manager, "rerank_candidates_with_qwen3", fake_rerank)
    # The suggester calls rerank_candidates_scored ((top_k results, scored indices)).
    monkeypatch.setattr(
        embedding_manager,
        "rerank_candidates_scored",
        lambda q, c, top_k=10, **k: (lambda r: (r, {i for i, _ in r}))(fake_rerank(q, c, top_k, **k)),
    )

    from cline_utils.dependency_system.analysis import reranker_history_tracker
    from cline_utils.dependency_system.utils import tracker_utils

    monkeypatch.setattr(
        reranker_history_tracker, "get_historical_pairs", lambda *a, **k: set()
    )
    monkeypatch.setattr(tracker_utils, "find_all_tracker_paths", lambda *a, **k: [])
    monkeypatch.setattr(
        dependency_suggester, "_load_ast_verified_links", lambda *_a, **_k: state["ast_links"]
    )
    return state


def _run(root, infos, source_name: str) -> Dict[str, str]:
    source = norm(os.path.join(root, source_name))
    result = dependency_suggester.suggest_semantic_dependencies_path_based(
        source,
        infos,
        root,
        threshold=0.7,
        project_symbol_map=None,
        shared_scan_counter=None,
        tracked_paths_globally=set(infos),
    )
    return {os.path.basename(p): c for p, c in result}


combine = dependency_suggester.combine_suggestions_path_based_with_char_priority


# --- one file's opposed '<' / '>' for the same target never become 'x' -------------


def test_combine_opposed_arrows_for_one_target_do_not_merge_to_x():
    assert dict(combine([("t", "<"), ("t", ">")], "src")) == {"t": "<"}
    assert dict(combine([("t", ">"), ("t", "<")], "src")) == {"t": ">"}
    assert dict(combine([("t", "<")], "src")) == {"t": "<"}
    # Priority still decides between different characters.
    assert dict(combine([("t", "S"), ("t", "<")], "src")) == {"t": "<"}
    assert dict(combine([("t", "<"), ("t", "x")], "src")) == {"t": "x"}


def test_build_ast_map_opposed_relations_do_not_merge_to_x():
    a, b = "H:/p/a.js", "H:/p/b.js"
    forward = {"source_path": a, "target_path": b, "char": "<"}  # a depends on b
    reverse = {"source_path": b, "target_path": a, "char": "<"}  # b depends on a
    build = dependency_suggester._build_ast_map
    # a's own link wins in either order; b's link is b's to report (its own ast_map).
    assert build(a, [forward, reverse]) == {b: "<"}
    assert build(a, [reverse, forward]) == {b: "<"}
    assert build(b, [forward, reverse]) == {a: "<"}
    assert build(b, [reverse, forward]) == {a: "<"}
    # Only the other file reports: the inverse of its edge.
    assert build(a, [reverse]) == {b: ">"}


# --- SQL: inverse generation must skip tables the other file defines -----------


def _sql(tmp_path, analyses):
    root = norm(str(tmp_path))
    paths = {}
    symbol_map = {}
    path_to_key_info = {}
    for i, (name, (defined, referenced)) in enumerate(analyses.items()):
        full = norm(os.path.join(root, name))
        (tmp_path / name).write_text("-- sql", encoding="utf-8")
        paths[name] = full
        symbol_map[full] = {
            "tables_defined": defined,
            "tables_referenced": referenced,
        }
        path_to_key_info[full] = dependency_suggester.KeyInfo(
            key_string=f"1A{i + 1}",
            norm_path=full,
            parent_path=root,
            tier=1,
            is_directory=False,
        )

    def suggest(name):
        return dict(
            dependency_suggester.suggest_sql_dependencies(
                paths[name],
                path_to_key_info,
                root,
                {p: dict(v) for p, v in symbol_map.items()},
                0.7,
                project_symbol_map=symbol_map,
            )
        )

    return paths, suggest


def test_sql_shared_table_name_does_not_make_mutual_x(tmp_path):
    """migration ALTERs 'users' (counts as defined) and references schema tables;
    schema defines+references 'users' itself. Only migration -> schema is real."""
    paths, suggest = _sql(
        tmp_path,
        {
            "mig.sql": (["users"], ["users", "perf", "alerts"]),
            "schema.sql": (
                ["users", "perf", "alerts"],
                ["users", "perf", "alerts"],
            ),
        },
    )
    assert suggest("mig.sql") == {paths["schema.sql"]: "<"}
    assert suggest("schema.sql") == {paths["mig.sql"]: ">"}


def test_sql_mutual_table_references_each_file_reports_the_same_char(tmp_path):
    """a references b's table and b references a's table: each file reports '<'
    toward the other (the grid turns that into 'x'); neither emits its own reverse."""
    paths, suggest = _sql(
        tmp_path,
        {
            "a.sql": (["ta"], ["tb"]),
            "b.sql": (["tb"], ["ta"]),
        },
    )
    assert suggest("a.sql") == {paths["b.sql"]: "<"}
    assert suggest("b.sql") == {paths["a.sql"]: "<"}


# --- Tracker cells must not be re-emitted as structural suggestions ------------


def _fake_tracker(monkeypatch, headers, rows):
    monkeypatch.setattr(tracker_utils, "find_all_tracker_paths", lambda *a, **k: ["t"])
    monkeypatch.setattr(
        tracker_utils,
        "read_tracker_file_structured",
        lambda *_a, **_k: {
            "grid_headers_ordered": headers,
            "grid_rows_ordered": [(k, compress(v)) for k, v in rows.items()],
        },
    )


def test_existing_tracker_arrow_is_carried_into_suggestions(
    project, mocked_semantics, monkeypatch
):
    """By design, verified tracker relations (including ones an LLM recorded with no
    code import) are merged into ast_map and carried forward with the file's suggestions."""
    root, build = project
    infos = build(["src/a.js", "src/b.js", "src/c.js"])  # keys 1A1, 1A2, 1A3
    mocked_semantics["similarity"] = 0.8  # below the promotion threshold
    # On-disk cell: a '<' b, with no AST link for the pair.
    _fake_tracker(monkeypatch, ["1A1", "1A2", "1A3"], {"1A1": "o<p"})

    out = _run(root, infos, "src/a.js")

    assert out == {"b.js": "<", "c.js": "S"}



def test_existing_tracker_arrow_still_skips_reranking_of_that_pair(
    project, mocked_semantics, monkeypatch
):
    root, build = project
    infos = build(["src/a.js", "src/b.js", "src/c.js"])
    mocked_semantics["rerank_scores"] = [0.97, 0.97]
    _fake_tracker(monkeypatch, ["1A1", "1A2", "1A3"], {"1A1": "o<p"})

    _run(root, infos, "src/a.js")

    # b is already verified in the tracker, so only c is left: no rerank call.
    assert mocked_semantics["rerank_calls"] == 0


def test_ast_link_takes_precedence_over_disagreeing_tracker_cell(
    project, mocked_semantics, monkeypatch
):
    """Guard: the AST arrow is what gets emitted, whatever a stale tracker cell says."""
    root, build = project
    infos = build(["src/a.js", "src/b.js", "src/c.js"])
    a = norm(os.path.join(root, "src/a.js"))
    b = norm(os.path.join(root, "src/b.js"))
    mocked_semantics["similarity"] = 0.8  # below the promotion threshold
    mocked_semantics["ast_links"] = [{"source_path": a, "target_path": b, "char": "<"}]
    _fake_tracker(monkeypatch, ["1A1", "1A2", "1A3"], {"1A1": "o>p"})

    assert _run(root, infos, "src/a.js") == {"b.js": "<", "c.js": "S"}


# --- tracker_io mirror rule: 'x' only when both files report the same char now ---


def test_both_files_report_the_same_char_in_this_pass_is_x(tmp_path, monkeypatch):
    rows = _grid(tmp_path, monkeypatch, {"1A": [("1B", "<")], "1B": [("1A", "<")]}, False)
    assert rows == {"1A": "ox", "1B": "xo"}


def test_stale_reverse_arrow_plus_fresh_arrow_is_not_x(tmp_path, monkeypatch):
    # Disk holds [B][A] = '<' from an earlier run (B depends on A); this pass only
    # re-reports "A depends on B". B does not report it now, so no mutual.
    rows = _grid(
        tmp_path,
        monkeypatch,
        {"1B": [("1A", "<")]},
        False,
        then=[({"1A": [("1B", "<")]}, False)],
    )
    assert "x" not in "".join(rows.values())
    assert rows["1A"][1] == "<" and rows["1B"][0] == "<"


def test_inverted_pair_on_disk_plus_correct_mirror_suggestions_is_not_x(
    tmp_path, monkeypatch
):
    """The types.ts <-> useAutomatedTasks.ts shape: disk has the pair inverted
    ([B][A]='>'), this pass reports the correct pair ([A][B]='>', [B][A]='<')."""
    rows = _grid(
        tmp_path,
        monkeypatch,
        {"1B": [("1A", ">")]},
        False,
        then=[({"1A": [("1B", ">")], "1B": [("1A", "<")]}, False)],
    )
    assert "x" not in "".join(rows.values())


# --- semantic promotion is unchanged (tunable via reranker_promotion_threshold) ---


def test_same_role_promotion_unchanged_and_tunable_by_threshold(
    project, mocked_semantics, monkeypatch
):
    """Same-role pairs at/above reranker_promotion_threshold are promoted via
    _determine_dependency_direction ('x'); the threshold decides who is promoted."""
    from cline_utils.dependency_system.utils.config_manager import ConfigManager

    root, build = project
    infos = build(["src/a.js", "src/b.js", "src/c.js"])
    mocked_semantics["rerank_scores"] = [0.97, 0.95]
    assert _run(root, infos, "src/a.js") == {"b.js": "x", "c.js": "x"}

    real = ConfigManager.get_threshold

    def stricter(self, name, *a, **k):
        if name == "reranker_promotion_threshold":
            return 0.99
        return real(self, name, *a, **k)

    monkeypatch.setattr(ConfigManager, "get_threshold", stricter)
    assert _run(root, infos, "src/a.js") == {"b.js": "S", "c.js": "S"}


# --- reranked results: no inherited characters, only scored pairs logged --------------


def test_low_rerank_score_after_a_high_one_gets_no_character(project, mocked_semantics):
    """assigned_char_semantic is reset per candidate: a score below every threshold
    used to inherit the previous candidate's character (here 'x')."""
    root, build = project
    infos = build(["src/a.js", "src/b.js", "src/c.js"])
    mocked_semantics["rerank_scores"] = [0.97, 0.05]
    assert _run(root, infos, "src/a.js") == {"b.js": "x"}


def _scans(root):
    path = os.path.join(root, "cline_utils", "dependency_system", "analysis", "reranker_scans.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_only_scored_pairs_are_logged_as_scanned(project, mocked_semantics, monkeypatch):
    root, build = project
    infos = build(["src/a.js", "src/b.js", "src/c.js"])
    monkeypatch.setattr(
        embedding_manager, "rerank_candidates_scored",
        lambda q, c, top_k=10, **k: ([(0, 0.97)], {0}),  # candidate 1 left unscored
    )
    out = _run(root, infos, "src/a.js")
    scans = _scans(root)
    assert len(scans) == 1
    assert os.path.basename(scans[0]["target"]) in out  # the scored one


def test_nothing_logged_when_nothing_was_scored(project, mocked_semantics, monkeypatch):
    root, build = project
    infos = build(["src/a.js", "src/b.js", "src/c.js"])
    monkeypatch.setattr(
        embedding_manager, "rerank_candidates_scored", lambda q, c, top_k=10, **k: ([], set())
    )
    _run(root, infos, "src/a.js")
    assert _scans(root) == []
