"""
The upfront similarity index must give exactly the scores calculate_similarity gives
pair by pair (dot of stored vectors, clamped to [0, 1], 0.0 when an embedding is
missing or has the wrong dimension), while loading each embedding only once. A
wrong-dimension file is never deleted by the index. No model is loaded: the selected
model config is stubbed, and the vectors are small arrays in tmp_path.
"""

import os
from typing import Dict

import numpy as np
import pytest

from cline_utils.dependency_system.analysis import embedding_manager as em
from cline_utils.dependency_system.analysis import similarity_index
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.cache_manager import clear_all_caches
from cline_utils.dependency_system.utils.path_utils import normalize_path

DIM = 8


@pytest.fixture
def project(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(em, "_selected_model_config", {"embedding_dim": DIM})
    clear_all_caches()
    similarity_index.invalidate()
    root = normalize_path(str(tmp_path / "proj"))
    emb = normalize_path(str(tmp_path / "proj" / "embeddings"))
    rng = np.random.default_rng(7)
    infos: Dict[str, KeyInfo] = {}
    for i, name in enumerate(["a.ts", "b.ts", "c.ts", "d.ts", "missing.ts", "wrongdim.ts"]):
        path = normalize_path(os.path.join(root, "src", name))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        infos[path] = KeyInfo(f"1A{i + 1}", path, normalize_path(os.path.dirname(path)), 1, False)
        npy = os.path.join(emb, "src", name) + ".npy"
        os.makedirs(os.path.dirname(npy), exist_ok=True)
        if name == "missing.ts":
            continue
        vec = rng.normal(size=DIM * 2 if name == "wrongdim.ts" else DIM).astype(np.float32)
        if name == "d.ts":
            vec = -np.load(os.path.join(emb, "src", "a.ts.npy"))  # negative -> clamped to 0
        np.save(npy, vec / np.linalg.norm(vec))
    yield root, emb, infos
    similarity_index.invalidate()
    clear_all_caches()


def test_bulk_scores_equal_pairwise_calculate_similarity(project) -> None:
    root, emb, infos = project
    wrongdim_npy = os.path.join(emb, "src", "wrongdim.ts.npy")
    good = [ki for ki in infos.values() if not ki.norm_path.endswith("wrongdim.ts")]

    for src in infos.values():
        row = similarity_index.get_scores(src.key_string, emb, root, infos)
        for tgt in good:
            if src.norm_path.endswith("wrongdim.ts"):
                continue
            expected = em.calculate_similarity(
                src.key_string, tgt.key_string, emb, infos, root, [], []
            )
            assert row.get(tgt.key_string, 0.0) == pytest.approx(expected, abs=1e-6)
        # A wrong-dimension embedding scores 0.0 and is NOT deleted by the index.
        assert row.get(infos[next(p for p in infos if p.endswith("wrongdim.ts"))].key_string, 0.0) == 0.0
    assert os.path.exists(wrongdim_npy)

    a, d = (next(k for p, k in infos.items() if p.endswith(n)) for n in ("a.ts", "d.ts"))
    assert similarity_index.get_scores(a.key_string, emb, root, infos)[d.key_string] == 0.0
    assert similarity_index.get_scores(a.key_string, emb, root, infos)[a.key_string] == pytest.approx(1.0)
    missing = next(k for p, k in infos.items() if p.endswith("missing.ts"))
    assert similarity_index.get_scores(missing.key_string, emb, root, infos) == {}


def test_source_without_embedding_never_consults_the_model(project, monkeypatch) -> None:
    root, emb, infos = project
    monkeypatch.setattr(em, "_selected_model_config", None)
    calls = []
    monkeypatch.setattr(em, "_select_best_model", lambda: calls.append(1) or {"embedding_dim": DIM})
    missing = next(k for p, k in infos.items() if p.endswith("missing.ts"))

    assert similarity_index.get_scores(missing.key_string, emb, root, infos) == {}
    assert similarity_index.get_scores("no-such-key", emb, root, infos) == {}
    assert calls == []


def test_each_embedding_is_loaded_once_per_index(project, monkeypatch) -> None:
    root, emb, infos = project
    loads = []
    real_load = np.load
    monkeypatch.setattr(similarity_index.np, "load", lambda p, *a, **k: loads.append(p) or real_load(p, *a, **k))

    similarity_index.prepare(emb, root, infos)
    for ki in infos.values():
        similarity_index.get_scores(ki.key_string, emb, root, infos)

    assert len(loads) == 5  # every existing .npy once; missing.ts has none
    assert len(set(loads)) == 5


def test_index_rebuilds_for_new_key_map_and_after_invalidate(project, monkeypatch) -> None:
    root, emb, infos = project
    builds = []
    real_init = similarity_index.SimilarityIndex.__init__

    def counting_init(self, *a, **k):
        builds.append(1)
        real_init(self, *a, **k)

    monkeypatch.setattr(similarity_index.SimilarityIndex, "__init__", counting_init)
    key = next(iter(infos.values())).key_string

    similarity_index.get_scores(key, emb, root, infos)
    similarity_index.get_scores(key, emb, root, infos)
    assert len(builds) == 1

    similarity_index.get_scores(key, emb, root, dict(infos))  # a different key map
    assert len(builds) == 2

    similarity_index.invalidate()
    similarity_index.get_scores(key, emb, root, infos)
    assert len(builds) == 3


def test_prepared_get_scores_skips_key_scan_and_disk_stat(project, monkeypatch) -> None:
    root, emb, infos = project
    similarity_index.prepare(emb, root, infos)

    monkeypatch.setattr(
        similarity_index.os.path,
        "exists",
        lambda _p: pytest.fail("get_scores must not stat disk when index is already prepared"),
    )
    for ki in infos.values():
        similarity_index.get_scores(ki.key_string, emb, root, infos)
    assert similarity_index.get_scores("no-such-key", emb, root, infos) == {}
