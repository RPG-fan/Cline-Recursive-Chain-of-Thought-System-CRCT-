"""
The reranker never invents a score. A candidate it cannot score is left out, and
a result with any unscored candidate is not cached. The cases: no VRAM grant before
the deadline, out-of-memory even as a single item, a failed batch, or no reranker.
Out-of-memory batches are retried in halves first. Previously such candidates got
0.0 (or 1.0 when the reranker was unavailable), the result was cached, and the
pairs entered history. Model-free: fake tokenizer and model on CPU, plus a fake VRAM
manager.
"""

import logging
from typing import Any, Dict, List

import pytest
import torch

import cline_utils.dependency_system.analysis.embedding_manager as em
from cline_utils.dependency_system.utils.cache_manager import cache_manager, clear_all_caches

NO_ID, YES_ID, VOCAB, POISON_ID = 1, 2, 8, 7


class _Param:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device) if device == "cpu" else type("D", (), {"type": device})()


class _FakeModel:
    """'yes' logit = item length / 100. Raises CUDA OOM for batches above
    `max_batch`, and (poison_token) for any batch holding a POISON item."""

    def __init__(self, device: str = "cpu", max_batch: int = 99, poison_token: bool = False):
        self.device = device
        self.max_batch = max_batch
        self.poison_token = poison_token
        self.batch_sizes: List[int] = []

    def parameters(self):
        return iter([_Param(self.device)])

    def __call__(self, input_ids, attention_mask, **kwargs: Any):
        n = input_ids.shape[0]
        self.batch_sizes.append(n)
        poisoned = self.poison_token and bool((input_ids == POISON_ID).any())
        if n > self.max_batch or poisoned:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 9 GiB")
        logits = torch.zeros(n, 1, VOCAB)
        logits[:, -1, YES_ID] = attention_mask.sum(dim=1).float() / 100

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


class _FakeTokenizer:
    def __call__(self, texts: List[str], **kwargs: Any) -> Dict[str, List[List[int]]]:
        return {
            "input_ids": [
                [5] * (len(t) // 10 + 1) + ([POISON_ID] if "POISON" in t else [])
                for t in texts
            ]
        }

    def pad(self, items, **kwargs: Any) -> Dict[str, torch.Tensor]:
        width = max(len(i["input_ids"]) for i in items)
        ids = [[0] * (width - len(i["input_ids"])) + i["input_ids"] for i in items]
        mask = [[0] * (width - len(i["input_ids"])) + [1] * len(i["input_ids"]) for i in items]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    clear_all_caches()
    em.reset_rerank_run_state()
    monkeypatch.setattr(em, "reranker_false_id", NO_ID)
    monkeypatch.setattr(em, "reranker_true_id", YES_ID)
    yield
    clear_all_caches()
    em.reset_rerank_run_state()


DOCS = ["a" * 50, "b" * 300, "c" * 900]  # distinct lengths -> distinct scores


def _use(monkeypatch, model) -> None:
    monkeypatch.setattr(em, "_load_reranker_model", lambda: (_FakeTokenizer(), model))


def _cached_entries() -> int:
    cache = cache_manager.get_cache("reranking", em.RERANK_CACHE_TTL_SEC)
    return len(set(cache.data) | set(cache.l1_data))  # an entry may sit in both layers


def test_oom_batch_is_split_and_every_item_scored(monkeypatch) -> None:
    model = _FakeModel(max_batch=1)
    _use(monkeypatch, model)

    results, scored = em.rerank_candidates_scored("q", DOCS, top_k=3)

    assert scored == {0, 1, 2}
    assert [i for i, _ in results] == [2, 1, 0]  # longer -> higher fake score
    assert all(s > 0 for _, s in results)
    assert model.batch_sizes[0] == 3 and set(model.batch_sizes[1:]) == {1}
    assert _cached_entries() == 1  # complete result is cached


def test_item_that_fails_even_alone_is_omitted_not_zero_and_not_cached(monkeypatch, caplog) -> None:
    model = _FakeModel(poison_token=True)
    _use(monkeypatch, model)
    docs = [DOCS[0], "POISON " + DOCS[1], DOCS[2]]

    with caplog.at_level(logging.WARNING, logger=em.logger.name):
        results, scored = em.rerank_candidates_scored("q", docs, top_k=3)

    assert scored == {0, 2}
    assert 1 not in {i for i, _ in results}
    assert all(s > 0 for _, s in results)  # no invented 0.0
    assert _cached_entries() == 0  # incomplete: not cached
    assert any("could not score 1 of 3" in r.getMessage() for r in caplog.records)
    assert 1 in model.batch_sizes  # the OOM batch was split down to single items


class _DenyingVram:
    def __init__(self) -> None:
        self.requests: List[float] = []

    def get_available_for_allocation(self) -> float:
        return 4.0

    def wait_for_available_vram(self, *_a, **_k) -> bool:
        return False

    def request_allocation(self, size_gb, blocking=True, timeout=None):
        self.requests.append(size_gb)
        return False, None

    def release_allocation(self, _id) -> bool:
        return True


def test_no_vram_grant_before_deadline_leaves_everything_unscored(monkeypatch) -> None:
    model = _FakeModel(device="cuda")
    _use(monkeypatch, model)
    vram = _DenyingVram()
    import cline_utils.dependency_system.utils.resource_validator as rv

    monkeypatch.setattr(rv, "get_vram_manager", lambda: vram)
    monkeypatch.setattr(em.torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(em.torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(em, "RERANK_SCORE_DEADLINE_SEC", 0.3)

    results, scored = em.rerank_candidates_scored("q", DOCS, top_k=3)

    assert results == [] and scored == set()
    assert model.batch_sizes == []  # nothing ran without a grant
    assert len(vram.requests) >= 2  # retried (in smaller batches) until the deadline
    assert _cached_entries() == 0

    # Run-wide give-up: later calls in this run do not each wait out the deadline.
    before = len(vram.requests)
    assert em.rerank_candidates_scored("q2", DOCS, top_k=3) == ([], set())
    assert len(vram.requests) == before

    em.reset_rerank_run_state()  # next run (project_analyzer / clear_caches)
    em.rerank_candidates_scored("q3", DOCS, top_k=3)
    assert len(vram.requests) > before


def test_unavailable_reranker_scores_nothing(monkeypatch) -> None:
    monkeypatch.setattr(em, "_load_reranker_model", lambda: (None, None))
    assert em.rerank_candidates_scored("q", DOCS, top_k=3) == ([], set())
    assert em.rerank_candidates_with_qwen3("q", DOCS, top_k=3) == []  # was 1.0 each
    assert _cached_entries() == 0


def test_complete_result_is_served_from_cache(monkeypatch) -> None:
    model = _FakeModel()
    _use(monkeypatch, model)
    first = em.rerank_candidates_with_qwen3("q", DOCS, top_k=2)
    calls = len(model.batch_sizes)
    assert em.rerank_candidates_with_qwen3("q", DOCS, top_k=2) == first
    assert len(model.batch_sizes) == calls
    assert len(first) == 2
