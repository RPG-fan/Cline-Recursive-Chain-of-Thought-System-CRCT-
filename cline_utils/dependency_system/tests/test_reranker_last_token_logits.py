"""
The reranker scores only the last token, so its forward pass must ask the model for
last-position logits only (logits_to_keep=1). Without it the model materializes a
batch x seq_len x vocab (151,669) logits tensor: about 296MB per 1k tokens per item, which
the VRAM batch estimate does not budget for. Batches of 7-11 items at around 3k tokens
then tried 6-9GB single allocations on an 8GB card. No real model is loaded here.
"""

from typing import Any, Dict, List

import pytest
import torch

import cline_utils.dependency_system.analysis.embedding_manager as em
from cline_utils.dependency_system.utils.cache_manager import clear_all_caches

NO_ID, YES_ID, VOCAB = 1, 2, 8


class _Param:
    device = torch.device("cpu")


class _FakeReranker:
    """Records forward kwargs; the 'yes' logit grows with the item's real (unpadded) length."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def parameters(self):
        return iter([_Param()])

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any):
        self.calls.append(kwargs)
        keep = kwargs.get("logits_to_keep", 0)
        seq = input_ids.shape[1] if keep == 0 else keep
        logits = torch.zeros(input_ids.shape[0], seq, VOCAB)
        logits[:, -1, YES_ID] = attention_mask.sum(dim=1).float() / 100  # unsaturated

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


class _FakeTokenizer:
    def __call__(self, texts: List[str], **kwargs: Any) -> Dict[str, List[List[int]]]:
        return {"input_ids": [[5] * (len(t) // 10 + 1) for t in texts]}

    def pad(self, items: List[Dict[str, List[int]]], **kwargs: Any) -> Dict[str, torch.Tensor]:
        width = max(len(i["input_ids"]) for i in items)
        ids = [[0] * (width - len(i["input_ids"])) + i["input_ids"] for i in items]
        mask = [[0] * (width - len(i["input_ids"])) + [1] * len(i["input_ids"]) for i in items]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


@pytest.fixture(autouse=True)
def _clear_rerank_cache():
    clear_all_caches()
    yield
    clear_all_caches()


def test_reranker_requests_last_token_logits_only(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeReranker()
    monkeypatch.setattr(em, "_load_reranker_model", lambda: (_FakeTokenizer(), model))
    monkeypatch.setattr(em, "reranker_false_id", NO_ID)
    monkeypatch.setattr(em, "reranker_true_id", YES_ID)

    short_doc, long_doc = "s" * 50, "l" * 2000
    results = em.rerank_candidates_with_qwen3(
        "last-token query",
        [short_doc, long_doc],
        top_k=2,
        source_file_path="/project/src/a.py",
        instruction="test instruction",
    )

    assert model.calls, "reranker forward pass never ran"
    assert all(call.get("logits_to_keep") == 1 for call in model.calls)
    # Scores still come from the last position: the longer item has the larger 'yes' logit.
    assert [idx for idx, _ in results] == [1, 0]
    assert results[0][1] > results[1][1]
