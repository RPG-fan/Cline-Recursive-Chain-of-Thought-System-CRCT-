"""Reranker cache key must vary with instruction and source file path."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from cline_utils.dependency_system.analysis.embedding_manager import (
    _get_rerank_cache_key,
    rerank_candidates_with_qwen3,
)
from cline_utils.dependency_system.utils.cache_manager import cache_manager, clear_all_caches


@pytest.fixture(autouse=True)
def _clear_rerank_cache():
    clear_all_caches()
    yield
    clear_all_caches()


def test_rerank_cache_key_differs_by_instruction() -> None:
    query = "same query"
    candidates = ["doc1", "doc2"]
    base_kwargs = {
        "top_k": 10,
        "source_file_path": "/project/src/foo.py",
    }

    key_a = _get_rerank_cache_key(
        query, candidates, instruction="Instruction A", **base_kwargs
    )
    key_b = _get_rerank_cache_key(
        query, candidates, instruction="Instruction B", **base_kwargs
    )

    assert key_a != key_b


def test_rerank_cache_key_differs_by_source_file_path() -> None:
    query = "same query"
    candidates = ["doc1", "doc2"]
    instruction = "same instruction"

    key_a = _get_rerank_cache_key(
        query,
        candidates,
        top_k=10,
        source_file_path="/project/src/foo.py",
        instruction=instruction,
    )
    key_b = _get_rerank_cache_key(
        query,
        candidates,
        top_k=10,
        source_file_path="/project/src/bar.py",
        instruction=instruction,
    )

    assert key_a != key_b


def test_rerank_cache_key_stable_for_identical_inputs() -> None:
    kwargs = {
        "top_k": 5,
        "source_file_path": "h:/project/src/foo.py",
        "instruction": "Retrieve related code.",
    }
    key_one = _get_rerank_cache_key("query", ["a", "b"], **kwargs)
    key_two = _get_rerank_cache_key("query", ["a", "b"], **kwargs)
    assert key_one == key_two


@patch("cline_utils.dependency_system.analysis.embedding_manager._load_reranker_model")
def test_rerank_candidates_cache_miss_on_different_instructions(
    mock_load_model: MagicMock,
) -> None:
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer.pad.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_tokenizer.convert_tokens_to_ids.side_effect = (
        lambda token: 1 if token == "no" else (2 if token == "yes" else 0)
    )
    mock_tokenizer.side_effect = lambda texts, **kwargs: {
        "input_ids": [[1, 2, 3]] * len(texts)
    }
    mock_model.return_value.logits = torch.randn(1, 3, 10)
    mock_model.parameters.side_effect = lambda: iter(
        [MagicMock(device=MagicMock(type="cpu"))]
    )
    mock_load_model.return_value = (mock_tokenizer, mock_model)

    import cline_utils.dependency_system.analysis.embedding_manager as em

    em.reranker_false_id = 1
    em.reranker_true_id = 2

    query = "test query"
    candidates = ["doc1", "doc2"]
    shared = {
        "top_k": 2,
        "source_file_path": "/project/src/module.py",
    }

    rerank_candidates_with_qwen3(
        query, candidates, instruction="Instruction A", **shared
    )
    rerank_candidates_with_qwen3(
        query, candidates, instruction="Instruction B", **shared
    )

    cache = cache_manager.get_cache("reranking")
    assert cache.metrics.misses == 2
    assert cache.metrics.hits == 0
