"""VRAM cleanup when Qwen3 reranker loading fails."""

from unittest.mock import MagicMock, patch

import pytest

import cline_utils.dependency_system.analysis.embedding_manager as em
from cline_utils.dependency_system.analysis.embedding_manager import (
    _clear_reranker_load_state,
    _load_reranker_model,
)


@pytest.fixture(autouse=True)
def _reset_reranker_singletons():
    em._reranker_model = None
    em._reranker_tokenizer = None
    em.reranker_false_id = None
    em.reranker_true_id = None
    em._reranker_prefix_tokens = None
    em._reranker_suffix_tokens = None
    yield
    _clear_reranker_load_state()


def test_clear_reranker_load_state_resets_globals() -> None:
    em._reranker_model = MagicMock()
    em._reranker_tokenizer = MagicMock()
    em.reranker_false_id = 1
    em.reranker_true_id = 2
    em._reranker_prefix_tokens = [1]
    em._reranker_suffix_tokens = [2]

    with (
        patch("cline_utils.dependency_system.analysis.embedding_manager.gc.collect") as mock_gc,
        patch(
            "cline_utils.dependency_system.analysis.embedding_manager._flush_accelerator_memory_cache"
        ) as mock_flush,
    ):
        _clear_reranker_load_state()

    assert em._reranker_model is None
    assert em._reranker_tokenizer is None
    assert em.reranker_false_id is None
    assert em.reranker_true_id is None
    assert em._reranker_prefix_tokens is None
    assert em._reranker_suffix_tokens is None
    mock_gc.assert_called_once()
    mock_flush.assert_called_once()


@patch("cline_utils.dependency_system.analysis.embedding_manager._verify_reranker_model", return_value=True)
@patch("cline_utils.dependency_system.analysis.embedding_manager._select_device", return_value="cuda")
@patch("cline_utils.dependency_system.analysis.embedding_manager.get_project_root", return_value="/proj")
@patch("transformers.AutoTokenizer")
@patch("transformers.AutoModelForCausalLM")
def test_load_reranker_failure_invokes_gc_and_cuda_flush(
    mock_model_cls: MagicMock,
    mock_tokenizer_cls: MagicMock,
    _mock_project_root: MagicMock,
    _mock_device: MagicMock,
    _mock_verify: MagicMock,
) -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 1 if token == "no" else 2
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    mock_model_cls.from_pretrained.side_effect = RuntimeError("CUDA out of memory")

    with (
        patch("cline_utils.dependency_system.analysis.embedding_manager.gc.collect") as mock_gc,
        patch(
            "cline_utils.dependency_system.analysis.embedding_manager._flush_accelerator_memory_cache"
        ) as mock_flush,
    ):
        tokenizer, model = _load_reranker_model()

    assert tokenizer is None
    assert model is None
    mock_gc.assert_called()
    mock_flush.assert_called()


@patch("cline_utils.dependency_system.analysis.embedding_manager._verify_reranker_model", return_value=True)
@patch("cline_utils.dependency_system.analysis.embedding_manager._select_device", return_value="cuda")
@patch("cline_utils.dependency_system.analysis.embedding_manager.get_project_root", return_value="/proj")
@patch("transformers.AutoTokenizer")
@patch("transformers.AutoModelForCausalLM")
def test_load_reranker_optimization_fallback_purges_before_retry(
    mock_model_cls: MagicMock,
    mock_tokenizer_cls: MagicMock,
    _mock_project_root: MagicMock,
    _mock_device: MagicMock,
    _mock_verify: MagicMock,
) -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 1 if token == "no" else 2
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

    mock_model_cls.from_pretrained.side_effect = [
        RuntimeError("optimized load failed"),
        RuntimeError("fallback also failed"),
    ]

    with patch(
        "cline_utils.dependency_system.analysis.embedding_manager._release_reranker_model_allocation"
    ) as mock_release:
        tokenizer, model = _load_reranker_model()

    mock_release.assert_called_once()
    assert tokenizer is None
    assert model is None
