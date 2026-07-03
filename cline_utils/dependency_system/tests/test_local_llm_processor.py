import pytest
from typing import Any
from unittest.mock import MagicMock
from cline_utils.dependency_system.analysis import local_llm_processor as llm_mod
from cline_utils.dependency_system.utils import tokenizer_factory


class _FakeLlama:
    created: list[dict[str, int]] = []

    def __init__(
        self,
        model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
        verbose: bool,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        _FakeLlama.created.append({"n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers})

    def tokenize(self, payload: bytes) -> list[int]:
        text = payload.decode("utf-8", errors="ignore")
        return [1] * max(1, len(text) // 4)

    def __call__(
        self,
        prompt: str,
        max_tokens: int,
        stop: list[str] | None,
        echo: bool,
        **kwargs: Any
    ) -> dict[str, list[dict[str, str]]]:
        return {"choices": [{"text": "a.py b.py -> n\nReasoning: No relationship."}]}


def test_token_based_context_sizing_avoids_unneeded_32768_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeLlama.created.clear()
    # Disable central tokenizer to force Llama tokenization fallback
    monkeypatch.setattr(tokenizer_factory, "get_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_mod, "Llama", _FakeLlama)

    processor = llm_mod.LocalLLMProcessor(model_path="models/fake.gguf")
    result, _ = processor.determine_dependency(
        source_content="source",
        target_content="target",
        source_basename="source.md",
        target_basename="target.md",
        source_tokens=4171,
        target_tokens=8508,
        instructional_prompt="Check dependency.",
    )
    processor.close()

    assert result == "n"
    assert _FakeLlama.created
    assert max(entry["n_ctx"] for entry in _FakeLlama.created) <= 16384


def test_local_llm_calculates_gpu_layers_dynamically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeLlama.created.clear()
    # Disable central tokenizer to force Llama tokenization fallback
    monkeypatch.setattr(tokenizer_factory, "get_tokenizer", lambda *args, **kwargs: None)

    class FakeResourceValidator:
        def validate_gpu(self) -> dict[str, Any]:
            return {"gpu_available": True, "vram_available_mb": 4000.0}  # 4GB VRAM

        def wait_for_vram_release(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(llm_mod, "ResourceValidator", FakeResourceValidator)
    monkeypatch.setattr(llm_mod, "Llama", _FakeLlama)

    processor = llm_mod.LocalLLMProcessor(model_path="models/fake.gguf")
    processor.determine_dependency(
        source_content="short source",
        target_content="short target",
        source_basename="a.py",
        target_basename="b.py",
        source_tokens=120,
        target_tokens=120,
        instructional_prompt="Analyze.",
    )
    processor.close()

    assert len(_FakeLlama.created) == 1
    # The inference load dynamically calculated layers > 0
    assert _FakeLlama.created[0]["n_gpu_layers"] > 0


def test_local_llm_uses_tokenizer_factory_optimization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeLlama.created.clear()

    # Mock a successful tokenizer from tokenizer_factory
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1] * 50
    monkeypatch.setattr(tokenizer_factory, "get_tokenizer", lambda *args, **kwargs: mock_tok)
    monkeypatch.setattr(tokenizer_factory, "count_tokens", lambda text, tok: len(tok.encode(text)))

    monkeypatch.setattr(llm_mod, "Llama", _FakeLlama)

    processor = llm_mod.LocalLLMProcessor(model_path="models/fake.gguf")
    processor.determine_dependency(
        source_content="short source",
        target_content="short target",
        source_basename="a.py",
        target_basename="b.py",
        source_tokens=120,
        target_tokens=120,
        instructional_prompt="Analyze.",
    )
    processor.close()

    # With the optimization, Llama is never loaded for token counting.
    # It is only loaded once for the actual inference run.
    assert len(_FakeLlama.created) == 1


def test_get_fresh_resources() -> None:
    processor = llm_mod.LocalLLMProcessor(model_path="models/fake.gguf")
    ram, vram = processor._get_fresh_resources()
    assert isinstance(ram, float)
    assert ram > 0
    assert vram is None or isinstance(vram, float)
    processor.close()


def test_save_pinned_state_resource_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = llm_mod.LocalLLMProcessor(model_path="models/fake.gguf")

    # Mock model with save_state
    mock_model = MagicMock()
    mock_model.save_state.return_value = b"some_state"
    mock_model.ctx = MagicMock()
    processor._model = mock_model

    # Mock llama_cpp.llama_get_state_size to return 10MB
    import llama_cpp
    monkeypatch.setattr(llama_cpp, "llama_get_state_size", lambda ctx: 10 * 1024 * 1024)

    # Scenario 1: Model uses GPU, VRAM is sufficient, RAM is sufficient
    processor._current_n_gpu_layers = 16
    monkeypatch.setattr(processor, "_get_fresh_resources", lambda: (4000.0, 1000.0))
    processor.save_pinned_state()
    assert processor._pinned_state == b"some_state"
    mock_model.save_state.assert_called_once()

    # Scenario 2: Model uses GPU, VRAM falls short, RAM is sufficient (fallback to RAM)
    processor.clear_pinned_state()
    mock_model.save_state.reset_mock()
    monkeypatch.setattr(processor, "_get_fresh_resources", lambda: (4000.0, 200.0))  # VRAM falls short
    processor.save_pinned_state()
    assert processor._pinned_state == b"some_state"  # Pinning still succeeds to RAM
    mock_model.save_state.assert_called_once()

    # Scenario 3: Model uses GPU, VRAM falls short AND RAM falls short (bypass pinning)
    processor.clear_pinned_state()
    mock_model.save_state.reset_mock()
    monkeypatch.setattr(processor, "_get_fresh_resources", lambda: (400.0, 200.0))  # both fall short
    processor.save_pinned_state()
    assert processor._pinned_state is None
    mock_model.save_state.assert_not_called()

    # Scenario 4: Model uses CPU, RAM is sufficient
    processor._current_n_gpu_layers = 0
    processor.clear_pinned_state()
    mock_model.save_state.reset_mock()
    monkeypatch.setattr(processor, "_get_fresh_resources", lambda: (4000.0, None))
    processor.save_pinned_state()
    assert processor._pinned_state == b"some_state"
    mock_model.save_state.assert_called_once()

    # Scenario 5: Model uses CPU, RAM falls short
    processor.clear_pinned_state()
    mock_model.save_state.reset_mock()
    monkeypatch.setattr(processor, "_get_fresh_resources", lambda: (400.0, None))  # RAM falls short
    processor.save_pinned_state()
    assert processor._pinned_state is None
    mock_model.save_state.assert_not_called()


def test_save_pinned_state_exception_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = llm_mod.LocalLLMProcessor(model_path="models/fake.gguf")
    mock_model = MagicMock()
    mock_model.save_state.side_effect = MemoryError("Out of memory during bytes conversion")
    mock_model.ctx = MagicMock()
    processor._model = mock_model

    # Ensure sufficient resources so save_state is actually called
    monkeypatch.setattr(processor, "_get_fresh_resources", lambda: (16000.0, 8000.0))
    import llama_cpp
    monkeypatch.setattr(llama_cpp, "llama_get_state_size", lambda ctx: 10 * 1024 * 1024)

    # Should catch MemoryError and clear state
    processor.save_pinned_state()
    assert processor._pinned_state is None

    # Test restore exception safety
    processor._pinned_state = b"some_pinned_state"
    mock_model.load_state.side_effect = Exception("Runtime error in load_state")
    processor.restore_pinned_state()
    assert processor._pinned_state is None  # Should clear state on failure
