import pytest
from typing import Any
from cline_utils.dependency_system.analysis import local_llm_processor as llm_mod


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

    assert len(_FakeLlama.created) >= 2
    # The first load is the tokenizer-only load with n_gpu_layers=0
    assert _FakeLlama.created[0]["n_gpu_layers"] == 0
    # The second load is the inference load which dynamically calculated layers > 0
    assert _FakeLlama.created[1]["n_gpu_layers"] > 0
