from cline_utils.dependency_system.analysis import local_llm_processor as llm_mod


class _FakeConfigManager:
    def __init__(self):
        self.config = {}

    def get_embedding_setting(self, setting_name, default=None):
        if setting_name == "qwen3_context_length":
            return 32768
        if setting_name == "qwen3_gpu_layers":
            return 5
        return default


class _FakeLlama:
    created = []

    def __init__(self, model_path, n_ctx, n_gpu_layers, verbose):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        _FakeLlama.created.append({"n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers})

    def tokenize(self, payload):
        text = payload.decode("utf-8", errors="ignore")
        return [1] * max(1, len(text) // 4)

    def __call__(self, prompt, max_tokens, stop, echo):
        return {"choices": [{"text": "d"}]}


def test_token_based_context_sizing_avoids_unneeded_32768_reload(monkeypatch):
    _FakeLlama.created.clear()
    monkeypatch.setattr(llm_mod, "ConfigManager", _FakeConfigManager)
    monkeypatch.setattr(llm_mod, "Llama", _FakeLlama)

    processor = llm_mod.LocalLLMProcessor(model_path="models/fake.gguf")
    result = processor.determine_dependency(
        source_content="source",
        target_content="target",
        source_basename="source.md",
        target_basename="target.md",
        source_tokens=4171,
        target_tokens=8508,
        instructional_prompt="Check dependency.",
    )
    processor.close()

    assert result == "d"
    assert _FakeLlama.created
    assert max(entry["n_ctx"] for entry in _FakeLlama.created) <= 16384


def test_local_llm_uses_configured_gpu_layers(monkeypatch):
    _FakeLlama.created.clear()

    class _ConfiguredLayers(_FakeConfigManager):
        def get_embedding_setting(self, setting_name, default=None):
            if setting_name == "qwen3_context_length":
                return 16384
            if setting_name == "qwen3_gpu_layers":
                return 11
            return default

    monkeypatch.setattr(llm_mod, "ConfigManager", _ConfiguredLayers)
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

    assert _FakeLlama.created
    assert all(entry["n_gpu_layers"] == 11 for entry in _FakeLlama.created)
