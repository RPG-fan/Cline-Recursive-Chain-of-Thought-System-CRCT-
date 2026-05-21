# cline_utils/dependency_system/tests/test_tokenizer_factory.py

import os
import pytest
from unittest.mock import MagicMock
from cline_utils.dependency_system.utils import tokenizer_factory


def test_count_tokens_fallback() -> None:
    """Verifies that token counting falls back correctly if no tokenizer is provided."""
    assert tokenizer_factory.count_tokens("") == 0
    # Consistently len(text) // 4
    assert tokenizer_factory.count_tokens("12345678") == 2
    assert tokenizer_factory.count_tokens("12") == 0


def test_count_tokens_with_mock_tokenizer() -> None:
    """Verifies that token counting uses the provided tokenizer's encode method."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    
    count = tokenizer_factory.count_tokens("some text", mock_tokenizer)
    assert count == 5
    mock_tokenizer.encode.assert_called_once_with("some text", add_special_tokens=False)


def test_count_tokens_with_mock_llama_tokenizer() -> None:
    """Verifies that token counting uses the provided tokenizer's tokenize method if encode fails."""
    mock_tokenizer = MagicMock()
    del mock_tokenizer.encode  # Remove encode to test fallback to tokenize
    mock_tokenizer.tokenize.return_value = [1, 2, 3]
    
    count = tokenizer_factory.count_tokens("some text", mock_tokenizer)
    assert count == 3
    mock_tokenizer.tokenize.assert_called_once_with(b"some text")


def test_get_tokenizer_caching() -> None:
    """Verifies that tokenizer factory caches successfully loaded instances."""
    tokenizer_factory._tokenizer_cache.clear()
    
    mock_tokenizer = MagicMock()
    tokenizer_factory._tokenizer_cache["dummy_model"] = mock_tokenizer
    
    resolved = tokenizer_factory.get_tokenizer("dummy_model")
    assert resolved is mock_tokenizer


def test_get_tokenizer_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that tokenizer_factory handles ImportError of transformers gracefully."""
    tokenizer_factory._tokenizer_cache.clear()
    
    # Simulate transformers import error
    import sys
    orig_modules = sys.modules.copy()
    monkeypatch.setitem(sys.modules, "transformers", None)
    
    # Clear out local cache for clean test
    resolved = tokenizer_factory.get_tokenizer("Qwen/Qwen2.5-7B-Instruct")
    assert resolved is None
    
    # Restore modules
    sys.modules.update(orig_modules)


def test_get_tokenizer_local_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that tokenizer_factory searches candidates and loads locally if files exist."""
    tokenizer_factory._tokenizer_cache.clear()
    
    mock_tokenizer = MagicMock()
    
    class MockAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path: str, local_files_only: bool = False) -> MagicMock:
            assert local_files_only is True
            return mock_tokenizer
            
    # Mock OS path checking and from_pretrained loading
    monkeypatch.setattr(os.path, "exists", lambda x: True)
    
    import sys
    from types import ModuleType
    mock_transformers = ModuleType("transformers")
    mock_transformers.AutoTokenizer = MockAutoTokenizer  # type: ignore
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)
    
    resolved = tokenizer_factory.get_tokenizer("local_gguf_model.gguf")
    assert resolved is mock_tokenizer
