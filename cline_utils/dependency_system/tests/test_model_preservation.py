"""
The Qwen3 embedding model is a multi-GB download and many CRCT users have metered
bandwidth, so an intact model file must never be deleted or re-downloaded just because
a load test failed (busy VRAM, drivers, a concurrent process). Only a structurally
corrupt file may be replaced. A relative `qwen3_model_path` resolves against the
project root, never the current working directory.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cline_utils.dependency_system.analysis import embedding_manager as em


def _intact_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + b"\0" * 2_000_000)
    return path


def test_intact_model_that_fails_to_load_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _intact_gguf(tmp_path / "model.gguf")
    download = MagicMock(return_value=True)
    monkeypatch.setattr(em, "_verify_qwen3_model", lambda p: False)  # load test fails
    monkeypatch.setattr(em, "_download_with_retry", download)

    assert em._download_qwen3_model(str(model)) is False  # pyright: ignore[reportPrivateUsage]
    assert model.exists()
    download.assert_not_called()


def test_corrupt_model_is_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not a model")
    download = MagicMock(return_value=False)
    monkeypatch.setattr(em, "_download_with_retry", download)

    assert em._download_qwen3_model(str(model)) is False  # pyright: ignore[reportPrivateUsage]
    assert not model.exists()
    download.assert_called_once()


def test_downloaded_model_that_fails_to_load_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model.gguf"

    def fake_download(url: str, path: str, *args: object, **kwargs: object) -> bool:
        _intact_gguf(Path(path))
        return True

    monkeypatch.setattr(em, "_download_with_retry", fake_download)
    monkeypatch.setattr(em, "_verify_qwen3_model", lambda p: False)

    assert em._download_qwen3_model(str(model)) is False  # pyright: ignore[reportPrivateUsage]
    assert model.exists()


def test_relative_model_path_resolves_against_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = MagicMock()
    config.get_embedding_setting.return_value = "models/Qwen3-Embedding-4B-Q6_K.gguf"
    monkeypatch.setattr(em, "get_project_root", lambda: str(tmp_path))

    resolved = em._configured_qwen3_path(config)  # pyright: ignore[reportPrivateUsage]

    assert Path(resolved) == tmp_path / "models" / "Qwen3-Embedding-4B-Q6_K.gguf"
