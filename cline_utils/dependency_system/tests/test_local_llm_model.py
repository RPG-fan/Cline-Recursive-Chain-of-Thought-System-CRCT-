"""
The local LLM (resolve-placeholders, determine-dependency, normalize-docs) is downloaded
on first use. No real download or model load: the downloader is mocked.
"""

import os
from typing import List
from unittest.mock import patch

import pytest

import cline_utils.dependency_system.analysis.embedding_manager as em
from cline_utils.dependency_system.analysis import local_llm_model as llm

GGUF_BYTES = b"GGUF" + b"\0" * 2_000_000


def _fake_download(content: bytes, ok: bool = True, calls: List[str] = None):
    def download(url: str, path: str, description: str, **_k) -> bool:
        if calls is not None:
            calls.append(path)
        if ok:
            with open(path, "wb") as f:
                f.write(content)
        return ok

    return download


def test_an_intact_default_model_is_used_without_downloading(tmp_path) -> None:
    path = llm.default_local_llm_path(str(tmp_path))
    os.makedirs(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(GGUF_BYTES)
    with patch.object(em, "_download_with_retry") as download:
        assert llm.ensure_local_llm_model(None, str(tmp_path)) == path
    download.assert_not_called()


def test_a_missing_default_model_is_downloaded_via_a_part_file(tmp_path) -> None:
    calls: List[str] = []
    with patch.object(em, "_download_with_retry", side_effect=_fake_download(GGUF_BYTES, calls=calls)):
        path = llm.ensure_local_llm_model(None, str(tmp_path))
    assert path == llm.default_local_llm_path(str(tmp_path))
    assert calls == [path + ".part"]
    assert os.path.getsize(path) == len(GGUF_BYTES)
    assert not os.path.exists(path + ".part")


def test_the_download_url_is_the_unsloth_q8_0_file() -> None:
    assert llm.DEFAULT_LOCAL_LLM_URL == (
        "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/"
        "Qwen3-4B-Instruct-2507-Q8_0.gguf"
    )


def test_a_failed_download_leaves_no_model_and_returns_none(tmp_path, capsys) -> None:
    with patch.object(em, "_download_with_retry", side_effect=_fake_download(b"", ok=False)):
        assert llm.ensure_local_llm_model(None, str(tmp_path)) is None
    assert not os.path.exists(llm.default_local_llm_path(str(tmp_path)))
    assert "could not download" in capsys.readouterr().out


def test_a_download_that_is_not_gguf_is_discarded(tmp_path) -> None:
    with patch.object(em, "_download_with_retry", side_effect=_fake_download(b"<html>" * 400_000)):
        assert llm.ensure_local_llm_model(None, str(tmp_path)) is None
    path = llm.default_local_llm_path(str(tmp_path))
    assert not os.path.exists(path)
    assert not os.path.exists(path + ".part")


def test_a_corrupt_default_model_is_downloaded_again(tmp_path) -> None:
    path = llm.default_local_llm_path(str(tmp_path))
    os.makedirs(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(b"oops")
    with patch.object(em, "_download_with_retry", side_effect=_fake_download(GGUF_BYTES)):
        assert llm.ensure_local_llm_model(None, str(tmp_path)) == path
    assert os.path.getsize(path) == len(GGUF_BYTES)


@pytest.mark.parametrize("exists", [True, False])
def test_an_explicit_model_path_is_never_downloaded(tmp_path, exists) -> None:
    custom = tmp_path / "custom.gguf"
    if exists:
        custom.write_bytes(GGUF_BYTES)
    with patch.object(em, "_download_with_retry") as download:
        result = llm.ensure_local_llm_model(str(custom), str(tmp_path))
    download.assert_not_called()
    assert result == (str(custom) if exists else None)
