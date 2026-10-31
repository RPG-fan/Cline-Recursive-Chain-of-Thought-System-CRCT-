"""
The selected embedding model is never swapped at run time (2026-10-30, cline_LLMRPG run):
the Qwen3 GGUF failed to load, the run switched to mpnet, metadata.json was rewritten
to name mpnet with every ses_hash cleared, and the mpnet batches crashed with a CUDA
device-side assert (max_seq_length raised to 12,800 on a 512-position model).

Now:
- a GGUF load failure is retried once, then embedding generation is skipped for the run;
  vectors and metadata.json are left exactly as they were;
- metadata.json never names a new model unless every vector was rewritten with it, and
  a file whose vector could not be written keeps its previous ses_hash;
- a failed load reports llama.cpp's own log lines and the free VRAM;
- a SentenceTransformer's max_seq_length never exceeds the model's context_length.
No model is loaded: llama-cpp and the encoder are mocked.
"""

import contextlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import cline_utils.dependency_system.analysis.embedding_manager as em
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.path_utils import normalize_path

QWEN = dict(em.MODEL_CONFIGS["qwen3-4b"], path="unused.gguf")
QWEN_DIM = int(QWEN["embedding_dim"])


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(em.time, "sleep", lambda s: None)


class Project:
    def __init__(self, tmp_path: Path, metadata_model: str) -> None:
        self.root = tmp_path / "proj"
        (self.root / "src").mkdir(parents=True)
        self.files = []
        for name in ("a.py", "b.py"):
            src = self.root / "src" / name
            src.write_text(f"def {name[0]}():\n    return 1\n", encoding="utf-8")
            self.files.append(normalize_path(str(src)))
        self.emb_dir = self.root / "embeddings"
        (self.emb_dir / "src").mkdir(parents=True)
        self.infos = {
            p: KeyInfo(f"1A{i + 1}", p, normalize_path(str(self.root / "src")), 1, False)
            for i, p in enumerate(self.files)
        }
        # An existing index: old vectors and metadata whose ses_hash no longer matches.
        for p in self.files:
            np.save(str(self.emb_dir / "src" / (Path(p).name + ".npy")), np.ones(QWEN_DIM, np.float32))
        self.metadata = {
            "version": em.EMBEDDING_METADATA_VERSION,
            "model": metadata_model,
            "keys": {
                ki.key_string: {"path": p, "mtime": 0, "content_hash": "", "ses_hash": "old", "ses_tokens": 5, "full_tokens": 9}
                for p, ki in self.infos.items()
            },
        }
        (self.emb_dir / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")

    def snapshot(self) -> Dict[str, bytes]:
        return {str(p): p.read_bytes() for p in self.emb_dir.rglob("*") if p.is_file()}

    def metadata_on_disk(self) -> Dict[str, Any]:
        return json.loads((self.emb_dir / "metadata.json").read_text(encoding="utf-8"))

    @contextlib.contextmanager
    def patched(self, load: Any) -> Iterator[None]:
        config = MagicMock()
        config.get_path.return_value = str(self.emb_dir)
        with patch.object(em, "_load_model", side_effect=load), patch.object(
            em, "_model_instance", None
        ), patch.object(em, "_selected_model_config", dict(QWEN)), patch.object(
            em, "get_project_root", return_value=str(self.root)
        ), patch.object(em, "ConfigManager", return_value=config), patch.object(
            em, "_is_valid_file", return_value=True
        ), patch.object(em, "_get_tokenizer", return_value=None), patch.object(
            em, "_unload_model"
        ), patch.object(em, "read_file_transparently", return_value=("", None)), patch.object(
            em, "_get_text_content_for_embedding", side_effect=lambda *a, **k: "SES text"
        ), patch(
            "cline_utils.dependency_system.io.transparency_manager.get_transparency_manager",
            return_value=MagicMock(),
        ):
            yield

    def run(self) -> bool:
        return em.generate_embeddings([str(self.root / "src")], self.infos, symbol_map={})


def _gguf_encoder(fail_paths: List[str]) -> Any:
    """A loaded llama-cpp stand-in; embed() raises for texts of the listed files."""
    encoder = MagicMock()
    encoder.embed.side_effect = lambda text: (_ for _ in ()).throw(RuntimeError("boom")) if fail_paths else [0.5] * QWEN_DIM
    return encoder


def test_a_failing_gguf_load_is_retried_then_skips_without_touching_the_index(tmp_path) -> None:
    project = Project(tmp_path, QWEN["name"])
    before = project.snapshot()
    loads: List[str] = []

    def always_fails(*_a: Any, **_k: Any) -> Any:
        loads.append(em._selected_model_config["type"])  # pyright: ignore[reportPrivateUsage]
        raise RuntimeError("Failed to load model from file")

    with project.patched(always_fails):
        assert project.run() is False
        assert em._selected_model_config["type"] == "gguf"  # pyright: ignore[reportPrivateUsage]

    assert loads == ["gguf"] * em.GGUF_LOAD_ATTEMPTS  # never a sentence-transformer
    assert project.snapshot() == before  # vectors and metadata.json byte-identical


def test_a_load_that_succeeds_on_retry_embeds_with_the_selected_model(tmp_path) -> None:
    project = Project(tmp_path, QWEN["name"])
    attempts: List[int] = []
    encoder = _gguf_encoder([])

    def fails_once(*_a: Any, **_k: Any) -> Any:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("busy")
        em._model_instance = encoder  # pyright: ignore[reportPrivateUsage]
        return encoder

    with project.patched(fails_once):
        assert project.run() is True

    meta = project.metadata_on_disk()
    assert meta["model"] == QWEN["name"]
    assert all(v["ses_hash"] not in ("", "old") for v in meta["keys"].values())
    assert np.load(str(project.emb_dir / "src" / "a.py.npy")).shape == (QWEN_DIM,)


def test_metadata_never_names_a_model_the_index_was_not_rebuilt_with(tmp_path) -> None:
    project = Project(tmp_path, "sentence-transformers/some-other-model")
    before = project.snapshot()
    encoder = _gguf_encoder(["all"])  # every flush fails

    def loads(*_a: Any, **_k: Any) -> Any:
        em._model_instance = encoder  # pyright: ignore[reportPrivateUsage]
        return encoder

    with project.patched(loads):
        assert project.run() is False

    assert project.snapshot() == before


def test_failed_vectors_keep_their_previous_ses_hash(tmp_path) -> None:
    project = Project(tmp_path, QWEN["name"])
    encoder = _gguf_encoder(["all"])

    def loads(*_a: Any, **_k: Any) -> Any:
        em._model_instance = encoder  # pyright: ignore[reportPrivateUsage]
        return encoder

    with project.patched(loads):
        assert project.run() is False  # partial failure is reported

    meta = project.metadata_on_disk()
    assert meta["model"] == QWEN["name"]
    assert {v["ses_hash"] for v in meta["keys"].values()} == {"old"}  # re-embedded next run


def test_a_failed_load_reports_llama_cpp_log_lines(monkeypatch) -> None:
    def fake_llama(**_k: Any) -> Any:
        em._LLAMA_LOG_TAIL.append(b"llama_model_load: error loading model: out of memory\n")  # pyright: ignore[reportPrivateUsage]
        raise ValueError("Failed to load model from file: m.gguf")

    monkeypatch.setattr(em, "Llama", fake_llama)
    monkeypatch.setattr(em, "llama_cpp", MagicMock())
    monkeypatch.setattr(em, "_model_instance", None)
    monkeypatch.setattr(em, "_selected_model_config", None)
    monkeypatch.setattr(em, "_select_best_model", lambda: dict(QWEN))
    monkeypatch.setattr(em, "_select_device", lambda: "cpu")
    with pytest.raises(RuntimeError) as err:
        em._load_model(n_ctx=12800)  # pyright: ignore[reportPrivateUsage]
    message = str(err.value)
    assert "Failed to load model from file" in message
    assert "error loading model: out of memory" in message
    assert "n_ctx=12800" in message


@pytest.mark.parametrize("current, requested, expected", [(384, 12800, 512), (384, 400, 400), (12800, 600, 512)])
def test_sentence_transformer_sequence_length_is_capped(monkeypatch, current, requested, expected) -> None:
    model = MagicMock()
    model.max_seq_length = current
    monkeypatch.setattr(em, "_model_instance", model)
    monkeypatch.setattr(em, "_selected_model_config", em.MODEL_CONFIGS["mpnet"].copy())
    em._load_model(n_ctx=requested)  # pyright: ignore[reportPrivateUsage]
    assert model.max_seq_length == expected
