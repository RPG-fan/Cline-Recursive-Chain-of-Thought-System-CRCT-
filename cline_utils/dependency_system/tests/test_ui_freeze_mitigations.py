"""
Two causes of the VS Code window freezing during analyze-project:
- PhaseTracker redrew and flushed the terminal on every item (~300 redraws/s in fast
  phases); it now redraws at most MAX_REDRAWS_PER_SECOND, always drawing the final state.
- Model selection loaded the full Qwen3 GGUF onto the GPU (with a 16k-token context) on
  every run, even with nothing to embed. Selection now only checks the file structurally;
  the model is loaded when embeddings are generated, and a failed load aborts/skips
  embedding updates for that run so existing Qwen3 embeddings on disk are preserved.
No model is loaded: llama-cpp and the encoder are mocked.
"""

import json
from io import StringIO
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import cline_utils.dependency_system.analysis.embedding_manager as em
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils import phase_tracker as pt
from cline_utils.dependency_system.utils.path_utils import normalize_path


def test_progress_redraws_are_throttled_and_the_final_state_is_drawn() -> None:
    out = StringIO()
    with patch("sys.stdout", out):
        tracker = pt.PhaseTracker(total=500, phase_name="Fast")
        tracker.is_tty = True
        with tracker:
            for _ in range(500):
                tracker.update(1, "item")
    draws = out.getvalue().count("\r")
    assert draws <= 4  # first draw, completion, forced final draw (+1 if a tick passed)
    assert "100.0%" in out.getvalue().split("\r")[-1]


def _intact_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + b"\0" * 2_000_000)
    return path


@pytest.mark.parametrize("selection", ["auto", "qwen3-4b"])
def test_model_selection_never_load_tests_an_intact_model(tmp_path, monkeypatch, selection) -> None:
    model = _intact_gguf(tmp_path / "Qwen3-Embedding-4B-Q6_K.gguf")
    config = MagicMock()
    config.get_embedding_setting.side_effect = lambda key, default=None: {
        "model_selection": selection, "qwen3_model_path": str(model),
    }.get(key, default)
    monkeypatch.setattr(em, "ConfigManager", lambda: config)
    monkeypatch.setattr(em, "_selected_model_config", None)
    monkeypatch.setattr(em, "_select_device", lambda: "cuda")
    monkeypatch.setattr(em, "_get_available_vram", lambda: 64.0)

    def no_load_test(*_a: Any) -> bool:
        raise AssertionError("model selection must not load the model onto the GPU")

    monkeypatch.setattr(em, "_verify_qwen3_model", no_load_test)
    monkeypatch.setattr(em, "_download_with_retry", lambda *a, **k: pytest.fail("no download"))

    chosen = em._select_best_model()  # pyright: ignore[reportPrivateUsage]
    assert chosen["type"] == "gguf"
    assert chosen["path"] == str(model)


def test_failed_gguf_load_skips_embedding_updates_and_preserves_existing(tmp_path) -> None:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    existing_src = root / "src" / "existing.py"
    existing_src.write_text("def keep():\n    return 1\n", encoding="utf-8")
    new_src = root / "src" / "new.py"
    new_src.write_text("def added():\n    return 2\n", encoding="utf-8")

    emb_dir = root / "embeddings"
    (emb_dir / "src").mkdir(parents=True)
    gguf_dim = int(em.MODEL_CONFIGS["qwen3-4b"]["embedding_dim"])
    existing_vec = np.ones(gguf_dim, dtype=np.float32)
    existing_npy = emb_dir / "src" / "existing.py.npy"
    np.save(str(existing_npy), existing_vec)
    meta_path = emb_dir / "metadata.json"
    original_meta = {
        "version": em.EMBEDDING_METADATA_VERSION,
        "model": em.MODEL_CONFIGS["qwen3-4b"]["name"],
        "keys": {
            "1A1": {
                "path": normalize_path(str(existing_src)),
                "ses_tokens": 10,
                "full_tokens": 10,
                "ses_hash": em._compute_ses_hash("SES text"),  # pyright: ignore[reportPrivateUsage]
            }
        },
    }
    meta_path.write_text(json.dumps(original_meta), encoding="utf-8")

    ki_existing = KeyInfo("1A1", normalize_path(str(existing_src)), normalize_path(str(existing_src.parent)), 1, False)
    ki_new = KeyInfo("1A2", normalize_path(str(new_src)), normalize_path(str(new_src.parent)), 1, False)
    config = MagicMock()
    config.get_path.return_value = str(emb_dir)

    loads: List[str] = []

    def fake_load(*_a: Any, **_k: Any) -> Any:
        kind = em._selected_model_config["type"]  # pyright: ignore[reportPrivateUsage]
        loads.append(kind)
        raise RuntimeError("CUDA error: out of memory")

    gguf = dict(em.MODEL_CONFIGS["qwen3-4b"], path=str(tmp_path / "m.gguf"))
    with patch.object(em, "_load_model", side_effect=fake_load), patch.object(
        em, "_model_instance", None
    ), patch.object(em, "_selected_model_config", gguf), patch.object(
        em, "get_project_root", return_value=str(root)
    ), patch.object(em, "ConfigManager", return_value=config), patch.object(
        em, "_is_valid_file", return_value=True
    ), patch.object(em, "_get_tokenizer", return_value=None), patch.object(
        em, "_unload_model"
    ) as unload_mock, patch.object(
        em, "read_file_transparently", return_value=("", None)
    ), patch.object(
        em, "_get_text_content_for_embedding", side_effect=lambda *a, **k: "SES text"
    ), patch(
        "cline_utils.dependency_system.io.transparency_manager.get_transparency_manager",
        return_value=MagicMock(),
    ):
        result = em.generate_embeddings(
            [str(existing_src), str(new_src)],
            {ki_existing.norm_path: ki_existing, ki_new.norm_path: ki_new},
            symbol_map={},
        )
        assert result is False
        assert em._selected_model_config["type"] == "gguf"  # pyright: ignore[reportPrivateUsage]
        unload_mock.assert_called_once()

    assert loads == ["gguf"]
    assert not (emb_dir / "src" / "new.py.npy").exists()
    preserved_vec = np.load(str(existing_npy))
    assert preserved_vec.shape == (gguf_dim,)
    assert json.loads(meta_path.read_text(encoding="utf-8")) == original_meta
