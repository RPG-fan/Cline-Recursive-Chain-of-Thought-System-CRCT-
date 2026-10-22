"""SES-aware embedding invalidation in ``embedding_manager.generate_embeddings``.

The embedding model is mocked the same way as in ``test_e2e_workflow.py``
(``_load_model`` / ``_model_instance`` / ``_selected_model_config``), so no real
model is ever loaded.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import cline_utils.dependency_system.analysis.embedding_manager as em
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.path_utils import normalize_path

EMBED_DIM = 384


class _Env:
    """Holds the mutable SES text and the texts seen by the mocked model."""

    def __init__(self, root: Path, embeddings_dir: Path, src: Path) -> None:
        self.root = root
        self.embeddings_dir = embeddings_dir
        self.src = src
        self.ses_text = "[FILE: src/a.py | TYPE: py]\nFUNCTIONS:\n  foo()"
        self.embedded: List[str] = []
        self.key_info = KeyInfo(
            key_string="1A1",
            norm_path=normalize_path(str(src)),
            parent_path=normalize_path(str(src.parent)),
            tier=1,
            is_directory=False,
        )

    def run(self) -> bool:
        return em.generate_embeddings(
            [str(self.src)],
            {self.key_info.norm_path: self.key_info},
            symbol_map={},
        )

    @property
    def metadata_entry(self) -> Dict[str, Any]:
        with open(self.embeddings_dir / "metadata.json", encoding="utf-8") as f:
            return json.load(f)["keys"][self.key_info.key_string]


@pytest.fixture
def env(tmp_path: Path) -> Generator[_Env, None, None]:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    src = root / "src" / "a.py"
    src.write_text("def foo():\n    return 1\n", encoding="utf-8")
    embeddings_dir = root / "embeddings"
    state = _Env(root, embeddings_dir, src)

    config = MagicMock()
    config.get_path.return_value = str(embeddings_dir)

    mock_instance = MagicMock()

    def fake_encode(texts: List[str], **_: Any) -> np.ndarray:
        state.embedded.extend(texts)
        return np.random.rand(len(texts), EMBED_DIM).astype(np.float32)

    mock_instance.encode.side_effect = fake_encode

    def fake_load(*_: Any, **__: Any) -> MagicMock:
        em._model_instance = mock_instance  # pyright: ignore[reportPrivateUsage]
        return mock_instance

    fake_tm = MagicMock()
    with patch.object(em, "_load_model", side_effect=fake_load), patch.object(
        em, "_model_instance", mock_instance
    ), patch.object(
        em,
        "_selected_model_config",
        {"type": "sentence-transformer", "name": "mock-model", "embedding_dim": EMBED_DIM},
    ), patch.object(
        em, "get_project_root", return_value=str(root)
    ), patch.object(
        em, "ConfigManager", return_value=config
    ), patch.object(
        em, "_is_valid_file", return_value=True
    ), patch.object(
        em, "_get_tokenizer", return_value=None
    ), patch.object(
        em, "_unload_model"
    ), patch.object(
        em, "read_file_transparently", return_value=("", None)
    ), patch.object(
        em, "_get_text_content_for_embedding", side_effect=lambda *a, **k: state.ses_text
    ), patch(
        "cline_utils.dependency_system.io.transparency_manager.get_transparency_manager",
        return_value=fake_tm,
    ):
        yield state


def _age_embedding_behind_source(env: _Env) -> None:
    """Make the .npy look newer than the source (source mtime fast path)."""
    npy = env.embeddings_dir / "src" / "a.py.npy"
    future = time.time() + 60
    os.utime(npy, (future, future))


def test_first_run_embeds_and_records_ses_hash(env: _Env) -> None:
    assert env.run()
    assert env.embedded == [env.ses_text]
    assert env.metadata_entry["ses_hash"] == em._compute_ses_hash(env.ses_text)  # pyright: ignore[reportPrivateUsage]


def test_unchanged_ses_and_source_does_not_reembed(env: _Env) -> None:
    env.run()
    _age_embedding_behind_source(env)
    env.embedded.clear()

    assert env.run()

    assert env.embedded == []
    assert env.metadata_entry["ses_hash"] == em._compute_ses_hash(env.ses_text)  # pyright: ignore[reportPrivateUsage]


def test_changed_ses_with_unchanged_source_reembeds(env: _Env) -> None:
    env.run()
    _age_embedding_behind_source(env)
    env.embedded.clear()

    env.ses_text += "\nCALLS:\n  bar"  # analyzer output changed, source file untouched
    assert env.run()

    assert env.embedded == [env.ses_text]
    assert env.metadata_entry["ses_hash"] == em._compute_ses_hash(env.ses_text)  # pyright: ignore[reportPrivateUsage]

    # And it settles: the next run is a no-op again.
    _age_embedding_behind_source(env)
    env.embedded.clear()
    env.run()
    assert env.embedded == []


def test_changed_ses_after_comment_only_touch_reembeds(env: _Env) -> None:
    """Source mtime moves forward but content hash matches (deep-check skip path)."""
    env.run()
    env.embedded.clear()
    future = time.time() + 120
    os.utime(env.src, (future, future))  # newer than the .npy, same content

    env.ses_text = "[FILE: src/a.py | TYPE: py]\nEXPORTS:\n  foo"
    assert env.run()

    assert env.embedded == [env.ses_text]


def test_touched_source_with_same_ses_does_not_reembed(env: _Env) -> None:
    env.run()
    env.embedded.clear()
    future = time.time() + 120
    os.utime(env.src, (future, future))

    assert env.run()

    assert env.embedded == []


def test_missing_ses_hash_in_old_metadata_forces_one_reembed(env: _Env) -> None:
    env.run()
    meta_path = env.embeddings_dir / "metadata.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    del data["keys"][env.key_info.key_string]["ses_hash"]  # pre-upgrade metadata
    meta_path.write_text(json.dumps(data), encoding="utf-8")
    _age_embedding_behind_source(env)
    env.embedded.clear()

    assert env.run()
    assert env.embedded == [env.ses_text]  # one-time re-embed
    assert env.metadata_entry["ses_hash"] == em._compute_ses_hash(env.ses_text)  # pyright: ignore[reportPrivateUsage]

    _age_embedding_behind_source(env)
    env.embedded.clear()
    env.run()
    assert env.embedded == []  # and not again
