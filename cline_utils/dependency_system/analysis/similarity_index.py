"""
Bulk embedding similarity for the dependency-suggestion phase.

Scoring each file against every other one pair by pair (calculate_similarity) reloads
both .npy files and scans path_to_key_info twice per pair. That is about 830 pairs per
file and ~680k per run on an 830-file project, and the 10k-entry similarity cache cannot
hold them. This module loads every embedding ONCE into a single matrix; a file's
similarity to all others is then one matrix-vector product.

Scores match calculate_similarity: dot product of the stored (already normalized)
vectors, clamped to [0, 1], and 0.0 when either embedding is missing or has the wrong
dimension. Unlike calculate_similarity, a wrong-dimension file is never deleted here.
"""

import logging
import os
import threading
import time
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.path_utils import normalize_path

logger = logging.getLogger(__name__)


class SimilarityIndex:
    """All file embeddings of one run, stacked as rows of one float32 matrix."""

    def __init__(
        self,
        embeddings_dir: str,
        project_root: str,
        embedding_dim: int,
        path_to_key_info: Dict[str, KeyInfo],
    ) -> None:
        start = time.perf_counter()
        self.row_of: Dict[str, int] = {}
        vectors: List[np.ndarray] = []
        missing = mismatched = unreadable = 0
        for ki in path_to_key_info.values():
            if ki.is_directory or ki.key_string in self.row_of:
                continue
            npy_path = (
                os.path.join(embeddings_dir, os.path.relpath(ki.norm_path, project_root))
                + ".npy"
            )
            if not os.path.exists(npy_path):
                missing += 1
                continue
            try:
                vec = np.load(npy_path)
            except Exception as e:
                unreadable += 1
                logger.warning(f"SimilarityIndex: could not load {npy_path}: {e}")
                continue
            if vec.shape[0] != embedding_dim:
                mismatched += 1
                continue
            self.row_of[ki.key_string] = len(vectors)
            vectors.append(vec.flatten().astype(np.float32, copy=False))

        self.matrix = (
            np.vstack(vectors) if vectors else np.zeros((0, embedding_dim), np.float32)
        )
        logger.info(
            f"SimilarityIndex: loaded {len(vectors)} embeddings once "
            f"({missing} missing, {mismatched} wrong dimension, {unreadable} unreadable) "
            f"in {time.perf_counter() - start:.2f}s."
        )
        if mismatched:
            logger.warning(
                f"SimilarityIndex: {mismatched} embedding(s) do not match the selected "
                f"model's dimension ({embedding_dim}) and score 0.0. Re-run embedding "
                "generation to refresh them."
            )

    def scores_for(self, source_key: str) -> Dict[str, float]:
        """Similarity of `source_key` to every indexed key, clamped to [0, 1].
        Empty when the source has no usable embedding (every score is then 0.0)."""
        row = self.row_of.get(source_key)
        if row is None:
            return {}
        sims = np.clip(self.matrix @ self.matrix[row], 0.0, 1.0)
        return {key: float(sims[i]) for key, i in self.row_of.items()}


_lock = threading.Lock()
_index: Optional[SimilarityIndex] = None
_index_key: Optional[Tuple[Hashable, ...]] = None


def _key(
    embeddings_dir: str, project_root: str, dim: int, path_to_key_info: Dict[str, KeyInfo]
) -> Tuple[Hashable, ...]:
    return (
        normalize_path(embeddings_dir),
        normalize_path(project_root),
        dim,
        id(path_to_key_info),
        len(path_to_key_info),
    )


def _embedding_dim() -> int:
    from .embedding_manager import _select_best_model  # pyright: ignore[reportPrivateUsage]

    return int(_select_best_model()["embedding_dim"])


def prepare(
    embeddings_dir: str, project_root: str, path_to_key_info: Dict[str, KeyInfo]
) -> SimilarityIndex:
    """Build the index now (upfront, before parallel suggestion workers start)."""
    global _index, _index_key
    dim = _embedding_dim()
    with _lock:
        _index = SimilarityIndex(embeddings_dir, project_root, dim, path_to_key_info)
        _index_key = _key(embeddings_dir, project_root, dim, path_to_key_info)
        return _index


def get_scores(
    source_key: str,
    embeddings_dir: str,
    project_root: str,
    path_to_key_info: Dict[str, KeyInfo],
) -> Dict[str, float]:
    """Scores for `source_key`; builds the index once (lock-guarded) if it was not
    prepared for this embeddings dir, model dimension and key map."""
    global _index, _index_key
    index = _index
    index_key = _index_key
    if index is not None and index_key is not None:
        from . import embedding_manager as em

        cfg = em._selected_model_config  # pyright: ignore[reportPrivateUsage]
        dim_match = cfg is None or int(cfg["embedding_dim"]) == index_key[2]
        if (
            dim_match
            and index_key[3] == id(path_to_key_info)
            and index_key[4] == len(path_to_key_info)
            and index_key[0] == normalize_path(embeddings_dir)
            and index_key[1] == normalize_path(project_root)
        ):
            return index.scores_for(source_key)

    # As in calculate_similarity, when the index is not yet built for this scope, a
    # source without an embedding scores 0.0 before the model is consulted for its dimension.
    source = next(
        (ki for ki in path_to_key_info.values() if ki.key_string == source_key), None
    )
    if source is None or not os.path.exists(
        os.path.join(embeddings_dir, os.path.relpath(source.norm_path, project_root))
        + ".npy"
    ):
        return {}
    dim = _embedding_dim()
    key = _key(embeddings_dir, project_root, dim, path_to_key_info)
    index = _index
    if index is None or _index_key != key:
        with _lock:
            if _index is None or _index_key != key:
                _index = SimilarityIndex(embeddings_dir, project_root, dim, path_to_key_info)
                _index_key = key
            index = _index
    return index.scores_for(source_key)


def invalidate() -> None:
    """Drop the index (embeddings were regenerated)."""
    global _index, _index_key
    with _lock:
        _index = None
        _index_key = None
