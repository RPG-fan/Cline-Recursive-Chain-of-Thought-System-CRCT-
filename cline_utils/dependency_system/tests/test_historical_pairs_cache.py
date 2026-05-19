"""Cache invalidation for get_historical_pairs when cycle history files change."""

import json
import os

import pytest

from cline_utils.dependency_system.analysis.reranker_history_tracker import (
    HISTORY_DIR,
    get_historical_pairs,
)
from cline_utils.dependency_system.utils.cache_manager import cache_manager, clear_all_caches


def _write_cycle_file(history_dir: str, cycle_num: int, pairs: list[tuple[str, str]]) -> str:
    os.makedirs(history_dir, exist_ok=True)
    path = os.path.join(history_dir, f"cycle_{cycle_num}.json")
    payload = {
        "cycle": cycle_num,
        "metrics": {"avg_confidence": 0.5},
        "all_pairs": [{"source": s, "target": t} for s, t in pairs],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


@pytest.fixture(autouse=True)
def _clear_caches():
    clear_all_caches()
    yield
    clear_all_caches()


def test_get_historical_pairs_invalidates_when_cycle_file_added(tmp_path):
    project_root = str(tmp_path)
    history_dir = os.path.join(project_root, HISTORY_DIR.replace("/", os.sep))

    _write_cycle_file(history_dir, 1, [("a.py", "b.py")])
    first = get_historical_pairs(project_root)
    assert first == {("a.py", "b.py")}

    cache = cache_manager.get_cache("reranker_history")
    assert cache.metrics.misses == 1
    assert cache.metrics.hits == 0

    second = get_historical_pairs(project_root)
    assert second == first
    assert cache.metrics.hits == 1

    _write_cycle_file(history_dir, 2, [("c.py", "d.py")])
    third = get_historical_pairs(project_root)
    assert ("c.py", "d.py") in third
    assert cache.metrics.misses == 2


def test_get_historical_pairs_invalidates_when_cycle_file_updated(tmp_path):
    project_root = str(tmp_path)
    history_dir = os.path.join(project_root, HISTORY_DIR.replace("/", os.sep))
    cycle_path = _write_cycle_file(history_dir, 5, [("x.py", "y.py")])

    assert get_historical_pairs(project_root) == {("x.py", "y.py")}
    assert get_historical_pairs(project_root) == {("x.py", "y.py")}

    payload = {
        "cycle": 5,
        "metrics": {"avg_confidence": 0.5},
        "all_pairs": [
            {"source": "x.py", "target": "y.py"},
            {"source": "m.py", "target": "n.py"},
        ],
    }
    with open(cycle_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    updated = get_historical_pairs(project_root)
    assert ("m.py", "n.py") in updated
    assert cache_manager.get_cache("reranker_history").metrics.misses == 2
