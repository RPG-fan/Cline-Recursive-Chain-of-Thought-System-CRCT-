"""
Cache.is_expired() is called by CacheManager.get_cache() on every cached-function call.
It used to flush all of L1 to L2 each time, which was most of the suggestion phase's
CPU. It must now answer without flushing, and give the same answer as the old
flush-then-check, so dirty L1 data is never discarded with an "expired" cache.
"""

import itertools
import time

import pytest

from cline_utils.dependency_system.utils import cache_manager as cm

_names = (f"test_no_flush_{i}" for i in itertools.count())


@pytest.fixture
def fresh():
    created = []

    def make(ttl: int = 600) -> cm.Cache:
        name = next(_names)
        cache = cm.Cache(name, ttl)
        cm.cache_manager.caches[name] = cache
        created.append(name)
        return cache

    yield make
    for name in created:
        cm.cache_manager.caches.pop(name, None)


def _old_is_expired(cache: cm.Cache) -> bool:
    cache.flush()
    return (time.time() - cache.creation_time) > cache.default_ttl and not cache.data


def _age(cache: cm.Cache) -> None:
    cache.creation_time -= cache.default_ttl + 10


@pytest.mark.parametrize(
    "state",
    ["empty", "dirty_l1_only", "l2_only", "clean_l1_and_l2", "young_empty"],
)
def test_is_expired_matches_the_old_flush_then_check(fresh, state) -> None:
    cache = fresh()
    if state == "dirty_l1_only":
        cache.set("k", "v")
    elif state == "l2_only":
        cache.set("k", "v")
        cache.flush()
        cache.l1_data.clear()
    elif state == "clean_l1_and_l2":
        cache.set("k", "v")
        cache.flush()
    if state != "young_empty":
        _age(cache)

    new = cache.is_expired()
    assert new == _old_is_expired(cache)
    assert new is (state == "empty")


def test_get_cache_never_flushes(fresh, monkeypatch) -> None:
    cache = fresh()
    for i in range(50):
        cache.set(f"k{i}", i)
    flushes = []
    monkeypatch.setattr(cache, "flush", lambda: flushes.append(1))

    for _ in range(1000):
        assert cm.cache_manager.get_cache(cache.name) is cache

    assert flushes == []


def test_dirty_l1_data_survives_get_cache_past_ttl(fresh) -> None:
    cache = fresh()
    cache.set("only-in-l1", 42)
    _age(cache)

    again = cm.cache_manager.get_cache(cache.name)

    assert again is cache
    assert again.get("only-in-l1") == 42
