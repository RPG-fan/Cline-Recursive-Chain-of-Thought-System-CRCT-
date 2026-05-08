import time
import pytest
from pathlib import Path
from cline_utils.dependency_system.utils.cache_manager import (
    cache_manager,
    cached,
    invalidate_dependent_entries,
    clear_all_caches,
)
from cline_utils.dependency_system.utils.path_utils import normalize_path


@pytest.fixture
def clear_cache():
    clear_all_caches()
    yield
    clear_all_caches()


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


def test_track_path_args(temp_dir, clear_cache):
    test_file = temp_dir / "test_file.txt"
    test_file.write_text("initial content")
    path_str = normalize_path(str(test_file))

    call_count = 0

    @cached("test_cache", track_path_args=[0])
    def get_content(file_path):
        nonlocal call_count
        call_count += 1
        return Path(file_path).read_text()

    # First call
    assert get_content(path_str) == "initial content"
    assert call_count == 1

    # Second call (cached)
    assert get_content(path_str) == "initial content"
    assert call_count == 1

    # Invalidate by file path using the system API
    from cline_utils.dependency_system.utils.cache_manager import file_modified

    file_modified(path_str, str(temp_dir))

    # Third call (re-run after invalidation)
    assert get_content(path_str) == "initial content"
    assert call_count == 2


def test_throttled_cleanup(clear_cache):
    # Reset last cleanup time to force the first one
    cache_manager._last_cleanup_time = 0

    @cached("throttle_test", ttl=1)  # Increased to 1s
    def func_to_cache(x):
        return x

    # Fill cache
    func_to_cache(1)
    cache = cache_manager.get_cache("throttle_test")
    assert "func_to_cache::1" in cache.data

    time.sleep(1.2)  # Expire key1

    # Reset last cleanup time to force cleanup on next call
    cache_manager._last_cleanup_time = 0
    func_to_cache(2)
    assert "func_to_cache::1" not in cache.data  # Should be cleaned up now
    time_cleanup_was_executed = cache_manager._last_cleanup_time
    assert time_cleanup_was_executed > 0

    # Create another item that will expire soon
    cache.set("manual_key", "val", ttl=1)
    time.sleep(1.2)

    # Second call within 60s should NOT trigger cleanup
    # We don't reset _last_cleanup_time here
    func_to_cache(3)
    assert "manual_key" in cache.data  # Still there because cleanup was throttled
    assert cache_manager._last_cleanup_time == time_cleanup_was_executed

    # Force cleanup by resetting time again
    cache_manager._last_cleanup_time = 0
    func_to_cache(4)
    assert "manual_key" not in cache.data
    assert cache_manager._last_cleanup_time > time_cleanup_was_executed
