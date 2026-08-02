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


def test_non_persistent_cache_skipped(clear_cache):
    """Caches in NON_PERSISTENT_CACHES should never be saved to disk."""
    import os
    from cline_utils.dependency_system.utils.cache_support import (
        NON_PERSISTENT_CACHES,
    )
    from cline_utils.dependency_system.utils.cache_manager import CACHE_DIR

    # Create a non-persistent cache and add data
    for cache_name in NON_PERSISTENT_CACHES:
        cache = cache_manager.get_cache(cache_name, ttl=0)
        cache.set("test_key", "test_value", ttl=0)

    # Flush to ensure data is in L2
    for cache_name in NON_PERSISTENT_CACHES:
        cache_manager.get_cache(cache_name).flush()

    # Call save_all - should skip non-persistent caches
    cache_manager.save_all()

    # Verify no .pkl or .pkl.gz file was created for non-persistent caches
    for cache_name in NON_PERSISTENT_CACHES:
        for ext in (".pkl", ".pkl.gz"):
            p = os.path.join(CACHE_DIR, f"{cache_name}{ext}")
            assert not os.path.exists(
                p
            ), f"Non-persistent cache '{cache_name}' was saved to disk!"


def test_non_picklable_keys_stripped(clear_cache):
    """_strip_non_picklable_keys should remove _ts_tree from dict values."""
    from cline_utils.dependency_system.utils.cache_support import (
        strip_non_picklable_keys,
    )

    # Test with a dict containing _ts_tree
    test_data = {
        "imports": ["os", "sys"],
        "functions": [{"name": "foo", "line": 1}],
        "_ts_tree": "fake_tree_object",
        "classes": [{"name": "MyClass", "line": 10}],
    }

    stripped = strip_non_picklable_keys(test_data)

    # _ts_tree should be removed
    assert "_ts_tree" not in stripped
    # Other keys should be preserved
    assert "imports" in stripped
    assert "functions" in stripped
    assert "classes" in stripped
    assert stripped["imports"] == ["os", "sys"]
    assert stripped["functions"] == [{"name": "foo", "line": 1}]

    # Test with a non-dict value (should be returned unchanged)
    assert strip_non_picklable_keys("string_value") == "string_value"
    assert strip_non_picklable_keys(42) == 42
    assert strip_non_picklable_keys(["a", "b"]) == ["a", "b"]


def test_file_analysis_cache_persists_without_ts_tree(clear_cache):
    """file_analysis cache should be saveable even when entries contain _ts_tree."""
    import os
    import pickle
    import gzip
    from cline_utils.dependency_system.utils.cache_manager import (
        cache_manager,
        CACHE_DIR,
    )

    # Simulate a file_analysis entry with _ts_tree (non-picklable in real usage)
    fake_analysis = {
        "file_path": "test.py",
        "file_type": "py",
        "imports": ["os", "sys"],
        "functions": [{"name": "foo", "line": 1}],
        "_ts_tree": None,  # Simulating the key exists (real value would be a Tree object)
    }

    cache = cache_manager.get_cache("file_analysis_test", ttl=0)
    cache.set("test_key", fake_analysis, ttl=0)
    cache.flush()

    # Save the cache
    cache_manager._save_cache("file_analysis_test")

    # Verify the cache file was created
    cache_file = os.path.join(CACHE_DIR, "file_analysis_test.pkl.gz")
    if not os.path.exists(cache_file):
        cache_file = os.path.join(CACHE_DIR, "file_analysis_test.pkl")
    assert os.path.exists(cache_file), "Cache file was not created!"

    # Load and verify _ts_tree was stripped
    if cache_file.endswith(".pkl.gz"):
        with gzip.open(cache_file, "rb") as f:
            loaded = pickle.load(f)
    else:
        with open(cache_file, "rb") as f:
            loaded = pickle.load(f)

    loaded_data = loaded.get("data", {})
    assert "test_key" in loaded_data
    entry = loaded_data["test_key"]
    assert "_ts_tree" not in entry, "_ts_tree was not stripped from saved cache!"
    assert "imports" in entry, "Useful data was lost!"
    assert "functions" in entry, "Useful data was lost!"

    # Cleanup
    try:
        os.remove(cache_file)
    except OSError:
        pass


def test_grid_cache_deterministic_hash(clear_cache):
    from cline_utils.dependency_system.core.key_manager import KeyInfo
    from cline_utils.dependency_system.core.dependency_grid import (
        _deterministic_hash,
        validate_grid,
        get_dependencies_from_grid,
    )
    from cline_utils.dependency_system.utils.calculate_hash import (
        calculate_content_hash,
    )

    # 1. Define keys and grid
    keys = [
        KeyInfo("1A1", "h:/repo/src/a.py", "h:/repo/src", 1, False),
        KeyInfo("1A2", "h:/repo/src/b.py", "h:/repo/src", 1, False),
    ]
    grid = {"1A1": "o<", "1A2": ">o"}

    # 2. Check that _deterministic_hash produces the expected hash from calculate_content_hash
    expected_hash = calculate_content_hash(str(sorted(grid.items())))
    assert _deterministic_hash(grid) == expected_hash

    # 3. Verify that calling validate_grid caches the result with the deterministic key
    validate_grid(grid, keys)
    from cline_utils.dependency_system.utils.cache_manager import cache_manager

    cache = cache_manager.get_cache("grid_validation")

    # Construct expected cache key: f"validate_grid:{_deterministic_hash(grid)}:{sorted keys path list}"
    from cline_utils.dependency_system.core.key_manager import (
        sort_key_strings_hierarchically,
    )

    expected_cache_key = f"validate_grid:{expected_hash}:{':'.join(sort_key_strings_hierarchically([ki.key_string for ki in keys]))}"
    assert expected_cache_key in cache.data

    # 4. Verify that calling get_dependencies_from_grid caches with deterministic key
    get_dependencies_from_grid(grid, "1A1", keys)
    cache_deps = cache_manager.get_cache("grid_dependencies")
    expected_cache_key_deps = f"grid_deps:{expected_hash}:1A1:{':'.join(sort_key_strings_hierarchically([ki.key_string for ki in keys]))}"
    assert expected_cache_key_deps in cache_deps.data
