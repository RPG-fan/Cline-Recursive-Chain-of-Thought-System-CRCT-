import pytest
import os
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Any, Generator

from cline_utils.dependency_system.utils.cache_manager import (
    get_cache_stats,
    clear_all_caches,
)
from cline_utils.dependency_system.utils import config_manager
from cline_utils.dependency_system.io import tracker_io
from cline_utils.dependency_system.analysis import dependency_analyzer
from cline_utils.dependency_system.analysis.project_analyzer import analyze_project
from cline_utils.dependency_system.dependency_processor import handle_clear_caches


def touch(filepath: str | Path) -> None:
    """Updates the modification time of a file."""
    try:
        Path(filepath).touch()
    except OSError as e:
        print(f"Error touching file {filepath}: {e}")


@pytest.fixture(scope="function")
def clear_cache_fixture() -> Generator[None, None, None]:
    """Ensures a clean cache state before each test function runs."""
    clear_all_caches()
    yield
    clear_all_caches()


@pytest.fixture(scope="session")
def temp_test_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary directory unique to the test session."""
    return tmp_path_factory.mktemp("cache_tests_manual")


@pytest.fixture(scope="function")
def test_project(temp_test_dir: Path) -> Generator[Path, None, None]:
    """Sets up a minimal temporary project structure for testing."""
    project_dir = temp_test_dir / "test_project_mt"
    project_dir.mkdir(exist_ok=True)
    (project_dir / ".clinerules").touch()
    # Base config
    base_config: dict[str, Any] = {
        "paths": {"doc_dir": "docs", "memory_dir": "cline_docs", "cache_dir": "cache"},
        "exclusions": {"dirs": [".git", "venv"], "files": ["*.log"]},
        "thresholds": {"code_similarity": 0.7},
    }
    (project_dir / ".clinerules.config.json").write_text(json.dumps(base_config))

    # Files needed by tests
    (project_dir / "src").mkdir(exist_ok=True)
    (project_dir / "src" / "module_a.py").write_text(
        "import os\nprint('hello from module_a')"
    )
    (project_dir / "src" / "module_b.py").write_text("print('hello from module_b')")
    (project_dir / "docs").mkdir(exist_ok=True)
    (project_dir / "docs" / "readme.md").write_text("# Test Readme")
    (project_dir / "cline_docs").mkdir(exist_ok=True)
    (project_dir / "cache").mkdir(exist_ok=True)

    # Write initial tracker for MT-01, MT-02, MT-03
    tracker_path = project_dir / "cline_docs" / "module_relationship_tracker.md"
    if not tracker_path.exists():
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        tracker_path.write_text("# Placeholder Tracker\nkeyA,keyB\np,p\n")

    tracker_path1 = project_dir / "cline_docs" / "tracker.md"
    tracker_path1.write_text("# Tracker 1\nA,B\np,p")
    tracker_path_log = project_dir / "cline_docs" / "tracker_log_test.md"
    tracker_path_log.write_text("# Log Test Tracker\nLogKey1,LogKey2\np,p")

    original_cwd = os.getcwd()
    os.chdir(project_dir)
    yield project_dir
    os.chdir(original_cwd)


# --- Manual/Tooling Tests (MT-01 to MT-03) ---


# Test Case MT-01: Cache Clearing
def test_mt01_cache_clearing(test_project: Path) -> None:
    """Verify the clear-caches command handler clears caches effectively."""
    # Use cacheable functions that analyze_project likely calls
    cache_name_read = "tracker_data_structured"
    cache_name_analyze = "file_analysis"
    cache_name_config_get = "excluded_dirs"

    tracker_path = test_project / "cline_docs" / "module_relationship_tracker.md"
    file_to_analyze = test_project / "src" / "module_a.py"

    # 1. Run analyze_project to populate caches
    print("\nRunning analyze_project (to populate caches for MT-01)...")
    try:
        _ = analyze_project(force_analysis=True)
    except Exception as e:
        pytest.fail(f"analyze_project failed during cache population: {e}")

    # Ensure tracker exists so read_tracker_file can be called meaningfully
    if not tracker_path.exists():
        # If analyze_project doesn't guarantee tracker creation, manually create one
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        tracker_path.write_text("# Placeholder Tracker\nkeyA,keyB\np,p\n")
        print("Warning: Tracker file not found, created placeholder for test.")

    # 2. Call cached functions once to ensure they are populated if not called by analyze_project
    #    and then call again to verify a cache hit occurred before clearing.
    print("Calling cached functions to confirm population and get initial hit...")
    _ = tracker_io.read_tracker_file_structured(str(tracker_path))  # Populate
    _ = dependency_analyzer.analyze_file(str(file_to_analyze))  # Populate
    _ = config_manager.ConfigManager().get_excluded_dirs()  # Populate

    _ = tracker_io.read_tracker_file_structured(str(tracker_path))  # Hit?
    stats_read1 = get_cache_stats(cache_name_read)
    assert (
        stats_read1.get("hits", 0) >= 1
    ), f"Cache '{cache_name_read}' should have hits before clear."

    _ = dependency_analyzer.analyze_file(str(file_to_analyze))  # Hit?
    stats_analyze1 = get_cache_stats(cache_name_analyze)
    assert (
        stats_analyze1.get("hits", 0) >= 1
    ), f"Cache '{cache_name_analyze}' should have hits before clear."

    _ = config_manager.ConfigManager().get_excluded_dirs()  # Hit?
    stats_config1 = get_cache_stats(cache_name_config_get)
    assert (
        stats_config1.get("hits", 0) >= 1
    ), f"Cache '{cache_name_config_get}' should have hits before clear."

    # 3. Call the clear-caches handler
    print("Calling handle_clear_caches...")
    # The handler takes argparse Namespace object. Pass an empty one.
    exit_code = handle_clear_caches(argparse.Namespace())
    assert exit_code == 0, "handle_clear_caches returned non-zero exit code."

    # 4. Call cached functions again -> Expect Cache Misses
    print("Calling cached functions after clearing...")

    # a) read_tracker_file
    _ = tracker_io.read_tracker_file_structured(str(tracker_path))  # Should miss
    stats_read2 = get_cache_stats(cache_name_read)
    print(f"Cache stats for {cache_name_read} after clear: {stats_read2}")
    # Note: clear_all_caches() replaces the cache instances, so metrics are reset.
    assert (
        stats_read2.get("misses", 0) >= 1
    ), f"Cache '{cache_name_read}' should have misses after clear (got {stats_read2.get('misses',0)})."

    # b) analyze_file
    _ = dependency_analyzer.analyze_file(str(file_to_analyze))  # Should miss
    stats_analyze2 = get_cache_stats(cache_name_analyze)
    print(f"Cache stats for {cache_name_analyze} after clear: {stats_analyze2}")
    assert (
        stats_analyze2.get("misses", 0) >= 1
    ), f"Cache '{cache_name_analyze}' should have misses after clear (got {stats_analyze2.get('misses',0)})."

    # c) ConfigManager getter
    _ = config_manager.ConfigManager().get_excluded_dirs()  # Should miss
    stats_config2 = get_cache_stats(cache_name_config_get)
    print(f"Cache stats for {cache_name_config_get} after clear: {stats_config2}")
    assert (
        stats_config2.get("misses", 0) >= 1
    ), f"Cache '{cache_name_config_get}' should have misses after clear (got {stats_config2.get('misses',0)})."

    print("Verified: Cache clearing appears functional.")


# Test Case MT-02: Cache Statistics
def test_mt02_cache_statistics(test_project: Path, clear_cache_fixture: None) -> None:
    """Verify get_cache_stats returns accurate hit/miss counts."""
    clear_all_caches()  # Manual clear at start
    # Use a simple cacheable function like read_tracker_file
    cache_name = "tracker_data_structured"
    tracker_path1 = test_project / "cline_docs" / "tracker.md"
    tracker_path2 = test_project / "docs" / "readme.md"

    # Files exist from fixture setup

    print(f"\nTesting cache statistics for '{cache_name}'...")

    # Initial state should be 0 hits, 0 misses *after* manual clear
    stats_initial = get_cache_stats(cache_name)
    assert (
        stats_initial.get("hits", 0) == 0
    ), f"Initial hits should be 0 after clear, got {stats_initial}"
    assert (
        stats_initial.get("misses", 0) == 0
    ), f"Initial misses should be 0 after clear, got {stats_initial}"

    # Call 1 (tracker1) -> Miss 1
    print("Call 1 (tracker1)...")
    _ = tracker_io.read_tracker_file_structured(str(tracker_path1))
    stats1 = get_cache_stats(cache_name)
    assert stats1.get("hits", 0) == 0, f"Hits should be 0 after 1st call, got {stats1}"
    assert (
        stats1.get("misses", 0) == 1
    ), f"Misses should be 1 after 1st call, got {stats1}"

    # Call 2 (tracker1 again) -> Hit 1
    print("Call 2 (tracker1)...")
    _ = tracker_io.read_tracker_file_structured(str(tracker_path1))
    stats2 = get_cache_stats(cache_name)
    assert stats2.get("hits", 0) == 1, "Hits should be 1 after 2nd call (same args)."
    assert stats2.get("misses", 0) == 1, "Misses should still be 1."

    # Call 3 (tracker2) -> Miss 2
    print("Call 3 (tracker2)...")
    _ = tracker_io.read_tracker_file_structured(str(tracker_path2))
    stats3 = get_cache_stats(cache_name)
    assert stats3.get("hits", 0) == 1, "Hits should still be 1."
    assert (
        stats3.get("misses", 0) == 2
    ), "Misses should be 2 after 3rd call (diff args)."

    # Call 4 (tracker1 again) -> Hit 2
    print("Call 4 (tracker1)...")
    _ = tracker_io.read_tracker_file_structured(str(tracker_path1))
    stats4 = get_cache_stats(cache_name)
    assert stats4.get("hits", 0) == 2, "Hits should be 2 after 4th call."
    assert stats4.get("misses", 0) == 2, "Misses should still be 2."

    # Call 5 (tracker2 again) -> Hit 3
    print("Call 5 (tracker2)...")
    _ = tracker_io.read_tracker_file_structured(str(tracker_path2))
    stats5 = get_cache_stats(cache_name)
    assert stats5.get("hits", 0) == 3, "Hits should be 3 after 5th call."
    assert stats5.get("misses", 0) == 2, "Misses should still be 2."

    # Call 6 (touch tracker1, call tracker1) -> Miss 3
    print("Call 6 (touch tracker1, call tracker1)...")
    touch(tracker_path1)
    time.sleep(0.1)
    _ = tracker_io.read_tracker_file_structured(str(tracker_path1))
    stats6 = get_cache_stats(cache_name)
    assert stats6.get("hits", 0) == 3, "Hits should still be 3."
    assert stats6.get("misses", 0) == 3, "Misses should be 3 after invalidation."

    # Call 7 (tracker1 again) -> Hit 4
    print("Call 7 (tracker1)...")
    _ = tracker_io.read_tracker_file_structured(str(tracker_path1))
    stats7 = get_cache_stats(cache_name)
    assert stats7.get("hits", 0) == 4, "Hits should be 4 after re-cache."
    assert stats7.get("misses", 0) == 3, "Misses should still be 3."

    print(f"Final Stats for '{cache_name}': {stats7}")
    print("Verified: Cache statistics appear accurate.")


# Test Case MT-03: DEBUG Logging
def test_mt03_debug_logging(
    test_project: Path, clear_cache_fixture: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify cache initialization, clearing, and invalidation messages appear in DEBUG logs."""
    from cline_utils.dependency_system.utils.cache_manager import tracker_modified

    # Set logging level to DEBUG to capture cache messages
    caplog.set_level(logging.DEBUG)

    print(f"\nTesting DEBUG logging for cache...")

    # 1. Clear all caches -> Expect cache cleared/initialized debug logs
    clear_all_caches()
    log_text = caplog.text.lower()
    assert (
        "initialized" in log_text or "spun up" in log_text or "cleared" in log_text
    ), "Expected cache initialization/clearing logs in DEBUG output."
    print(" -> Initialization/clearing logs found.")
    caplog.clear()

    # 2. Trigger invalidation via tracker_modified -> Expect invalidation debug log
    tracker_path = test_project / "cline_docs" / "tracker_log_test.md"
    tracker_modified(str(tracker_path), "relationship", str(test_project))

    invalidation_log_text = caplog.text.lower()
    assert (
        "invalidated" in invalidation_log_text
    ), "Expected invalidation log in DEBUG output after tracker modification."
    print(" -> Invalidation log found.")
