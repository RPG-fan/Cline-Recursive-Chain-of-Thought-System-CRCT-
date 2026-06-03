import pytest
import os
import time
import json
from pathlib import Path
from typing import Any, Generator
import numpy as np

from cline_utils.dependency_system.core.key_manager import (
    KeyInfo,
    sort_key_strings_hierarchically,
)
from cline_utils.dependency_system.utils.cache_manager import (
    clear_all_caches,
)
from cline_utils.dependency_system.utils import config_manager, path_utils

from cline_utils.dependency_system.analysis import embedding_manager
from cline_utils.dependency_system.core import dependency_grid
from cline_utils.dependency_system.io import tracker_io
from cline_utils.dependency_system.analysis.project_analyzer import analyze_project


def touch(filepath: str | Path) -> None:
    """Updates the modification time of a file."""
    try:
        Path(filepath).touch()
    except OSError as e:
        print(f"Error touching file {filepath}: {e}")


def read_tracker_file_compat(tracker_path: str | Path) -> dict[str, Any] | None:
    data = tracker_io.read_tracker_file_structured(str(tracker_path))
    if not data or not data.get("definitions_ordered"):
        return None

    keys_map = {k: p for k, p in data["definitions_ordered"]}
    grid = {k: v for k, v in data["grid_rows_ordered"]}

    return {"keys": keys_map, "grid": grid}


def get_char_at_key(
    grid: dict[str, str], source_key: str, target_key: str, sorted_keys: list[str]
) -> str | None:
    if source_key not in grid:
        return None
    row_compressed = grid[source_key]
    try:
        target_idx = sorted_keys.index(target_key)
        return dependency_grid.get_char_at(row_compressed, target_idx)
    except ValueError:
        return None


def get_mini_tracker_path(project_root: Path, relative_dir: str | Path) -> Path | None:
    dir_path = project_root / relative_dir
    # Find first file ending in _module.md
    for f in dir_path.glob("*_module.md"):
        return f
    return None


class MockEmbeddingModel:
    embedding_dim: int

    def __init__(self, embedding_dim: int = 384) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def encode(self, texts: list[str], **kwargs: Any) -> list[Any]:
        # Return deterministic embeddings based on text content
        embeddings = []
        for text in texts:
            # Use hash of text to seed random generator for deterministic output
            # This ensures same text -> same embedding, different text -> different embedding
            seed = abs(hash(text)) % (2**32)
            rng = np.random.RandomState(seed)
            # Use randn + bias to ensure some positive similarity but not too high
            # We use + 0.6 to guarantee a positive dot product that never clips to 0.0,
            # avoiding flakiness.
            embeddings.append(rng.randn(self.embedding_dim).astype(np.float32) + 0.6)
        return embeddings


@pytest.fixture(scope="function", autouse=True)
def mock_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocks the embedding model to prevent downloads."""
    mock_model = MockEmbeddingModel()
    monkeypatch.setattr(embedding_manager, "_model_instance", mock_model)
    # Also mock the config so it thinks a model is selected
    monkeypatch.setattr(
        embedding_manager,
        "_selected_model_config",
        {"type": "sentence-transformer", "embedding_dim": 384, "name": "mock-model"},
    )
    # And ensure _load_model doesn't overwrite it if called
    monkeypatch.setattr(embedding_manager, "_load_model", lambda *args, **kwargs: True)
    # Ensure _unload_model doesn't clear the mock instance
    monkeypatch.setattr(embedding_manager, "_unload_model", lambda: None)

    # AGGRESSIVE MOCKING: Prevent any import of sentence_transformers or transformers
    # This ensures NO downloads can possibly happen even if _load_model is bypassed
    import sys
    from unittest.mock import MagicMock

    mock_st = MagicMock()
    mock_st.SentenceTransformer.return_value = mock_model
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st)

    mock_tf = MagicMock()
    mock_tf.AutoTokenizer.from_pretrained.return_value = MagicMock()
    mock_tf.AutoModel.from_pretrained.return_value = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", mock_tf)

    # Mock reranker to prevent download
    monkeypatch.setattr(
        embedding_manager, "rerank_candidates_with_qwen3", lambda *args, **kwargs: []
    )

    # Ensure _model_instance is set
    embedding_manager._model_instance = mock_model

    # Mock _get_tokenizer to prevent download
    monkeypatch.setattr(embedding_manager, "_get_tokenizer", lambda: MagicMock())


# --- Fixtures ---
# NOTE: Fixtures are redefined here. If running tests together, consider a shared conftest.py


@pytest.fixture(scope="function")
def clear_cache_fixture() -> (
    Generator[None, None, None]
):  # Needed by some tests implicitly via analyze_project potentially
    """Ensures a clean cache state before each test function runs."""
    clear_all_caches()
    yield
    clear_all_caches()


@pytest.fixture(scope="function")
def test_project(tmp_path: Path) -> Generator[Path, None, None]:  # Used by all IS tests
    """Sets up a minimal temporary project structure for testing."""
    project_dir = tmp_path / "test_project_is"  # Unique name
    project_dir.mkdir(exist_ok=True)

    # Reset ConfigManager singleton to ensure it picks up the new project root
    config_manager.ConfigManager._instance = None

    (project_dir / ".clinerules").touch()  # IS-03 needs this
    # Base config
    base_config: dict[str, Any] = {
        "paths": {
            "doc_dir": "docs",
            "memory_dir": "cline_docs",
            "cache_dir": "cache",
            "embeddings_dir": "cache/embeddings",  # Explicitly set to match test expectation
        },
        "exclusions": {"dirs": [".git", "venv"], "files": ["*.log"]},
        "thresholds": {"code_similarity": 0.7},
    }
    (project_dir / ".clinerules.config.json").write_text(
        json.dumps(base_config)
    )  # IS-02 needs this

    (project_dir / "src").mkdir(exist_ok=True)
    # Make content distinct to avoid accidental semantic similarity ('S')
    (project_dir / "src" / "module_a.py").write_text(
        "def calculate_sum(a, b):\n    return a + b"
    )
    (project_dir / "src" / "module_b.py").write_text(
        "class UserProfile:\n    def __init__(self, name):\n        self.name = name"
    )
    (project_dir / "docs").mkdir(exist_ok=True)  # IS-03 needs this
    (project_dir / "docs" / "readme.md").write_text("# Test Readme")  # IS-03 needs this
    (project_dir / "cline_docs").mkdir(exist_ok=True)  # Needed for tracker path
    (project_dir / "lib").mkdir(exist_ok=True)  # IS-03 needs this root dir
    (project_dir / "lib" / "helper.py").write_text(
        "def helper_func(): return 1"
    )  # IS-03 needs this file

    # Create dummy cache dir and embedding files for IS-04
    cache_dir = project_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    embedding_dir = cache_dir / "embeddings"
    embedding_dir.mkdir(exist_ok=True)

    # Write initial .clinerules for IS-03
    initial_clinerules_content = """
[COUNT]
n + 1 = (x)
[LAST_ACTION_STATE]
last_action: "Test Init IS03"
current_phase: "Execution"
next_action: "Complete Test"
next_phase: "Execution"
---
[CODE_ROOT_DIRECTORIES]
- src
[DOC_DIRECTORIES]
- docs
[LEARNING_JOURNAL]
- Test entry.
[Character_Definitions]
```
- <: Row depends on column.
```
---
**IMPORTANT**
1. Understand the Objective
"""
    (project_dir / ".clinerules").write_text(initial_clinerules_content)

    original_cwd = os.getcwd()
    os.chdir(project_dir)
    yield project_dir
    os.chdir(original_cwd)


# --- Integration Tests (IS-01 to IS-04) ---


# Test Case IS-01: Source File Modification
def test_is01_source_file_modification(test_project: Path) -> None:
    """Verify analyze_project updates trackers correctly after source file change."""
    project_root = test_project
    # Use mini-tracker for src to check file-level dependencies
    # tracker_path will be determined after analysis runs

    # No need to mock clear_all_caches for this test.
    # We rely on analyze_project's internal cache invalidation based on mtime.

    # 1. Initial analysis run
    print("\nRunning analyze_project (initial run)...")
    try:
        # Use force_analysis to ensure trackers are generated based on current state
        _results1 = analyze_project(force_analysis=True)
    except Exception as e:
        pytest.fail(f"Initial analyze_project failed: {e}")

    # Check if tracker file exists and read its initial state
    tracker_path = get_mini_tracker_path(
        project_root, "src"
    )  # Refresh path in case it was created
    if not tracker_path or not tracker_path.exists():
        # analyze_project MUST create the tracker if files are present
        pytest.fail(f"Initial run failed to create tracker file in src: {tracker_path}")

    tracker_data1 = read_tracker_file_compat(str(tracker_path))
    if not tracker_data1 or "grid" not in tracker_data1 or "keys" not in tracker_data1:
        pytest.fail("Could not read valid initial tracker data.")
    grid1 = tracker_data1["grid"]
    keys1_map = tracker_data1["keys"]  # Renamed for clarity

    # Find keys corresponding to the test files using normalize_path for consistency
    module_a_norm_path = path_utils.normalize_path(
        str(project_root / "src" / "module_a.py")
    )
    module_b_norm_path = path_utils.normalize_path(
        str(project_root / "src" / "module_b.py")
    )

    key_a = next(
        (
            k
            for k, p in keys1_map.items()
            if path_utils.normalize_path(p) == module_a_norm_path
        ),
        None,
    )
    key_b = next(
        (
            k
            for k, p in keys1_map.items()
            if path_utils.normalize_path(p) == module_b_norm_path
        ),
        None,
    )

    if not key_a or not key_b:
        print(f"Keys found: {keys1_map}")
        print(f"Looking for: {module_a_norm_path} and {module_b_norm_path}")
        pytest.fail(
            f"Could not find keys for module_a.py or module_b.py in initial tracker."
        )

    # Verify initially no dependency between a and b
    # module_a.py: "def calculate_sum(a, b): return a + b"
    # module_b.py: "class UserProfile: ..."
    # Neither imports the other, so dependency char should be 'p' (placeholder) or 'n' (none)
    sorted_keys1 = sort_key_strings_hierarchically(list(keys1_map.keys()))
    initial_dep_char = get_char_at_key(grid1, key_a, key_b, sorted_keys1)
    print(f"Initial dependency char {key_a} -> {key_b}: '{initial_dep_char}'")
    assert initial_dep_char in (
        "p",
        "n",
        "?",
    ), f"Initial dependency char expected to be 'p', 'n', or '?', but got '{initial_dep_char}'"

    # 2. Modify a source file to add a dependency
    module_a_path_obj = project_root / "src" / "module_a.py"
    print(f"Modifying {module_a_path_obj} to import module_b...")
    # Add import to create explicit dependency
    module_a_path_obj.write_text(
        "import module_b\n\ndef calculate_sum(a, b):\n    return a + b"
    )
    time.sleep(0.1)  # Ensure mtime change

    # 3. Run analysis again
    print("Running analyze_project (after modification)...")
    # Rely on analyze_project's internal cache handling (mtime checks in analyze_file)
    try:
        _results2 = (
            analyze_project()
        )  # Don't force, let caching work (except for modified file)
    except Exception as e:
        pytest.fail(f"Second analyze_project run failed: {e}")

    # 4. Compare tracker grid
    tracker_path = get_mini_tracker_path(project_root, "src")
    if not tracker_path or not tracker_path.exists():
        pytest.fail(f"Tracker file missing after second run: {tracker_path}")

    tracker_data2 = read_tracker_file_compat(str(tracker_path))
    if not tracker_data2 or "grid" not in tracker_data2 or "keys" not in tracker_data2:
        pytest.fail("Could not read valid tracker data after modification.")
    grid2 = tracker_data2["grid"]
    keys2_map = tracker_data2[
        "keys"
    ]  # Keys might change if analysis adds/removes files

    # Re-find keys in case they changed (unlikely here but good practice)
    key_a_new = next(
        (
            k
            for k, p in keys2_map.items()
            if path_utils.normalize_path(p) == module_a_norm_path
        ),
        None,
    )
    key_b_new = next(
        (
            k
            for k, p in keys2_map.items()
            if path_utils.normalize_path(p) == module_b_norm_path
        ),
        None,
    )

    if not key_a_new or not key_b_new:
        pytest.fail(f"Could not find keys in second tracker. Keys: {keys2_map}")
    assert key_a_new == key_a  # Keys should ideally be stable
    assert key_b_new == key_b

    # Verify the dependency character changed
    sorted_keys2 = sort_key_strings_hierarchically(list(keys2_map.keys()))
    final_dep_char = get_char_at_key(grid2, key_a, key_b, sorted_keys2)
    print(f"Final dependency char {key_a} -> {key_b}: '{final_dep_char}'")
    # Expect '<' (module_a depends on module_b, based on fixture definition)
    # or 'x' (if reciprocal detected)
    # Allow 's' or 'S' if semantic analysis picks it up instead of static import
    assert final_dep_char in (
        "<",
        "x",
        "s",
        "S",
    ), f"Expected dependency char '<', 'x', 's', or 'S', but got '{final_dep_char}'"
    assert (
        final_dep_char != initial_dep_char
    ), "Dependency character did not change after modification"


# Test Case IS-02: Config File Modification (Exclusion)
def test_is02_config_file_exclusion(
    test_project: Path, monkeypatch: pytest.MonkeyPatch, clear_cache_fixture: None
) -> None:
    """Verify analyze_project removes keys for newly excluded files."""
    # No need to mock clear_all_caches for this test.
    # We rely on analyze_project picking up config changes correctly,
    # implicitly using invalidated config-dependent caches.

    project_root = test_project
    # 1. Initial analysis run to establish baseline keys
    print("\nRunning analyze_project (initial run for IS-02)...")
    try:
        _results1 = analyze_project(force_analysis=True)
    except Exception as e:
        pytest.fail(f"Initial analyze_project failed: {e}")

    tracker_path = get_mini_tracker_path(project_root, "src")
    config_path = project_root / ".clinerules.config.json"
    file_to_exclude_rel = "src/module_b.py"
    file_to_exclude_abs_norm = path_utils.normalize_path(
        str(project_root / file_to_exclude_rel)
    )

    if not tracker_path or not tracker_path.exists():
        pytest.fail(f"Initial run did not create tracker file: {tracker_path}")
    tracker_data1 = read_tracker_file_compat(str(tracker_path))
    if not tracker_data1 or "keys" not in tracker_data1:
        pytest.fail("Could not read valid initial tracker data.")
    keys1_map = tracker_data1["keys"]
    key_b_initial = next(
        (
            k
            for k, p in keys1_map.items()
            if path_utils.normalize_path(p) == file_to_exclude_abs_norm
        ),
        None,
    )
    assert (
        key_b_initial is not None
    ), f"Key for {file_to_exclude_rel} not found in initial tracker."
    print(f"Initial tracker contains key '{key_b_initial}' for {file_to_exclude_rel}.")
    assert (
        key_b_initial is not None
    ), f"Key for {file_to_exclude_rel} not found in initial tracker, test setup issue."

    # 2. Modify config file to exclude module_b.py
    print(f"Modifying {config_path} to exclude {file_to_exclude_rel}...")
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        # Ensure excluded_paths list exists
        if "excluded_paths" not in config_data:
            config_data["excluded_paths"] = []

        # Add the relative path to excluded_paths
        config_data["excluded_paths"].append(file_to_exclude_rel)

        config_path.write_text(json.dumps(config_data, indent=4))
        time.sleep(0.1)  # Ensure mtime change
    except Exception as e:
        pytest.fail(f"Failed to modify config file: {e}")

    # Clear ConfigManager singleton instance to force reload on next call
    monkeypatch.setattr(config_manager, "_instance", None, raising=False)

    # Explicitly invalidate ConfigManager caches to avoid mtime resolution issues
    # (Filesystem mtime resolution might be > 0.1s, causing stale cache hits)
    from cline_utils.dependency_system.utils.cache_manager import cache_manager as cm

    cm.get_cache("config_data").invalidate(".*")
    cm.get_cache("excluded_paths").invalidate(".*")
    cm.get_cache("excluded_dirs").invalidate(".*")
    cm.get_cache("excluded_extensions").invalidate(".*")

    # 3. Run analysis again
    print("Running analyze_project (after config exclusion)...")
    try:
        # Rely on config mtime check in ConfigManager and downstream cache invalidation
        _results2 = analyze_project()
    except Exception as e:
        pytest.fail(f"Second analyze_project run failed: {e}")

    # 4. Verify key removal from tracker
    tracker_path = get_mini_tracker_path(project_root, "src")
    if not tracker_path or not tracker_path.exists():
        pytest.fail(f"Tracker file missing after second run: {tracker_path}")
    tracker_data2 = read_tracker_file_compat(str(tracker_path))
    if not tracker_data2 or "keys" not in tracker_data2:
        pytest.fail("Could not read valid tracker data after modification.")
    keys2_map = tracker_data2["keys"]

    key_b_final = next(
        (
            k
            for k, p in keys2_map.items()
            if path_utils.normalize_path(p) == file_to_exclude_abs_norm
        ),
        None,
    )
    assert (
        key_b_final is None
    ), f"Key for excluded file {file_to_exclude_rel} was found in final tracker (key: {key_b_final})."
    print(
        f"Verified: Key for excluded file {file_to_exclude_rel} correctly removed from tracker."
    )


# Test Case IS-03: .clinerules Modification (Roots)
def test_is03_clinerules_modification_roots(test_project: Path) -> None:
    """Verify analyze_project picks up new roots from modified .clinerules."""
    # No need to mock clear_all_caches for this test.
    # We rely on analyze_project picking up .clinerules changes correctly,
    # implicitly using invalidated caches (e.g., for root dirs).

    project_root = test_project
    # tracker_path = project_root / "cline_docs" / "module_relationship_tracker.md"
    # We will look for the NEW tracker for 'lib'
    clinerules_path = project_root / ".clinerules"
    new_root_rel = "lib"
    new_file_rel = f"{new_root_rel}/helper.py"
    new_file_abs_norm = path_utils.normalize_path(str(project_root / new_file_rel))

    # Initial .clinerules content is set up in the fixture now

    # 1. Initial analysis run (with only 'src' as code root)
    print("\nRunning analyze_project (initial run for IS-03)...")
    try:
        _results1 = analyze_project(force_analysis=True)
    except Exception as e:
        pytest.fail(f"Initial analyze_project failed: {e}")

    # Initial run only has src, so lib tracker shouldn't exist (or we check src tracker)
    # But we want to verify 'helper.py' is NOT tracked.
    # It shouldn't be in src tracker.
    tracker_path_src = get_mini_tracker_path(project_root, "src")
    if not tracker_path_src or not tracker_path_src.exists():
        pytest.fail("Initial run did not create src tracker.")

    tracker_data1 = read_tracker_file_compat(str(tracker_path_src))
    if not tracker_data1 or "keys" not in tracker_data1:
        pytest.fail("Could not read valid initial tracker data.")
    keys1_map = tracker_data1["keys"]
    key_helper_initial = next(
        (
            k
            for k, p in keys1_map.items()
            if path_utils.normalize_path(p) == new_file_abs_norm
        ),
        None,
    )
    assert (
        key_helper_initial is None
    ), f"Key for {new_file_rel} found in initial tracker, but 'lib' was not a root."

    # 2. Modify .clinerules to add 'lib' as a code root
    print(
        f"Modifying {clinerules_path} to add '{new_root_rel}' to CODE_ROOT_DIRECTORIES..."
    )
    try:
        current_content = clinerules_path.read_text()
        # Simple string replacement - might be fragile
        modified_content = current_content.replace(
            "[CODE_ROOT_DIRECTORIES]\n- src",
            f"[CODE_ROOT_DIRECTORIES]\n- src\n- {new_root_rel}",
        )
        if modified_content == current_content:  # Fallback if pattern didn't match
            pytest.fail(
                "Could not find '[CODE_ROOT_DIRECTORIES]\n- src' to modify in .clinerules"
            )

        clinerules_path.write_text(modified_content)
        time.sleep(0.1)  # Ensure mtime change
    except Exception as e:
        pytest.fail(f"Failed to modify .clinerules file: {e}")

    # Caches depending on .clinerules mtime should invalidate.
    # This includes ConfigManager reading roots, and potentially path_utils.get_project_root
    from cline_utils.dependency_system.utils.cache_manager import cache_manager as cm

    cm.get_cache("code_roots").invalidate(".*")
    cm.get_cache("doc_dirs").invalidate(".*")

    # 3. Run analysis again
    print("Running analyze_project (after .clinerules modification)...")
    try:
        # ConfigManager should reload due to mtime change check within it or its cache decorator
        # analyze_project uses ConfigManager to get roots
        _results2 = analyze_project()
    except Exception as e:
        pytest.fail(f"Second analyze_project run failed: {e}")

    # 4. Verify new key exists in tracker
    # Now check for lib tracker
    tracker_path_lib = get_mini_tracker_path(project_root, "lib")
    if not tracker_path_lib or not tracker_path_lib.exists():
        pytest.fail(f"Tracker file for 'lib' missing after second run.")
    tracker_data2 = read_tracker_file_compat(str(tracker_path_lib))
    if not tracker_data2 or "keys" not in tracker_data2:
        pytest.fail("Could not read valid tracker data after modification.")
    keys2_map = tracker_data2["keys"]

    key_helper_final = next(
        (
            k
            for k, p in keys2_map.items()
            if path_utils.normalize_path(p) == new_file_abs_norm
        ),
        None,
    )
    assert (
        key_helper_final is not None
    ), f"Key for newly added root file {new_file_rel} was NOT found in final tracker."
    print(
        f"Verified: Key '{key_helper_final}' for new root file {new_file_rel} correctly added to tracker."
    )


# # Test Case IS-04: Embedding Regeneration
def test_is04_embedding_regeneration(test_project: Path) -> None:
    """Verify --force-embeddings works and similarity cache updates."""
    # No mocking needed for clear_all_caches or calculate_similarity.
    # We want to test the *real* interaction after forced embedding regeneration.

    project_root = test_project
    # tracker_path = project_root / "cline_docs" / "module_relationship_tracker.md"
    tracker_path = get_mini_tracker_path(project_root, "src")
    module_a_rel = "src/module_a.py"
    module_b_rel = "src/module_b.py"
    module_a_abs_norm = path_utils.normalize_path(str(project_root / module_a_rel))
    module_b_abs_norm = path_utils.normalize_path(str(project_root / module_b_rel))

    # Define cache name for similarity if needed for stats (optional here)
    _cache_name_sim = "calculate_similarity"

    # Ensure embedding dir exists for calculate_similarity to work
    embedding_dir = project_root / "cache" / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    # ConfigManager in the test environment should point to this 'cache' dir
    # because we set "cache_dir": "cache" in the base_config fixture.
    # We verify this assumption by checking if the dir exists.
    if not embedding_dir.exists():
        pytest.fail(f"Embedding directory {embedding_dir} was not created.")

    # 1. Initial analysis run to establish baseline keys and embeddings
    print("\nRunning analyze_project (initial run for IS-04)...")
    try:
        _results1 = analyze_project(force_analysis=True)  # Ensure embeddings generated
    except Exception as e:
        pytest.fail(f"Initial analyze_project failed: {e}")

    tracker_path = get_mini_tracker_path(project_root, "src")
    if not tracker_path or not tracker_path.exists():
        pytest.fail("Tracker file not created.")
    tracker_data1 = read_tracker_file_compat(str(tracker_path))
    assert tracker_data1 is not None, "Could not read valid initial tracker data."
    keys_map = tracker_data1.get("keys", {})  # Get key map
    key_a = next(
        (
            k
            for k, p in keys_map.items()
            if path_utils.normalize_path(p) == module_a_abs_norm
        ),
        None,
    )
    key_b = next(
        (
            k
            for k, p in keys_map.items()
            if path_utils.normalize_path(p) == module_b_abs_norm
        ),
        None,
    )
    if not key_a or not key_b:
        pytest.fail("Could not find keys for module_a or module_b.")

    # Construct path_to_key_info for calculate_similarity
    path_to_key_info: dict[str, KeyInfo] = {}
    for k, p in keys_map.items():
        norm_p = path_utils.normalize_path(p)
        parent = path_utils.normalize_path(os.path.dirname(p))
        path_to_key_info[norm_p] = KeyInfo(
            key_string=k,
            norm_path=norm_p,
            parent_path=parent,
            tier=1,  # Dummy
            is_directory=False,  # Dummy
        )

    # Call REAL similarity function to get a baseline
    initial_sim = embedding_manager.calculate_similarity(
        key_a,
        key_b,
        str(embedding_dir),
        path_to_key_info,
        str(project_root),
        ["src"],  # code_roots
        ["docs"],  # doc_roots
    )
    print(f" -> Initial Similarity: {initial_sim}")
    assert (
        initial_sim > 0
    ), f"Initial similarity is 0.0, which implies mocking failed or empty embeddings. Sim: {initial_sim}"

    # 2. Modify a source file significantly
    module_a_path_obj = project_root / module_a_rel
    print(f"Modifying {module_a_path_obj} significantly...")
    new_content_a = "class NewClass:\n def method(self):\n  pass\n# Completely different content hash"
    module_a_path_obj.write_text(new_content_a)
    # content_a_new_hash = hash(new_content_a) # Not needed without mock
    time.sleep(0.1)

    # 3. Run analysis with force_embeddings=True AND force_analysis=True to ensure fresh SES
    print("Running analyze_project --force-embeddings...")
    try:
        _results2 = analyze_project(force_embeddings=True, force_analysis=True)
    except Exception as e:
        pytest.fail(f"analyze_project --force-embeddings run failed: {e}")

    # 4. Call similarity manually *after* the forced embedding run
    print("Calling REAL similarity calculation *after* force-embeddings run...")
    # Ensure the keys are still valid (they shouldn't change in this scenario)
    tracker_data2 = read_tracker_file_compat(str(tracker_path))
    assert (
        tracker_data2 is not None
    ), "Could not read valid tracker data after modification."
    keys_map_after = tracker_data2.get("keys", {})
    key_a_after = next(
        (
            k
            for k, p in keys_map_after.items()
            if path_utils.normalize_path(p) == module_a_abs_norm
        ),
        None,
    )
    key_b_after = next(
        (
            k
            for k, p in keys_map_after.items()
            if path_utils.normalize_path(p) == module_b_abs_norm
        ),
        None,
    )
    assert key_a_after == key_a, "Key for module_a changed unexpectedly."
    assert key_b_after == key_b, "Key for module_b changed unexpectedly."

    final_sim = embedding_manager.calculate_similarity(
        key_a,
        key_b,
        str(embedding_dir),
        path_to_key_info,  # Keys shouldn't have changed, so this is still valid
        str(project_root),
        ["src"],
        ["docs"],
    )
    print(f" -> Final Similarity: {final_sim}")

    # Assertions:
    # Check that the similarity score changed.
    # We use a deterministic mock embedding model based on text hash.
    # Since the file content changed significantly, the hash changed, so the embedding changed.
    # Therefore, the similarity score MUST be different.
    similarity_change = abs(final_sim - initial_sim)
    # With random vectors (randn), similarity is usually low (~0).
    # We just need to verify that the score CHANGED, proving cache invalidation.
    # A small change is expected if both scores are near 0.
    assert (
        similarity_change > 1e-6
    ), f"Similarity score did not change significantly ({similarity_change:.6f}) after modifying file. Cache might not have updated."
    assert (
        final_sim != initial_sim
    ), f"Similarity score ({final_sim}) did not change after modifying file (initial: {initial_sim}). Cache might not have updated."

    print(
        "Verified: Embedding regeneration appears to have triggered use of new data for similarity."
    )


def test_is05_embedding_regeneration_on_model_change(
    test_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that shifting model name in config triggers full regeneration of embeddings."""
    project_root = test_project

    embedding_dir = project_root / "cache" / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    # 1. Set model to mock-model-1
    monkeypatch.setattr(
        embedding_manager,
        "_selected_model_config",
        {"type": "sentence-transformer", "embedding_dim": 384, "name": "mock-model-1"},
    )

    # Run initial analysis
    analyze_project(force_analysis=True)

    # Check that metadata has mock-model-1
    metadata_path = embedding_dir / "metadata.json"
    assert metadata_path.exists()
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    assert meta_data.get("model") == "mock-model-1"

    # Set up spy for _flush_batch
    flush_calls = 0
    original_flush = embedding_manager._flush_batch

    def mock_flush(*args, **kwargs):
        nonlocal flush_calls
        flush_calls += 1
        return original_flush(*args, **kwargs)

    monkeypatch.setattr(embedding_manager, "_flush_batch", mock_flush)

    # 2. Re-run analysis WITHOUT changing the model - should skip and NOT call _flush_batch
    analyze_project(force_analysis=False)
    assert flush_calls == 0  # Should be cached/skipped

    # 3. Change model to mock-model-2
    monkeypatch.setattr(
        embedding_manager,
        "_selected_model_config",
        {"type": "sentence-transformer", "embedding_dim": 384, "name": "mock-model-2"},
    )

    # Run analysis - should detect model mismatch and force regeneration
    analyze_project(force_analysis=False)
    assert flush_calls > 0  # Must regenerate embeddings

    # Check that metadata now has mock-model-2
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data_updated = json.load(f)
    assert meta_data_updated.get("model") == "mock-model-2"


def test_is06_embedding_regeneration_on_dim_mismatch(
    test_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that a shape/dimension mismatch in an existing `.npy` file forces its regeneration."""
    project_root = test_project

    embedding_dir = project_root / "cache" / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    # 1. Set model config (expected dim = 384)
    monkeypatch.setattr(
        embedding_manager,
        "_selected_model_config",
        {"type": "sentence-transformer", "embedding_dim": 384, "name": "mock-model-1"},
    )

    # Write a dummy stale .npy file with a completely different dimension (e.g. 768)
    npy_path = embedding_dir / "src" / "module_a.py.npy"
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    stale_emb = np.zeros(768, dtype=np.float32)
    np.save(npy_path, stale_emb)

    # Set up spy for _flush_batch
    flush_calls = 0
    original_flush = embedding_manager._flush_batch

    def mock_flush(*args, **kwargs):
        nonlocal flush_calls
        flush_calls += 1
        return original_flush(*args, **kwargs)

    monkeypatch.setattr(embedding_manager, "_flush_batch", mock_flush)

    # 2. Run analysis - should detect shape mismatch (768 vs 384) and force regeneration
    analyze_project(force_analysis=False)
    assert flush_calls > 0  # Must regenerate due to dimension mismatch

    # Verify the saved file now has 384 elements
    saved_emb = np.load(npy_path)
    assert saved_emb.shape[0] == 384
