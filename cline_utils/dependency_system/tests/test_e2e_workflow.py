import pytest
import os
import json
import argparse
from pathlib import Path
from typing import Generator, Any
from unittest.mock import MagicMock, patch

from cline_utils.dependency_system.dependency_processor import command_handler_analyze_project
from cline_utils.dependency_system.utils.cache_manager import clear_all_caches
import numpy as np

# --- Fixtures ---

@pytest.fixture(autouse=True)
def mock_embedding_manager() -> Generator[None, None, None]:
    """Mocks the EmbeddingManager to avoid model downloads."""
    with patch("cline_utils.dependency_system.analysis.embedding_manager._load_model") as mock_load, \
         patch("cline_utils.dependency_system.analysis.embedding_manager._model_instance", new_callable=MagicMock) as mock_instance, \
         patch("cline_utils.dependency_system.analysis.embedding_manager._selected_model_config", {"type": "sentence-transformer", "name": "mock-model"}):
        
        # Mock model instance
        mock_load.return_value = mock_instance
        
        # Ensure global _model_instance is set when _load_model is called
        def side_effect_load(*args: Any, **kwargs: Any) -> MagicMock:
            import cline_utils.dependency_system.analysis.embedding_manager as em
            em._model_instance = mock_instance  # pyright: ignore[reportPrivateUsage]
            return mock_instance
            
        mock_load.side_effect = side_effect_load
        
        # Mock sentence transformer encode method
        mock_instance.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_instance.embed.return_value = np.random.rand(384).tolist()
        
        yield

@pytest.fixture(scope="function")
def test_project_e2e(tmp_path: Path) -> Path:
    """Sets up a realistic project structure for E2E testing."""
    project_dir = tmp_path / "test_project_e2e"
    project_dir.mkdir()
    
    # Create .clinerules and config
    (project_dir / ".clinerules").write_text("[CODE_ROOT_DIRECTORIES]\n- src\n\n[DOC_DIRECTORIES]\n- docs")
    (project_dir / ".clinerules.config.json").write_text(json.dumps({
        "paths": {
            "doc_dir": "docs", 
            "memory_dir": "cline_docs", 
            "cache_dir": "cache",
            "embeddings_dir": "cline_docs/embeddings"
        },
        "excluded_dirs": [".git", "venv"],
        "embedding": {"model_selection": "mpnet"}
    }))

    # Create source code
    src_dir = project_dir / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("import utils\nprint('Main running')")
    (src_dir / "utils.py").write_text("def helper(): return True")
    
    # Create docs
    docs_dir = project_dir / "docs"
    docs_dir.mkdir()
    (docs_dir / "readme.md").write_text("# Project Readme\nDescribes main.py")
    
    # Create memory dir
    memory_dir = project_dir / "cline_docs"
    memory_dir.mkdir()
    
    return project_dir

@pytest.fixture
def mock_args() -> MagicMock:
    """Helper to create mock arguments."""
    args = MagicMock(spec=argparse.Namespace)
    args.project_root = "."
    args.output = None
    args.force_analysis = False
    args.force_embeddings = False
    args.force_validate = False
    return args

# --- Tests ---

def test_e2e_analyze_project_fresh(test_project_e2e: Path, mock_args: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
    """Test a fresh run of analyze-project."""
    clear_all_caches()
    mock_args.project_root = str(test_project_e2e)
    
    # Run analysis
    exit_code = command_handler_analyze_project(mock_args)
    
    # Verify success
    assert exit_code == 0
    
    # Verify output files
    cline_docs = test_project_e2e / "cline_docs"
    assert (cline_docs / "embeddings" / "metadata.json").exists()
    assert (cline_docs / "doc_tracker.md").exists() or (cline_docs / "tracker.md").exists()
    
    # Verify captured output
    captured = capsys.readouterr()
    assert "Project analysis completed successfully" in captured.out

def test_e2e_analyze_project_cached(test_project_e2e: Path, mock_args: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that a second run uses cached data (idempotency)."""
    clear_all_caches()
    mock_args.project_root = str(test_project_e2e)
    
    # 1. First Run
    print("--- First Run ---")
    exit_code1 = command_handler_analyze_project(mock_args)
    assert exit_code1 == 0
    
    # 2. Second Run
    print("--- Second Run ---")
    exit_code2 = command_handler_analyze_project(mock_args)
    assert exit_code2 == 0
    
    # Verify output files still exist
    cline_docs = test_project_e2e / "cline_docs"
    assert (cline_docs / "embeddings" / "metadata.json").exists()

def test_e2e_force_analysis(test_project_e2e: Path, mock_args: MagicMock) -> None:
    """Test that --force-analysis triggers re-analysis."""
    clear_all_caches()
    mock_args.project_root = str(test_project_e2e)
    mock_args.force_analysis = True
    
    exit_code = command_handler_analyze_project(mock_args)
    assert exit_code == 0

def test_e2e_force_embeddings(test_project_e2e: Path, mock_args: MagicMock) -> None:
    """Test that --force-embeddings works."""
    clear_all_caches()
    mock_args.project_root = str(test_project_e2e)
    mock_args.force_embeddings = True
    
    exit_code = command_handler_analyze_project(mock_args)
    assert exit_code == 0

def test_e2e_force_validate(test_project_e2e: Path, mock_args: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    """Test that --force-validate clears validation cache."""
    clear_all_caches()
    mock_args.project_root = str(test_project_e2e)
    mock_args.force_validate = True
    
    # Pre-create the validation cache file to ensure it exists
    from cline_utils.dependency_system.utils.resource_validator import get_cache_path
    cache_path = get_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write("{}")
    
    # We need to mock the logger to check for the specific log message
    with caplog.at_level("INFO"):
        exit_code = command_handler_analyze_project(mock_args)
    
    assert exit_code == 0
    # Check logs for validation cache clearing message
    assert "Cleared validation cache" in caplog.text

