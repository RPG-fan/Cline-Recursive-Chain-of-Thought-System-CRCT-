# CRCT Dependency System Test Suite

This directory contains the comprehensive test suite for the CRCT Dependency System.

## Structure

- **`test_functional_cache.py`**: Functional tests for caching mechanisms (low-level).
- **`test_integration_cache.py`**: Integration tests for the full analysis workflow, including embedding generation and source modification.
- **`test_manual_tooling_cache.py`**: Tests for manual cache management tools.
- **`test_e2e_workflow.py`**: End-to-End tests simulating CLI usage.
- **`test_resource_validator.py`**: Tests for system resource validation and optimization.
- **`test_phase_tracker.py`**: Tests for progress tracking and UI feedback.
- **`test_config_manager_extended.py`**: Tests for configuration management, environment overrides, and resource adjustments.
- **`test_runtime_inspector.py`**: Tests for runtime symbol extraction and analysis.

## Running Tests

### Prerequisites
Ensure you have the development dependencies installed:
```bash
pip install -r requirements.txt
```

### Basic Usage
Run all tests:
```bash
pytest cline_utils/dependency_system/tests/
```

Run with verbose output:
```bash
pytest -v cline_utils/dependency_system/tests/
```

### Running Specific Tests
Run a specific test file:
```bash
pytest cline_utils/dependency_system/tests/test_integration_cache.py
```

Run a specific test case:
```bash
pytest cline_utils/dependency_system/tests/test_integration_cache.py::test_is04_embedding_regeneration
```

### Coverage
Generate a coverage report:
```bash
pytest --cov=cline_utils.dependency_system --cov-report=html cline_utils/dependency_system/tests/
```
The report will be generated in `htmlcov/index.html`.

## Adding New Tests

1.  **Identify the Component**: Determine which component you are testing (e.g., `project_analyzer`, `embedding_manager`).
2.  **Choose the Level**:
    *   **Unit/Functional**: Use `test_functional_*.py` or create a new `test_<component>.py`.
    *   **Integration**: Use `test_integration_*.py` if it involves multiple components or file I/O.
    *   **E2E**: Use `test_e2e_workflow.py` for full workflow verification.
3.  **Use Fixtures**: Leverage existing fixtures in `conftest.py` (if available) or define them locally. Common fixtures include `test_project` (creates a temp project structure) and `mock_embedding_model`.
4.  **Mock External Calls**: Avoid actual network requests (e.g., model downloads) by mocking `embedding_manager` or `transformers`.

## Troubleshooting

-   **Embedding Model Downloads**: If tests are slow or failing due to network issues, ensure `MockEmbeddingModel` is being used or that you have the model cached locally.
-   **Path Issues**: Use `pathlib` for robust path handling across operating systems.
-   **Cache Interference**: Tests should generally run in isolation. Ensure `clear_all_caches()` is called or that tests use unique temporary directories.
