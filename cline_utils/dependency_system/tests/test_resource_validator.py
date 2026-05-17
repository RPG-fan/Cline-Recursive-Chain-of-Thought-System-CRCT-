import pytest
import json
import datetime
from typing import Any, Generator
from unittest.mock import MagicMock, patch, mock_open

from cline_utils.dependency_system.utils.resource_validator import (
    ResourceValidator,
    get_cache_path as _get_cache_path,
    _load_validation_cache,
    _save_validation_cache,
    _is_cache_valid,
    validate_and_get_optimal_settings,
    MemoryLimitError,
    CACHE_VERSION,
)

# Mock data
MOCK_PROJECT_PATH = "/path/to/project"
MOCK_CACHE_PATH = "/path/to/cache/validation_cache.json"


@pytest.fixture
def validator() -> ResourceValidator:
    return ResourceValidator(strict_mode=False)


@pytest.fixture
def mock_psutil() -> Generator[MagicMock, None, None]:
    mock = MagicMock()
    with patch.dict("sys.modules", {"psutil": mock}):
        yield mock


@pytest.fixture
def mock_shutil() -> Generator[MagicMock, None, None]:
    with patch("cline_utils.dependency_system.utils.resource_validator.shutil") as mock:
        yield mock


@pytest.fixture
def mock_fs() -> Generator[tuple[MagicMock, MagicMock, MagicMock], None, None]:
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("os.path.exists") as mock_exists:
            with patch("os.makedirs") as mock_makedirs:
                yield mock_file, mock_exists, mock_makedirs


class TestResourceValidatorCache:
    def test_get_cache_path(self) -> None:
        path = _get_cache_path()
        assert path.endswith("validation_cache.json")

    def test_save_validation_cache(
        self, mock_fs: tuple[MagicMock, MagicMock, MagicMock]
    ) -> None:
        mock_file, mock_exists, mock_makedirs = mock_fs
        results: dict[str, Any] = {"valid": True, "errors": []}

        with patch(
            "cline_utils.dependency_system.utils.resource_validator.get_cache_path",
            return_value=MOCK_CACHE_PATH,
        ):
            _save_validation_cache(MOCK_PROJECT_PATH, results)

            mock_makedirs.assert_called_once()
            mock_file.assert_called_with(MOCK_CACHE_PATH, "w", encoding="utf-8")
            handle: Any = mock_file()
            # Verify json dump was called (checking write calls)
            assert handle.write.called

    def test_load_validation_cache_exists(
        self, mock_fs: tuple[MagicMock, MagicMock, MagicMock]
    ) -> None:
        mock_file, mock_exists, _ = mock_fs
        mock_exists.return_value = True

        cache_data: dict[str, Any] = {
            "version": CACHE_VERSION,
            "project_path": MOCK_PROJECT_PATH,
            "results": {"valid": True},
        }

        mock_file.return_value.read.return_value = json.dumps(cache_data)

        with patch(
            "cline_utils.dependency_system.utils.resource_validator.get_cache_path",
            return_value=MOCK_CACHE_PATH,
        ):
            loaded_data = _load_validation_cache()
            assert loaded_data == cache_data

    def test_load_validation_cache_not_exists(
        self, mock_fs: tuple[MagicMock, MagicMock, MagicMock]
    ) -> None:
        _, mock_exists, _ = mock_fs
        mock_exists.return_value = False

        with patch(
            "cline_utils.dependency_system.utils.resource_validator.get_cache_path",
            return_value=MOCK_CACHE_PATH,
        ):
            loaded_data = _load_validation_cache()
            assert loaded_data is None

    def test_is_cache_valid_success(self) -> None:
        cache_data: dict[str, Any] = {
            "version": CACHE_VERSION,
            "project_path": MOCK_PROJECT_PATH,
            "last_validated": datetime.datetime.now().isoformat(),
            "results": {
                "valid": True,
                "errors": [],
                "warnings": [],
                "resource_check": {"memory": {"available_mb": 4096}},
            },
        }
        assert _is_cache_valid(cache_data, MOCK_PROJECT_PATH) is True

    def test_is_cache_valid_version_mismatch(self) -> None:
        cache_data: dict[str, Any] = {
            "version": "0.9",
            "project_path": MOCK_PROJECT_PATH,
            "last_validated": datetime.datetime.now().isoformat(),
            "results": {"valid": True},
        }
        assert _is_cache_valid(cache_data, MOCK_PROJECT_PATH) is False

    def test_is_cache_valid_path_mismatch(self) -> None:
        cache_data: dict[str, Any] = {
            "version": CACHE_VERSION,
            "project_path": "/other/path",
            "last_validated": datetime.datetime.now().isoformat(),
            "results": {
                "valid": True,
                "resource_check": {"memory": {"available_mb": 4096}},
            },
        }
        assert _is_cache_valid(cache_data, MOCK_PROJECT_PATH) is False

    def test_is_cache_valid_expired(self) -> None:
        old_date = (datetime.datetime.now() - datetime.timedelta(days=8)).isoformat()
        cache_data: dict[str, Any] = {
            "version": CACHE_VERSION,
            "project_path": MOCK_PROJECT_PATH,
            "last_validated": old_date,
            "results": {
                "valid": True,
                "resource_check": {"memory": {"available_mb": 4096}},
            },
        }
        assert _is_cache_valid(cache_data, MOCK_PROJECT_PATH) is False

    def test_is_cache_valid_previous_errors(self) -> None:
        cache_data: dict[str, Any] = {
            "version": CACHE_VERSION,
            "project_path": MOCK_PROJECT_PATH,
            "last_validated": datetime.datetime.now().isoformat(),
            "results": {"valid": False, "errors": ["Some error"]},
        }

        assert _is_cache_valid(cache_data, MOCK_PROJECT_PATH) is False


class TestResourceValidatorMemory:
    def test_validate_memory_sufficient(
        self, validator: ResourceValidator, mock_psutil: MagicMock
    ) -> None:
        # Mock 16GB total, 8GB available
        memory_mock = MagicMock()
        memory_mock.total = 16 * 1024 * 1024 * 1024
        memory_mock.available = 8 * 1024 * 1024 * 1024
        memory_mock.percent = 50.0
        mock_psutil.virtual_memory.return_value = memory_mock

        result = validator._validate_memory()
        assert result["sufficient"] is True
        assert result["critical"] is False
        assert result["available_mb"] == 8192.0

    def test_validate_memory_critical(
        self, validator: ResourceValidator, mock_psutil: MagicMock
    ) -> None:
        # Mock 16GB total, 100MB available (below MIN_MEMORY_MB=512)
        memory_mock = MagicMock()
        memory_mock.total = 16 * 1024 * 1024 * 1024
        memory_mock.available = 100 * 1024 * 1024
        memory_mock.percent = 99.0
        mock_psutil.virtual_memory.return_value = memory_mock

        with pytest.raises(MemoryLimitError):
            validator._validate_memory()

    def test_validate_memory_fallback(self, validator: ResourceValidator) -> None:
        # Force ImportError for psutil
        with patch.dict("sys.modules", {"psutil": None}):
            # Mock sys.platform to not be win32 to test generic fallback
            with patch("sys.platform", "linux"):
                result = validator._validate_memory()
                assert result["fallback"] is True
                assert result["sufficient"] is True


class TestResourceValidatorDisk:
    def test_validate_disk_space_sufficient(
        self, validator: ResourceValidator, mock_shutil: MagicMock
    ) -> None:
        # Mock 100GB total, 50GB free
        mock_shutil.disk_usage.return_value = (
            100 * 1024**3,
            50 * 1024**3,
            50 * 1024**3,
        )

        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                validator, "_estimate_required_disk_space", return_value=500
            ):
                result = validator._validate_disk_space(MOCK_PROJECT_PATH)
                assert result["sufficient"] is True
                assert result["free_space_mb"] == 50 * 1024

    def test_validate_disk_space_insufficient(
        self, validator: ResourceValidator, mock_shutil: MagicMock
    ) -> None:
        # Mock 100GB total, 10MB free
        mock_shutil.disk_usage.return_value = (
            100 * 1024**3,
            99.99 * 1024**3,
            10 * 1024**2,
        )

        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                validator, "_estimate_required_disk_space", return_value=500
            ):
                # The method suppresses the error and returns a fallback with sufficient=False
                result = validator._validate_disk_space(MOCK_PROJECT_PATH)
                assert result["sufficient"] is False
                assert "error" in result


class TestResourceValidatorIntegration:
    def test_validate_system_resources_success(
        self, validator: ResourceValidator
    ) -> None:
        with patch.object(
            validator,
            "_validate_memory",
            return_value={"sufficient": True, "critical": False, "available_mb": 2048},
        ):
            with patch.object(
                validator,
                "_validate_disk_space",
                return_value={"sufficient": True, "free_space_mb": 5000},
            ):
                with patch.object(
                    validator,
                    "_validate_temporary_space",
                    return_value={"sufficient": True, "free_space_mb": 5000},
                ):
                    with patch.object(
                        validator,
                        "_validate_cpu",
                        return_value={"sufficient": True, "cores": 4},
                    ):
                        with patch.object(
                            validator,
                            "_validate_project_specific",
                            return_value={"sufficient": True},
                        ):
                            with patch(
                                "cline_utils.dependency_system.utils.resource_validator._save_validation_cache"
                            ) as mock_save:
                                results = validator.validate_system_resources(
                                    MOCK_PROJECT_PATH
                                )
                                assert results["valid"] is True
                                assert len(results["errors"]) == 0
                                mock_save.assert_called()

    def test_validate_and_get_optimal_settings(self) -> None:
        with patch(
            "cline_utils.dependency_system.utils.resource_validator.ResourceValidator.validate_system_resources"
        ) as mock_validate:
            # Mock high resources
            mock_validate.return_value = {
                "valid": True,
                "resource_check": {
                    "memory": {"available_mb": 4096},
                    "cpu": {"cores": 8},
                },
            }
            settings = validate_and_get_optimal_settings(MOCK_PROJECT_PATH)
            assert settings["batch_size"] == 32
            assert settings["enable_parallel"] is True

            # Mock low resources
            mock_validate.return_value = {
                "valid": True,
                "resource_check": {
                    "memory": {"available_mb": 512},
                    "cpu": {"cores": 1},
                },
            }
            settings = validate_and_get_optimal_settings(MOCK_PROJECT_PATH)
            assert settings["batch_size"] == 16
            assert settings["memory_efficient"] is True
            assert settings["enable_parallel"] is False
