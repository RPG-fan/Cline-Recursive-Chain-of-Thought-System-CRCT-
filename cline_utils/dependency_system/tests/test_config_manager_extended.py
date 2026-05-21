import pytest
import os
import json
from unittest.mock import MagicMock, patch, mock_open

from cline_utils.dependency_system.utils.config_manager import (
    ConfigManager,
    DEFAULT_CONFIG
)

@pytest.fixture(autouse=True)
def mock_cached():
    def no_op_cached(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
        
    with patch("cline_utils.dependency_system.utils.cache_manager.cached", side_effect=no_op_cached):
        yield

@pytest.fixture
def clean_config_manager():
    # Reset singleton instance
    ConfigManager._instance = None
    # Patch load methods to avoid file I/O
    with patch.object(ConfigManager, '_load_user_config_file', return_value=None):
        with patch.object(ConfigManager, '_save_config', return_value=True):
            manager = ConfigManager()
            yield manager
    ConfigManager._instance = None

class TestConfigManagerExtended:
    def test_singleton(self, clean_config_manager):
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_default_values(self, clean_config_manager):
        assert clean_config_manager.config["performance"]["default_batch_size"] == 32
        assert clean_config_manager.config["resources"]["min_memory_mb"] == 512

    def test_deep_update(self, clean_config_manager):
        original = {"a": 1, "b": {"c": 2}}
        updates = {"b": {"d": 3}, "e": 4}
        clean_config_manager._deep_update(original, updates)
        assert original == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}

    def test_environment_overrides(self):
        # Reset singleton
        ConfigManager._instance = None
        
        with patch.dict(os.environ, {
            "ANALYZER_BATCH_SIZE": "64",
            "ANALYZER_LOG_LEVEL": "DEBUG",
            "ANALYZER_USE_STREAMING": "false",
            "ANALYZER_SKIP_DISK_ESTIMATION": "true"
        }):
            with patch.object(ConfigManager, '_load_user_config_file', return_value=None):
                with patch.object(ConfigManager, '_save_config', return_value=True):
                    manager = ConfigManager()
                    
                    assert manager.get_performance_setting("default_batch_size") == 64
                    assert manager.get_output_setting("log_level") == "DEBUG"
                    assert manager.get_performance_setting("use_streaming_analysis") is False
                    assert manager.get_resource_setting("skip_disk_estimation") is True

    def test_resource_adjustments_low_memory(self, clean_config_manager):
        # Mock ResourceValidator to return low memory
        mock_validator = MagicMock()
        mock_validator.validate_system_resources.return_value = {
            "valid": True,
            "resource_check": {
                "memory": {"available_mb": 512}, # Low memory
                "cpu": {"cores": 4},
                "disk_space": {"free_space_mb": 10000}
            }
        }
        
        with patch("cline_utils.dependency_system.utils.config_manager.ResourceValidator", return_value=mock_validator):
            clean_config_manager._apply_resource_adjustments()
            
            # Should have adjusted batch sizes
            assert clean_config_manager.get_performance_setting("default_batch_size") <= 16
            assert clean_config_manager.get_performance_setting("embedding_batch_size") <= 8
            assert clean_config_manager.get_performance_setting("max_workers") == 1
            assert clean_config_manager.get_performance_setting("use_streaming_analysis") is True

    def test_resource_adjustments_low_disk(self, clean_config_manager):
        # Mock ResourceValidator to return low disk
        mock_validator = MagicMock()
        mock_validator.validate_system_resources.return_value = {
            "valid": True,
            "resource_check": {
                "memory": {"available_mb": 4096},
                "cpu": {"cores": 4},
                "disk_space": {"free_space_mb": 100} # Low disk
            }
        }
        
        with patch("cline_utils.dependency_system.utils.config_manager.ResourceValidator", return_value=mock_validator):
            clean_config_manager._apply_resource_adjustments()
            
            # Should have reduced cache size and disabled diagrams
            assert clean_config_manager.get_performance_setting("cache_size_limit") <= 1000
            assert clean_config_manager.get_output_setting("auto_generate_diagrams") is False

    def test_get_analysis_settings(self, clean_config_manager):
        settings = clean_config_manager.get_analysis_settings()
        assert "use_streaming" in settings
        assert "batch_size" in settings
        assert "memory_limit_mb" in settings
        assert settings["batch_size"] == 32 # Default

    def test_get_optimization_recommendations(self, clean_config_manager):
        # Manually set validation results
        clean_config_manager._resource_validation_results = {
            "resource_check": {
                "memory": {"available_mb": 512},
                "cpu": {"cores": 1},
                "disk_space": {"free_space_mb": 100}
            }
        }
        
        recommendations = clean_config_manager.get_optimization_recommendations()
        assert any("streaming analysis" in r for r in recommendations)
        assert any("limited CPU" in r for r in recommendations)
        assert any("freeing up disk space" in r for r in recommendations)

    def test_default_config_mutation_prevention(self):
        # Reset singleton to ensure fresh start
        ConfigManager._instance = None
        
        with patch.object(ConfigManager, '_load_user_config_file', return_value=None):
            with patch.object(ConfigManager, '_save_config', return_value=True):
                manager = ConfigManager()
                
                # Check mutation of excluded_dirs list retrieved from getter
                ex_dirs = manager.get_excluded_dirs()
                ex_dirs.append("MUTATED_DIR_TEST")
                assert "MUTATED_DIR_TEST" not in DEFAULT_CONFIG["excluded_dirs"]
                
                # Check allowed_dependency_chars mutation prevention
                chars = manager.get_allowed_dependency_chars()
                chars.append("MUTATED_CHAR")
                assert "MUTATED_CHAR" not in DEFAULT_CONFIG["allowed_dependency_chars"]
                
                # Mutate nested dictionary retrieved via config property
                manager.config["performance"]["default_batch_size"] = 9999
                assert DEFAULT_CONFIG["performance"]["default_batch_size"] == 32
                
                # Check that fresh manager instance has clean configuration
                ConfigManager._instance = None
                manager2 = ConfigManager()
                assert "MUTATED_DIR_TEST" not in manager2.get_excluded_dirs()
                assert "MUTATED_CHAR" not in manager2.get_allowed_dependency_chars()
                assert manager2.config["performance"]["default_batch_size"] == 32

