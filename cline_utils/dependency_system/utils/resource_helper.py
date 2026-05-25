# resource_helper.py

"""
Downstream resource helper utilities that coordinate configuration and validation.
Moved from resource_validator.py to prevent circular import dependencies.
"""

from typing import Any, Dict
from .config_manager import ConfigManager
from .resource_validator import ResourceValidator


def quick_resource_check(project_path: str) -> bool:
    """Quick check if basic resources are available for analysis."""
    try:
        config_mgr = ConfigManager()
        excluded = config_mgr.get_excluded_dirs()
    except Exception:
        excluded = None

    try:
        validator = ResourceValidator(strict_mode=False, excluded_dirs=excluded)
        results = validator.validate_system_resources(project_path)
        return results["valid"]
    except Exception:
        return False  # If validation fails, be conservative


def validate_and_get_optimal_settings(project_path: str) -> Dict[str, Any]:
    """Validate resources and return optimal analysis settings."""
    try:
        config_mgr = ConfigManager()
        excluded = config_mgr.get_excluded_dirs()
    except Exception:
        excluded = None

    validator = ResourceValidator(strict_mode=False, excluded_dirs=excluded)
    results = validator.validate_system_resources(project_path)

    settings = {
        "use_streaming": True,
        "batch_size": 32,
        "chunk_size": 8192,
        "enable_parallel": True,
        "memory_efficient": False,
    }

    # Adjust settings based on available resources
    memory_check = results.get("resource_check", {}).get("memory", {})
    available_mb = memory_check.get("available_mb", 0)

    if available_mb < 1024:
        settings.update(
            {"use_streaming": True, "batch_size": 16, "memory_efficient": True}
        )
    elif available_mb < 2048:
        settings.update({"batch_size": 24})

    # CPU-based adjustments
    cpu_check = results.get("resource_check", {}).get("cpu", {})
    cores = cpu_check.get("cores", 1)
    if cores < 2:
        settings["enable_parallel"] = False

    return settings
