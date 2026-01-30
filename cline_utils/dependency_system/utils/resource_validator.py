"""
System resource validation utilities for project analyzer.
Validates available memory, disk space, and other resources before analysis.
"""

import datetime
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.exceptions_enhanced import DiskSpaceError, MemoryLimitError, log_and_reraise
from ..utils.path_utils import normalize_path

logger = logging.getLogger(__name__)

# Cache configuration
VALIDATION_CACHE_FILE = "validation_cache.json"
DEFAULT_CACHE_TTL_SECONDS = 604800  # 7 days (hardware resources rarely change)


def _get_cache_path() -> str:
    """Get path to validation cache file."""
    from .. import core

    core_dir = os.path.dirname(os.path.abspath(core.__file__))
    return os.path.join(core_dir, VALIDATION_CACHE_FILE)


def _load_validation_cache() -> Optional[Dict[str, Any]]:
    """Load cached validation results if available."""
    try:
        cache_path = _get_cache_path()
        if not os.path.exists(cache_path):
            return None

        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        return cache_data
    except Exception as e:
        logger.warning(f"Failed to load validation cache: {e}")
        return None


def _save_validation_cache(project_path: str, results: Dict[str, Any]) -> None:
    """Save validation results to cache."""
    try:
        cache_path = _get_cache_path()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        cache_data = {
            "version": "1.0",
            "last_validated": datetime.datetime.now().isoformat(),
            "project_path": normalize_path(project_path),
            "results": results,
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

        logger.debug(f"Saved validation cache to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save validation cache: {e}")


def _is_cache_valid(
    cache_data: Dict[str, Any],
    project_path: str,
    ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
) -> bool:
    """Check if cached validation is still valid."""
    try:
        # Check version
        if cache_data.get("version") != "1.0":
            return False

        # Check project path
        if normalize_path(cache_data.get("project_path", "")) != normalize_path(
            project_path
        ):
            logger.debug("Cache invalid: project path mismatch")
            return False

        # Check age
        last_validated_str = cache_data.get("last_validated")
        if not last_validated_str:
            return False

        last_validated = datetime.datetime.fromisoformat(last_validated_str)
        age_seconds = (datetime.datetime.now() - last_validated).total_seconds()

        if age_seconds > ttl_seconds:
            logger.debug(
                f"Cache invalid: age {age_seconds:.1f}s exceeds TTL {ttl_seconds}s"
            )
            return False

        # Check if previous validation had errors/warnings
        results = cache_data.get("results", {})
        if not results.get("valid", False):
            logger.debug("Cache invalid: previous validation had errors")
            return False

        if results.get("errors") or results.get("warnings"):
            logger.debug("Cache invalid: previous validation had warnings/errors")
            return False

        return True
    except Exception as e:
        logger.warning(f"Error checking cache validity: {e}")
        return False


class ResourceValidator:
    """Validates system resources for project analysis."""

    # Minimum resource requirements (in MB)
    MIN_MEMORY_MB = 512
    MIN_DISK_SPACE_MB = 100
    MIN_FREE_SPACE_MB = 50

    # Recommended resource requirements (in MB)
    RECOMMENDED_MEMORY_MB = 2048
    RECOMMENDED_DISK_SPACE_MB = 500

    def __init__(self, strict_mode: bool = False):
        """
        Initialize resource validator.

        Args:
            strict_mode: If True, fail on warnings. If False, only fail on critical issues.
        """
        super().__init__()
        self.strict_mode = strict_mode
        self.validation_results = {}

    def validate_system_resources(
        self, project_path: str, estimated_files: int = 0
    ) -> Dict[str, Any]:
        """
        Comprehensive system resource validation.

        Args:
            project_path: Path to project directory
            estimated_files: Estimated number of files to analyze

        Returns:
            Dictionary with validation results and recommendations
        """
        logger.info("Starting comprehensive system resource validation...")

        # Try to use cached validation results
        cache_data = _load_validation_cache()
        if cache_data and _is_cache_valid(cache_data, project_path):
            cached_results = cache_data.get("results")
            if cached_results:
                logger.info("Using cached resource validation results (cache hit)")
                self.validation_results = cached_results
                return cached_results

        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
            "resource_check": {
                "memory": {},
                "disk_space": {},
                "cpu": {},
                "temporary_space": {},
            },
        }

        try:
            # Memory validation
            memory_check = self._validate_memory()
            results["resource_check"]["memory"] = memory_check

            if not memory_check["sufficient"]:
                results["valid"] = False
                if memory_check["critical"]:
                    results["errors"].append(
                        f"Insufficient memory: {memory_check['available_mb']} MB available, {memory_check['required_mb']} MB required"
                    )
                else:
                    results["warnings"].append(
                        f"Low memory: {memory_check['available_mb']} MB available, {memory_check['required_mb']} MB recommended"
                    )

            # Disk space validation
            disk_check = self._validate_disk_space(project_path)
            results["resource_check"]["disk_space"] = disk_check

            if not disk_check["sufficient"]:
                results["valid"] = False
                results["errors"].append(
                    f"Insufficient disk space: {disk_check['free_space_mb']} MB free, {disk_check['required_mb']} MB required"
                )

            # Temporary space validation
            temp_check = self._validate_temporary_space()
            results["resource_check"]["temporary_space"] = temp_check

            if not temp_check["sufficient"]:
                results["valid"] = False
                results["errors"].append(
                    f"Insufficient temporary space: {temp_check['free_space_mb']} MB free, {temp_check['required_mb']} MB required"
                )

            # CPU validation
            cpu_check = self._validate_cpu()
            results["resource_check"]["cpu"] = cpu_check

            if not cpu_check["sufficient"]:
                warning_msg = f"Limited CPU cores: {cpu_check['cores']} cores available, {cpu_check['recommended_cores']} recommended"
                if self.strict_mode:
                    results["valid"] = False
                    results["errors"].append(warning_msg)
                else:
                    results["warnings"].append(warning_msg)

            # Project-specific validation
            project_check = self._validate_project_specific(
                project_path, estimated_files
            )
            results["resource_check"]["project"] = project_check

            if not project_check["sufficient"]:
                results["valid"] = False
                results["errors"].append(
                    f"Project validation failed: {project_check['reason']}"
                )

            # Generate recommendations
            recommendations = self._generate_recommendations(results)
            results["recommendations"] = recommendations

            # Summary
            if results["valid"] and not results["warnings"]:
                logger.info("System resource validation passed successfully")
            elif results["valid"] and results["warnings"]:
                logger.warning(
                    f"System resource validation passed with {len(results['warnings'])} warnings"
                )
            else:
                logger.error(
                    f"System resource validation failed with {len(results['errors'])} errors"
                )

            self.validation_results = results

            # Cache successful validation results
            if results.get("valid") and not results.get("errors"):
                _save_validation_cache(project_path, results)

            return results

        except Exception as e:
            logger.error(f"Resource validation failed: {e}")
            results["valid"] = False
            results["errors"].append(f"Validation process error: {e}")
            raise log_and_reraise(logger, e, "resource_validation")

    def _validate_memory(self) -> Dict[str, Any]:
        """Validate system memory availability."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            total_mb = memory.total / (1024 * 1024)

            # Determine required memory based on project size and system capabilities
            required_mb = max(
                self.MIN_MEMORY_MB,
                min(
                    total_mb * 0.25, available_mb
                ),  # Use 25% of total or available, whichever is less
            )

            # Check if available memory meets requirements
            sufficient = available_mb >= required_mb
            critical = available_mb < self.MIN_MEMORY_MB

            check_result = {
                "sufficient": sufficient,
                "critical": critical,
                "available_mb": round(available_mb, 2),
                "total_mb": round(total_mb, 2),
                "required_mb": round(required_mb, 2),
                "usage_percent": memory.percent,
                "sufficient_for_streaming": available_mb
                >= 256,  # Minimum for streaming analysis
            }

            if critical:
                raise MemoryLimitError(
                    available_mb,
                    required_mb,
                    details={"total_memory": total_mb, "usage_percent": memory.percent},
                )

            return check_result

        except ImportError:
            logger.warning("psutil not available, using fallback memory detection")
            return self._validate_memory_fallback()

    def _validate_memory_fallback(self) -> Dict[str, Any]:
        """Fallback memory validation without psutil."""
        try:
            # Windows-specific memory detection
            if sys.platform == "win32":
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("dwTotalPhys", ctypes.c_ulonglong),
                        ("dwAvailPhys", ctypes.c_ulonglong),
                        ("dwTotalPageFile", ctypes.c_ulonglong),
                        ("dwAvailPageFile", ctypes.c_ulonglong),
                        ("dwTotalVirtual", ctypes.c_ulonglong),
                        ("dwAvailVirtual", ctypes.c_ulonglong),
                        ("sAvailVirtual", ctypes.c_ulonglong),
                        ("dwReserved", ctypes.c_ulong * 10),
                    ]

                memory_status = MEMORYSTATUSEX()
                memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))

                available_mb = memory_status.dwAvailPhys / (1024 * 1024)
                total_mb = memory_status.dwTotalPhys / (1024 * 1024)

                required_mb = max(
                    self.MIN_MEMORY_MB, min(total_mb * 0.25, available_mb)
                )
                sufficient = available_mb >= required_mb
                critical = available_mb < self.MIN_MEMORY_MB

                check_result = {
                    "sufficient": sufficient,
                    "critical": critical,
                    "available_mb": round(available_mb, 2),
                    "total_mb": round(total_mb, 2),
                    "required_mb": round(required_mb, 2),
                    "usage_percent": memory_status.dwMemoryLoad,
                    "sufficient_for_streaming": available_mb >= 256,
                }

                if critical:
                    raise MemoryLimitError(
                        available_mb, required_mb, details={"total_memory": total_mb}
                    )

                return check_result
            else:
                # Non-Windows fallback - very basic estimation
                logger.warning(
                    "Using very basic memory estimation on non-Windows system"
                )
                available_mb = 1024  # Conservative estimate
                total_mb = 2048
                required_mb = 512
                sufficient = True

                return {
                    "sufficient": sufficient,
                    "critical": False,
                    "available_mb": available_mb,
                    "total_mb": total_mb,
                    "required_mb": required_mb,
                    "usage_percent": 50,
                    "sufficient_for_streaming": True,
                    "fallback": True,
                }

        except Exception as e:
            logger.error(f"Fallback memory validation failed: {e}")
            # Return conservative estimates
            return {
                "sufficient": True,
                "critical": False,
                "available_mb": 512,
                "total_mb": 1024,
                "required_mb": 512,
                "usage_percent": 50,
                "sufficient_for_streaming": True,
                "error": str(e),
            }

    def _validate_disk_space(self, project_path: str) -> Dict[str, Any]:
        """Validate disk space for project analysis."""
        try:
            # Get free space for project directory
            project_dir = Path(project_path)
            if not project_dir.exists():
                project_dir = project_dir.parent

            total, used, free = shutil.disk_usage(project_dir)
            free_mb = free / (1024 * 1024)

            # Estimate required space (files + temp space + cache)
            estimated_required = max(
                self.MIN_DISK_SPACE_MB, self._estimate_required_disk_space(project_path)
            )

            sufficient = free_mb >= estimated_required
            check_result = {
                "sufficient": sufficient,
                "free_space_mb": round(free_mb, 2),
                "total_space_mb": round(total / (1024 * 1024), 2),
                "required_mb": estimated_required,
                "path": str(project_dir),
            }

            if not sufficient:
                raise DiskSpaceError(free_mb, estimated_required, str(project_path))

            return check_result

        except Exception as e:
            logger.error(f"Disk space validation failed: {e}")
            return {
                "sufficient": False,
                "free_space_mb": 100,
                "total_space_mb": 1000,
                "required_mb": 100,
                "path": project_path,
                "error": str(e),
            }

    def _validate_temporary_space(self) -> Dict[str, Any]:
        """Validate temporary disk space availability."""
        try:
            # Check system temp directory
            temp_dir = Path(tempfile.gettempdir())
            total, used, free = shutil.disk_usage(temp_dir)
            free_mb = free / (1024 * 1024)

            required_temp_mb = max(self.MIN_FREE_SPACE_MB, 100)  # 100MB minimum

            sufficient = free_mb >= required_temp_mb

            check_result = {
                "sufficient": sufficient,
                "free_space_mb": round(free_mb, 2),
                "required_mb": required_temp_mb,
                "temp_path": str(temp_dir),
            }

            if not sufficient:
                raise DiskSpaceError(free_mb, required_temp_mb, str(temp_dir))

            return check_result

        except Exception as e:
            logger.error(f"Temporary space validation failed: {e}")
            return {
                "sufficient": True,  # Don't fail due to temp dir check
                "free_space_mb": 100,
                "required_mb": 100,
                "temp_path": tempfile.gettempdir(),
                "error": str(e),
            }

    def _validate_cpu(self) -> Dict[str, Any]:
        """Validate CPU availability."""
        try:
            import psutil

            cores = psutil.cpu_count(logical=False)  # Physical cores
            logical_cores = psutil.cpu_count(logical=True)  # Logical processors

            recommended_cores = 2  # Minimum recommended for efficient analysis

            sufficient = cores >= 1 and logical_cores >= 2

            # Consider CPU usage
            current_usage = psutil.cpu_percent(interval=1)
            high_usage = current_usage > 80

            check_result = {
                "sufficient": sufficient and not high_usage,
                "cores": cores,
                "logical_cores": logical_cores,
                "recommended_cores": recommended_cores,
                "current_usage_percent": current_usage,
                "high_usage": high_usage,
            }

            return check_result

        except ImportError:
            # Fallback without psutil
            cores = max(1, os.cpu_count() or 1)
            sufficient = cores >= 1

            return {
                "sufficient": sufficient,
                "cores": cores,
                "logical_cores": cores,
                "recommended_cores": 2,
                "current_usage_percent": 50,
                "high_usage": False,
                "fallback": True,
            }

    def _validate_project_specific(
        self, project_path: str, estimated_files: int
    ) -> Dict[str, Any]:
        """Validate project-specific constraints."""
        try:
            project_dir = Path(project_path)

            if not project_dir.exists():
                return {
                    "sufficient": False,
                    "reason": "Project directory does not exist",
                }

            if not project_dir.is_dir():
                return {
                    "sufficient": False,
                    "reason": "Project path is not a directory",
                }

            # Check if directory is readable
            try:
                list(project_dir.iterdir())
            except PermissionError:
                return {
                    "sufficient": False,
                    "reason": "Project directory is not readable",
                }

            # Validate file count estimation
            if estimated_files > 10000:
                logger.warning(f"Large number of files estimated: {estimated_files}")

            # Check for excessive nesting
            try:
                max_depth = self._calculate_directory_depth(project_dir)
                if max_depth > 20:
                    logger.warning(
                        f"Deep directory nesting detected: {max_depth} levels"
                    )
            except Exception as e:
                logger.warning(f"Could not calculate directory depth: {e}")

            return {"sufficient": True}

        except Exception as e:
            return {"sufficient": False, "reason": f"Project validation error: {e}"}

    def _estimate_required_disk_space(self, project_path: str) -> int:
        """Estimate required disk space for analysis."""
        try:
            project_dir = Path(project_path)
            if not project_dir.exists():
                return 200  # Conservative estimate

            # Calculate total size of project files
            total_size_mb = 0
            file_count = 0

            for file_path in project_dir.rglob("*"):
                if file_path.is_file():
                    try:
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        total_size_mb += size_mb
                        file_count += 1
                    except (OSError, PermissionError):
                        continue

            # Add overhead for analysis results (estimated 20% of source size)
            analysis_overhead = total_size_mb * 0.2

            # Add cache space (estimated 50MB base + 1MB per 100 files)
            cache_space = 50 + (file_count / 100)

            total_required = total_size_mb + analysis_overhead + cache_space

            return max(100, int(total_required))  # Minimum 100MB

        except Exception as e:
            logger.warning(f"Could not estimate disk space: {e}")
            return 200  # Conservative fallback

    def _calculate_directory_depth(self, path: Path, max_depth: int = 50) -> int:
        """Calculate maximum directory depth."""
        max_depth_found = 0

        for root, dirs, files in os.walk(path):
            try:
                relative_depth = len(Path(root).relative_to(path).parts)
                max_depth_found = max(max_depth_found, relative_depth)
            except ValueError:
                continue

        return max_depth_found

    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> list:
        """Generate recommendations based on validation results."""
        recommendations = []

        memory_check = validation_results["resource_check"].get("memory", {})
        disk_check = validation_results["resource_check"].get("disk_space", {})
        cpu_check = validation_results["resource_check"].get("cpu", {})

        # Memory recommendations
        if memory_check.get("available_mb", 0) < 1024:
            recommendations.append(
                "Consider closing other applications to free up memory"
            )

        if not memory_check.get("sufficient_for_streaming", True):
            recommendations.append(
                "Enable streaming analysis mode for better memory usage"
            )

        # Disk space recommendations
        if disk_check.get("free_space_mb", 0) < 500:
            recommendations.append(
                "Free up disk space or analyze a smaller project subset"
            )

        # CPU recommendations
        if cpu_check.get("cores", 1) < 4:
            recommendations.append("Analysis may be slower due to limited CPU cores")

        # General recommendations
        if validation_results["warnings"]:
            recommendations.append(
                "Review warnings before proceeding with large projects"
            )

        return recommendations

    def get_optimization_suggestions(self) -> Dict[str, Any]:
        """Get optimization suggestions based on current validation results."""
        if not self.validation_results:
            return {
                "error": "No validation results available. Run validate_system_resources() first."
            }

        suggestions = {
            "memory_optimization": [],
            "performance_optimization": [],
            "storage_optimization": [],
        }

        memory_check = self.validation_results.get("resource_check", {}).get(
            "memory", {}
        )

        # Memory-based suggestions
        available_mb = memory_check.get("available_mb", 0)
        if available_mb < 1024:
            suggestions["memory_optimization"].extend(
                [
                    "Enable streaming analysis for large files",
                    "Reduce batch size for embedding generation",
                    "Use smaller model configurations",
                ]
            )
        elif available_mb < 2048:
            suggestions["memory_optimization"].extend(
                [
                    "Monitor memory usage during analysis",
                    "Consider processing files in smaller batches",
                ]
            )

        # Performance suggestions
        cpu_check = self.validation_results.get("resource_check", {}).get("cpu", {})
        if cpu_check.get("cores", 1) < 4:
            suggestions["performance_optimization"].extend(
                [
                    "Analysis will run single-threaded for some operations",
                    "Consider using a more powerful machine for large projects",
                ]
            )

        # Storage suggestions
        disk_check = self.validation_results.get("resource_check", {}).get(
            "disk_space", {}
        )
        if disk_check.get("free_space_mb", 0) < 1000:
            suggestions["storage_optimization"].extend(
                [
                    "Clear temporary files after analysis",
                    "Use external storage for large embedding caches",
                ]
            )

        return suggestions


# Convenience functions
def quick_resource_check(project_path: str) -> bool:
    """Quick check if basic resources are available for analysis."""
    try:
        validator = ResourceValidator(strict_mode=False)
        results = validator.validate_system_resources(project_path)
        return results["valid"]
    except Exception:
        return False  # If validation fails, be conservative


def validate_and_get_optimal_settings(project_path: str) -> Dict[str, Any]:
    """Validate resources and return optimal analysis settings."""
    validator = ResourceValidator(strict_mode=False)
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
