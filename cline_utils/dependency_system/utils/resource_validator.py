"""
System resource validation utilities for project analyzer.
Validates available memory, disk space, VRAM, and other resources before analysis.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

from ..core.exceptions_enhanced import DiskSpaceError, MemoryLimitError, log_and_reraise
from ..utils.path_utils import normalize_path

# Try to import torch for VRAM management
try:
    import torch as _torch

    torch = _torch
    _torch_available: bool = True
except ImportError:
    _torch_available = False
    torch = None

TORCH_AVAILABLE = _torch_available

logger = logging.getLogger(__name__)

# Cache configuration
VALIDATION_CACHE_FILE = "validation_cache.json"
CACHE_VERSION = "1.1"  # Bumped: now includes GPU/VRAM metrics
DEFAULT_CACHE_TTL_SECONDS = 604800  # 7 days (hardware resources rarely change)


def get_cache_path() -> str:
    """Get path to validation cache file."""
    from .. import core

    core_dir = os.path.dirname(os.path.abspath(core.__file__))
    return core.resolve_state_path(VALIDATION_CACHE_FILE, core_dir)


def _load_validation_cache() -> Optional[Dict[str, Any]]:
    """Load cached validation results if available."""
    try:
        cache_path = get_cache_path()
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
        cache_path = get_cache_path()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        cache_data = {
            "version": CACHE_VERSION,
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
        if cache_data.get("version") != CACHE_VERSION:
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

        # Check for invalid/zero metrics that suggest a failed validation
        resource_check = results.get("resource_check", {})

        # Memory should never be zero
        mem_check = resource_check.get("memory", {})
        if mem_check.get("available_mb", 0) <= 0:
            logger.debug("Cache invalid: memory metrics are zero/missing")
            return False

        # GPU: if available, VRAM should be > 0
        gpu_check = resource_check.get("gpu", {})
        if gpu_check.get("gpu_available") and gpu_check.get("vram_total_mb", 0) <= 0:
            logger.debug("Cache invalid: GPU marked available but VRAM is zero")
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

    def __init__(self, strict_mode: bool = False, skip_disk_estimation: bool = False):
        """
        Initialize resource validator.

        Args:
            strict_mode: If True, fail on warnings. If False, only fail on critical issues.
            skip_disk_estimation: If True, skip disk estimation and space validation.
        """
        super().__init__()
        self.strict_mode = strict_mode
        self.skip_disk_estimation = skip_disk_estimation
        self.validation_results: Dict[str, Any] = {}

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

        # Use separate typed containers to avoid mixed-type dict inference issues
        warnings_list: list[str] = []
        errors_list: list[str] = []
        resource_check: Dict[str, Dict[str, Any]] = {
            "memory": {},
            "disk_space": {},
            "cpu": {},
            "gpu": {},
            "temporary_space": {},
        }
        is_valid = True

        try:
            # Memory validation
            memory_check = self._validate_memory()
            resource_check["memory"] = memory_check

            if not memory_check["sufficient"]:
                is_valid = False
                if memory_check["critical"]:
                    errors_list.append(
                        f"Insufficient memory: {memory_check['available_mb']} MB available, {memory_check['required_mb']} MB required"
                    )
                else:
                    warnings_list.append(
                        f"Low memory: {memory_check['available_mb']} MB available, {memory_check['required_mb']} MB recommended"
                    )

            # Disk space validation
            disk_check = self._validate_disk_space(project_path)
            resource_check["disk_space"] = disk_check

            if not disk_check["sufficient"]:
                is_valid = False
                errors_list.append(
                    f"Insufficient disk space: {disk_check['free_space_mb']} MB free, {disk_check['required_mb']} MB required"
                )

            # Temporary space validation
            temp_check = self._validate_temporary_space()
            resource_check["temporary_space"] = temp_check

            if not temp_check["sufficient"]:
                is_valid = False
                errors_list.append(
                    f"Insufficient temporary space: {temp_check['free_space_mb']} MB free, {temp_check['required_mb']} MB required"
                )

            # CPU validation
            cpu_check = self._validate_cpu()
            resource_check["cpu"] = cpu_check

            if not cpu_check["sufficient"]:
                warning_msg = f"Limited CPU cores: {cpu_check['cores']} cores available, {cpu_check['recommended_cores']} recommended"
                if self.strict_mode:
                    is_valid = False
                    errors_list.append(warning_msg)
                else:
                    warnings_list.append(warning_msg)

            # GPU/VRAM validation
            gpu_check = self.validate_gpu()
            resource_check["gpu"] = gpu_check

            # Project-specific validation
            project_check = self._validate_project_specific(
                project_path, estimated_files
            )
            resource_check["project"] = project_check

            if not project_check["sufficient"]:
                is_valid = False
                errors_list.append(
                    f"Project validation failed: {project_check['reason']}"
                )

            # Assemble final results dict
            results: Dict[str, Any] = {
                "valid": is_valid,
                "warnings": warnings_list,
                "errors": errors_list,
                "resource_check": resource_check,
            }

            # Generate recommendations
            recommendations: list[str] = self._generate_recommendations(results)
            results["recommendations"] = recommendations

            # Summary
            if is_valid and not warnings_list:
                logger.info("System resource validation passed successfully")
            elif is_valid and warnings_list:
                logger.warning(
                    f"System resource validation passed with {len(warnings_list)} warnings"
                )
            else:
                logger.error(
                    f"System resource validation failed with {len(errors_list)} errors"
                )

            self.validation_results = results

            # Cache successful validation results
            if is_valid and not errors_list:
                _save_validation_cache(project_path, results)

            return results

        except Exception as e:
            logger.error(f"Resource validation failed: {e}")
            results_err: Dict[str, Any] = {
                "valid": False,
                "warnings": warnings_list,
                "errors": errors_list + [f"Validation process error: {e}"],
                "resource_check": resource_check,
                "recommendations": [],
            }
            self.validation_results = results_err
            exc = log_and_reraise(logger, e, "resource_validation", reraise=False)
            if exc:
                raise exc
            raise e  # Fallback if exc is somehow None

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
            if self.skip_disk_estimation:
                logger.info("Disk space estimation and validation skipped via configuration")
                return {
                    "sufficient": True,
                    "free_space_mb": 999999.0,
                    "total_space_mb": 999999.0,
                    "required_mb": 0,
                    "path": project_path,
                    "skipped": True,
                }

            # Get free space for project directory
            project_dir = Path(project_path)
            if not project_dir.exists():
                project_dir = project_dir.parent

            total, _used, free = shutil.disk_usage(project_dir)
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
            _total, _used, free = shutil.disk_usage(temp_dir)
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

            cores: int = psutil.cpu_count(logical=False) or 1  # Physical cores
            logical_cores: int = (
                psutil.cpu_count(logical=True) or 1
            )  # Logical processors

            recommended_cores = 2  # Minimum recommended for efficient analysis

            sufficient: bool = cores >= 1 and logical_cores >= 2

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

    def validate_gpu(self) -> Dict[str, Any]:
        """Validate GPU/VRAM availability and collect hardware details."""
        check_result: Dict[str, Any] = {
            "gpu_available": False,
            "gpu_name": None,
            "gpu_count": 0,
            "vram_total_mb": 0.0,
            "vram_available_mb": 0.0,
            "vram_used_mb": 0.0,
            "driver_version": None,
            "cuda_version": None,
        }
        if not TORCH_AVAILABLE or torch is None:
            logger.debug("GPU validation skipped: torch not available")
            return check_result

        try:
            if not torch.cuda.is_available():
                logger.debug("GPU validation: CUDA not available")
                return check_result

            gpu_count = torch.cuda.device_count()
            check_result["gpu_available"] = gpu_count > 0
            check_result["gpu_count"] = gpu_count

            if gpu_count > 0:
                props = cast(Any, torch.cuda).get_device_properties(0)
                check_result["gpu_name"] = str(getattr(props, "name", "Unknown GPU"))
                total_bytes = float(getattr(props, "total_memory", 0))
                check_result["vram_total_mb"] = round(total_bytes / (1024 * 1024), 2)

                try:
                    torch.cuda.synchronize()
                    free_bytes_raw, _ = torch.cuda.mem_get_info(0)
                    free_bytes = float(free_bytes_raw)
                    check_result["vram_available_mb"] = round(
                        free_bytes / (1024 * 1024), 2
                    )
                    check_result["vram_used_mb"] = round(
                        (total_bytes - free_bytes) / (1024 * 1024), 2
                    )
                except Exception as e_mem:
                    logger.debug(f"Could not query live VRAM usage: {e_mem}")

                # Driver / CUDA version (best-effort)
                try:
                    from torch.version import cuda as cuda_version

                    cuda_ver = cuda_version
                    check_result["cuda_version"] = cuda_ver
                except Exception:
                    pass

            logger.debug(
                f"GPU validation: {check_result['gpu_name']} "
                f"({check_result['vram_total_mb']:.0f} MB total, "
                f"{check_result['vram_available_mb']:.0f} MB free)"
            )
        except Exception as e:
            logger.warning(f"GPU validation failed: {e}")
            check_result["gpu_available"] = False
        return check_result

    def wait_for_vram_release(
        self,
        target_free_mb: float,
        poll_interval: float = 0.5,
        stall_tolerance: int = 3,
        hard_cap_seconds: float = 120.0,
        tolerance_mb: float = 50.0,
    ) -> bool:
        """
        Polls VRAM usage until at least `target_free_mb` is available or convergence occurs.

        Args:
            target_free_mb: Target available VRAM in MB.
            poll_interval: Time between polls in seconds.
            stall_tolerance: Number of consecutive polls without growth to consider converged.
            hard_cap_seconds: Absolute maximum time to wait in seconds.

        Returns:
            True if target VRAM (within tolerance) is available, False if timed out or stalled.
        """
        if not TORCH_AVAILABLE or torch is None or not torch.cuda.is_available():
            return False

        torch.cuda.empty_cache()  # Flush allocator cache to driver immediately

        def _query_free_mb() -> Optional[float]:
            if torch is None or not hasattr(torch, "cuda"):
                return None
            try:
                torch.cuda.synchronize()
                free_bytes, _ = torch.cuda.mem_get_info(0)
                return float(free_bytes) / (1024 * 1024)
            except Exception as e:
                logger.debug(f"VRAM query failed: {e}")
                return None

        start_time = time.time()
        prev_free_mb = _query_free_mb() or 0.0
        stall_count = 0

        while True:
            elapsed = time.time() - start_time

            if elapsed >= hard_cap_seconds:
                logger.warning(
                    f"VRAM wait hit hard cap ({hard_cap_seconds}s). "
                    f"Last free: {prev_free_mb:.1f} MB, "
                    f"target: {target_free_mb:.1f} MB."
                )
                return False

            time.sleep(poll_interval)
            free_mb = _query_free_mb()

            if free_mb is None:
                stall_count += 1
            else:
                # EXACT VERIFICATION — target is a known baseline, not an estimate
                if free_mb >= (target_free_mb - tolerance_mb):
                    if free_mb < target_free_mb:
                        logger.debug(
                            f"VRAM within tolerance: {free_mb:.1f} MB free (target {target_free_mb:.1f} MB, "
                            f"delta {target_free_mb - free_mb:.1f} MB). Proceeding."
                        )
                    else:
                        logger.debug(
                            f"VRAM verified: {free_mb:.1f} MB free >= "
                            f"target {target_free_mb:.1f} MB. "
                            f"Elapsed: {elapsed:.2f}s."
                        )
                    return True

                growth = free_mb - prev_free_mb
                stall_count = 0 if growth > 0 else stall_count + 1
                prev_free_mb = free_mb

            if stall_count >= stall_tolerance:
                delta = target_free_mb - prev_free_mb
                if delta <= tolerance_mb:
                    logger.info(
                        f"VRAM converged near baseline at {prev_free_mb:.1f} MB "
                        f"(target: {target_free_mb:.1f} MB, delta: {delta:.1f} MB). "
                        f"Close enough to proceed."
                    )
                    return True

                logger.warning(
                    f"VRAM converged at {prev_free_mb:.1f} MB "
                    f"(target: {target_free_mb:.1f} MB, "
                    f"delta: {delta:.1f} MB not reclaimed). "
                    f"Elapsed: {elapsed:.2f}s."
                )
                return False

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
        """
        Estimate required disk space for analysis using an optimized os.scandir traversal.
        Respects excluded directories and limits scanning overhead.
        """
        try:
            project_dir = Path(project_path)
            if not project_dir.exists():
                return 200  # Conservative estimate

            try:
                from .config_manager import ConfigManager
                config_mgr = ConfigManager()
                excluded_dirs = set(config_mgr.get_excluded_dirs())
            except Exception:
                # Fallback if config manager is not fully initialized
                excluded_dirs = {
                    "__pycache__", ".git", ".svn", ".hg", ".vscode", ".idea", 
                    "venv", "env", ".venv", "node_modules", "build", "dist", 
                    "target", "out", "tmp", "temp", "tests", "__tests__", "embeddings"
                }

            total_size_bytes = 0
            file_count = 0
            max_scan_files = 10000  # Cap the scan at 10,000 files to prevent endless scan
            max_depth = 8  # Limit recursive depth for estimation purposes

            def _scan_dir(dir_path: str, current_depth: int) -> None:
                nonlocal total_size_bytes, file_count
                if file_count >= max_scan_files or current_depth > max_depth:
                    return

                try:
                    with os.scandir(dir_path) as entries:
                        for entry in entries:
                            if file_count >= max_scan_files:
                                break

                            if entry.is_symlink():
                                continue

                            if entry.is_dir(follow_symlinks=False):
                                if entry.name in excluded_dirs:
                                    continue
                                _scan_dir(entry.path, current_depth + 1)
                            elif entry.is_file(follow_symlinks=False):
                                try:
                                    # entry.stat() is cached on Windows during scandir
                                    stat_val = entry.stat()
                                    total_size_bytes += stat_val.st_size
                                    file_count += 1
                                except (OSError, PermissionError):
                                    continue
                except (OSError, PermissionError):
                    pass

            _scan_dir(str(project_dir), 1)

            total_size_mb = total_size_bytes / (1024 * 1024)

            # If we hit the file limit, extrapolate the size
            if file_count >= max_scan_files:
                logger.warning(
                    f"Disk estimation capped at {max_scan_files} files. Extrapolating disk requirement."
                )
                total_size_mb = total_size_mb * 1.5

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

        for root, _, _ in os.walk(path):
            try:
                relative_depth = len(Path(root).relative_to(path).parts)
                max_depth_found = max(max_depth_found, relative_depth)
            except ValueError:
                continue

        return max_depth_found

    def _generate_recommendations(
        self, validation_results: Dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations: list[str] = []

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

        suggestions: Dict[str, List[str]] = {
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


# =============================================================================
# VRAM RESOURCE MANAGEMENT
# =============================================================================


class CrossProcessLock:
    """
    A cross-process file-based lock to synchronize access to VRAM checks/allocations.
    Uses platform-specific locking: msvcrt on Windows, fcntl on POSIX.
    """

    def __init__(self, lockfile_path: str):
        self.lockfile_path = lockfile_path
        self._fd: Optional[int] = None

    def acquire(self, timeout: float = 60.0, poll_interval: float = 0.05) -> bool:
        """Acquire the lock, blocking until timeout."""
        start_time = time.time()
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.lockfile_path), exist_ok=True)

        while True:
            try:
                # Open with read-write and create
                self._fd = os.open(
                    self.lockfile_path, os.O_RDWR | os.O_CREAT
                )

                if sys.platform == "win32":
                    import msvcrt
                    # Ensure the file contains at least 1 byte so locking 1 byte is safe
                    if os.lseek(self._fd, 0, os.SEEK_END) == 0:
                        os.write(self._fd, b"\0")
                    os.lseek(self._fd, 0, os.SEEK_SET)

                    try:
                        # Non-blocking lock
                        msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                        return True
                    except (OSError, IOError):
                        # Lock not available, close file descriptor
                        os.close(self._fd)
                        self._fd = None
                else:
                    import fcntl
                    try:
                        # Exclusive lock, non-blocking
                        fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return True
                    except (OSError, IOError):
                        os.close(self._fd)
                        self._fd = None

            except Exception as e:
                # Handle unexpected errors gracefully
                if self._fd is not None:
                    try:
                        os.close(self._fd)
                    except Exception:
                        pass
                    self._fd = None

            if time.time() - start_time >= timeout:
                return False

            time.sleep(poll_interval)

    def release(self) -> None:
        """Release the lock."""
        if self._fd is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    try:
                        os.lseek(self._fd, 0, os.SEEK_SET)
                        msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                    except Exception:
                        pass
                else:
                    import fcntl
                    try:
                        fcntl.flock(self._fd, fcntl.LOCK_UN)
                    except Exception:
                        pass
            finally:
                try:
                    os.close(self._fd)
                except Exception:
                    pass
                self._fd = None

    def __enter__(self) -> "CrossProcessLock":
        if not self.acquire():
            raise RuntimeError(
                f"Could not acquire cross-process lock on {self.lockfile_path}"
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


def is_pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h_process = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if h_process:
                kernel32.CloseHandle(h_process)
                return True
            return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False



class AllocationStatus(Enum):
    """Status of a VRAM allocation request."""

    GRANTED = "granted"
    DENIED = "denied"
    PENDING = "pending"
    RELEASED = "released"


@dataclass
class VRAMAllocation:
    """Represents a single VRAM allocation."""

    allocation_id: str
    size_gb: float
    requested_at: float
    granted_at: Optional[float] = None
    released_at: Optional[float] = None
    status: AllocationStatus = field(default=AllocationStatus.PENDING)
    worker_id: Optional[str] = None
    batch_id: Optional[str] = None


class VRAMResourceManager:
    """
    Thread-safe singleton for managing VRAM allocations across all workers.

    Solves the race condition problem by providing atomic allocation requests
    instead of the problematic check-then-allocate pattern. Integrates with
    ResourceValidator to provide comprehensive resource management.

    Key Features:
    - Singleton pattern ensures global coordination across all workers
    - Atomic allocation requests prevent race conditions
    - Blocking mode with timeout for backpressure
    - Statistics tracking for monitoring and tuning
    - Integration with ResourceValidator for unified resource checks
    """

    _instance: Optional["VRAMResourceManager"] = None
    _instance_lock: threading.Lock = threading.Lock()

    # Default configuration
    DEFAULT_RESERVATION_PERCENT = 0.10  # 10% system reservation
    DEFAULT_SAFETY_BUFFER_GB = 0.5  # Minimum buffer per allocation
    MIN_VRAM_FOR_OPERATION_GB = 0.1  # Absolute minimum to proceed

    # Model footprint estimates (GB)
    MODEL_FOOTPRINTS = {
        "qwen3_reranker_0.6b": 0.7,
        "qwen3_embedding_4b": 3.5,
        "mpnet_base": 0.5,
    }

    def __new__(cls, *args: Any, **kwargs: Any) -> "VRAMResourceManager":
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        reservation_percent: float = DEFAULT_RESERVATION_PERCENT,
        safety_buffer_gb: float = DEFAULT_SAFETY_BUFFER_GB,
    ):
        """
        Initialize the VRAM resource manager.

        Args:
            reservation_percent: Percentage of total VRAM to reserve for system/OS
            safety_buffer_gb: Minimum buffer to maintain per allocation
        """
        super().__init__()
        # Avoid re-initialization
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Core synchronization primitives
        self._allocation_lock = threading.RLock()
        self._condition = threading.Condition(self._allocation_lock)

        # Configuration
        self._reservation_percent = reservation_percent
        self._safety_buffer_gb = safety_buffer_gb

        # State tracking
        self._active_allocations: Dict[str, VRAMAllocation] = {}
        self._allocation_counter = 0
        self._total_allocated_gb = 0.0
        self._peak_allocated_gb = 0.0

        # Statistics
        self._stats = {
            "total_requests": 0,
            "granted": 0,
            "denied": 0,
            "deferred": 0,
            "released": 0,
            "peak_concurrent_gb": 0.0,
        }

        # Batch scheduling state
        self._batch_queue: List[Dict[str, Any]] = []
        self._batch_counter = 0

        # Cross-process lock and registry paths
        temp_dir = tempfile.gettempdir()
        self._lockfile_path = os.path.join(temp_dir, "vram_lock.lock")
        self._registry_path = os.path.join(temp_dir, "vram_registry.json")

        self._initialized = True
        logger.debug(
            f"VRAMResourceManager initialized (reservation={reservation_percent:.0%}, buffer={safety_buffer_gb}GB)"
        )

    def _load_registry(self) -> Dict[str, Any]:
        """Load the shared registry of active allocations."""
        if not os.path.exists(self._registry_path):
            return {}
        try:
            with open(self._registry_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return cast(Dict[str, Any], json.loads(content))
        except Exception as e:
            logger.warning(f"Failed to load VRAM registry: {e}")
            return {}

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Save the shared registry of active allocations."""
        try:
            # Atomic write via temporary file in the same directory
            temp_fd, temp_path = tempfile.mkstemp(
                dir=os.path.dirname(self._registry_path), prefix="vram_reg_tmp_"
            )
            try:
                with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                    json.dump(registry, f, indent=2)
                os.replace(temp_path, self._registry_path)
            except Exception as e:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e
        except Exception as e:
            logger.error(f"Failed to save VRAM registry: {e}")

    def _prune_dead_allocations(self, registry: Dict[str, Any]) -> None:
        """Prune allocations of processes that are no longer running."""
        dead_ids = []
        for alloc_id, alloc in registry.items():
            pid = alloc.get("pid")
            if pid is not None and not is_pid_alive(pid):
                dead_ids.append(alloc_id)

        for alloc_id in dead_ids:
            logger.info(
                f"Pruning stale VRAM allocation {alloc_id} from dead process {registry[alloc_id].get('pid')}"
            )
            del registry[alloc_id]


    def _get_physical_vram_gb(self) -> float:
        """Get total physical VRAM in GB."""
        if not TORCH_AVAILABLE or torch is None or not torch.cuda.is_available():
            return 0.0
        try:
            _, total_memory = torch.cuda.mem_get_info(0)
            return total_memory / (1024**3)
        except Exception as e:
            logger.warning(f"Failed to get physical VRAM: {e}")
            return 0.0

    def _get_available_vram_gb(self) -> float:
        """Get actually free VRAM in GB (not accounting for reservations)."""
        if not TORCH_AVAILABLE or torch is None or not torch.cuda.is_available():
            return 0.0
        try:
            torch.cuda.synchronize()
            free_memory, _ = torch.cuda.mem_get_info(0)
            return free_memory / (1024**3)
        except Exception as e:
            logger.warning(f"Failed to get available VRAM: {e}")
            return 0.0

    def get_reserved_vram_gb(self) -> float:
        """
        Calculate system-wide reserved VRAM amount.
        This is the amount we always keep free for system stability.
        """
        physical = self._get_physical_vram_gb()
        if physical == 0:
            return 0.0
        return max(
            physical * self._reservation_percent,
            self.MODEL_FOOTPRINTS.get("qwen3_reranker_0.6b", 0.7)
            + self._safety_buffer_gb,
        )

    def _get_available_for_allocation_locked(self, registry: Dict[str, Any]) -> float:
        """
        Calculate available VRAM using an already-loaded registry.
        Must be called under _allocation_lock and while holding the CrossProcessLock.
        """
        free_vram = self._get_available_vram_gb()
        reserved = self.get_reserved_vram_gb()
        total_allocated = sum(
            float(alloc.get("size_gb", 0.0))
            for alloc in registry.values()
        )
        self._total_allocated_gb = total_allocated
        available = free_vram - reserved - total_allocated
        return max(0.0, available)

    def get_available_for_allocation(self) -> float:
        """
        Get VRAM available for new allocations.
        Accounts for: system reservation + pending (not-yet-materialized) allocations.

        mem_get_info() returns physically free VRAM, which already reflects
        any memory consumed by active tensors. We subtract:
        - reserved: system stability buffer
        - total allocations registered in the cross-process registry (sum of all allocations of alive processes).

        Returns:
            Available VRAM in GB for new allocations
        """
        with self._allocation_lock:
            lock = CrossProcessLock(self._lockfile_path)
            try:
                if lock.acquire(timeout=5.0):
                    try:
                        registry = self._load_registry()
                        self._prune_dead_allocations(registry)
                        self._save_registry(registry)
                        return self._get_available_for_allocation_locked(registry)
                    finally:
                        lock.release()
                else:
                    logger.warning("Could not acquire cross-process lock for VRAM query, using cached/local state")
            except Exception as e:
                logger.warning(f"Error checking cross-process VRAM availability: {e}")

            free_vram = self._get_available_vram_gb()
            reserved = self.get_reserved_vram_gb()
            available = free_vram - reserved - self._total_allocated_gb
            return max(0.0, available)

    def get_model_footprint(self, model_name: str) -> float:
        """Get the estimated VRAM footprint for a model.

        Args:
            model_name: Name of the model to get footprint for.

        Returns:
            Estimated VRAM footprint in GB.
        """
        # First, check if we can get actual measured size from the model itself
        if model_name == "qwen3_reranker_0.6b":
            try:
                # Use torch directly to measure VRAM without importing from embedding_manager
                # This avoids a circular import cycle
                if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    # Get actual memory usage - this will be the model's footprint if loaded
                    model_memory_gb: float = torch.cuda.memory_allocated() / (1024**3)
                    if model_memory_gb > 0.1:  # If we have valid measurement
                        logger.debug(
                            f"Using actual measured footprint for {model_name}: {model_memory_gb:.2f}GB"
                        )
                        return model_memory_gb
            except Exception as e:
                logger.debug(f"Failed to get actual model footprint: {e}")

        # Fallback to configured or default values
        return self.MODEL_FOOTPRINTS.get(model_name, 1.0)

    def request_allocation(
        self,
        size_gb: float,
        worker_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        blocking: bool = False,
        timeout: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Atomically request VRAM allocation.

        This is the core method that prevents race conditions. Instead of checking
        available memory and then allocating (which creates a race window), this
        method performs an atomic check-and-allocate operation.

        Args:
            size_gb: Amount of VRAM requested in GB
            worker_id: Identifier for the requesting worker
            batch_id: Identifier for the batch this allocation belongs to
            blocking: If True, wait until allocation is possible
            timeout: Maximum wait time in seconds (None = forever)

        Returns:
            Tuple of (granted: bool, allocation_id: Optional[str])
            If granted is True, allocation_id can be used to release the allocation later.
        """
        self._stats["total_requests"] += 1

        # Generate unique allocation ID with PID to ensure no cross-process collision
        allocation_id = (
            f"alloc_{os.getpid()}_{self._allocation_counter}_{int(time.time() * 1000)}"
        )
        self._allocation_counter += 1

        allocation = VRAMAllocation(
            allocation_id=allocation_id,
            size_gb=size_gb,
            requested_at=time.time(),
            worker_id=worker_id,
            batch_id=batch_id,
        )

        wait_start = time.time()
        poll_interval = 0.2  # Poll interval in seconds for cross-process wait

        lock = CrossProcessLock(self._lockfile_path)

        while True:
            # Try to acquire CrossProcessLock to verify and update the global registry
            try:
                if lock.acquire(timeout=5.0):
                    try:
                        registry = self._load_registry()
                        self._prune_dead_allocations(registry)
                        
                        # Calculate available VRAM using the loaded registry
                        available = self._get_available_for_allocation_locked(registry)
                        
                        # Check if we can allocate (adding safety buffer)
                        if available >= (size_gb + self._safety_buffer_gb):
                            # Grant allocation
                            allocation.status = AllocationStatus.GRANTED
                            allocation.granted_at = time.time()
                            
                            # Add to local allocations
                            with self._allocation_lock:
                                self._active_allocations[allocation_id] = allocation
                                self._total_allocated_gb += size_gb
                                self._stats["granted"] += 1
                                self._peak_allocated_gb = max(self._peak_allocated_gb, self._total_allocated_gb)
                            
                            # Add to cross-process registry
                            registry[allocation_id] = {
                                "allocation_id": allocation_id,
                                "size_gb": size_gb,
                                "pid": os.getpid(),
                                "requested_at": allocation.requested_at,
                                "granted_at": allocation.granted_at,
                                "worker_id": worker_id,
                                "batch_id": batch_id,
                                "status": "granted",
                            }
                            self._save_registry(registry)
                            return True, allocation_id
                    finally:
                        lock.release()
                else:
                    logger.warning("Could not acquire cross-process lock for VRAM allocation")
            except Exception as e:
                logger.error(f"Error requesting cross-process allocation: {e}")

            # If we reached here, allocation was not granted in this turn
            if not blocking:
                self._stats["denied"] += 1
                logger.debug(
                    f"VRAM allocation denied: {size_gb:.2f}GB"
                )
                return False, None

            # Blocking mode: check timeout
            elapsed = time.time() - wait_start
            if timeout is not None and elapsed >= timeout:
                self._stats["denied"] += 1
                logger.warning(
                    f"VRAM allocation timeout after {timeout}s: {allocation_id}"
                )
                return False, None

            # Wait on the thread condition variable to yield CPU and wake up on local releases
            with self._condition:
                self._stats["deferred"] += 1
                if timeout is not None:
                    remaining = timeout - elapsed
                    wait_time = min(poll_interval, remaining)
                else:
                    wait_time = poll_interval
                
                if wait_time > 0:
                    self._condition.wait(timeout=wait_time)

    def _can_allocate(self, size_gb: float) -> bool:
        """
        Check if allocation can be granted without exceeding limits.
        Must be called while holding _allocation_lock.
        """
        available = self.get_available_for_allocation()
        return available >= (size_gb + self._safety_buffer_gb)

    def _grant_allocation(self, allocation: VRAMAllocation) -> None:
        """
        Grant an allocation and update tracking.
        Must be called while holding _allocation_lock.
        """
        allocation.status = AllocationStatus.GRANTED
        allocation.granted_at = time.time()
        self._active_allocations[allocation.allocation_id] = allocation
        self._total_allocated_gb += allocation.size_gb

        self._stats["granted"] += 1
        self._peak_allocated_gb = max(self._peak_allocated_gb, self._total_allocated_gb)

    def release_allocation(self, allocation_id: str) -> bool:
        """
        Release a previously granted allocation.

        Args:
            allocation_id: The ID returned by request_allocation

        Returns:
            True if released successfully, False if not found
        """
        released_global = False
        
        lock = CrossProcessLock(self._lockfile_path)
        try:
            if lock.acquire(timeout=10.0):
                try:
                    registry = self._load_registry()
                    if allocation_id in registry:
                        del registry[allocation_id]
                        self._save_registry(registry)
                        released_global = True
                finally:
                    lock.release()
            else:
                logger.warning(f"Could not acquire cross-process lock to release allocation {allocation_id}")
        except Exception as e:
            logger.error(f"Error releasing allocation in cross-process registry: {e}")

        with self._condition:
            allocation = self._active_allocations.get(allocation_id)
            if not allocation:
                if released_global:
                    self._stats["released"] += 1
                    return True
                logger.warning(
                    f"Attempted to release unknown allocation: {allocation_id}"
                )
                return False

            if allocation.status == AllocationStatus.GRANTED:
                self._total_allocated_gb -= allocation.size_gb

            allocation.status = AllocationStatus.RELEASED
            allocation.released_at = time.time()
            del self._active_allocations[allocation_id]

            self._stats["released"] += 1

            # Notify waiting threads
            self._condition.notify_all()
            return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current allocation statistics.

        Returns:
            Dictionary with statistics including:
            - total_requests, granted, denied, deferred, released
            - current_allocated_gb, peak_allocated_gb
            - available_for_allocation_gb, physical_vram_gb, reserved_vram_gb
            - active_allocations count
        """
        global_allocated_gb = 0.0
        global_active_count = 0
        
        lock = CrossProcessLock(self._lockfile_path)
        try:
            if lock.acquire(timeout=5.0):
                try:
                    registry = self._load_registry()
                    self._prune_dead_allocations(registry)
                    self._save_registry(registry)
                    
                    global_allocated_gb = sum(
                        float(alloc.get("size_gb", 0.0))
                        for alloc in registry.values()
                    )
                    global_active_count = len(registry)
                finally:
                    lock.release()
            else:
                logger.warning("Could not acquire cross-process lock for stats, using fallback local tracking")
                global_allocated_gb = self._total_allocated_gb
                global_active_count = len(self._active_allocations)
        except Exception as e:
            logger.warning(f"Error loading registry for stats: {e}")
            global_allocated_gb = self._total_allocated_gb
            global_active_count = len(self._active_allocations)

        with self._allocation_lock:
            return {
                **self._stats,
                "current_allocated_gb": global_allocated_gb,
                "peak_allocated_gb": max(self._peak_allocated_gb, global_allocated_gb),
                "available_for_allocation_gb": self.get_available_for_allocation(),
                "physical_vram_gb": self._get_physical_vram_gb(),
                "reserved_vram_gb": self.get_reserved_vram_gb(),
                "active_allocations": global_active_count,
            }

    def get_active_allocations(self) -> List[VRAMAllocation]:
        """Get list of currently active allocations."""
        with self._allocation_lock:
            return list(self._active_allocations.values())

    def get_recommended_max_workers(
        self, model_name: str = "qwen3_reranker_0.6b"
    ) -> int:
        """
        Calculate recommended maximum workers based on available VRAM.

        Args:
            model_name: Name of the model being used (determines footprint)

        Returns:
            Recommended maximum number of workers
        """
        with self._allocation_lock:
            available = self.get_available_for_allocation()

            # Get model footprint
            model_footprint = self.get_model_footprint(model_name)

            # Calculate workers: available VRAM / model footprint
            if model_footprint <= 0:
                return 1

            max_workers = int(available / model_footprint)

            # Cap at reasonable limits (but allow more workers if VRAM allows)
            return max(1, min(max_workers, 16))

    def should_pause_for_backpressure(
        self, threshold_gb: Optional[float] = None
    ) -> bool:
        """
        Check if batch processing should pause due to VRAM pressure.

        Args:
            threshold_gb: Optional override for backpressure threshold

        Returns:
            True if processing should pause
        """
        if threshold_gb is None:
            threshold_gb = self._safety_buffer_gb

        available = self.get_available_for_allocation()
        return available < threshold_gb

    def wait_for_available_vram(
        self, required_gb: float, timeout: Optional[float] = None
    ) -> bool:
        """
        Wait until the specified amount of VRAM is available.

        Args:
            required_gb: Amount of VRAM required
            timeout: Maximum wait time in seconds

        Returns:
            True if VRAM became available, False if timeout
        """
        wait_start = time.time()
        poll_interval = 0.2
        
        while True:
            if self.get_available_for_allocation() >= required_gb:
                return True
                
            elapsed = time.time() - wait_start
            if timeout is not None and elapsed >= timeout:
                return False
                
            with self._condition:
                if timeout is not None:
                    remaining = timeout - elapsed
                    wait_time = min(poll_interval, remaining)
                else:
                    wait_time = poll_interval
                
                if wait_time > 0:
                    self._condition.wait(timeout=wait_time)


class VRAMBatchScheduler:
    """
    Scheduler for coordinating VRAM-intensive batch operations.

    This class manages a queue of batch operations that require VRAM,
    ensuring that batches are executed in order and that VRAM constraints
    are respected. It provides backpressure mechanisms to prevent VRAM
    exhaustion.

    Use this for reranker batch operations to ensure proper coordination
    across multiple workers.
    """

    def __init__(self, vram_manager: Optional[VRAMResourceManager] = None):
        """
        Initialize the batch scheduler.

        Args:
            vram_manager: VRAMResourceManager instance (creates default if None)
        """
        super().__init__()
        self._vram_manager = vram_manager or VRAMResourceManager()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending_batches: List[Dict[str, Any]] = []
        self._running_batches: Set[str] = set()
        self._batch_counter = 0
        self._paused = False

    def submit_batch(
        self,
        batch_func: Callable[..., Any],
        batch_args: tuple[Any, ...] = (),
        batch_kwargs: Optional[Dict[str, Any]] = None,
        vram_required_gb: float = 1.0,
        priority: int = 1,
        blocking: bool = False,
        timeout: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Submit a batch operation for execution.

        Args:
            batch_func: Function to execute for the batch
            batch_args: Positional arguments for batch_func
            batch_kwargs: Keyword arguments for batch_func
            vram_required_gb: Amount of VRAM required for this batch
            priority: Priority (higher = executed earlier)
            blocking: If True, wait until batch can be executed
            timeout: Maximum wait time if blocking

        Returns:
            Tuple of (submitted: bool, batch_id: Optional[str])
        """
        batch_kwargs = batch_kwargs or {}

        with self._lock:
            self._batch_counter += 1
            batch_id = f"batch_{self._batch_counter}_{int(time.time() * 1000)}"

            batch_info: Dict[str, Any] = {
                "batch_id": batch_id,
                "func": batch_func,
                "args": batch_args,
                "kwargs": batch_kwargs,
                "vram_required_gb": vram_required_gb,
                "priority": priority,
                "submitted_at": time.time(),
                "status": "pending",
            }

            # Insert based on priority (higher first)
            insert_idx = len(self._pending_batches)
            for i, existing in enumerate(self._pending_batches):
                if existing["priority"] < priority:
                    insert_idx = i
                    break
            self._pending_batches.insert(insert_idx, batch_info)

            if not blocking:
                logger.debug(
                    f"Batch {batch_id} submitted (VRAM: {vram_required_gb:.2f}GB, priority: {priority})"
                )
                return True, batch_id

            # Blocking mode: wait for batch to complete
            wait_start = time.time()
            while batch_info["status"] in ("pending", "running"):
                if timeout:
                    elapsed = time.time() - wait_start
                    if elapsed >= timeout:
                        return False, batch_id

                self._condition.wait(timeout=1.0)

            return batch_info["status"] == "completed", batch_id

    def execute_next_batch(self) -> Optional[Dict[str, Any]]:
        """
        Execute the next pending batch if VRAM is available.

        Returns:
            Batch result if a batch was executed, None otherwise
        """
        with self._lock:
            if self._paused or not self._pending_batches:
                return None

            # Check if we should apply backpressure
            if self._vram_manager.should_pause_for_backpressure():
                logger.debug("Pausing batch execution due to VRAM pressure")
                return None

            # Get next batch
            batch_info = self._pending_batches.pop(0)
            batch_id = str(batch_info["batch_id"])
            vram_required = float(batch_info["vram_required_gb"])

            alloc_id: Optional[str] = None

            # Request VRAM allocation
            allocated, alloc_id = self._vram_manager.request_allocation(
                size_gb=vram_required,
                batch_id=batch_id,
                blocking=False,
            )

            if not allocated:
                # Put back in queue and try later
                batch_info["status"] = "pending"
                self._pending_batches.insert(0, batch_info)
                return None

            # Execute batch
            batch_info["status"] = "running"
            batch_info["allocation_id"] = alloc_id
            batch_info["started_at"] = time.time()
            self._running_batches.add(batch_id)

        # Execute outside lock
        try:
            logger.debug(f"Executing batch {batch_id} (VRAM: {vram_required:.2f}GB)")
            result = batch_info["func"](*batch_info["args"], **batch_info["kwargs"])
            batch_info["result"] = result
            batch_info["status"] = "completed"
            batch_info["completed_at"] = time.time()
        except Exception as e:
            logger.error(f"Batch {batch_id} failed: {e}")
            batch_info["error"] = str(e)
            batch_info["status"] = "failed"
            batch_info["failed_at"] = time.time()
        finally:
            # Release VRAM
            if alloc_id:
                self._vram_manager.release_allocation(alloc_id)

            with self._lock:
                self._running_batches.discard(batch_id)
                self._condition.notify_all()

        return batch_info

    def pause(self) -> None:
        """Pause batch execution (backpressure)."""
        with self._lock:
            self._paused = True
            logger.info("Batch execution paused")

    def resume(self) -> None:
        """Resume batch execution."""
        with self._lock:
            self._paused = False
            self._condition.notify_all()
            logger.info("Batch execution resumed")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self._lock:
            return {
                "paused": self._paused,
                "pending_batches": len(self._pending_batches),
                "running_batches": len(self._running_batches),
                "pending_vram_gb": sum(
                    b["vram_required_gb"] for b in self._pending_batches
                ),
            }


# =============================================================================
# Convenience functions
# =============================================================================


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


def get_vram_manager() -> VRAMResourceManager:
    """Get the global VRAM resource manager instance."""
    return VRAMResourceManager()


def get_batch_scheduler() -> VRAMBatchScheduler:
    """Get the global VRAM batch scheduler instance."""
    return VRAMBatchScheduler()


def get_cached_resource_metrics() -> Optional[Dict[str, Any]]:
    """
    Retrieve the cached resource_check metrics from validation_cache.json.

    Returns the 'resource_check' dict (memory, cpu, gpu, disk_space, etc.)
    if the cache is present and its version matches, else None.
    This is a lightweight read-only helper — it does NOT re-run validation.
    """
    cache_data = _load_validation_cache()
    if cache_data is None:
        return None
    if cache_data.get("version") != CACHE_VERSION:
        logger.debug(
            f"Validation cache version mismatch "
            f"(found {cache_data.get('version')!r}, expected {CACHE_VERSION!r}). "
            f"Returning None — cache will be refreshed on next validation run."
        )
        return None
    return cache_data.get("results", {}).get("resource_check")
