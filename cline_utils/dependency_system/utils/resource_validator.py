"""
System resource validation utilities for project analyzer.
Validates available memory, disk space, VRAM, and other resources before analysis.
"""

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
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.exceptions_enhanced import DiskSpaceError, MemoryLimitError, log_and_reraise
from ..utils.path_utils import normalize_path

# Try to import torch for VRAM management
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

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


# =============================================================================
# VRAM RESOURCE MANAGEMENT
# =============================================================================


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
        "qwen3_reranker_0.6b": 0.07,
        "qwen3_embedding_4b": 3.5,
        "mpnet_base": 0.5,
    }

    def __new__(cls, *args, **kwargs) -> "VRAMResourceManager":
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

        self._initialized = True
        logger.debug(
            f"VRAMResourceManager initialized (reservation={reservation_percent:.0%}, buffer={safety_buffer_gb}GB)"
        )

    def _get_physical_vram_gb(self) -> float:
        """Get total physical VRAM in GB."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0
        try:
            _, total_memory = torch.cuda.mem_get_info(0)
            return total_memory / (1024**3)
        except Exception as e:
            logger.warning(f"Failed to get physical VRAM: {e}")
            return 0.0

    def _get_available_vram_gb(self) -> float:
        """Get actually free VRAM in GB (not accounting for reservations)."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
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
            self.MODEL_FOOTPRINTS.get("qwen3_reranker_0.6b", 0.07)
            + self._safety_buffer_gb,
        )

    def get_available_for_allocation(self) -> float:
        """
        Get VRAM available for new allocations.
        Accounts for: active allocations + system reservation

        Returns:
            Available VRAM in GB for new allocations
        """
        with self._allocation_lock:
            free_vram = self._get_available_vram_gb()
            reserved = self.get_reserved_vram_gb()

            # Available = free - reserved - already_allocated
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
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    # Get actual memory usage - this will be the model's footprint if loaded
                    model_memory_gb = torch.cuda.memory_allocated() / (1024**3)
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
        with self._condition:
            self._stats["total_requests"] += 1

            # Generate unique allocation ID
            allocation_id = (
                f"alloc_{self._allocation_counter}_{int(time.time() * 1000)}"
            )
            self._allocation_counter += 1

            allocation = VRAMAllocation(
                allocation_id=allocation_id,
                size_gb=size_gb,
                requested_at=time.time(),
                worker_id=worker_id,
                batch_id=batch_id,
            )

            # Check if we can grant immediately
            if self._can_allocate(size_gb):
                self._grant_allocation(allocation)
                # logger.debug(
                #     f"VRAM allocation granted: {allocation_id} ({size_gb:.2f}GB)"
                # )
                return True, allocation_id

            # Can't allocate now
            if not blocking:
                self._stats["denied"] += 1
                logger.debug(
                    f"VRAM allocation denied: {size_gb:.2f}GB (available: {self.get_available_for_allocation():.2f}GB)"
                )
                return False, None

            # Blocking mode: wait for availability
            self._stats["deferred"] += 1
            allocation.status = AllocationStatus.PENDING
            self._active_allocations[allocation_id] = allocation

            wait_start = time.time()
            while not self._can_allocate(size_gb):
                remaining = None
                if timeout:
                    elapsed = time.time() - wait_start
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        # Timeout - remove from pending
                        if allocation_id in self._active_allocations:
                            del self._active_allocations[allocation_id]
                        self._stats["denied"] += 1
                        logger.warning(
                            f"VRAM allocation timeout after {timeout}s: {allocation_id}"
                        )
                        return False, None

                self._condition.wait(timeout=remaining)

            # Granted after waiting
            self._grant_allocation(allocation)
            # logger.debug(
            #     f"VRAM allocation granted after wait: {allocation_id} ({size_gb:.2f}GB)"
            # )
            return True, allocation_id

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
        with self._condition:
            allocation = self._active_allocations.get(allocation_id)
            if not allocation:
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

            # logger.debug(f"VRAM allocation released: {allocation_id}")
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
        with self._allocation_lock:
            return {
                **self._stats,
                "current_allocated_gb": self._total_allocated_gb,
                "peak_allocated_gb": self._peak_allocated_gb,
                "available_for_allocation_gb": self.get_available_for_allocation(),
                "physical_vram_gb": self._get_physical_vram_gb(),
                "reserved_vram_gb": self.get_reserved_vram_gb(),
                "active_allocations": len(self._active_allocations),
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
        with self._condition:
            wait_start = time.time()
            while self.get_available_for_allocation() < required_gb:
                remaining = None
                if timeout:
                    elapsed = time.time() - wait_start
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        return False
                self._condition.wait(timeout=remaining)
            return True


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
        batch_func: callable,
        batch_args: tuple = (),
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

            batch_info = {
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
            batch_id = batch_info["batch_id"]
            vram_required = batch_info["vram_required_gb"]

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
