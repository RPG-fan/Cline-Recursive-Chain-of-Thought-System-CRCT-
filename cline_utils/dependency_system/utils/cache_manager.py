"""
Cache management module with dynamic, TTL-based caching for dependency tracking system.
Supports on-demand cache creation, automatic expiration, and granular invalidation.
"""

from __future__ import annotations

import functools
import gzip
import json
import logging
import os
import pickle
import re
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Configuration
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DEFAULT_MAX_SIZE = 10000  # Default max items per cache
DEFAULT_TTL = 600
CACHE_SIZES = {
    "embeddings_generation": 300,  # Smaller for heavy data
    "key_generation": 5000,  # Larger for key maps
    "reranking": 35000,  # Reranking results
    "reranker_history": 200,  # Historical pairs (Set/List)
    "ast_verified_links": 200,  # Structural data
    "default": DEFAULT_MAX_SIZE,
}

# Advanced cache configuration
ENABLE_COMPRESSION = True
COMPRESSION_THRESHOLD = 10485760  # Only compress items larger than 10MB
COMPRESSION_MIN_SAVINGS = 0.1  # 10% minimum savings

# Global registry for memory monitoring
_CACHE_REGISTRY: list[weakref.ref[Cache]] = []
_REGISTRY_LOCK = threading.Lock()
_global_budget_mb: Optional[int] = None  # Calculated on first access


class EvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    ADAPTIVE = "adaptive"  # Hybrid LRU/LFU
    FIFO = "fifo"  # First In, First Out
    RANDOM = "random"  # Random eviction


@dataclass
class CacheMetrics:
    """Enhanced cache performance metrics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    compression_saves: int = 0
    total_size_bytes: int = 0
    access_count: Dict[str, int] = field(default_factory=dict)
    last_access: Dict[str, float] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class Cache:
    """Enhanced cache instance with LRU/LFU eviction, compression, TTL, and dependency tracking."""

    def __init__(
        self,
        name: str,
        ttl: int = DEFAULT_TTL,
        max_size: int = DEFAULT_MAX_SIZE,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        enable_compression: bool = ENABLE_COMPRESSION,
    ):
        super().__init__()
        self.name = name
        # Initialize dictionaries with explicit types for Pyright
        self.data: Dict[str, Tuple[Any, float, Optional[float]]] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.reverse_deps: Dict[str, List[str]] = {}
        self.metrics = CacheMetrics()  # This will call __post_init__ automatically
        self.creation_time = time.time()
        self.default_ttl = ttl
        self.max_size = CACHE_SIZES.get(name, max_size)
        self.eviction_policy = eviction_policy
        self.enable_compression = enable_compression

        # Enhanced features
        self._lock = threading.RLock()  # Thread safety
        self.compression_threshold = COMPRESSION_THRESHOLD

        # Register cache for global monitoring
        with _REGISTRY_LOCK:
            _CACHE_REGISTRY.append(weakref.ref(self))

        # Adaptive parameters
        self.lru_weight = 0.6 if eviction_policy == EvictionPolicy.ADAPTIVE else 1.0
        self.min_accesses = 3

        logger.debug(
            f"Cache '{name}' initialized: policy={eviction_policy.value}, "
            f"max_size={self.max_size}, compression={enable_compression}"
        )

    def get(self, key: str) -> Any:
        with self._lock:
            if key not in self.data:
                self.metrics.misses += 1
                # logger.debug(f"Cache '{self.name}': Miss for key '{key}'")
                return None

            # Get value and metadata
            value, current_time, expiry = self.data[key]

            # Check if expired
            if expiry and time.time() > expiry:
                self._remove_key(key)
                self.metrics.misses += 1
                # logger.debug(f"Cache '{self.name}': Miss (expired) for key '{key}'")
                return None

            # Update access information
            current_time = time.time()
            self.data[key] = (value, current_time, expiry)  # Update access time
            self.metrics.last_access[key] = current_time
            self.metrics.access_count[key] = self.metrics.access_count.get(key, 0) + 1
            self.metrics.hits += 1
            # logger.debug(f"Cache '{self.name}': Hit for key '{key}'")

            # Decompress if necessary
            if isinstance(value, bytes) and self.enable_compression:
                value = self._decompress_value(value)

            return value

    def _compress_value(self, value: Any) -> bytes:
        """Compress value for storage."""
        if isinstance(value, str):
            return gzip.compress(value.encode("utf-8"))
        else:
            return gzip.compress(pickle.dumps(value))

    def _decompress_value(self, value: bytes) -> Any:
        """Decompress value from storage."""
        try:
            decompressed = gzip.decompress(value)
            # Try to decode as string first
            try:
                return decompressed.decode("utf-8")
            except UnicodeDecodeError:
                # Fall back to unpickling
                return pickle.loads(decompressed)
        except Exception as e:
            logger.warning(f"Failed to decompress value: {e}")
            # Return raw bytes if decompression fails
            return value

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of a value in bytes."""
        try:
            if isinstance(value, str):
                return len(value.encode("utf-8"))
            elif isinstance(value, bytes):
                return len(value)
            else:
                return len(pickle.dumps(value))
        except:
            return len(str(value))

    def _should_compress(self, key: str, value: Any) -> bool:
        """Determine if value should be compressed."""
        if not self.enable_compression:
            return False

        size = self._estimate_size(value)

        # Only compress if above threshold
        if size < self.compression_threshold:
            return False

        # Estimate compression ratio
        try:
            compressed = gzip.compress(pickle.dumps(value))
            ratio = len(compressed) / size
            return ratio < (1 - COMPRESSION_MIN_SAVINGS)
        except:
            return False

    def set(
        self,
        key: str,
        value: Any,
        dependencies: Optional[List[str]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        with self._lock:
            # Compress if beneficial
            if self._should_compress(key, value):
                value = self._compress_value(value)
                self.metrics.compression_saves += 1

            # Calculate size for metrics
            size_estimate = self._estimate_size(value)

            # Evict if necessary
            if len(self.data) >= self.max_size:
                self._evict_items()

            # Set with new metadata
            expiry = (
                time.time() + (ttl if ttl is not None else self.default_ttl)
                if ttl != 0
                else None
            )
            self.data[key] = (value, time.time(), expiry)
            self.metrics.total_size_bytes += size_estimate

            # Track dependencies
            if dependencies:
                for dep in dependencies:
                    if dep not in self.dependencies:
                        self.dependencies[dep] = []
                    self.dependencies[dep].append(key)
                    if key not in self.reverse_deps:
                        self.reverse_deps[key] = []
                    self.reverse_deps[key].append(dep)

    def _evict_items(self) -> None:
        """Enhanced eviction with multiple policies."""
        items_to_evict = len(self.data) - self.max_size

        if items_to_evict <= 0:
            return

        if self.eviction_policy == EvictionPolicy.LFU:
            # Evict least frequently used items
            freq_items = sorted(
                self.data.keys(), key=lambda k: self.metrics.access_count.get(k, 0)
            )
            keys_to_evict = freq_items[:items_to_evict]
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Evict first in, first out
            keys_to_evict = list(self.data.keys())[:items_to_evict]
        elif self.eviction_policy == EvictionPolicy.RANDOM:
            import random

            keys = list(self.data.keys())
            keys_to_evict = random.sample(keys, min(items_to_evict, len(keys)))
        else:  # LRU or ADAPTIVE
            # For LRU/ADAPTIVE, use access time sorting
            time_sorted = sorted(
                self.data.keys(), key=lambda k: self.data[k][1]  # access_time
            )
            keys_to_evict = time_sorted[:items_to_evict]

        # Remove evicted items
        for key in keys_to_evict:
            self._remove_key(key)
            self.metrics.evictions += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced cache statistics."""
        with self._lock:
            total_items = len(self.data)
            total_size = self.metrics.total_size_bytes

            return {
                "name": self.name,
                "total_items": total_items,
                "max_size": self.max_size,
                "utilization": (
                    (total_items / self.max_size * 100) if self.max_size > 0 else 0
                ),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "hit_rate": self.metrics.hit_rate,
                "hits": self.metrics.hits,
                "misses": self.metrics.misses,
                "evictions": self.metrics.evictions,
                "compression_saves": self.metrics.compression_saves,
                "eviction_policy": self.eviction_policy.value,
                "compression_enabled": self.enable_compression,
            }

    def _evict_lru(self) -> None:
        if not self.data:
            return
        try:
            lru_key = min(self.data, key=lambda k: self.data[k][1])
            self._remove_key(lru_key)
        except ValueError:
            pass
        except RuntimeError:
            logger.warning(
                f"Cache '{self.name}': RuntimeError during LRU eviction. Cache may be highly contended."
            )

    def _remove_key(self, key: str) -> None:
        try:
            if key in self.data:
                # Update size metrics before removal
                value, _, _ = self.data[key]
                size = self._estimate_size(value)
                del self.data[key]
                self.metrics.total_size_bytes = max(
                    0, self.metrics.total_size_bytes - size
                )

            # Clean up reverse dependencies: remove 'key' from the dependency lists of its dependents
            if key in self.reverse_deps:
                for dependent_key in list(
                    self.reverse_deps.get(key, [])
                ):  # Iterate a copy
                    if (
                        dependent_key in self.dependencies
                    ):  # Check if dependent_key still in self.dependencies
                        try:
                            self.dependencies[dependent_key].remove(key)
                            if not self.dependencies[dependent_key]:
                                del self.dependencies[dependent_key]
                        except (ValueError, KeyError):
                            pass  # Item already removed or dep dict changed
                try:
                    del self.reverse_deps[key]  # Remove key from reverse_deps itself
                except KeyError:
                    pass

        except KeyError:
            pass

    def cleanup_expired(self) -> None:
        """Remove all expired entries."""
        current_time = time.time()
        try:
            items_to_check = list(self.data.items())
        except RuntimeError:
            logger.warning(
                f"Cache '{self.name}': RuntimeError getting items for cleanup, likely high contention. Skipping this cleanup cycle."
            )
            return

        expired_keys_to_remove = [
            k
            for k, (_, _, expiry) in items_to_check
            if expiry and current_time > expiry
        ]

        if expired_keys_to_remove:
            keys_actually_removed_count = 0
            for key_to_remove in expired_keys_to_remove:
                if key_to_remove in self.data:
                    _val, _acc_time, exp_check_final = self.data[key_to_remove]
                    if exp_check_final and current_time > exp_check_final:
                        self._remove_key(key_to_remove)
                        keys_actually_removed_count += 1
            if keys_actually_removed_count > 0:
                logger.debug(
                    f"Cache '{self.name}': Cleaned up {keys_actually_removed_count} expired entries."
                )

    def is_expired(self) -> bool:
        return (time.time() - self.creation_time) > self.default_ttl and not self.data

    def invalidate(self, key_pattern: str) -> None:
        """Invalidate entries matching a key pattern (supports regex). Also invalidates dependent entries."""
        with self._lock:  # <<< SUGGESTION: Add lock here
            compiled_pattern = re.compile(key_pattern)
            # Iterate over a copy of keys for safety during removal
            keys_to_remove_initial = [
                k for k in list(self.data.keys()) if compiled_pattern.match(k)
            ]

            processed_for_invalidation = set()
            queue_to_invalidate = list(keys_to_remove_initial)

            while queue_to_invalidate:
                key_to_invalidate = queue_to_invalidate.pop(0)
                if key_to_invalidate in processed_for_invalidation:
                    continue

                self._remove_key(
                    key_to_invalidate
                )  # Handles removal from self.data and basic reverse_deps cleanup
                processed_for_invalidation.add(key_to_invalidate)

                if key_to_invalidate in self.dependencies:
                    dependent_keys_list = list(
                        self.dependencies.get(key_to_invalidate, [])
                    )
                    for dep_key in dependent_keys_list:
                        if dep_key not in processed_for_invalidation:
                            queue_to_invalidate.append(dep_key)
                    try:
                        del self.dependencies[key_to_invalidate]
                    except KeyError:
                        pass

            if processed_for_invalidation:
                logger.debug(
                    f"Cache '{self.name}': Invalidated {len(processed_for_invalidation)} entries matching pattern '{key_pattern}'."
                )

    def stats(self) -> Dict[str, int]:
        return {
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "size": len(self.data),
            "evictions": self.metrics.evictions,
            "compression_saves": self.metrics.compression_saves,
        }


class CacheManager:
    """Manages multiple caches with persistence and cleanup."""

    def __init__(self, persist: bool = False):
        super().__init__()
        self.caches: Dict[str, Cache] = {}
        self.persist = persist
        self._last_cleanup_time = 0.0  # Throttling for cleanup()
        if persist:
            os.makedirs(CACHE_DIR, exist_ok=True)
            self._load_persistent_caches()

    def get_cache(self, cache_name: str, ttl: int = DEFAULT_TTL) -> Cache:
        """Retrieve or create a cache by name."""
        if cache_name not in self.caches or self.caches[cache_name].is_expired():
            self.caches[cache_name] = Cache(cache_name, ttl)
            logger.debug(f"Spun up new cache: {cache_name} with TTL {ttl}s")
        return self.caches[cache_name]

    def cleanup(self, force: bool = False) -> None:
        """Remove expired caches and items, with throttling, and enforce global budget."""
        now = time.time()
        if not force and (now - self._last_cleanup_time < 60):
            return

        self._last_cleanup_time = now

        # Enforce global budget
        self._enforce_global_budget()

        # Existing cleanup logic...
        for name, cache in list(self.caches.items()):
            # Clean up items within the cache
            cache.cleanup_expired()

            # If the whole cache is now empty and expired, remove it
            if cache.is_expired():
                if self.persist:
                    self._save_cache(name)
                del self.caches[name]
                logger.debug(f"Spun down expired cache: {name}")

    def _enforce_global_budget(self) -> None:
        """Monitor total memory usage across all caches and evict if budget exceeded."""
        budget_mb = self.get_global_budget_mb()
        budget_bytes = budget_mb * 1024 * 1024
        total_usage = self.get_total_cache_usage()

        if total_usage > budget_bytes:
            logger.warning(
                f"Global cache limit exceeded: {total_usage // (1024*1024)}MB / {budget_mb}MB. "
                "Starting proactive eviction."
            )
            # Find all active caches and prune dead refs
            active_caches: list[Cache] = []
            with _REGISTRY_LOCK:
                new_registry: list[weakref.ref[Cache]] = []
                for ref in _CACHE_REGISTRY:
                    cache: Optional[Cache] = ref()
                    if cache is not None:
                        active_caches.append(cache)
                        new_registry.append(ref)
                _CACHE_REGISTRY[:] = new_registry

            if not active_caches:
                return

            # Sort caches by usage descending
            active_caches.sort(key=lambda c: c.metrics.total_size_bytes, reverse=True)

            # Evict from each cache until we are under budget
            # Or at least until we've reduced usage significantly
            target_usage = int(budget_bytes * 0.8)  # Aim for 80% to provide a buffer
            for active_cache in active_caches:
                if total_usage <= target_usage:
                    break

                # Evict 20% of items or until we meet target
                initial_size: int = active_cache.metrics.total_size_bytes
                if initial_size == 0:
                    continue

                items_to_evict = max(1, len(active_cache.data) // 5)
                # Ensure we don't evict everything if it's small
                if len(active_cache.data) < 5 and total_usage > budget_bytes:
                    items_to_evict = 1

                with active_cache._lock:
                    # Logic similar to _evict_items but forced
                    time_sorted: list[str] = sorted(
                        active_cache.data.keys(),
                        key=lambda k: active_cache.data[k][1],  # access_time
                    )
                    keys_to_evict: list[str] = time_sorted[:items_to_evict]
                    for key in keys_to_evict:
                        active_cache._remove_key(key)
                        active_cache.metrics.evictions += 1

                total_usage -= initial_size - active_cache.metrics.total_size_bytes

    @staticmethod
    def get_global_budget_mb() -> int:
        """Get the global cache budget in MB, scaling based on system resources."""
        global _global_budget_mb
        if _global_budget_mb is not None:
            return _global_budget_mb

        # Default fallback
        _global_budget_mb = 512

        try:
            # Use psutil directly instead of ConfigManager to avoid an
            # import cycle (cache_manager ↔ config_manager).
            import psutil

            available_mb: int = psutil.virtual_memory().available // (1024 * 1024)
            if available_mb > 16384:  # > 16GB available
                _global_budget_mb = 2048
            elif available_mb > 4096:  # > 4GB available
                _global_budget_mb = 512
            elif available_mb > 0:
                _global_budget_mb = 128  # Very tight
        except ImportError:
            logger.debug("psutil not available, using default cache budget")
        except Exception as e:
            logger.debug(f"Could not calculate dynamic budget, using default: {e}")

        return _global_budget_mb

    @staticmethod
    def get_total_cache_usage() -> int:
        """Sum total_size_bytes across all registered caches."""
        total: int = 0
        with _REGISTRY_LOCK:
            for ref in _CACHE_REGISTRY:
                cache_ref: Optional[Cache] = ref()
                if cache_ref is not None:
                    total += cache_ref.metrics.total_size_bytes
        return total

    def clear_all(self) -> None:
        if self.persist:
            for name in list(self.caches.keys()):
                self._save_cache(name)
        self.caches.clear()
        logger.info("All caches cleared.")

    def _save_cache(self, cache_name: str) -> None:
        if cache_name in self.caches:
            cache_file = os.path.join(CACHE_DIR, f"{cache_name}.json")
            # Use UUID to ensure uniqueness across threads/processes
            temp_file = f"{cache_file}.{uuid.uuid4()}.tmp"
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    # Ensure there's data to write
                    if not self.caches[cache_name].data:
                        # If cache is empty, don't write an empty file, just ensure old one is gone
                        if os.path.exists(cache_file):
                            try:
                                os.remove(cache_file)
                            except OSError:
                                pass
                        return

                    current_cache_data_items = list(
                        self.caches[cache_name].data.items()
                    )

                    def _json_safe(obj: Any) -> Any:
                        """
                        Convert cache values into JSON-serializable form.
                        - Pass through primitives and plain containers.
                        - For bytes and other non-serializable objects, store a safe representation.
                        """
                        # Fast path: already JSON-native
                        if isinstance(obj, (str, int, float, bool)) or obj is None:
                            return obj
                        if isinstance(obj, dict):
                            return {str(k): _json_safe(v) for k, v in obj.items()}
                        if isinstance(obj, (list, tuple, set)):
                            return [_json_safe(v) for v in obj]
                        if isinstance(obj, (bytes, bytearray)):
                            # Avoid breaking on compressed/serialized cache entries
                            # Represent as tagged hex string instead of raw bytes
                            return {
                                "__type__": "bytes",
                                "encoding": "hex",
                                "data": obj.hex(),
                            }
                        # Fallback: string representation
                        return repr(obj)

                    data = {
                        "data": {
                            k: _json_safe(v[0])
                            for k, v in current_cache_data_items
                            if v[2] is None or v[2] > time.time()
                        },
                        "dependencies": dict(self.caches[cache_name].dependencies),
                    }
                    json.dump(data, f)

                # Atomic rename with retries for Windows file locking
                max_retries = 3
                for i in range(max_retries):
                    try:
                        os.replace(temp_file, cache_file)
                        break
                    except OSError as e:
                        if i == max_retries - 1:
                            raise  # Re-raise on last attempt
                        logger.warning(
                            f"Retry {i+1}/{max_retries} saving cache {cache_name}: {e}"
                        )
                        time.sleep(0.1)  # Short wait before retry

            except Exception as e:
                logger.error(f"Failed to save cache {cache_name}: {e}")
            finally:
                # Clean up temp file if it still exists
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass

    def _json_revive(self, obj: Any) -> Any:
        """Recursively revive objects from their JSON-safe representation."""
        if isinstance(obj, dict):
            if obj.get("__type__") == "bytes" and obj.get("encoding") == "hex":
                try:
                    return bytes.fromhex(obj.get("data", ""))
                except (ValueError, TypeError):
                    return obj  # Return as-is if hex is invalid
            return {k: self._json_revive(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._json_revive(v) for v in obj]
        return obj

    def _load_persistent_caches(self) -> None:
        if not os.path.exists(CACHE_DIR):
            return
        for cache_file in os.listdir(CACHE_DIR):
            if cache_file.endswith(".json"):
                cache_name = cache_file[:-5]
                cache_path = os.path.join(CACHE_DIR, cache_file)
                try:
                    # Check if file is empty to prevent JSONDecodeError
                    if os.path.getsize(cache_path) == 0:
                        logger.warning(
                            f"Cache file {cache_file} is empty. Deleting it."
                        )
                        os.remove(cache_path)
                        continue

                    with open(cache_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        cache = Cache(cache_name)
                        for key, value in data.get("data", {}).items():
                            revived_value = self._json_revive(value)
                            cache.set(key, revived_value, ttl=0)
                        cache.dependencies = data.get("dependencies", {})
                        self.caches[cache_name] = cache
                    logger.debug(f"Loaded persistent cache: {cache_name}")
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to decode JSON from {cache_file}: {e}. Deleting corrupt cache file."
                    )
                    try:
                        os.remove(cache_path)
                    except OSError as oe:
                        logger.error(
                            f"Error removing corrupt cache file {cache_path}: {oe}"
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to load cache {cache_name} from {cache_file}: {e}"
                    )


cache_manager = CacheManager(persist=False)


def get_tracker_cache_key(tracker_path: str, tracker_type: str) -> str:
    from .path_utils import normalize_path

    return f"tracker:{normalize_path(tracker_path)}:{tracker_type}"


def clear_all_caches() -> None:
    """Clear all caches in the manager."""
    cache_manager.clear_all()


def invalidate_dependent_entries(cache_name: str, key_pattern: str) -> None:
    """Invalidate cache entries matching a key pattern in a specific cache."""
    cache = cache_manager.get_cache(cache_name)
    cache.invalidate(key_pattern)


def file_modified(file_path: str, project_root: str, cache_type: str = "all") -> None:
    """Invalidate caches when a file is modified."""
    from .path_utils import normalize_path

    norm_path = normalize_path(file_path)
    # Use raw f-string for regex pattern
    key_pattern_to_invalidate = (
        rf".*(?::|\||^){re.escape(norm_path)}(?:\||$).*"  # FIXED
    )

    caches_to_scan = (
        [
            cache_manager.get_cache(cache_name)
            for cache_name in list(cache_manager.caches.keys())
        ]
        if cache_type == "all"
        else [cache_manager.get_cache(cache_type)]
    )

    dep_key = f"file:{norm_path}"

    for cache_instance in caches_to_scan:
        if cache_instance:
            # 1. Invalidate by key pattern check (existing logic)
            cache_instance.invalidate(key_pattern_to_invalidate)

            # 2. Invalidate by dependency lookup (new logic)
            with cache_instance._lock:
                if dep_key in cache_instance.dependencies:
                    # Get list of keys dependent on this file
                    dependent_keys = list(cache_instance.dependencies[dep_key])
                    if dependent_keys:
                        # logger.debug(f"Cache '{cache_instance.name}': Invalidating {len(dependent_keys)} entries dependent on '{norm_path}'")
                        for key in dependent_keys:
                            cache_instance._remove_key(key)

    logger.debug(
        f"Invalidated entries matching path '{norm_path}' in cache(s) type '{cache_type}'."
    )


def tracker_modified(
    tracker_path: str, tracker_type: str, project_root: str, cache_type: str = "all"
) -> None:
    """Invalidate caches when a tracker is modified."""
    from .path_utils import normalize_path

    norm_path = normalize_path(tracker_path)
    key_pattern_for_tracker: str

    if cache_type == "all":
        # Use raw f-string for regex pattern
        generic_path_pattern = rf".*(?::|\||^){re.escape(norm_path)}(?:\||$).*"  # FIXED
        specific_tracker_data_pattern = (
            rf"^tracker_data_structured:{re.escape(norm_path)}:.*"  # FIXED
        )

        cache_data_struct = cache_manager.get_cache("tracker_data_structured")
        if cache_data_struct:
            cache_data_struct.invalidate(specific_tracker_data_pattern)
            logger.debug(
                f"Invalidated '{specific_tracker_data_pattern}' in 'tracker_data_structured'."
            )

        for cache_name_iter in list(cache_manager.caches.keys()):
            if cache_name_iter != "tracker_data_structured":
                cache_instance_iter = cache_manager.get_cache(cache_name_iter)
                if cache_instance_iter:
                    cache_instance_iter.invalidate(generic_path_pattern)
        logger.debug(
            f"Additionally scanned other caches for generic pattern '{generic_path_pattern}'."
        )

    else:
        if cache_type == "tracker_data_structured":
            key_pattern_for_tracker = (
                rf"^tracker_data_structured:{re.escape(norm_path)}:.*"  # FIXED
            )
        else:
            key_pattern_for_tracker = (
                rf".*(?::|\||^){re.escape(norm_path)}(?:\||$).*"  # FIXED
            )

        cache_instance_tracker = cache_manager.get_cache(cache_type)
        if cache_instance_tracker:
            cache_instance_tracker.invalidate(key_pattern_for_tracker)
        logger.debug(
            f"Invalidated entries for tracker '{norm_path}' (pattern '{key_pattern_for_tracker}') in cache type '{cache_type}'."
        )


def cached(
    cache_name: str,
    key_func: Optional[Callable[..., str]] = None,
    ttl: Optional[int] = DEFAULT_TTL,
    file_deps: Optional[Callable[..., List[str]]] = None,
    check_mtime: bool = False,
    track_path_args: Optional[List[int]] = None,
):
    """
    Decorator for caching with dynamic dependencies and TTL.

    Args:
        cache_name: Name of the cache to use.
        key_func: Optional function to generate cache key from args/kwargs.
        ttl: Time-to-live in seconds. None uses DEFAULT_TTL, 0 means no expiry.
        file_deps: Optional callable that returns list of file paths whose
                   modification times should invalidate this cache entry.
                   Signature: file_deps(*args, **kwargs) -> List[str]
        check_mtime: If True and file_deps is provided, incorporates file mtimes
                     into the cache key, causing automatic invalidation when
                     dependent files are modified.
        track_path_args: Optional list of indices for arguments that are file paths
            to be added as dependencies for cascading invalidation.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # --- Step 1: Generate base cache key ---
            if key_func is not None:
                base_key = key_func(*args, **kwargs)
            else:
                # Default key: function name + stringified args/kwargs
                # (excluding "self"/"cls" heuristically)
                arg_list = list(args)
                if (
                    arg_list
                    and hasattr(args[0], "__class__")
                    # Only omit if it's an instance of a non-builtin class (likely self/cls)
                    and type(args[0]).__module__ != "builtins"
                    and not isinstance(
                        args[0], (str, int, float, bool, list, dict, set, tuple)
                    )
                ):
                    # Heuristic: if first arg looks like an instance/class, omit it
                    arg_list_for_key = arg_list[1:]
                else:
                    arg_list_for_key = arg_list
                d_key_parts = [str(a) for a in arg_list_for_key] + [
                    f"{k}={v}" for k, v in sorted(kwargs.items())
                ]
                base_key = f"{func.__name__}::{'|'.join(d_key_parts)}"

            # --- Step 2: Resolve file dependencies ---
            dep_file_paths: List[str] = []
            if file_deps is not None:
                try:
                    dep_file_paths = file_deps(*args, **kwargs)
                    # Filter to only existing files
                    dep_file_paths = [
                        p for p in dep_file_paths if p and os.path.exists(p)
                    ]
                except Exception as e:
                    logger.debug(f"Error resolving file_deps for {func.__name__}: {e}")
                    dep_file_paths = []

            # --- Step 3: Build final cache key (optionally with mtimes) ---
            key = base_key
            if check_mtime and dep_file_paths:
                mtime_list: List[str] = []
                for fp in dep_file_paths:
                    try:
                        # Use absolute path to ensure sorting is consistent across different relative paths
                        from .path_utils import normalize_path

                        abs_path = normalize_path(fp)
                        mtime = os.path.getmtime(fp)
                        mtime_list.append(f"{abs_path}:{mtime:.6f}")
                    except OSError:
                        mtime_list.append(f"{fp}:missing")

                # Sort the mtime list to ensure order invariance
                mtime_list.sort()
                key = f"{base_key}|mtime:{hash('|'.join(mtime_list))}"

            cache_ttl_to_use = ttl if ttl is not None else DEFAULT_TTL
            cache = cache_manager.get_cache(cache_name, cache_ttl_to_use)

            # --- Step 4: Check cache ---
            cached_val = cache.get(key)
            if cached_val is not None:
                return cached_val

            # --- Step 5: Execute function ---
            result = func(*args, **kwargs)

            # --- Step 6: Build dependency list for cascading invalidation ---
            dependencies_list: List[str] = []

            # Add explicit file dependencies (normalized) from file_deps
            if dep_file_paths:
                from .path_utils import normalize_path

                for fp in dep_file_paths:
                    norm_path = normalize_path(fp)
                    dependencies_list.append(f"file:{norm_path}")

            # Add dependencies from tracked path arguments
            if track_path_args:
                from .path_utils import normalize_path

                for idx in track_path_args:
                    if idx < len(args):
                        path_arg = args[idx]
                        if isinstance(path_arg, str) and os.path.exists(path_arg):
                            norm_dep_path = normalize_path(path_arg)
                            dependencies_list.append(f"file:{norm_dep_path}")

            # --- Step 7: Store in cache ---
            cache.set(key, result, dependencies_list, ttl=cache_ttl_to_use)
            # Throttled cleanup
            cache_manager.cleanup()
            return result

        return cast(F, wrapper)

    return decorator


def _get_normalize_path_cache_key(p: str) -> str:
    """Key generator for normalize_path_cached."""
    return f"normalize:{p if p else 'empty'}"


@cached("path_normalization", key_func=_get_normalize_path_cache_key)
def normalize_path_cached(path: str) -> str:
    """Cached version of normalize_path."""
    from .path_utils import normalize_path

    return normalize_path(path)


def _get_project_root_cache_key() -> str:
    """Key generator for get_project_root_cached."""
    return f"project_root:{normalize_path_cached(os.getcwd())}"


@cached("project_root", key_func=_get_project_root_cache_key)
def get_project_root_cached() -> str:
    """Cached version of get_project_root."""
    from .path_utils import get_project_root

    return get_project_root()


def _get_is_valid_project_path_cache_key(p: str) -> str:
    """Key generator for is_valid_project_path_cached."""
    return f"valid_project_path:{normalize_path_cached(p)}:{get_project_root_cached()}"


@cached("valid_project_paths", key_func=_get_is_valid_project_path_cache_key)
def is_valid_project_path_cached(path: str) -> bool:
    """Cached version of is_valid_project_path."""
    from .path_utils import is_valid_project_path

    return is_valid_project_path(path)


def check_file_modified(file_path: str) -> bool:
    """Check if a file has been modified, updating metadata cache."""
    from .path_utils import get_project_root, normalize_path

    norm_path = normalize_path(file_path)
    cache_key = f"timestamp:{norm_path}"
    cache = cache_manager.get_cache("metadata")
    current_project_root = get_project_root()

    if not os.path.exists(file_path):
        if cache.get(cache_key) is not None:
            cache.invalidate(cache_key)
            file_modified(norm_path, current_project_root)
            return True
        return False

    current_timestamp = os.path.getmtime(file_path)
    cached_timestamp_val = cache.get(cache_key)

    if cached_timestamp_val is None or current_timestamp > cached_timestamp_val:
        cache.set(cache_key, current_timestamp, ttl=None)
        file_modified(norm_path, current_project_root)
        return True
    return False


def get_file_type_cached(file_path: str) -> str:
    """Cached version of get_file_type."""
    from .path_utils import get_file_type

    return get_file_type(file_path)


def get_cache_stats(cache_name: str) -> Dict[str, int]:
    """Get hit/miss stats for a cache."""
    cache = cache_manager.get_cache(cache_name)
    return cache.stats()
