# Cache System Documentation (v8.4)

## Overview

The CRCT cache system is a production-grade infrastructure designed to boost performance by storing the results of potentially costly operations (e.g., embeddings, re-ranking, AST analysis). Version 8.4 continues to refine this with stable hashing, hardware-adaptive resource management, and integration with the Bolt Optimization for faster dependency resolution.

#### Core Components:

1.  **`Cache` Class**: A high-performance instance holding data (`key -> value`), access metrics, and dependency information. It manages eviction policies (LRU, LFU) and handles automated compression for large entries.
2.  **`CacheManager` Class**: The central orchestrator that oversees all active `Cache` instances. It enforces the **Global Memory Budget**, manages persistent storage using **Pickle**, and handles automatic migration from legacy JSON caches.
3.  **`@cached` Decorator**: The primary interface for developers. It handles automated key generation, stable SHA256 hashing of file modification times, and dependency-aware invalidation.

#### Key Features (v8.4):

-   **Stable Hashing**: Uses SHA256 hashes of file modification times (mtimes) to ensure cache hits persist even when the process is restarted or moved.
-   **Persistent Storage (Pickle)**: Caches are saved as `.pkl` files in `cline_utils/dependency_system/utils/cache/`, allowing results to be reused across different CLI executions.
-   **Global Resource Management**: Automatically throttles total cache memory usage based on available system RAM (e.g., 512MB for 16GB systems).
-   **Intelligent Invalidation**: Caches automatically invalidate when dependent files (tracked via `file_deps`) are modified on disk.
-   **Compression**: Uses Gzip to compress large cache entries (>10MB), typically saving 30-50% disk space in large projects.

---

## How to Use the Cache System

The primary interface for enabling caching is the `@cached` decorator.

### 1. Basic Usage

To cache a function's results, decorate it with `@cached`, providing a unique `cache_name` and a `key_func` to generate a unique string key based on the function's arguments.

```python
from cline_utils.dependency_system.utils.cache_manager import cached

# Define a function to generate a key based on input 'x'
def create_cache_key(x):
    return f"computation_key:{x}"

@cached(cache_name="math_cache", key_func=create_cache_key)
def expensive_computation(x):
    print(f"Executing expensive_computation({x})...")
    # Simulate work
    return x * x

# First call: executes the function, result stored in "math_cache"
result1 = expensive_computation(5)

# Second call: returns cached result instantly
result2 = expensive_computation(5)
```

-   `"math_cache"`: This name identifies the specific cache file (`math_cache.pkl`) on disk.
-   `key_func`: Ensures that different inputs get different cache entries.

### 2. Advanced Usage: TTL and Dependencies

You can customize the Time-To-Live (TTL) or link cache entries to physical files.

**Custom TTL:**
```python
# Set a 1-hour (3600s) TTL for this specific cache
@cached(cache_name="hourly_cache", key_func=lambda arg: f"hc:{arg}", ttl=3600)
def fetch_api_data(arg):
    ...
```

**File-Based Invalidation (check_mtime):**
If a result depends on a file's state, use `check_mtime=True`. The system will hash the file's modification time into the cache key.

```python
@cached(
    cache_name="file_analysis",
    key_func=lambda path: f"analyze:{path}",
    check_mtime=True,
    file_deps=lambda path: [path] # Track this file for invalidation
)
def analyze_source_code(path):
    # This will automatically invalidate if the file at 'path' changes
    ...
```

### 3. Manual Invalidation

While the system handles most invalidation automatically, you can manually clear entries using `invalidate_dependent_entries` or clear entire caches via the CLI.

**In Code:**
```python
from cline_utils.dependency_system.utils.cache_manager import invalidate_dependent_entries

# Invalidate specific entry using a regex pattern
invalidate_dependent_entries(cache_name="file_analysis", key_pattern="analyze:src/main.py")

# Invalidate ALL entries in a specific cache
invalidate_dependent_entries(cache_name="file_analysis", key_pattern=".*")
```

**Via CLI:**
The most straightforward way to reset the system is the `clear-caches` command:
```bash
python -m cline_utils.dependency_system.dependency_processor clear-caches
```

The LLM can execute this command if you suspect caching issues are causing problems.

---

## Cache Management Details

### On-Demand Creation & Persistence
Caches are created by the `CacheManager` when first requested. If persistence is enabled (default in v8.3), the manager loads existing data from `cline_utils/dependency_system/utils/cache/*.pkl`. 

### Global Memory Budget & Eviction
The `CacheManager` monitors total memory usage across all active caches. If the **Global Budget** is exceeded:
1.  It identifies the least recently used (LRU) items across all caches.
2.  It evicts items until memory usage drops to 80% of the budget.
3.  This ensures the system remains responsive even with massive dependency graphs.

### Legacy Migration
If you have caches from v7.x or v8.0-8.2 in `.json` format, the `CacheManager` will:
-   Detect the `.json` files on startup.
-   Load and convert them to the new `.pkl` format.
-   Delete the old `.json` files to keep the directory clean.

---

## Configuration & Tuning

Cache behavior can be tuned by modifying constants in `cline_utils/dependency_system/utils/cache_manager.py` or via the project config.

| Constant | Default | Description |
| :--- | :--- | :--- |
| `DEFAULT_TTL` | `604800` | 7 days (in seconds). |
| `DEFAULT_MAX_SIZE` | `10000` | Maximum items per individual cache instance. |
| `ENABLE_COMPRESSION` | `True` | Whether to gzip large items. |
| `CACHE_DIR` | `./cache/` | Location of `.pkl` storage files. |

For detailed hardware-specific tuning (e.g., overriding the memory budget), see **[CACHE_TUNING.md](CACHE_TUNING.md)**.

---

## Monitoring & Statistics

You can retrieve real-time statistics for any cache to analyze hit rates and performance.

```python
from cline_utils.dependency_system.utils.cache_manager import get_cache_stats

stats = get_cache_stats("file_analysis")
print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")
print(f"Hit Rate: {stats['hit_rate']:.2%}")
print(f"Memory Usage: {stats['size_bytes'] / 1024:.2f} KB")
```

---

## Troubleshooting

-   **Cold Starts**: High analysis times after a major project change mean the system is rebuilding caches. This is normal.
-   **Unexpected Stale Data**: If analysis results seem wrong, run `clear-caches` via the CLI to force a full rebuild.
-   **Pickle Errors**: If you see "UnpicklingError", it likely means a cache file was corrupted during a crash. Simply delete the `cache/` directory contents.
-   **High Disk Usage**: Large projects can generate gigabytes of cache data. Check the `cache/` directory size and use `clear-caches` if necessary.