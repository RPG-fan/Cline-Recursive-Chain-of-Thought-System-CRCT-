# Advanced Cache Tuning (v8.4)

Version 8.4 continues to refine the production-grade caching infrastructure introduced in v8.3, focusing on integration with the Bolt Optimization and improved VRAM efficiency for local LLM tasks.

## 1. Stable Hashing (SHA256)
The system now uses SHA256 stable hashing for cache keys involving file modification times (mtimes). This ensures that cache hits are reliable across different process runs and environments, provided the underlying file content hasn't changed.

- **Implementation**: The `@cached` decorator with `check_mtime=True` now generates a stable hash of the sorted `path:mtime` list.
- **Benefit**: Eliminates "cold starts" after restarting the CLI or moving to a different environment.

## 2. Persistent Storage (Pickle vs JSON)
We have migrated from JSON to **Pickle** for cache persistence.
- **Why?**: Pickle supports complex Python objects (sets, custom classes, numpy arrays) that JSON cannot handle without expensive serialization.
- **Migration**: The `CacheManager` automatically detects legacy `.json` caches and migrates them to the new `.pkl` format on first run.

## 3. Global Memory Budget
To prevent Out-Of-Memory (OOM) errors on systems with limited RAM, the `CacheManager` now enforces a global memory budget.

- **Dynamic Scaling**: The budget is automatically calculated based on available system RAM:
  - **>16GB RAM**: 2048 MB budget.
  - **>4GB RAM**: 512 MB budget.
  - **Low RAM**: 128 MB budget.
- **Proactive Eviction**: When the total size of all active caches exceeds the budget, the system triggers a global cleanup, evicting the least recently used items across *all* cache instances until usage drops to 80% of the budget.

## 4. Tuning Constants
Advanced users can tune the following constants in `cline_utils/dependency_system/utils/cache_manager.py`:

| Constant | Default | Description |
| :--- | :--- | :--- |
| `ENABLE_COMPRESSION` | `True` | Enables gzip compression for large cache items. |
| `COMPRESSION_THRESHOLD` | `10 MB` | Items larger than this will be compressed. |
| `COMPRESSION_MIN_SAVINGS` | `0.1` | Only compress if it saves at least 10% space. |
| `DEFAULT_MAX_SIZE` | `10000` | Max items per cache instance. |
| `_global_budget_mb` | `Auto` | Override for the global memory budget (in MB). |

## 5. Eviction Policies
Caches now support multiple eviction policies via the `EvictionPolicy` Enum:
- `LRU` (Default): Least Recently Used.
- `LFU`: Least Frequently Used.
- `ADAPTIVE`: Hybrid approach for dynamic workloads.
- `FIFO`: First In, First Out.
- `RANDOM`: For low-overhead cleanup.

## 6. Troubleshooting
- **Cache Invalidation**: Use `python -m cline_utils.dependency_system.dependency_processor clear-caches` if you suspect stale data.
- **Metrics**: Call `get_cache_stats(cache_name)` in Python to see hit rates and memory usage.
