# Changelog

All notable changes to the Cline Recursive Chain-of-Thought System (CRCT) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-06-17

### Added
- **Unified JS/TS/TSX Analyzer**: One query-table-driven tree-sitter analyzer replaces the three copy-pasted JS/TS/TSX analyzers. It extracts import symbols (named, default, namespace, type; `require()` bindings; dynamic `import()`; `export … from` edges), every export form (including CommonJS), classes with methods and properties, interfaces, type aliases, enums, arrow-function consts, `extends`/`implements` heritage, type references and JSX component usage. Validated against the TypeScript compiler API on 322 files. Tests: `tests/test_js_ts_analyzer.py`, fixtures under `tests/fixtures/jsts/`.
- **TypeScript-Accurate Module Resolution** (`analysis/js_tsconfig.py`): JSONC tsconfig parsing with `extends` chains, project `references` and per-file tsconfig selection; `paths` matching as in TypeScript (an exact key wins, else the longest wildcard prefix; no `baseUrl` fallback once a `paths` pattern matched); `.js`/`.jsx`/`.mjs`/`.cjs` specifiers resolve to their TypeScript sources first; extended probe list (`.mts`, `.cts`, `.d.ts`, `index.*`). Tests: `tests/test_js_resolution.py`.
- **Upfront Bulk Passes For The Suggestion Phase**: Before the parallel workers start, `project_analyzer` builds a similarity index (`analysis/similarity_index.py`, every embedding loaded once into one matrix), resolves every JS/TS import and verifies every structural link (`prepare_js_structural_index`), indexes each source's tracker relations (`prepare_tracker_relations_index`) and the set of `x`/`n` pairs to skip. Workers only look results up; each step falls back to the per-file path on error. On an 820-file project the phase went from 1,433s to about 60s (about 3s when nothing needs reranking).
- **Test Guards For Live Resources**: Autouse fixtures stop tests from deleting or downloading into the real `models/` directory and from touching the live VRAM lock and registry in the temp directory.
- **One-Time Mutual-Dependency Reset (`reset-mutual-dependencies`)**: New command that resets every mutual (`x`) cell in every tracker listed in `tracker_map.json` (main, doc and mini) to the placeholder `p`. **Run it once after upgrading, before re-verifying, and do not repeat it afterwards (it would also wipe genuine `x` cells restored by re-verification)**; then re-verify with `analyze-project` and `resolve-placeholders`. Why: an earlier defect in `tracker_io.py` turned directional import edges into `x` in both cells of a pair, and the old JS/TS resolver linked to compiled `.js` twins. CRCT never overwrites `x` (the suggester skips pairs holding `x`/`n`, and the tracker write path keeps the higher-priority character), and trackers keep no per-cell provenance, so valid and bogus `x` cannot be told apart and the bad ones never self-heal. The command is a dry run by default (`--dry-run`, per-tracker and total counts plus examples); `--apply` applies all trackers in one `TrackerBatchCollector` commit (temp backups with rollback of the whole batch if any write fails; the collector's grid-consolidation pass is disabled, because a reset must not re-derive cells), backs up each tracker it changes (`backup_tracker_file`, plus a complete run snapshot of `.bak` copies under `<backups_dir>/x_reset_<timestamp>/`), rewrites it through the normal tracker writers, re-reads it to verify that only `x` cells changed, and restores the backup if that check fails. Only `x` cells change (both cells of a pair; never the diagonal `o`); `n`, `<`, `>`, `d`, `s`, `S` and `p` are left alone. Logic lives in `utils/reset_mutual.py`; tests in `tests/test_reset_mutual.py`.

- **Malformed Transparency Recovery Guards**: Added regression coverage for malformed transparency metadata, including malformed ranges, invalid `start_line`, invalid/missing `total_lines`, and malformed connection-map metadata.
- **Centralized Reciprocal Dependency Helpers**: Added shared reciprocal dependency helpers for building reciprocal suggestions and applying reciprocal grid characters in a single batch pass.
- **KV-Cache Source Grouping Tests**: Added coverage for placeholder resolver KV-cache source grouping behavior.
- **Dependency Ignore Rules**: Ignored local `.kilo/` state and generated dependency-system cache files.

### Changed
- **One Suggestion Workflow For Every Language**: `analysis/structural_engine.py` gives each language a small adapter (how its imports, links or references resolve and which uses are verified); the upfront pass, per-file lookup, semantic step, character-priority combination and skip filter are shared. The per-language `suggest_*` functions remain as thin wrappers. Output is unchanged, verified file by file on an 834-file JS/TS project and a 557-file Python/SQL project. SQL's table map is now built once per run instead of once per SQL file. The Python analyzer records `import_bindings` (every import and the names it binds); the analysis cache key gains a schema version, so every file is re-analysed once after upgrading (no re-embed).
- **Embedding Model Is Never Deleted For A Failed Load Test**: Only a structurally corrupt GGUF (too small or wrong header) is removed and re-downloaded; an intact model that fails to load (busy VRAM, drivers) is kept and embedding updates are skipped for that run so existing Qwen3 embeddings on disk are preserved. A relative `qwen3_model_path` resolves against the project root.
- **SES-Aware Embedding Invalidation**: Each embedding records the hash of the exact text it was built from (`ses_hash`); files whose Symbol Essence String changed are re-embedded even when the source is unchanged. The first run after upgrading re-embeds everything once.
- **`tracker_map.json` Is Authoritative**: Tracker reads, updates and consolidation are limited to the trackers listed in `tracker_map.json`; mini-tracker discovery honours `excluded_dirs` (including dot-directories such as `.agents/`).
- **Structured Debug Output**: Per-item DEBUG lines (Python import-ledger hits and resolution attempts, structural verified/not-verified items, JS verification, per-candidate semantic assignments, about 15 lines per tracker update plus one per ignored grid cell, per-file cache invalidations) are replaced by one line per file (`Suggest <file> [adapter]: edges N (by reason) + semantic M (by character) -> K kept, S skipped`) and one per tracker update (`Tracker summary <file>: type= keys= suggestions= applied= kept_existing= key_delta= ...`). `debug.txt` shrinks by roughly an order of magnitude; output is unchanged.
- **Reranker Backpressure Pause Removed**: The separate pause at the top of the batch loop waited up to 20s and then continued regardless; the VRAM allocation request is the gate.
- **Opposed `<` / `>` Are One Relationship, Not A Mutual `x`**: `[A][B] = '<'` ("A depends on B") and `[B][A] = '>'` (the column requires the row) are the same relationship seen from the two files, so an opposed pair must not become `x`. A mutual `x` needs both files to report the same character from their own side in the current pass. Three places changed:
  - `combine_suggestions_path_based_with_char_priority` no longer merges an opposed `<` and `>` for the same target into `x`; at equal priority it keeps the existing character.
  - `_build_ast_map` no longer merges a forward link and the inverse of a reverse link into `x`; the file's own (forward) link wins in either order, which also removes the order dependence. A real mutual import still gives `x`: each file reports `<` toward the other and the grid upgrade in `tracker_io` forms the `x`.
  - `tracker_io.update_tracker` mirror-pair rule: the `<`/`<` or `>`/`>` upgrade to `x` now also requires the reverse cell's same arrow to be among the suggestions of the current pass. An arrow left in the reverse cell by an earlier run (possibly stale or inverted, e.g. `types.ts` ↔ `useAutomatedTasks.ts`) no longer counts. Forced / manual applies (`force_apply_suggestions`, e.g. `add-dependency`) keep the old rule, and existing `x` cells are unaffected. A forced edit that writes the opposite arrow into a cell already holding an arrow makes the pair `x`: it adds the other direction to the one on record. Unforced suggestions never do this.
  - Not changed: the module-level `<` + `>` → `x` merges in `update_main_tracker.py`. They merge characters that are all read from the row module's side (different file pairs), so they read as "M depends on N" plus "N depends on M".
- **Transparency Recovery Hardening**: `TransparencyManager.recover_alignment()` now validates required line metadata before use, derives original line counts from section metadata when possible, safely ignores malformed optional connection metadata, and locks unrecoverable entries.
- **Transparent Read Failure Handling**: `read_file_transparently()` now locks the transparency entry and returns `None` metadata when recovery raises an exception.
- **Public Transparency Context Manager**: Renamed/exposed `TransparencyManager.update_context()` and updated dependency-system callers away from the private `_update_context()` name.
- **Reciprocal Dependency Consolidation**: Moved reciprocal dependency application out of duplicated call-site logic and into `TrackerBatchCollector`, with `handle_add_dependency()` using the shared suggestion helper.
- **Placeholder Resolver KV-Cache Pinning**: Placeholder resolution now tracks source-key boundaries and pins/restores local LLM KV-cache state within contiguous source groups.
- **Reranker Load Handling**: Reranker loading now unconditionally sets the loaded model to eval mode and logs CUDA memory footprint after load.

### Fixed
- **Tracker Writes Failing While Another Process Holds The File**: Main and doc trackers were truncated and rewritten in place, which on Windows fails with `[Errno 22] Invalid argument` while an antivirus scan or indexer briefly holds the file. All tracker writes now go to a temp file beside the tracker and are swapped in, retrying for about 1.5s on transient lock errors, so a failure can never leave a partial file. A failed write in a batch is now reported as left unchanged, and the batch summary says how many of its trackers failed (it previously claimed a rollback and full success).
- **Editor Freezes During analyze-project**: The terminal progress bar redrew on every item (hundreds of times a second in fast phases, enough to freeze the VS Code integrated terminal and window); it now redraws at most 10 times a second and always shows the final state. Embedding model selection no longer load-tests the Qwen3 GGUF on the GPU on every run (a multi-GB VRAM spike even when nothing needed embedding); it checks the file structurally, and the model is loaded only when embeddings are generated, skipping embedding updates for the run if that load fails so existing Qwen3 embeddings are preserved.
- **Python Relative Imports Missed By The Structural Step**: The structural import map climbed one directory too many for relative imports (`from . import x` resolved in the parent package), so calls, attributes and type hints through relative imports never produced structural links. They now resolve from the file's own package, as the explicit-import step does.
- **Svelte Script Imports Never Resolved**: Svelte imports were read from a `source` key that import normalization renames to `path`; no Svelte file produced import edges. They are now read from `path`.
- **Stale Python Import Maps**: The Python structural step's import map was cached by file path only, persisted to disk and never expired or invalidated, so after a file's imports changed its structural links kept resolving through the old map (only `--force-analysis` cleared it). It is now built from the file's current analysis on every run.
- **Reranker CUDA Out-Of-Memory**: The reranker forward pass now requests last-position logits only (`logits_to_keep=1`). It used to materialize batch × tokens × 151,669 fp16 logits (7.4 GB for 7 items of 3.7k tokens), which the VRAM batch estimate did not budget for.
- **Reranker Never Invents Scores**: A batch that cannot get a VRAM grant (retried in halves until a 600s deadline) or runs out of memory (retried in halves down to one item) leaves its candidates unscored instead of scoring them 0.0, and an unavailable reranker no longer scores every candidate 1.0 (which promoted all of them to arrows). Only complete results are cached (new `rerank2:` key, so older cached results are never read), only scored pairs are logged to `reranker_scans.jsonl` / history, and after one deadline the rest of the run skips reranking instead of waiting again.
- **Semantic Character Leaked Between Candidates**: `assigned_char_semantic` was never reset per candidate, so a candidate below every threshold inherited the previous candidate's `s`, `S` or arrow. It is now reset in both the reranked and the similarity loop.
- **VRAM Lock Stalls**: Fixed a lock-order inversion (the availability query held `_allocation_lock` while waiting for the cross-process lock that `request_allocation` held while waiting for `_allocation_lock`), made threads queue on a per-path in-process lock instead of polling the lock file, moved every CUDA call (including `torch.cuda.synchronize()`) out of the locked sections, and read total VRAM once.
- **Cache Expiry Check Flushed On Every Lookup**: `Cache.is_expired()`, called by `get_cache()` on every cached call, flushed all of L1 each time (~40µs per call). It now checks both layers without flushing (~1.8µs).
- **Transparency Symbol Map Re-Parsed Per File**: `_load_symbol_map_for_transparency` checked `get_cache(...)` (a `Cache` object) for a dict, never matched, and re-parsed the whole `project_symbol_map.json` for every file during connection-map virtualization (~115s per run on 702 files). It is now parsed once per file version.
- **Repeated Per-File Work In The Semantic Step**: tracker discovery and reading, per-target skip checks and `metadata.json` parsing ran once per file; they now run once per run (per-file CPU 0.53s → 0.018s).
- **SQL Inverse Dependencies Ignored Self-Defined Tables**: `suggest_sql_dependencies` skipped tables a file defines itself for its forward `<` edges but not for the inverse `>` edges. Two files that both mention a shared table (e.g. a migration that `ALTER`s `users` and the schema that defines it) each produced `<` (it references our other tables) and `>` (inverse) for the same target, and the combine step merged them into `x` (it no longer merges, see above; the source no longer emits both). The inverse now excludes tables the other file defines itself. On the project's own symbol map the only SQL `x` (`add_foreign_key_constraints.sql` → `phase3_schema.sql`) becomes `<` with the reverse `>`.
- **Reciprocal Application Duplication**: Removed duplicated reciprocal dependency propagation from `update_tracker()` and `handle_add_dependency()` paths by centralizing it in shared helpers.
- **Transparency Metadata Drift Crashes**: Prevented malformed transparency metadata from crashing recovery and ensured failed recovery invalidates the affected entry.

## [8.6.0] - 2026-05-21

### Added
- **Platform-Agnostic Cross-Process Locking**: Implemented a robust file-based lock (`CrossProcessLock`) leveraging standard Windows `msvcrt` and POSIX `fcntl` APIs, handling 0-byte locking safely.
- **Coordinated VRAM Allocation Registry**: Created a dynamic, shared registry (`vram_registry.json`) under the cross-process lock to prevent resource contention and OOMs across concurrent worker processes, including automatic process liveness checks and zombie allocation pruning.
- **Comprehensive Unit Testing**: Added robust test suites (`test_calculate_hash.py`, `test_cross_process_lock.py`, and `test_vram_process_coordination.py`) covering all new hashing and locking mechanics.

### Changed
- **Stable Content Hashing Normalization**: Overhauled the hash generation logic in `calculate_hash.py` to strip and normalize carriage returns, varying indent styles, comment prefixes, and empty lines, ensuring stable hashes across development environments.
- **Transactional Bulk Operations**: Optimized `TransparencyManager` and `dependency_processor` by introducing batch marker transactions (`bulk_restore_markers`, `bulk_remove_markers`), running all alignments and checks in memory and writing to disk exactly once at the end of the transaction.
- **Parallel Embedding Preparation**: Accelerated `embedding_manager.py` by multithreading file reads, Symbol Essence generation, and tokenization (up to 16 threads), and integrated transparency batching to bypass lock overhead.
- **Calculated Path Migration Pass-Through**: Passed down pre-computed `path_migration_info` straight through `PlaceholderResolver` and project analyzer to save redundant computational passes.

## [8.5.0] - 2026-05-17

### Added
- **Precise Symbol-Level Connection Maps**: Connection maps now track precise target symbols and 1-indexed definition lines rather than just file-level mappings, formatted as `KEY(symbol:line) {char}`.
- **Low-Value Signal Filtration**: Implemented filtering using `_LOW_VALUE_SYMBOL_NAMES` to exclude generic standard built-ins and logger names, focusing maps on actionable business logic dependencies.
- **Comprehensive Unit Testing**: Added `test_precise_connection_maps.py` with 10 robust test cases covering parsing, symbol resolving, filtering, virtualization, and dry-run code paths.

### Changed
- **Comment Virtualization**: Upgraded `TransparencyManager` to extract, parse, and store connection maps inside `transparency_registry.json`, stripping physical comments on disk to keep working trees perfectly clean.
- **Embedding Integration**: Integrated the transparency layer with `embedding_manager.py` to transparently load stripped code contents and append virtual connection maps to the Symbol Essence Strings (SES) to enrich embedding search quality.
- **Centralized Hash Computation**: Moved all content hash calculations into `cline_utils/dependency_system/utils/calculate_hash.py` to prevent cyclic imports.
- **Refactored Caching Patterns**: Replaced inline lambda key functions inside decorators with stable named function key generators.

---

## [8.4.0] - 2026-05-08

### Added
- **Automated Dependency Annotation**: New `populate_comments.py` utility for injecting Station Headers and Connection Maps into source files.
- **Experimental Native SVG Renderer**: Python-native dependency visualization with intelligent routing and pipe-bundling.
- **Comment-Aware Reporting**: Integrated `comment_index` scanner into the report generation pipeline.
- **Agent Comment-Skill Suite**: Comprehensive documentation and plugins for dependency-driven code comments.

### Changed
- **Deterministic LLM Tasking**: Optimized placeholder resolution by sorting tasks by descending context size to maximize VRAM efficiency.
- **Improved Reporting**: Enhanced `report_generator` to surface architectural metadata from the new comment system.

### Performance
- **Optimization**: Refactored dependency resolution to $O(M)$ complexity using global set sharing and batched directory resolution.
- **Global Path Sharing**: Eliminated redundant path set reconstruction across the dependency resolution loop.

---

## [8.3.0] - 2026-05-03

### Added
- **Modular Visualization Package**: Extracted dependency visualization logic into a dedicated `cline_utils.dependency_system.utils.viz` sub-package.
- **Report Generator Modularization**: Refactored monolithic `report_generator.py` into specialized `code_analysis.scanner` and `code_analysis.reporting` packages.
- **Stable Hashing**: SHA256-based stable hashing for cache keys to ensure persistence across sessions and environments.
- **Automatic Cache Migration**: Automatic migration of legacy `.json` caches to the new `.pkl` format.
- **LLM-assisted Placeholder Resolution**: New `PlaceholderResolver` leveraging local LLMs for intelligent dependency verification.

### Changed
- **Persistent Caching Overhaul**: Switched from JSON to Pickle for more robust Python object serialization in `CacheManager`.
- **Refactored Visualization**: Decomposed monolithic `visualize_dependencies.py` into specialized modules for DSL building, rendering, and configuration.
- **Improved Cache Key Generation**: Integrated stable mtime hashing and file dependency tracking in the `@cached` decorator.
- **Multi-threaded Prefetching**: Optimized dependency resolution loop to overlap CPU tasks with GPU inference.
- **VRAM Footprint Correction**: Adjusted reranker model memory usage estimates to 0.7GB for better OOM prevention.
- **Global Cache Budget**: Implemented system-aware memory limits for caches to optimize resource usage.

### Fixed
- **Documentation Audit & Reconciliation**: Conducted a comprehensive audit of all project documentation (`README.md`, `DEPENDENCY_RESOLUTION.md`, `VIZ_PACKAGE.md`, `REPORTING_SYSTEM.md`) to eliminate hallucinations and align with the v8.3 modular architecture.

---

## [8.2.0] - 2026-02-14

### Added
- **Local LLM Dependency Resolution**: New commands `resolve-placeholders` and `determine-dependency` for automated verification of 'p' placeholders using GGUF models.
- **Dual-Token Metadata Schema**: Implementation of `ses_tokens` and `full_tokens` for precise context window management.
- **Structured Documentation Support**: Parser for `structured_doc_template.md` to enhance semantic embedding quality for requirements and design files.
- **LOCAL_LLM_GUIDE.md**: New handbook for local LLM resolution and command usage.

### Changed
- **Dependency-Aware Cache Invalidation**: Cascading invalidation based on logical file dependencies to ensure analysis correctness.
- **Enhanced Document Analysis**: Improved SES extraction logic for markdown/text files.

---

## [8.1.0] - 2026-01-20

### Added
- **Tracker Batch Collection**: `TrackerBatchCollector` for atomic, high-performance tracker writes with rollback support.
- **Report Filtering**: `EXCLUSION_PATTERNS` in `report_generator` to eliminate false positives in code quality analysis.

### Changed
- **Advanced Caching**: Enhanced `@cached` decorator with dynamic file dependencies and mtime-based invalidation.
- **Optimized I/O**: Refactored `tracker_io` with local lookup caching.
- **Windows Path Consistency**: Standardized Uppercase drive letters for cross-environment compatibility.

---

## [8.0.0] - 2025-12-02

> [!IMPORTANT]
> **MAJOR RELEASE** - Significant architectural changes to embedding and dependency analysis systems. See [MIGRATION_v7.x_to_v8.0.md](MIGRATION_v7.x_to_v8.0.md) for upgrade instructions.

### 💥 Breaking Changes

- **Embedding System Rewrite**: Migrated from simple content-based to Symbol Essence String (SES) architecture
  - Embeddings now include runtime type info, inheritance, decorators, and comprehensive symbol metadata
  - **Action Required**: Regenerate all embeddings with `analyze-project --force-embeddings`

- **New Dependencies**: Added `llama-cpp-python` and `huggingface_hub`
  - Required for GGUF model support and automatic model downloads
  - **Action Required**: Run `pip install -r requirements.txt`

- **Runtime Symbol Inspection**: Requires valid, importable Python modules
  - Syntax errors in project files may prevent symbol extraction
  - **Action Required**: Fix syntax errors before running `analyze-project`

- **CLI Deprecation**: `set_char` command is now **unsafe** and deprecated
  - Operates on outdated grid structure and can corrupt tracker files
  - **Action Required**: Use `add-dependency` with `--source-key` and `--target-key` instead

### 🎯 Major Features

#### Symbol Essence Strings (SES) - Revolutionary Embedding Architecture
- Constructs rich, structured embeddings from runtime + AST analysis
- Includes: type annotations, inheritance hierarchies, method resolution order, decorators, docstrings, import graphs, call relationships
- Configurable max length (default 4000 chars, supports up to 32k)
- Dramatically improved semantic understanding for dependency suggestions

#### Qwen3 Reranker Integration - AI-Powered Dependency Scoring
- Integrated ManiKumarAdapala/Qwen3-Reranker-0.6B-Q8_0 for semantic reranking
- Specialized instructions for doc↔doc, doc↔code, code↔code relationship types
- Automatic model download with progress tracking (~600MB)
- Global scan limiter for performance control
- VRAM management with automatic model unloading
- Score caching with 7-day TTL

#### Hardware-Adaptive Model Selection - Intelligent Resource Management
- Automatic detection of CUDA VRAM and system RAM
- Multi-model support:
  - **GGUF**: Qwen3-Embedding-4B-Q6_K (for systems with ≥8GB VRAM or ≥16GB RAM)
  - **SentenceTransformer**: all-mpnet-base-v2 (for lower-end systems)
- Dynamic batch size optimization (32-256 based on available VRAM)
- Context length up to 32,768 tokens for large files

#### Runtime Symbol Inspection - Deep Metadata Extraction
- **NEW MODULE**: `runtime_inspector.py` - Extracts type annotations, inheritance, MRO, closures, decorators from live Python modules
- **NEW MODULE**: `symbol_map_merger.py` - Merges runtime data with AST analysis for comprehensive symbol maps
- Generates `project_symbol_map.json` combining best of both approaches
- Validation with categorized issue reporting

#### Enhanced Dependency Analysis - Smarter, More Accurate
- Advanced call filtering: Filters 20+ generic methods, resolves import aliases with `_is_useful_call()`
- Internal vs external module detection
- Call result deduplication and consolidation
- Improved accuracy with reduced false positives
- AST-verified link extraction with structured metadata

### ✨ Enhancements

#### User Experience
- **PhaseTracker**: Real-time progress bars with ETA for long-running operations
  - Clean terminal output (no more scrolling spam)
  - Accurate time estimates based on processing rate
  - Graceful TTY vs non-TTY handling
- Reduced console verbosity (info → debug for routine operations)
- Better progress reporting throughout analysis
- Detailed debug logs still available with verbose mode

####Performance
- Optimal batch sizing (32-256) based on hardware
- Reranker model unloading after suggestions to free VRAM
- Smart caching for reranker scores (7-day TTL)
- Parallel processing with shared scan counter for global limits
- Cache compression for items >1KB with 10% minimum savings

#### Data Quality
- AST link consolidation (merges duplicate links, combines reasons)
- Expanded symbol map coverage (16 symbol categories vs 5 in v7.x)
- Runtime + AST merging for richer metadata
- Only stores non-empty symbol data
- Enhanced validation with categorized reporting

#### Caching System (cache_manager.py)
- **NEW**: Compression support with gzip for large cache items
- **NEW**: Multiple eviction policies (LRU, LFU, FIFO, Random, Adaptive)
- **NEW**: Enhanced metrics with CacheMetrics dataclass
  - Hit rate calculation
  - Access count tracking
  - Memory usage estimation
- **NEW**: Smart persistence with JSON-safe serialization
- Improved size estimation for cache entries
- Compression threshold: 1KB minimum, 10% savings required

#### Configuration System (config_manager.py)
- **NEW**: Reranker threshold settings
  - `reranker_promotion_threshold`: 0.92 (promotes to `<`)
  - `reranker_strong_semantic_threshold`: 0.78 (assigns `S`)
  - `reranker_weak_semantic_threshold`: 0.65 (assigns `s`)
- **NEW**: Embedding configuration options
  - `batch_size`: Auto-sizing or manual override
  - `max_context_length`: Up to 32,768 tokens
  - `auto_select_model`: Hardware-adaptive selection
- **NEW**: Resource management settings
  - `min_memory_mb`, `recommended_memory_mb`
  - `min_disk_space_mb`, `min_free_space_mb`
  - `max_workers`, `cpu_threshold`
- **NEW**: Analysis controls
  - Binary detection settings
  - Docstring extraction toggles
  - Min function/class lengths
- **NEW**: Resource validation method
  - `perform_resource_validation_and_adjustments()`
  - Pre-flight system checks with recommendations

### 🧪 Testing & Quality

- **NEW**: Comprehensive test suite (4 test files)
  - `test_functional_cache.py` - Cache functionality tests
  - `test_integration_cache.py` - Integration testing
  - `test_manual_tooling_cache.py` - Manual tooling verification
  - `verify_rerank_caching.py` - Reranker cache validation
- Enhanced exception handling system (`exceptions_enhanced.py` - 261 lines vs 27 in old `exceptions.py`)
- More specific, actionable exception types

### 🔧 Developer Tools

- **NEW**: `report_generator.py` - AST-based code quality analysis
  - Detects incomplete code using Tree-sitter
  - Supports Python, JavaScript, TypeScript
  - Integrates with Pyright for type checking
- **NEW**: `resource_validator.py` - Pre-analysis system validation
  - Validates memory, disk, CPU before analysis
  - 7-day cache with TTL for validation results
  - Generates optimization recommendations
- **NEW**: `phase_tracker.py` - Terminal progress bars with ETA
  - Context manager for clean progress tracking
  - Real-time ETA calculations
  - Improved user experience for long operations
- Improved error messages with detailed context
- Validation tools for merged symbol maps

### 📦 Internal Improvements

- Thread-safe model loading with locks
- Graceful model download with progress reporting
- GGUF model validation (size checks, format verification)
- Configurable context lengths and batch sizes
- Better memory management across the board
- Module-level cache for AST trees (ast_cache)
- Enhanced logging with structured context
- Parser architecture change for thread safety (local parsers vs global)

### 📊 Performance Metrics

- **Embedding Generation**: 2-3x faster with optimal batch sizing
- **Dependency Suggestions**: 5-10x more accurate with reranker
- **Analysis Time**: Similar or slightly slower on first run (runtime inspection overhead), faster on subsequent runs (better caching)
- **Memory Usage**: Higher peak during reranker operations, better managed with unloading
- **Cache Efficiency**: 30-50% memory savings with compression for large projects

### ⚠️ Known Issues

- Reranker may timeout on very large dependency graphs (4000+ edges) - use visualization sparingly
- Runtime inspection requires importable modules (fix syntax errors first)
- GGUF model download requires stable internet connection (600MB)
- First-run analysis slower due to model downloads and SES generation complexity

### 🐛 Bug Fixes

- Fixed parser state conflicts with local parser instances (vs global)
- Improved call filtering to reduce noise in suggestions
- Better handling of relative imports in Python
- Enhanced error recovery in runtime inspection
- Resolved cache key collisions with improved hashing

### 🗑️ Removed

- **DEPRECATED**: `exceptions.py` → Replaced by `exceptions_enhanced.py`
  - Migration: Update imports from `core.exceptions` to `core.exceptions_enhanced`
  - Old exception classes are still available for backward compatibility

### 📝 Documentation

- **NEW**: [MIGRATION_v7.x_to_v8.0.md](MIGRATION_v7.x_to_v8.0.md) - Comprehensive migration guide
- Updated README with v8.0 features and system requirements
- Enhanced inline documentation throughout codebase

### 🔄 Migration Notes

1. **Install New Dependencies**:
   ```bash
   pip install -r requirements.txt
   npm install  # for mermaid-cli if using visualization
   ```

2. **Regenerate Embeddings**:
   ```bash
   python -m cline_utils.dependency_system.dependency_processor analyze-project --force-embeddings
   ```

3. **Run Runtime Inspector** (if using Python projects):
   ```bash
   python -m cline_utils.dependency_system.analysis.runtime_inspector
   ```

4. **Expected First-Run Behavior**:
   - Automatic download of Qwen3 reranker model (~600MB)
   - Longer initial embedding generation (due to SES complexity)
   - Runtime inspection may fail on files with syntax errors

5. **System Requirements Update**:
   - **Recommended**: 8GB+ VRAM or 16GB+ RAM for optimal performance
   - **Minimum**: 4GB RAM for CPU-only mode with reduced batch sizes

---

## [7.90] - 2024-11-XX

### Added
- **Dependency Visualization** (`visualize-dependencies` command)
  - Generate Mermaid diagrams for project overview, module-focused, and multi-key views
  - Auto-generates overview and module diagrams during `analyze-project`
  - Integrated mermaid-cli to render diagrams as .svg files
- Enhanced dependency analysis with tree-sitter support for .js, .ts, .tsx, .html, .css
- Strategy Phase overhaul with iterative, area-based workflow

### Changed
- Improved AST analysis for Python files
- Refined state management (`.clinerules` vs. `activeContext.md`)
- Split strategy into Dispatch and Worker prompts

### Fixed
- Diagram rendering performance (works well under 1000 edges, struggles with 1500+, times out with 4000+)

---

## [7.7] - 2024-XX-XX

### Added
- Restructured core prompt/plugins
- `cleanup_consolidation_plugin.md` phase (use with caution)
- `hdta_review_progress` and `hierarchical_task_checklist` templates

---

## [7.5] - 2024-XX-XX

### Added
- Significant baseline restructuring
- Core architecture establishment
- Contextual Keys (`KeyInfo`) system
- Hierarchical Dependency Aggregation
- Configurable embedding device
- File exclusion patterns

### Changed
- Enhanced `show-dependencies` command
- Improved caching & batch processing

---

[8.0.0]: https://github.com/your-repo/compare/v7.90...v8.0.0
[7.90]: https://github.com/your-repo/compare/v7.7...v7.90
[7.7]: https://github.com/your-repo/compare/v7.5...v7.7
[7.5]: https://github.com/your-repo/releases/tag/v7.5
