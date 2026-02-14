# Changelog

All notable changes to the Cline Recursive Chain-of-Thought System (CRCT) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
