# Cline Recursive Chain-of-Thought System (CRCT) - v8.4

Welcome to the **Cline Recursive Chain-of-Thought System (CRCT)**, a framework designed to manage context, dependencies, and tasks in large-scale Cline projects within VS Code. Built for the Cline extension, CRCT leverages a recursive, file-based approach with a modular dependency tracking system to maintain project state and efficiency as complexity increases.

- Version **v8.4**: **OPTIMIZATION & NATIVE VIZ** - High-Performance Scaling & Dependency Annotation
    - **Optimization**: Refactored dependency resolution to $O(M)$ complexity using global set sharing, eliminating redundant path reconstruction.
    - **Deterministic LLM Tasking**: Implemented monotonically decreasing context-size sorting for LLM resolution tasks to maximize VRAM efficiency and minimize model reloads.
    - **Native SVG Renderer (EXPERIMENTAL)**: Introduced a native Python-based SVG renderer for high-density dependency diagrams with intelligent routing avoidance. *Note: Currently experimental; edges may occasionally overlap nodes.*
    - **Automated Dependency Annotation**: New `populate_comments.py` utility for injecting "Station Headers" and "Connection Maps" directly into source code to enhance agent navigability.
    - **Agent Comment-Skill Suite**: Deployed a comprehensive suite of plugins and documentation for dependency-driven code comments across multiple languages.
    - **Comment-Aware Reporting**: Integrated a new `comment_index` scanner into the reporting pipeline to surface architectural metadata alongside code quality issues.
- Version **v8.3**: 🎨 **MODULAR ARCHITECTURE & STABLE CACHING** - Architectural Refinement & Reliability
    - **Modular Visualization Refactor**: The dependency visualization system has been decomposed into a dedicated `viz` sub-package, separating Mermaid DSL construction, image rendering, and layout configuration.
    - **Report Generator Decomposition**: Refactored the monolithic `report_generator.py` into specialized `scanner` and `reporting` packages for better maintainability and code quality analysis.
    - **Stable Persistent Caching**: Implemented SHA256-based stable hashing for cache keys, ensuring cache hits persist across different process runs.
    - **Pickle-based Storage**: Migrated from JSON to Pickle for more robust Python object serialization in persistent caches, with automatic legacy migration.
- Version **v8.2**: 🤖 **LOCAL LLM & DUAL-TOKEN EMBEDDINGS** - Automated Resolution & Precise Context
    - **Automated Placeholder Resolution**: New `resolve-placeholders` command uses local LLMs (via `llama-cpp-python`) to verify 'p' dependencies in batches, significantly reducing manual verification time and API costs.
    - **Dual-Token Schema**: Refactored metadata to track both `ses_tokens` (for embeddings) and `full_tokens` (for total file size), enabling smarter context window management.
    - **Enhanced Documentation Parsing**: Specialized SES extraction for structured docs, improving semantic search for requirements and design specs.
    - **Dependency-Aware Cache**: Cache invalidation now cascades through logical file dependencies, ensuring analysis consistency.
- Version **v8.1**: ⚡ **PERFORMANCE & STABILITY** - Batching and Caching Overhaul
    - **Tracker Batch Collection**: New `TrackerBatchCollector` system for atomic, high-performance tracker writes with rollback support.
    - **Advanced Caching**: Enhanced `@cached` decorator with dynamic file dependencies (`file_deps`), mtime-based invalidation, and automated path tracking.
    - **Optimized I/O**: Refactored `tracker_io` with local lookup caching to eliminate redundant file parsing during updates.
    - **Report Filtering**: `report_generator` now supports `EXCLUSION_PATTERNS` to eliminate false positives and groups multiple occurrences for cleaner analysis.
    - **Windows Path Consistency**: Standardized on Uppercase drive letters for improved compatibility across different Python environments.
- Version **v8.0**: 🚀 **MAJOR RELEASE** - Embedding & analysis system overhaul
    - **Symbol Essence Strings (SES)**: Revolutionary embedding architecture combining runtime + AST metadata for 10x better accuracy
    - **Qwen3 Reranker**: AI-powered semantic dependency scoring with automatic model download
    - **Hardware-Adaptive Models**: Automatically selects between GGUF (Qwen3-4B) and SentenceTransformer based on available resources
    - **Runtime Symbol Inspection**: Deep metadata extraction from live Python modules (types, inheritance, decorators)
    - **PhaseTracker UX**: Real-time progress bars with ETA for all long-running operations
    - **Enhanced Analysis**: Advanced call filtering, deduplication, internal/external detection
    - **Breaking Changes**: `set_char` deprecated, `exceptions.py` removed, new dependencies (`llama-cpp-python`), requires re-run of `analyze-project`. See [MIGRATION_v7.x_to_v8.0.md](MIGRATION_v7.x_to_v8.0.md)
- Version **v7.90**: Introduces dependency visualization, overhauls the Strategy phase for iterative roadmap planning, and refines Hierarchical Design Token Architecture (HDTA) templates.
    - **Dependency Visualization (`visualize-dependencies`)**:
        - Added a new command to generate Mermaid diagrams visualizing project dependencies.
        - Supports project overview, module-focused (internal + interface), and multi-key focused views.
        - Auto-generates overview and module diagrams during `analyze-project` (configurable).
        - Diagrams saved by default to `<memory_dir>/dependency_diagrams/`.
        - **NEW** integrated mermaid-cli to render dependency diagrams as .svg files. (experimental stage, subject to change in rendering process)
            - Performs well under 1000 edges to render, struggles with more than 1500 edges. Will reliably time-out with large 4000+ edge diagrams.
            - Requires additional dependency installation, should work via `npm install`
    - **Dependency Analysis and Suggestions**
        - Enhanced with python AST (for python)
        - Enhanced with tree-sitter (for .js, .ts, .tsx, .html, .css)
        - More to come!
    - **Strategy Phase Overhaul (`strategy_plugin.md`):**
        - Replaced monolithic planning with an **iterative, area-based workflow** focused on minimal context loading, making it more robust for LLM execution.
        - Clarified primary objective as **hierarchical project roadmap construction and maintenance** using HDTA.
        - Integrated instructions for leveraging dependency diagrams (auto-generated or on-demand) to aid analysis.
        - Refined state management (`.clinerules` vs. `activeContext.md`).
        - Split into Dispatch and Worker prompts to take advantage of new_task
    - **HDTA Template Updates**:
        - Reworked `implementation_plan_template.md` for objective/feature focus.
        - Added clarifying instructions to `module_template.md` and `task_template.md`.
        - Created new `roadmap_summary_template.md` for unified cycle plans.
- Version **v7.7**: Restructured core prompt/plugins, introduced `cleanup_consolidation_plugin.md` phase (use with caution due to file operations), added `hdta_review_progress` and `hierarchical_task_checklist` templates.
- Version **v7.5**: Significant baseline restructuring, establishing core architecture, Contextual Keys (`KeyInfo`), Hierarchical Dependency Aggregation, enhanced `show-dependencies`, configurable embedding device, file exclusion patterns, improved caching & batch processing.

---

## System Requirements

### Recommended (v8.0+)
- **VRAM**: 8GB+ (NVIDIA GPU) for optimal Qwen3-4B model performance
- **RAM**: 16GB+ for large projects
- **Disk**: 8GB+ for models and embeddings
- **Python**: 3.8+
- **Node.js**: 16+ (for mermaid-cli visualization)

### Minimum
- **RAM**: 4GB (CPU-only mode with reduced batch sizes)
- **Disk**: 500MB+ (lightweight models)
- **Python**: 3.8+

*The system automatically adapts to available hardware.*

---

## Key Features

- **Recursive Decomposition**: Breaks tasks into manageable subtasks, organized via directories and files for isolated context management.
- **Minimal Context Loading**: Loads only essential data, expanding via dependency trackers as needed.
- **Persistent State**: Uses the VS Code file system to store context, instructions, outputs, and dependencies. State integrity is rigorously maintained via a **Mandatory Update Protocol (MUP)** applied after actions and periodically during operation.
- **Modular Dependency System**: Fully modularized dependency tracking system.
- **Contextual Keys**: Introduces `KeyInfo` for context-rich keys, enabling more accurate and hierarchical dependency tracking.
- **Hierarchical Dependency Aggregation**: Implements hierarchical rollup and foreign dependency aggregation for the main tracker, providing a more comprehensive view of project dependencies.
- **Enhanced Dependency Workflow**: A refined workflow simplifies dependency management.
    - `show-keys` identifies keys needing attention ('p', 's', 'S') within a specific tracker.
    - `show-dependencies` aggregates dependency details (inbound/outbound, paths) from *all* trackers for a specific key, eliminating manual tracker deciphering.
    - `add-dependency` resolves placeholder ('p') or suggested ('s', 'S') relationships identified via this process. **Crucially, when targeting a mini-tracker (`*_module.md`), `add-dependency` now allows specifying a `--target-key` that doesn't exist locally, provided the target key is valid globally (known from `analyze-project`). The system automatically adds the foreign key definition and updates the grid, enabling manual linking to external dependencies.**
      *   **Tip:** This is especially useful for manually linking relevant documentation files (e.g., requirements, design specs, API descriptions) to code files within a mini-tracker, even if the code file is incomplete or doesn't trigger an automatic suggestion. This provides the LLM with crucial context during code generation or modification tasks, guiding it towards the intended functionality described in the documentation (`doc_key < code_key`).
    - `resolve-placeholders`: **(NEW in v8.2)** Automates the verification of unverified dependencies ('p') using a local LLM. This makes the resolution process significantly more efficient and less costly than using larger API-based models.
      *   Example: `python -m cline_utils.dependency_system.dependency_processor resolve-placeholders --tracker <path_to_tracker.md>`
    - `determine-dependency`: **(NEW in v8.2)** Performs a detailed local LLM analysis between two specific keys to verify their relationship with full reasoning.
      *   Example: `python -m cline_utils.dependency_system.dependency_processor determine-dependency --source-key <key> --target-key <key>`
   - **Dependency Visualization (`visualize-dependencies`)**: **(NEW in v7.8)**
    - Generates Mermaid diagrams for project overview, module scope (internal + interface), or specific key focus.
    - Auto-generates overview/module diagrams via `analyze-project`.
    - **NEW in v7.90** Now generates .svg image files for diagram visualization if the mermaid-cli dependency is installed.
- **Iterative Strategy Phase**: **(NEW in v7.8)**
    - Plans the project roadmap iteratively, focusing on one area (module/feature) at a time.
    - Explicitly integrates dependency analysis (textual + visual) into planning.
- **Refined HDTA Templates**: **(NEW in v7.8)**
    - Improved templates for Implementation Plans, Modules, and Tasks.
    - New template for Roadmap Summaries.
- **Configurable Embedding Device**: Allows users to configure the embedding device (`cpu`, `cuda`, `mps`) via `.clinerules.config.json` for optimized performance on different hardware. (Note: *the system does not yet install the requirements for cuda or mps automatically, please install the requirements manually or with the help of the LLM.*)
- **File Exclusion Patterns**: Users can now define file exclusion patterns in `.clinerules.config.json` to customize project analysis.
- **Automated Dependency Annotation**: **(NEW in v8.4)**
    - Automatically injects **Station Headers** (ROLE, LAYER, CRCT_KEY) at the top of source files.
    - Adds **Connection Maps** (dependency rails) before function and class definitions.
    - Preserves agent-authored prose in `[FILL: ...]` tags while refreshing architectural metadata.
- **Experimental Native SVG Visualization**: **(NEW in v8.4)**
    - A Python-native renderer for high-density dependency graphs.
    - Bypasses `mermaid-cli` overhead for local visualizations.
    - Supports grouped layouts and "pipe-bundling" for complex modules.
    - > [!CAUTION]
    - > **Experimental Status**: The native renderer is currently in an experimental state. Edge routing may occasionally overlap nodes or pass through unrelated labels. For mission-critical diagrams, the default Mermaid backend is recommended.
- **Code Quality Analysis**: **(NEW in v8.0, Improved in v8.1)**
    - **Report Generator**: A tool (`report_generator.py`) that performs AST-based code quality analysis.
    - **Incomplete Code Detection**: Identifies `TODO`, `FIXME`, empty functions/classes, and `pass` statements using robust Tree-sitter parsing.
    - **False Positive Filtering**: **(NEW in v8.1)** Added `EXCLUSION_PATTERNS` to filter out common false positives like `sql.Placeholder`.
    - **Condensed Reporting**: **(NEW in v8.1)** Groups multiple occurrences of the same issue in a file for more readable reports.
    - **Unused Item Detection**: Integrates with Pyright to report unused variables, imports, and functions.
- **Advanced Caching and Batching**: **(MAJOR UPDATE in v8.1)**
    - **Tracker Batching**: Collects all tracker updates in memory and writes them in a single operation, significantly reducing disk I/O.
    - **Atomic Updates**: Features rollback support to maintain tracker integrity if a batch write fails.
    - **Intelligent Invalidation**: Caches now automatically invalidate based on file modification times (mtime) and specific path arguments.
- **Modular Dependency Tracking**:
    - Utilizes main trackers (`module_relationship_tracker.md`, `doc_tracker.md`) and module-specific mini-trackers (`{module_name}_module.md`).
    - Mini-tracker files also serve as the HDTA Domain Module documentation for their respective modules.
    - Employs hierarchical keys and RLE compression for efficiency.
- **Automated Operations**: System operations are now largely automated and condensed into single commands, streamlining workflows and reducing manual command execution.
- **Phase-Based Workflow**: Operates in distinct phases: Set-up/Maintenance -> Strategy -> Execution -> Cleanup/Consolidation, controlled by `.clinerules`.
- **Chain-of-Thought Reasoning**: Ensures transparency with step-by-step reasoning and reflection.

---

## Quickstart

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/RPG-fan/Cline-Recursive-Chain-of-Thought-System-CRCT-.git
   cd Cline-Recursive-Chain-of-Thought-System-CRCT-
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   npm install  # For mermaid-cli visualization
   ```

3. **Set Up Cline or RooCode Extension**:
   - Open the project in VS Code with the Cline or RooCode extension installed.
   - Copy `cline_docs/prompts/core_prompt(put this in Custom Instructions).md` into the Cline Custom Instructions field. (new process to be updated)

4. **Start the System**:
   - Type `Start.` in the Cline input to initialize the system.
   - The LLM will bootstrap from `.clinerules`, creating missing files and guiding you through setup if needed.

*Note*: The Cline extension's LLM automates most commands and updates to `cline_docs/`. Minimal user intervention is required (in theory!)

---

## Project Structure

```
Cline-Recursive-Chain-of-Thought-System-CRCT-/
│   .clinerules/
│   .clinerules.config.json       # Configuration for dependency system
│   .gitignore
│   LICENSE
│   README.md
│   requirements.txt
│   package.json                  # Node dependencies for visualization
│
├───cline_docs/                   # Operational memory
│   │  activeContext.md           # Current state and priorities
│   │  changelog.md               # Logs significant changes
│   │  userProfile.md             # User profile and preferences
│   │  progress.md                # High-level project checklist
│   │
│   ├──backups/                   # Tracker backups
│   ├──CRCT_Documentation/        # Detailed v8.x Tech Guides
│   │    CHANGELOG.md             # Detailed version history
│   │    SES_ARCHITECTURE.md      # Embedding system deep-dive
│   │    Cache_System_Documentation.md
│   │    CACHE_TUNING.md          # Advanced cache config
│   │    DEPENDENCY_RESOLUTION.md # LLM-assisted resolution & prefetching
│   │    MIGRATION_v7.x_to_v8.0.md
│   │    VIZ_PACKAGE.md           # Modular visualization guide
│   │    REPORTING_SYSTEM.md      # Modular reporting guide
│   │    ...
│   ├──dependency_diagrams/       # Auto-generated diagrams
│   ├──prompts/                   # System prompts and plugins
│   │    core_prompt(put this in Custom Instructions).md
│   │    cleanup_consolidation_plugin.md
│   │    execution_plugin.md
│   │    setup_maintenance_plugin.md
│   │    setup_worker.md
│   │    strategy_dispatcher_plugin.md
│   │    strategy_worker_plugin.md
│   ├──templates/                 # Templates for HDTA documents
│   │    dispatcher_area_log_template.md
│   │    hdta_review_progress_template.md
│   │    hierarchical_task_checklist_template.md
│   │    implementation_plan_template.md
│   │    module_template.md
│   │    project_roadmap_template.md
│   │    roadmap_summary_template.md
│   │    structured_doc_template.md
│   │    system_manifest_template.md
│   │    task_template.md
│   │    worker_sub_task_output_template.md
│
├───cline_utils/                  # Utility scripts
│   └─dependency_system/
│     │ dependency_processor.py   # Dependency management orchestrator
│     ├──analysis/                # Analysis modules
│     │    dependency_analyzer.py
│     │    dependency_suggester.py
│     │    embedding_manager.py
│     │    local_llm_processor.py
│     │    project_analyzer.py
│     │    reranker_history_tracker.py
│     │    runtime_inspector.py
│     │    symbol_map_merger.py   # Merges runtime + AST symbol maps
│     ├──core/                    # Core logic and exceptions
│     │    dependency_grid.py     # Grid compression/decompression
│     │    exceptions_enhanced.py
│     │    key_manager.py         # Key generation & resolution
│     ├──io/                      # IO and tracker management
│     │    tracker_io.py          # Read/write tracker files
│     │    update_doc_tracker.py
│     │    update_main_tracker.py
│     │    update_mini_tracker.py
│     └──utils/                   # Utility modules
│          batch_processor.py      # Enhanced with PhaseTracker
│          cache_manager.py        # Persistent Pickle caching & stable hashing
│          config_manager.py
│          path_utils.py           # Path normalization utilities
│          phase_tracker.py
│          placeholder_resolver.py # LLM-assisted resolution
│          resource_validator.py
│          tracker_batch_collector.py
│          tracker_utils.py        # Tracker read/aggregate helpers
│          visualize_dependencies.py # Orchestrator for viz
│          viz/                    # Modular visualization package
│               mermaid_builder.py # DSL construction
│               renderer.py        # Image rendering (mmdc)
│               layout_config.py   # Mermaid styling
│
├───code_analysis/                # Code quality analysis
│   │ report_generator.py         # Orchestrator for reporting
│   ├──scanner/                   # Static & Runtime engines
│   │    static_engine.py
│   │    runtime_bridge.py
│   │    heuristics.py
│   └──reporting/                 # Export and formatting
│        json_exporter.py
│        markdown_formatter.py
│
├───docs/                         # Project documentation
├───models/                       # AI models (auto-downloaded)
└───src/                          # Source code root

```
*(Added/Updated relevant files/dirs)*

---

## Current Status & Future Plans

- **v8.4**: ⚡ **Optimization & Native Viz** - Refactored dependency resolution to $O(M)$, introduced experimental native SVG rendering, and deployed the automated comment-skill annotation system.
- **v8.3**: 🎨 **Modular Viz & Stable Caching** - Refactored visualization architecture into a dedicated `viz` package and implemented stable SHA256 hashing for reliable persistent caching.
- **v8.2**: 🤖 **Local LLM & Dual-Tokens** - Automated resolution of dependency placeholders and precise token-based context management.
- **v8.1**: ⚡ **Performance Optimization** - Introduced batch tracker updates and advanced caching mechanisms to handle larger projects with lower I/O overhead.
- **v8.0**: 🚀 **Major architecture evolution** - Symbol Essence Strings, Qwen3 reranker, hardware-adaptive models, runtime symbol inspection, enhanced UX with PhaseTracker. See [CHANGELOG.md](CHANGELOG.md) for complete details.
- **v7.8**: Focus on **visual comprehension and planning robustness**. Introduced Mermaid dependency diagrams (`visualize-dependencies`, auto-generation via `analyze-project`). Overhauled the Strategy phase (`strategy_plugin.md`) for iterative, area-based roadmap planning, explicitly using visualizations. Refined HDTA templates, including a new `roadmap_summary_template.md`.
- **v7.7**: Introduced `cleanup_consolidation` phase, added planning/review tracker templates.
- **v7.5**: Foundational restructure: Contextual Keys, Hierarchical Aggregation, `show-dependencies`, configuration enhancements, performance improvements (cache/batch).

**Future Focus**: Continue refining performance, usability, and robustness. v8.x series will focus on optimizing the new reranking and SES systems based on real-world usage. Future versions may include MCP-based tool use and transition from filesystem to database-focused operations.

Feedback is welcome! Please report bugs or suggestions via GitHub Issues.

---

## Getting Started (Optional - Existing Projects)

To test on an existing project:
1. Copy your project into `src/`.
2. Use these prompts to kickstart the LLM:
   - `Perform initial setup and populate dependency trackers.`
   - `Review the current state and suggest next steps.`

The system will analyze your codebase, initialize trackers, and guide you forward.

---

## Thanks!

This is a labor of love to make Cline projects more manageable. I'd love to hear your thoughts—try it out and let me know what works (or doesn't)!
