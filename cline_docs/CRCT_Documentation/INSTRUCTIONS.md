# INSTRUCTIONS.md

## Cline Recursive Chain-of-Thought System (CRCT) - v8.4 Instructions

These instructions provide a guide to setting up and using the Cline Recursive Chain-of-Thought System (CRCT) v8.4. This system is designed to enhance the Cline extension in VS Code by providing robust context and dependency management for complex projects through advanced algorithmic analysis and LLM-assisted verification.

---

## Prerequisites

- **VS Code**: Installed with the Cline extension.
- **Python**: 3.8+ with `pip`.
- **Git**: To clone the repo.

---

## Step 1: Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RPG-fan/Cline-Recursive-Chain-of-Thought-System-CRCT-.git
   cd Cline-Recursive-Chain-of-Thought-System-CRCT-
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Includes `sentence-transformers` for embeddings.*

3. **Open in VS Code**:
   - Launch VS Code and open the `cline/` folder.

4. **Configure Cline**:
   - Open the Cline extension settings.
   - Paste the contents of `.clinerules/core_prompt(put this in Custom Instructions).md` into the "Custom Instructions" field.

---

## Step 2: Initialize the System (v8.4+)

1. **Start the System**:
   - In the Cline input, type `Start.` and run it.
   - The LLM will perform these initialization steps:
     - Read `.clinerules` to determine the current phase.
     - Load the corresponding phase plugin (e.g., `Set-up/Maintenance`).
     - Initialize core files in `cline_docs/`, including tracker files and context documents.

2. **Automated Analysis**:
   - Run: `analyze-project`
   - This command now handles the entire "Grounding" process:
     - Scans `[CODE_ROOT_DIRECTORIES]` and `[DOC_DIRECTORIES]`.
     - Generates/Updates the `KeyMap` (Contextual Keys).
     - Populates all trackers (`module_relationship_tracker.md`, `doc_tracker.md`, and mini-trackers).
     - Generates Symbol Essence Strings (SES) and embeddings.

3. **Verify Setup**:
   - Check `cline_docs/` for new files (e.g., `doc_tracker.md`).
   - Ensure `[CODE_ROOT_DIRECTORIES]` in `.clinerules` lists `src/` (edit manually if needed).

---

## Step 3: Populate Dependency Trackers

1. **Run Initial Setup**:
   - Input: `Perform initial setup and populate dependency trackers.`
   - The LLM will:
     - Identify code root directories (e.g., `src/`).
     - Identify documentation directories (e.g., `docs/`)
     - Generate `module_relationship_tracker.md`, `doc_tracker.md`, and all mini-trackers using `dependency_processor.py`.
     - Suggest and validate module dependencies.

2. **Resolve Placeholders (Bolt Pass)**:
   - Run: `resolve-placeholders --limit <N>`
   - The system utilizes the **Bolt Optimization** ($O(M)$ complexity) to algorithmically resolve directory-level and structural dependencies.
   - For remaining ambiguous dependencies, the LLM processes tasks in **Deterministic Order** (sorted by context size from high to low) to optimize VRAM and prevent redundant model reloads.

3. **Comment-Skill Integration**:
   - After resolution, the system automatically triggers `populate_comments`.
   - This utility injects **Station Headers** and **Connection Maps** directly into source files, providing agent-navigable infrastructure that stays in sync with the trackers.

---

## Step 4: Plan and Execute

1. **Enter Strategy Phase**:
   - Once trackers are populated, the LLM will transition to `Strategy` (check `.clinerules`).
   - Input: `Plan the next steps for my project.`
   - Output: New instruction files in `strategy_tasks/` or `src/`.

2. **Execute Tasks**:
   - Input: `Execute the planned tasks.`
   - The LLM will follow instruction files, update files, and apply the MUP.

---

## Tips

- **Monitor `activeContext.md`**: Tracks current state and priorities.
- **Check `.clinerules`**: Shows the current phase and next action.
- **Debugging**: If stuck, try `Review the current state and suggest next steps.`

---

## Using CRCT v8.4

1. **Understanding Phases**:
   - CRCT operates in three distinct phases, controlled by the `.clinerules` file:
     - **Set-up/Maintenance**: For initial setup, project configuration, and ongoing maintenance of the system's operational files and dependency trackers. Key operations in this phase include identifying code and documentation roots, and running `analyze-project` for initial setup and after significant codebase changes.
     - **Strategy**: Phase focused on planning, task decomposition, and creating detailed instructions. This phase leverages dependency information gathered in the Set-up/Maintenance phase to inform strategic planning.
     - **Execution**: For carrying out tasks, modifying code, and implementing planned strategies. This phase relies on the context and plans developed in the Strategy phase.
   - The `current_phase` in `.clinerules` dictates the active operational mode and the plugin loaded by the system.
   *Note: The initial Set-up phase will likely take quite some time if used with a large pre-existing project, as the LLM will need to create the necessary support documentation for all files and folders and populate them with your project specific details. It is suggested that you divide the workload among multiple tasks that concentrate on creating/populating the documentation in a single system at a time. Perform a validation pass to confirm all details are sufficiently covered.*

  **Phase Transition Checklists:**

  Before transitioning between phases, ensure the following checklists are complete:

  - **Set-up/Maintenance → Strategy**:
    - Confirm `doc_tracker.md` and `module_relationship_tracker.md` in `cline_docs/` have no 'p' placeholders (all dependencies verified via Bolt Pass or LLM resolution).
    - Ensure `activeContext.md` and `project_roadmap.md` are synchronized with the latest tracker state.
    - Verify that `[CODE_ROOT_DIRECTORIES]` and `[DOC_DIRECTORIES]` sections in `.clinerules` are correctly populated.

  - **Strategy → Execution**:
    - Ensure all instruction files created in the Strategy phase (e.g., in `strategy_tasks/` or `src/`) contain complete "Steps" and "Dependencies" sections, providing clear guidance for the Execution phase.

  Completing these checklists helps ensure a smooth and effective transition between CRCT phases, maintaining system integrity and task continuity.


2. **Key CRCT Operations**:

   - **Project Analysis**:
     - The LLM will use `analyze-project` command to fully analyze the project, generate **contextual keys**, update dependency trackers (main, doc, and mini-trackers), and generate embeddings. This command is central to maintaining up-to-date dependency information and should be run in the Set-up/Maintenance phase and after significant code changes.
   - **Key/Placeholder Inspection**:
     - The LLM will use `show-keys --tracker <tracker_file>` to view definitions and identify keys needing verification.
     - The LLM will use `show-placeholders [--tracker <tracker_file>]` to list 'p', 's', and 'S' dependencies. If `--tracker` is omitted, provides a project-wide aggregate summary of all unverified dependencies.
     - The LLM will use `visualize-dependencies [--key <key1> <key2> ...]` to generate Mermaid or SVG diagrams. Multi-key focus is supported for complex relationship analysis.
   - **Dependency Inspection**:
     - The LLM will utilize the `show-dependencies --key <key>` command to inspect dependencies for a specific **contextual key**. Replace `<key>` with the desired file or module key. This command aggregates dependency information from all trackers and provides a comprehensive view of inbound and outbound dependencies, significantly simplifying dependency analysis.
   - **Manual Dependency Management**:
     - The LLM will use `add-dependency --tracker <tracker_file> --source-key <key> --target-key <key1> [<key2>...] --dep-type <char>` to manually set dependency relationships in tracker files. This is useful for correcting or verifying suggested dependencies and for marking verified relationships. **Ensure you are using contextual keys when using this command.**
       *(Note: --target-key accepts multiple keys. The specified `--dep-type` is applied to *all* targets.)*
       *(Recommendation: Specify no more than five target keys at once for clarity.)*
     - The LLM will use `remove-key <tracker_file> <key>` to remove a key and its associated data from a tracker file, typically used when files are deleted or refactored. **Ensure you are using contextual keys when using this command.**

   **Dependency Characters for Manual Management:**

   When manually managing dependencies using `add-dependency`, you need to specify the dependency type using a character code. Here's a breakdown of the available characters, as defined in the `[Character_Definitions]` section of `.clinerules`:

   - `<`: Row depends on column (Source depends on Target).
   - `>`: Column depends on row (Target depends on Source).
   - `x`: Mutual dependency (Both depend on each other).
   - `d`: Documentation dependency (Source documents Target).
   - `o`: Self dependency (Used only on the diagonal of tracker grids, indicating a file's dependency on itself - usually for structural elements).
   - `n`: Verified no dependency (Explicitly marks that no dependency exists).
   - `p`: Placeholder (Indicates an unverified or suggested dependency, needs manual review).
   - `s`: Semantic dependency (weak - similarity score between 0.05 and 0.06).
   - `S`: Semantic dependency (strong - similarity score 0.07 or higher).

   **Example Scenarios:**

   - If `moduleA.py` imports `moduleB.py`, you would use `>` to indicate that `moduleB.py` (Target/Column) depends on `moduleA.py` (Source/Row).
   - If `docs_about_moduleA.md` documents `moduleA.py`, use `d`.
   - If you have manually verified that there is no dependency between two modules, use `n` to explicitly mark it as such and prevent future suggestions.

   Using these characters correctly ensures accurate and meaningful dependency tracking within the CRCT system. Remember to use contextual keys for all manual dependency management operations.

  **Understanding Contextual Keys:**

  CRCT v8.4 uses a hierarchical key system (`KeyInfo`) to represent files and directories. Keys follow a pattern like `Tier` + `DirLetter` + `[SubdirLetter]` + `[FileNumber]` (e.g., `1A`, `1Aa`, `1Aa1`).

  - **Tier:** The initial number indicates the nesting level. Top-level roots are Tier 1.
  - **DirLetter:** An uppercase letter ('A', 'B', ...) identifies top-level directories within defined roots.
  - **SubdirLetter:** A lowercase letter ('a', 'b', ...) identifies subdirectories within a directory.
  - **FileNumber:** A number identifies files within a directory.

  **Tier Promotion:** To keep keys manageable in deeply nested projects, CRCT uses "tier promotion". When a directory *already represented by a subdirectory key* (e.g., `1Ba`) contains further subdirectories requiring keys, those sub-subdirectories will start a *new tier*. For example, a directory inside `1Ba` might get the key `2Ca` (Tier 2, promoted Dir 'C', Subdir 'a'), instead of `1Baa`. This prevents excessively long keys.

   **Understanding Dependency Trackers:**

   CRCT uses three types of tracker files to manage dependencies:

   - **`module_relationship_tracker.md` (Main Tracker):** Located in `cline_docs/`, this tracker provides a high-level view of dependencies *between modules*. Modules typically correspond to the top-level directories defined in `[CODE_ROOT_DIRECTORIES]` within `.clinerules`. Dependencies shown here are *aggregated* from lower-level trackers; if a file in `moduleA` depends on a file in `moduleB`, this dependency is "rolled up" and represented as a dependency from `moduleA` to `moduleB` in this main tracker.
   - **`doc_tracker.md`:** Located in `cline_docs/`, this tracks dependencies between documentation files found within the directories specified in `[DOC_DIRECTORIES]` in `.clinerules`.
   - **Mini-Trackers (`{module_name}_module.md`):** Located within each module directory (as defined in `[CODE_ROOT_DIRECTORIES]`), these files track dependencies *between files within that specific module* and also dependencies from internal files to *external* files (files outside the module or in documentation roots). The tracker data is embedded within the module's documentation file between `---mini_tracker_start---` and `---mini_tracker_end---` markers.

   **Suggestion Priority:** When different analysis methods (e.g., explicit import vs. semantic similarity) suggest conflicting dependencies between the same two files, CRCT uses a priority system to decide which dependency character (`<`, `>`, `x`, `d`, `S`, `s`, `p`) to record. The approximate priority order (highest first) is: `x` (mutual), `<`/`>` (direct structural/import), `d` (documentation link), `S` (strong semantic), `s` (weak semantic), `n`/`p`/`o` (no/placeholder/self). Explicit dependencies generally override semantic ones.

 3.  **Mandatory Update Protocol (MUP)**:
     - CRCT employs a Mandatory Update Protocol (MUP) to ensure system state consistency. The LLM will perform a full MUP regularly (every 5 turns) to update `activeContext.md`, `changelog.md`, and `.clinerules`, and to clean up completed tasks. **The MUP is crucial for maintaining system integrity and should not be skipped.**
     - If the LLM does not perform this step within a reasonable number of turns, prompt it to follow the MUP protocol. Remember that the context window is limited and LLMs quickly lose track of what the have and haven't done as the context window grows. *Be very wary of LLM hallucinations and mis-steps, especially in context sizes above 100k*

 4. **Configuration**:
    - **`.clinerules.config.json`**: Configure system settings in this file.
      - `embedding_device`:  Set the embedding device (`cpu`, `cuda`, `mps`) to optimize performance based on your hardware.
      - `excluded_file_patterns`: Define file exclusion patterns (glob patterns) to customize project analysis and exclude specific files or directories from dependency tracking.
    - *Note: Other settings like specific path exclusions (`excluded_paths`) and system directory locations (`memory_dir`, `embeddings_dir`, `backups_dir`, etc.) are also configurable in `.clinerules.config.json`. Refer to the file or defaults for more details.*
    - **`.clinerules`**:  Manage core system settings directly in the `.clinerules` file.
      - `current_phase`: Set the current operational phase of CRCT.
      - `[CODE_ROOT_DIRECTORIES]`: Define directories considered as code roots for project analysis. **Modify this section to specify directories containing source code that should be analyzed for dependencies.**
      - `[DOC_DIRECTORIES]`: Define directories considered as documentation roots. **Modify this section to specify directories containing project documentation that should be tracked.**
      - `[LEARNING_JOURNAL]`: Review the learning journal for insights into system operations and best practices.

---

## Hierarchical Design Token Architecture (HDTA)

CRCT utilizes the Hierarchical Design Token Architecture (HDTA) to organize project documentation and planning. HDTA provides a structured approach to manage system-level information, broken down into four tiers of documents:

1. **System Manifest (`system_manifest.md` in `cline_docs/`)**: Provides a top-level overview of the entire project, its goals, architecture, and key components. This document is created during the Set-up/Maintenance phase and serves as the central point of reference for the project.

2. **Domain Modules (`{module_name}_module.md` in `cline_docs/`)**: Describes major functional areas or modules within the project. Each module document details its purpose, functionalities, and relationships with other modules. These are also created during Set-up/Maintenance.

3. **Implementation Plans (Files within modules)**: Contains detailed plans for specific implementations within a module. These documents outline the steps, dependencies, and considerations for implementing particular features or functionalities. Implementation plans are typically created during the Strategy phase.

4. **Task Instructions (`{task_name}.md` in `strategy_tasks/` or `src/`)**: Provides procedural guidance for individual tasks. Task instructions break down complex tasks into smaller, manageable steps, ensuring clarity and efficient execution. Task instructions are created during the Strategy phase to guide the Execution phase.

HDTA helps maintain a clear and organized project documentation structure, facilitating better understanding, collaboration, and management of complex projects within CRCT. The templates for HDTA documents are located in `cline_docs/templates/`.

---

## Troubleshooting

- **Initialization Issues**: If the system fails to initialize, ensure that the core prompt is correctly loaded into the Cline extension and that all prerequisites are met. **Double-check that you have copied the content of `.clinerules/core_prompt(put this in Custom Instructions).md` into the Cline custom instructions field.**
- **Dependency Tracking Problems**: If you encounter issues with dependency tracking, use `analyze-project` to refresh the trackers and embeddings. Check `.clinerules` for correct configuration of code and documentation roots. **Ensure that the `[CODE_ROOT_DIRECTORIES]` and `[DOC_DIRECTORIES]` sections in `.clinerules` are correctly populated.**
  - You may need to manually delete the module_relationship_tracker and doc_tracker in cline_docs.
  - For mini-trackers I suggest manually deleting the content between the mini-tracker start and end markers. *IMPORTANT: do not remove the mini-tracker start and end markers, this will cause the entire file content to be overwritten, losing any information previously recorded.*
- **Command Errors**: Ensure commands are typed correctly and used in the appropriate phase. Refer to command documentation for syntax and usage. **Pay close attention to the required arguments for each command, especially when using `show-dependencies`, `add-dependency`, and `remove-key`.**
- **Key Generation Limit (`KeyGenerationError`)**: The key system supports up to 26 direct subdirectories ('a' through 'z') within any single directory before requiring "tier promotion". If you have a directory structure exceeding this limit (e.g., 27+ immediate subfolders needing keys), the `analyze-project` command may fail with a `KeyGenerationError`. To resolve this, either restructure the directory or add some of the problematic subdirectory paths to the `"excluded_paths"` list in `.clinerules.config.json`.

---

## Advanced Usage & Troubleshooting

- **Semantic Analysis Details:** Semantic similarity analysis (leading to 's' or 'S' dependencies) relies on the `sentence-transformers` library (specifically the `"sentence-transformers/all-mpnet-base-v2"` model by default). Embeddings are stored as `.npy` files in a mirrored structure within the directory configured by `embeddings_dir` in `.clinerules.config.json` (default: `cline_utils/dependency_system/analysis/embeddings/`). The system avoids regenerating embeddings for unchanged files by checking file modification times (mtime) against a `metadata.json` file.
- **Tracker Backups:** Before overwriting tracker files (`module_relationship_tracker.md`, `doc_tracker.md`, `*_module.md`) during updates (e.g., via `analyze-project`), the system automatically creates a timestamped backup in the directory configured by `backups_dir` (default: `cline_docs/backups/`). The two most recent backups for each tracker are kept.
- **Batch Processing Tuning:** The system uses parallel batch processing for tasks like file analysis (`analyze-project`). While it attempts adaptive tuning, performance might vary. For very large projects or specific hardware, if analysis seems slow or uses excessive resources, you can instruct the LLM to try specific parameters for the `BatchProcessor` by suggesting values for `max_workers` (number of threads) or `batch_size` when invoking relevant commands.
- **Additional Utility Commands:** The `dependency_processor.py` script provides several utility commands beyond the main workflow ones. These might be useful for advanced inspection or manual intervention (ask the LLM to use them if needed):
    - `python -m cline_utils.dependency_system.dependency_processor analyze-file <file_path> [--output <json_path>]`: Analyzes a single file and outputs detailed findings (imports, calls, etc.) as JSON.
    - `python -m cline_utils.dependency_system.dependency_processor merge-trackers <primary_path> <secondary_path> [--output <out_path>]`: Merges two tracker files, with the primary taking precedence. Useful for combining tracker data, though typically managed automatically.
    - `python -m cline_utils.dependency_system.dependency_processor export-tracker <tracker_path> [--format <json|csv|dot>] [--output <out_path>]`: Exports tracker data into different formats for external analysis or visualization (e.g., DOT format for Graphviz).
    - `python -m cline_utils.dependency_system.dependency_processor update-config <key_path> <value>`: Updates a specific setting in `.clinerules.config.json` (e.g., `python -m cline_utils.dependency_system.dependency_processor update-config thresholds.code_similarity 0.75`).
    - `python -m cline_utils.dependency_system.dependency_processor reset-config`: Resets `.clinerules.config.json` to its default settings.
    - `python -m cline_utils.dependency_system.dependency_processor clear-caches`: Clears internal caches used by the dependency system (embeddings metadata, analysis results, etc.). Useful if you suspect stale cache data is causing issues.

---

## Notes

- CRCT v8.4 introduces high-performance optimizations (Bolt) and automated source-code annotation (Comment-Skill) to bridge the gap between documentation and code.
- The LLM automates most dependency management and system commands (e.g., `analyze-project`, `show-dependencies`, etc.).
- For custom projects, ensure `src/` and `docs/` directories are populated before initializing the system.
  - `src/` and `docs/` are the default directories used for quick-start.
  - The system supports custom code and documentation directory structures, so feel free to create your own if you feel comfortable.
  (if any issues arise let me know and I will do my best to offer assistance or modify the system to support your unique project structure)

## Getting Help

For further assistance, questions, or bug reports, please refer to the project's GitHub repository and issue tracker.

---

Thank you for using CRCT v8.4! I hope it enhances your Cline project workflows.