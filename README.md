# Cline Recursive Chain-of-Thought System (CRCT) - v7.0

Welcome to the **Cline Recursive Chain-of-Thought System (CRCT)**, a framework designed to manage context, dependencies, and tasks in large-scale Cline projects within VS Code. Built for the Cline extension, CRCT leverages a recursive, file-based approach with a modular dependency tracking system to keep your project's state persistent and efficient, even as complexity grows.

This is **v7.0**, a basic but functional release of an ongoing refactor to improve dependency tracking modularity. While the full refactor is still in progress (stay tuned!), this version offers a stable starting point for community testing and feedback. It includes base templates for all core files and the new `dependency_processor.py` script.

---

## Key Features

- **Recursive Decomposition**: Breaks tasks into manageable subtasks, organized via directories and files for isolated context management.
- **Minimal Context Loading**: Loads only essential data, expanding via dependency trackers as needed.
- **Persistent State**: Uses the VS Code file system to store context, instructions, outputs, and dependencies—kept up-to-date via a **Mandatory Update Protocol (MUP)**.
- **Modular Dependency Tracking**: 
  - `dependency_tracker.md` (module-level dependencies)
  - `doc_tracker.md` (documentation dependencies)
  - Mini-trackers (file/function-level within modules)
  - Uses hierarchical keys and RLE compression for efficiency (~90% fewer characters vs. full names in initial tests).
- **Phase-Based Workflow**: Operates in distinct phases—**Set-up/Maintenance**, **Strategy**, **Execution**—controlled by `.clinerules`.
- **Chain-of-Thought Reasoning**: Ensures transparency with step-by-step reasoning and reflection.

---

## Quickstart (Recommended Approach)

1. **Clone CRCT and Configure Project**:
   ```bash
   # Clone the repository
   git clone https://github.com/RPG-fan/Cline-Recursive-Chain-of-Thought-System-CRCT-.git
   cd Cline-Recursive-Chain-of-Thought-System-CRCT-
   
   # Install dependencies (optional, handled by container)
   # pip install -r requirements.txt 
   
   # Configure CRCT to include your project
   python cline_utils/create_crct_workspace.py 
   ```
   - The script will prompt you for your project's folder path.
   - It will add a mount for your project to `.devcontainer/devcontainer.json`.
   - It will create a symlink in the CRCT root pointing to your project's container path.
   - It will update `.clinerules` with the correct container path for your project.

2. **Open CRCT Folder in VS Code**:
   ```bash
   # Open the CRCT root folder (NOT a workspace file)
   code . 
   ```

3. **Reopen in Container**:
   - When VS Code opens, click "Reopen in Container" when prompted (or use F1 > Dev Containers: Reopen in Container).
   - The first time, Docker will build the container image (this might take a few minutes).
   - Your project will appear as a symlink in the Explorer sidebar.

4. **Configure Cline**:
   - Copy `cline_docs/prompts/core_prompt(put this in Custom Instructions).md` into the Cline system prompt field.

5. **Start the System**:
   - Type `Start.` in the Cline input to initialize the system.
   - The LLM will bootstrap from `.clinerules` (using your configured code root), creating missing files and guiding you through setup if needed.

*Note*: The Cline extension's LLM automates most commands and updates to `cline_docs/`. Minimal user intervention is required (in theory!).

---

## Project Structure

```
cline/
│   .clinerules              # Controls phase and state
│   README.md                # This file
│   requirements.txt         # Python dependencies
│
├───cline_docs/              # Operational memory
│   │   activeContext.md     # Current state and priorities
│   │   changelog.md         # Logs significant changes
│   │   productContext.md    # Project purpose and user needs
│   │   progress.md          # Tracks progress
│   │   projectbrief.md      # Mission and objectives
│   │   dependency_tracker.md # Module-level dependencies
│   │   ...                  # Additional templates
│   └───prompts/             # System prompts and plugins
│       core_prompt.md       # Core system instructions
│       setup_maintenance_plugin.md
│       strategy_plugin.md
│       execution_plugin.md
│
├───cline_utils/             # Utility scripts
│   └───dependency_system/
│       dependency_processor.py # Dependency management script
│
├───docs/                    # Project documentation
│   │   doc_tracker.md       # Documentation dependencies
│
├───src/                     # Source code root
│
└───strategy_tasks/          # Strategic plans
```

---

## Current Status & Future Plans

- **v7.0**: A basic, functional release with modular dependency tracking via `dependency_processor.py`. Includes templates for all `cline_docs/` files.
- **Efficiency**: Achieves a ~1.9 efficiency ratio (90% fewer characters) for dependency tracking vs. full names—improving with scale.
- **Ongoing Refactor**: I'm enhancing modularity and token efficiency further. The next version will refine dependency storage and extend savings to simpler projects.

Feedback is welcome! Please report bugs or suggestions via GitHub Issues.

---

## Using CRCT with Your Project (Single-Root Dev Container Approach)

CRCT now uses a single-root VS Code Dev Container setup to work directly with your project folder via mounts and symlinks:

1. **Configure Your Project**:
   ```bash
   # Clone CRCT if you haven't already
   git clone https://github.com/RPG-fan/Cline-Recursive-Chain-of-Thought-System-CRCT-.git
   cd Cline-Recursive-Chain-of-Thought-System-CRCT-
   
   # Run the setup script
   python cline_utils/create_crct_workspace.py 
   ```
   - The script prompts for your project's absolute path.
   - It adds a mount for your project to `.devcontainer/devcontainer.json`.
   - It creates a symlink in the CRCT root (e.g., `YourProjectName`) pointing to the container mount path (`/workspaces/YourProjectName`).
   - It updates `.clinerules` with the container path (`/workspaces/YourProjectName`).
   - **New**: It automatically detects your project's language/framework and adds appropriate Dev Container Features to support it.

2. **Open CRCT Folder in VS Code**:
   ```bash
   # Open the CRCT root folder
   code . 
   ```

3. **Reopen in Container**:
   - When VS Code opens, click "Reopen in Container" when prompted (or use F1 > Dev Containers: Reopen in Container).
   - The first time, Docker builds the container image.
   - If project-specific features were detected, VS Code may prompt you to rebuild the container.

4. **Access Your Project**:
   - Inside the container, your project will appear as a symlink in the Explorer sidebar within the CRCT folder structure.
   - CRCT tools will access your project via the container path configured in `.clinerules` (e.g., `/workspaces/YourProjectName`).

5. **Initialize CRCT**:
   - Type `Start.` in the Cline input to initialize the system.
   - CRCT analyzes your codebase (based on the configured code root), initializes trackers, and guides you forward.

*Note*: This single-root approach avoids multi-root workspace complexities and ensures reliable container startup while still providing seamless access to your external project.

---

## Thanks!

This is a labor of love to make Cline projects more manageable. I'd love to hear your thoughts—try it out and let me know what works (or doesn't)!
