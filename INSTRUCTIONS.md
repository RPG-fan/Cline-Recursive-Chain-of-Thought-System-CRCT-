# Instructions for Using CRCT v7.0

These instructions assume you're using the Cline VS Code extension with CRCT v7.0. The system automates most tasks via the LLM, but some manual steps are needed to get started.

---

## Prerequisites

- **VS Code**: Installed with the Cline extension.
- **Python**: 3.8+ with `pip`.
- **Git**: To clone the repo.
- **Docker**: For the Dev Container functionality.

---

## Step 1: Setup CRCT Dev Container with Your Project

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/RPG-fan/Cline-Recursive-Chain-of-Thought-System-CRCT-.git
    cd Cline-Recursive-Chain-of-Thought-System-CRCT-
    ```

2.  **Configure Project Mount**:
    ```bash
    python cline_utils/create_crct_workspace.py 
    ```
    - The script prompts for your project's absolute path.
    - It adds a mount for your project to `.devcontainer/devcontainer.json`.
    - It creates a symlink in the CRCT root (e.g., `YourProjectName`) pointing to the container mount path (`/workspaces/YourProjectName`).
    - It updates `.clinerules` with the container path (`/workspaces/YourProjectName`).

3.  **Open CRCT Folder in VS Code**:
    ```bash
    # Open the CRCT root folder
    code . 
    ```

4.  **Reopen in Container**:
    - When VS Code opens, click "Reopen in Container" when prompted (or use F1 > Dev Containers: Reopen in Container).
    - The first time, Docker builds the container image.

5.  **Configure Cline Extension**:
    - Once the container is running, open the Cline extension settings within VS Code.
    - Paste the contents of `cline_docs/prompts/core_prompt(put this in Custom Instructions).md` into the system prompt/custom instructions field.

---

## Step 2: Verify Setup and Start CRCT

1.  **Access Your Project**:
    - Inside the container, your project will appear as a symlink in the Explorer sidebar within the CRCT folder structure.
    - CRCT tools will access your project via the container path configured in `.clinerules` (e.g., `/workspaces/YourProjectName`).

2.  **Verify Code Root in `.clinerules`**:
    - In the VS Code Explorer, navigate to the CRCT files.
    - Open `.clinerules`.
    - Confirm the `[CODE_ROOT_DIRECTORIES]` section points to your project's container path (e.g., `/workspaces/YourProjectName`). The setup script should have configured this automatically.

3.  **Start CRCT Initialization**:
    - In the Cline input panel, type `Start.` and run it.
    - The LLM will:
        - Read `.clinerules` (using your configured code root).
        - Load the `Set-up/Maintenance` plugin.
        - Initialize core CRCT files in `cline_docs/`.

4.  **Follow Prompts**:
    - The LLM may ask for input (e.g., project goals for `projectbrief.md`). Provide concise answers.

5.  **Verify Setup**:
    - Check `cline_docs/` for newly created files
    - Confirm `.clinerules` reflects the correct `[CODE_ROOT_DIRECTORIES]` for your project

---

## Step 3: Populate Dependency Trackers

1.  **Run Initial Setup**:
    - Input: `Perform initial setup and populate dependency trackers.`
    - The LLM will:
        - Identify the code root(s) specified in `.clinerules`
        - Use `dependency_processor.py` to scan your project code based on those roots
        - Generate `dependency_tracker.md` and `doc_tracker.md` within the CRCT `cline_docs/` directory
        - Suggest and validate dependencies based on its analysis of your code

2. **Manual Validation (if prompted)**:
   - The LLM may present dependency suggestions (e.g., JSON output)
   - Confirm or adjust characters (`<`, `>`, `x`, etc.) as prompted

---

## Step 4: Plan and Execute

1. **Enter Strategy Phase**:
   - Once trackers are populated, the LLM will transition to `Strategy` (check `.clinerules`)
   - Input: `Plan the next steps for my project.`
   - Output: New instruction files in `strategy_tasks/` or within your project's directories

2. **Execute Tasks**:
   - Input: `Execute the planned tasks.`
   - The LLM will follow instruction files, update files, and apply the MUP

---

## Tips

- **Monitor `activeContext.md`**: Tracks current state and priorities
- **Check `.clinerules`**: Shows the current phase and next action
- **Debugging**: If stuck, try `Review the current state and suggest next steps.`
- **Path References**: When referring to files in your project, use the container paths (e.g., `/workspaces/YourProjectName/src/file.py`)

---

## Notes

- CRCT v7.0 is a work-in-progress. Expect minor bugs (e.g., tracker initialization).
- The LLM handles most commands (`generate-keys`, `set_char`, etc.) automatically based on the configured code root in `.clinerules`.
- Ensure your project's documentation is accessible if you want `doc_tracker.md` populated correctly (you might need to add its path to `[CODE_ROOT_DIRECTORIES]` or ensure it's within a scanned root).

Questions? Open an issue on GitHub!
