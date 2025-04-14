# Using External Projects with CRCT Dev Containers

This guide explains how to use the CRCT system with external project folders within the VS Code Dev Container environment using a single-root setup.

## Overview

Instead of using multi-root workspaces (which can cause issues with Dev Containers), CRCT now uses a single-root approach where:
1. You open the main CRCT folder in VS Code.
2. You run a setup script to configure your external project.
3. The script adds a volume mount for your project to the dev container configuration.
4. The script creates a symlink inside the CRCT folder pointing to your project's location within the container.
5. You work within the CRCT folder, accessing your project via the symlink in the Explorer.

## Why This Approach?

- **Reliability:** Avoids common errors like "workspace does not exist" associated with multi-root workspaces in containers.
- **Simplicity:** Easier setup and configuration compared to managing multi-root workspace files.
- **Seamless Access:** Your project appears directly in the CRCT Explorer via a symlink.
- **Automation:** The setup script handles mounting, symlinking, and `.clinerules` configuration.

## Setting Up Your Project

### Automatic Setup (Recommended)

Use the provided script to configure your project:

```bash
# Run from the CRCT root directory
python cline_utils/create_crct_workspace.py 
```

This script will:
1. Prompt for your project's absolute path on your host machine.
2. Add a volume mount for this path to `.devcontainer/devcontainer.json`. The target inside the container will typically be `/workspaces/YourProjectName`.
3. Create a symlink in the CRCT root directory (e.g., `YourProjectName`) pointing to the container target path (`/workspaces/YourProjectName`).
4. Update `.clinerules`'s `[CODE_ROOT_DIRECTORIES]` section with the container target path.

**Mount Management:**
- Each time you configure a new project, the script appends a mount for it to `.devcontainer/devcontainer.json`.
- To remove all project mounts, run:  
  `python cline_utils/create_crct_workspace.py --clean-mounts` 
  *(Note: This currently doesn't remove the symlinks, which might need manual cleanup or a script enhancement).*

**Note on Tab Completion:**
- Tab completion for the project path prompt works only on Unix-like systems (macOS, Linux).
- On Windows, type or paste the full path.

### Manual Setup

1.  **Add Mount:** Manually edit `.devcontainer/devcontainer.json` and add your project to the `mounts` array:
    ```json
    "mounts": [
        "source=/path/to/your/project,target=/workspaces/YourProjectName,type=bind,consistency=cached"
    ]
    ```
2.  **Create Symlink:** After the container is running, create a symlink inside the container:
    ```bash
    # Run inside the container terminal
    ln -s /workspaces/YourProjectName /workspaces/CRCT\ System/YourProjectName 
    ```
3.  **Update .clinerules:** Manually edit `.clinerules` and add `/workspaces/YourProjectName` to `[CODE_ROOT_DIRECTORIES]`.

## Working with Your Project

1.  **Open CRCT Folder:** Open the main CRCT folder in VS Code: `code .`
2.  **Reopen in Container:** Click "Reopen in Container" when prompted (or use F1).
3.  **Access Project:** Your project will appear as a symlink (e.g., `YourProjectName`) in the VS Code Explorer sidebar. You can browse and edit files directly.
4.  **CRCT Interaction:** CRCT tools and the LLM will interact with your project using the container path specified in `.clinerules` (e.g., `/workspaces/YourProjectName`).

## Understanding Container Paths

- The CRCT root folder is mounted, typically at `/workspaces/CRCT System`.
- Your external project is mounted via the `mounts` configuration (e.g., to `/workspaces/YourProjectName`).
- A symlink in `/workspaces/CRCT System/` points to `/workspaces/YourProjectName`.
- `.clinerules` uses the container mount path (`/workspaces/YourProjectName`) for analysis.

## Troubleshooting

- **"Symlink broken" or "Cannot access project files"**: Ensure the container is running and the mount in `.devcontainer/devcontainer.json` is correct. Rebuild the container if you changed mounts. Check permissions on the host directory.
- **"CRCT can't find my project files"**: Verify the path in `.clinerules` matches the container mount target path (e.g., `/workspaces/YourProjectName`).
- **"Container fails to build"**: Check Docker is running, has resources, and review build logs. Ensure mount source path is correct.

## Advanced Configuration

### Multiple Project Folders

Run the setup script (`python cline_utils/create_crct_workspace.py`) for each external project you want to include. The script will:
- Add a separate mount for each project in `.devcontainer/devcontainer.json`.
- Create a symlink for each project in the CRCT root.
- Add each project's container path to `.clinerules`.

You can then access all configured projects via their symlinks in the Explorer.

## Further Reading

- [VS Code Dev Containers](https://code.visualstudio.com/docs/remote/containers)
- [Docker Bind Mounts](https://docs.docker.com/storage/bind-mounts/)
