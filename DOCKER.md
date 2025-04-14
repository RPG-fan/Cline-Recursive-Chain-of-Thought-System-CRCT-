# Using CRCT with Docker and VSCode

This guide explains how to use the CRCT system in a Docker container with Visual Studio Code.

## Prerequisites

- Docker installed on your system
- Visual Studio Code with the following extensions:
  - Dev Containers
  - Cline AI

## Quick Start (Single-Root Dev Container Approach)

1. Clone the repository:
   ```bash
   git clone https://github.com/RPG-fan/Cline-Recursive-Chain-of-Thought-System-CRCT-.git
   cd Cline-Recursive-Chain-of-Thought-System-CRCT-
   ```

2. Configure your external project for the container:
   ```bash
   python cline_utils/create_crct_workspace.py 
   ```
   - The script prompts for your project's absolute path.
   - It adds a mount for your project to `.devcontainer/devcontainer.json`.
   - It creates a symlink in the CRCT root (e.g., `YourProjectName`) pointing to the container mount path (`/workspaces/YourProjectName`).
   - It updates `.clinerules` with the container path (`/workspaces/YourProjectName`).

3. Open the CRCT root folder in VS Code:
   ```bash
   code .
   ```

4. When prompted, click "Reopen in Container" (or use F1 > Dev Containers: Reopen in Container).

5. VS Code builds the Docker container and opens the CRCT folder inside it.

6. Inside the container, your project will appear as a symlink in the Explorer.

7. Follow the normal setup instructions in INSTRUCTIONS.md (configuring Cline, starting the system).

## Alternative: Single-Folder Container (Legacy Approach)

If you prefer to open just the CRCT folder in a container:

1. Open the CRCT project in VSCode:
   ```bash
   code .
   ```

2. When prompted, click "Reopen in Container" or use the Command Palette (F1) and select "Remote-Containers: Reopen in Container"

3. Once the container is running, you'll need to copy your project files into the CRCT structure (typically into `/app/src/`) or configure CRCT to access files elsewhere.

## Manual Container Setup

If you prefer to run the container manually:

```bash
# Build the container
docker-compose build

# Run the container
docker-compose up -d

# Execute commands in the container
docker-compose exec crct python cline_utils/dependency_system/dependency_processor.py
```

## Understanding Container Paths

When using the single-root approach with Dev Containers:

- The CRCT root folder is mounted at `/workspaces/CRCT System` (or similar, check `.devcontainer/devcontainer.json`).
- Your external project is mounted via the `mounts` configuration in `.devcontainer/devcontainer.json` (e.g., to `/workspaces/YourProjectName`).
- A symlink is created in the CRCT root pointing to the project's mount path.
- The `[CODE_ROOT_DIRECTORIES]` section in `.clinerules` should use the container mount path (e.g., `/workspaces/YourProjectName`).

## Notes

- All files are volume-mounted from your host into the container, so changes made in the container are saved to your local filesystem.
- The Cline extension should be configured as outlined in INSTRUCTIONS.md.
- Python dependencies are pre-installed in the container.
- The `.devcontainer` configuration automatically installs required VSCode extensions.

## Troubleshooting

- **"I can't see my project in the container"**: Check the symlink in the CRCT root and the `mounts` in `.devcontainer/devcontainer.json`. Ensure the container is running.
- **"CRCT can't find my project files"**: Check the `[CODE_ROOT_DIRECTORIES]` section in `.clinerules` and make sure it points to the correct container path (e.g., `/workspaces/YourProjectName`).
- **"The container fails to build"**: Make sure Docker is running and has sufficient resources. Check the Docker build logs.
- If you encounter permission issues with files created inside the container, ensure `"updateRemoteUserUID": true` is set in `.devcontainer/devcontainer.json`.
- For Cline extension configuration issues, refer to the extension documentation.

## Further Reading

For more detailed information about using the single-root approach with CRCT, see [docs/multi-root-workspace-guide.md](docs/multi-root-workspace-guide.md).
