# Using CRCT with Docker and VSCode

This guide explains how to use the CRCT system in a Docker container with Visual Studio Code.

## Prerequisites

- Docker installed on your system
- Visual Studio Code with the following extensions:
  - Dev Containers
  - Cline AI

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/RPG-fan/Cline-Recursive-Chain-of-Thought-System-CRCT-.git
   cd Cline-Recursive-Chain-of-Thought-System-CRCT-
   ```

2. Open the project in VSCode:
   ```bash
   code .
   ```

3. When prompted, click "Reopen in Container" or use the Command Palette (F1) and select "Remote-Containers: Reopen in Container"

4. VSCode will build the Docker container and open the project inside it. This may take a few minutes on first run.

5. Once the container is running, you can follow the normal setup instructions in INSTRUCTIONS.md.

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

## Notes

- All files are volume-mounted from your host into the container, so changes made in the container are saved to your local filesystem.
- The Cline extension should be configured as outlined in INSTRUCTIONS.md.
- Python dependencies are pre-installed in the container.
- The `.devcontainer` configuration automatically installs required VSCode extensions.

## Troubleshooting

- If you encounter permission issues with files created inside the container, you may need to adjust file ownership on your host.
- For Cline extension configuration issues, refer to the extension documentation.
