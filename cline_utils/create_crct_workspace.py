#!/usr/bin/env python3
"""
CRCT Dev Container Setup Script

This script configures the CRCT dev container to include a user's external
project folder via mounts and symlinks, enabling seamless use of CRCT
with any project.
"""

import os
import json
import subprocess
import sys
from pathlib import Path

DEVCONTAINER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".devcontainer", "devcontainer.json")

def read_devcontainer_json():
    """Read the devcontainer.json file and return its contents as a dict."""
    if not os.path.exists(DEVCONTAINER_PATH):
        print(f"devcontainer.json not found at {DEVCONTAINER_PATH}")
        return None
    with open(DEVCONTAINER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def write_devcontainer_json(data):
    """Write the given data to devcontainer.json."""
    with open(DEVCONTAINER_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

def add_mount_to_devcontainer(source_path, target_path):
    """Add a mount entry to devcontainer.json if not already present."""
    data = read_devcontainer_json()
    if data is None:
        return
    mount_str = f"source={source_path},target={target_path},type=bind,consistency=cached"
    mounts = data.get("mounts", [])
    if mount_str not in mounts:
        mounts.append(mount_str)
        data["mounts"] = mounts
        write_devcontainer_json(data)
        print(f"✓ Added mount: {mount_str}")
    else:
        print(f"Mount already exists: {mount_str}")

def clean_mounts_in_devcontainer():
    """Remove all mounts from devcontainer.json."""
    data = read_devcontainer_json()
    if data is None:
        return
    if "mounts" in data:
        del data["mounts"]
        write_devcontainer_json(data)
        print("✓ All mounts removed from devcontainer.json")
    else:
        print("No mounts found in devcontainer.json")

# Try to import readline for tab completion (Unix systems)
try:
    import readline
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False

def setup_tab_completion():
    """Set up tab completion for file paths if readline is available."""
    if not READLINE_AVAILABLE:
        return
        
    # Tab completion function for directory paths
    def complete_path(text, state):
        # Expand ~ to user's home directory
        expanded_text = os.path.expanduser(text)
        
        # Get the directory part of the path and the prefix to match
        if os.path.isdir(expanded_text):
            # If the text is a directory, list its contents
            dir_path = expanded_text
            prefix = ""
        else:
            # Otherwise, list contents of the parent directory that match the prefix
            dir_path = os.path.dirname(expanded_text) or "."
            prefix = os.path.basename(expanded_text)
        
        # Get matching directories
        try:
            matches = []
            for item in os.listdir(dir_path):
                full_path = os.path.join(dir_path, item)
                if item.startswith(prefix) and os.path.isdir(full_path):
                    # Add trailing slash to indicate it's a directory
                    matches.append(os.path.join(dir_path, item) + os.sep)
        except (OSError, PermissionError):
            matches = []
        
        # Return the state-th match or None if no more matches
        return matches[state] if state < len(matches) else None
    
    # Configure readline
    readline.set_completer_delims(' \t\n;')
    
    # Different binding for different platforms
    if sys.platform == 'darwin':  # macOS
        readline.parse_and_bind("bind ^I rl_complete")
    else:  # Linux and other Unix
        readline.parse_and_bind("tab: complete")
    
    readline.set_completer(complete_path)
    
    # Print a hint about tab completion
    print("Hint: Use Tab key to complete directory paths.")

def get_project_path():
    """Get the user's project path with tab completion support."""
    # Set up tab completion if available
    if READLINE_AVAILABLE:
        setup_tab_completion()
    else:
        # Print a note about tab completion not being available
        if sys.platform == "win32":
            print("Note: Tab completion is not available on Windows.")
        else:
            print("Note: Tab completion is not available (readline module not found).")
        print("Tip: You can type the path in your shell (where tab completion works) and paste it here.")
        
    # Prompt for project path
    while True:
        try:
            if READLINE_AVAILABLE:
                prompt = "Enter the absolute path to your project's root folder (Tab for completion): "
            else:
                prompt = "Enter the absolute path to your project's root folder: "
                
            path_input = input(prompt).strip()
            
            # Handle empty input
            if not path_input:
                print("Please enter a path.")
                continue
                
            # Expand ~ to home directory
            expanded_path = os.path.expanduser(path_input)
            # Get absolute path
            abs_path = os.path.abspath(expanded_path)
            if os.path.exists(abs_path) and os.path.isdir(abs_path):
                return abs_path
            else:
                print(f"Error: Path not found or not a directory: {abs_path}")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)

def is_running_in_container():
    """Check if the script is running inside a dev container."""
    # Check for common container environment indicators
    if os.path.exists("/.dockerenv"):
        return True
    if os.environ.get("REMOTE_CONTAINERS") or os.environ.get("CODESPACES"):
        return True
    if os.path.exists("/workspaces") and os.path.isdir("/workspaces"):
        return True
    return False

def get_container_project_path(project_dir_name):
    """Get the expected path of the project inside the container."""
    # In VS Code Dev Containers, workspace folders are mounted at /workspaces/FolderName
    return f"/workspaces/{project_dir_name}"

def update_clinerules(crct_root_dir, project_dir_name, project_root_dir=None):
    """Updates the .clinerules file to include both the project path and CRCT docs path."""
    clinerules_path = os.path.join(crct_root_dir, ".clinerules")
    
    if not os.path.exists(clinerules_path):
        print(f"Warning: .clinerules file not found at {clinerules_path}. Skipping update.")
        return
    
    # Define the required paths inside the container
    user_project_container_path = get_container_project_path(project_dir_name)
    # Assuming CRCT root is mounted at /workspaces/CRCT System
    crct_docs_container_path = "/workspaces/CRCT System/cline_docs" 
    
    required_paths = {user_project_container_path, crct_docs_container_path}
    found_paths = set()
    
    try:
        with open(clinerules_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        in_code_root_section = False
        updated_lines = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            if stripped_line == "[CODE_ROOT_DIRECTORIES]":
                in_code_root_section = True
                updated_lines.append(line)
                # Add comments if they aren't already there (simple check)
                if i + 1 >= len(lines) or not lines[i+1].strip().startswith("# Your project folder"):
                    updated_lines.append("# Your project folder is available at /workspaces/User Project\n")
                    updated_lines.append(f"# For this workspace, consider using: {user_project_container_path}\n")
                    updated_lines.append(f"# CRCT docs path: {crct_docs_container_path}\n")
                continue

            if in_code_root_section:
                # Check if we've left the section
                if stripped_line.startswith("[") and stripped_line != "[CODE_ROOT_DIRECTORIES]":
                    # Add any missing required paths before leaving the section
                    missing_paths = required_paths - found_paths
                    for path in sorted(list(missing_paths)): # Sort for consistent order
                         updated_lines.append(f"- {path}\n")
                    found_paths.update(missing_paths) # Mark as added
                    
                    in_code_root_section = False
                    updated_lines.append(line)
                    continue

                # Check if the line is one of the required paths
                if stripped_line.startswith("-"):
                    path_entry = stripped_line[1:].strip()
                    if path_entry in required_paths:
                        found_paths.add(path_entry)
                
                # Append the original line (including comments, other entries, etc.)
                updated_lines.append(line)

            else:
                # Outside the target section
                updated_lines.append(line)

        # If the file ended while still in the section, add missing paths
        if in_code_root_section:
            missing_paths = required_paths - found_paths
            for path in sorted(list(missing_paths)):
                updated_lines.append(f"- {path}\n")

        # Write the updated content back
        with open(clinerules_path, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)
            
        print(f"✓ Updated .clinerules [CODE_ROOT_DIRECTORIES] with:")
        for path in sorted(list(required_paths)):
             print(f"  - {path}")
             
    except Exception as e:
        print(f"Warning: Could not update .clinerules: {e}")

def ensure_symlink(crct_root_dir, project_dir_name, container_target_path):
    """Create a symlink in the CRCT root pointing to the container target path."""
    symlink_path = os.path.join(crct_root_dir, project_dir_name)
    if not os.path.lexists(symlink_path): # Use lexists to check for broken links too
        try:
            # We create the symlink relative to the container target path
            # This assumes the script is run where the container path is valid,
            # or the link might be broken until the container runs.
            os.symlink(container_target_path, symlink_path, target_is_directory=True)
            print(f"✓ Created symlink: {symlink_path} -> {container_target_path}")
        except OSError as e:
            # Handle cases where symlink creation might fail (e.g., permissions)
            # Or if run outside container where target path isn't valid yet.
            print(f"Warning: Could not create symlink {symlink_path}: {e}")
            print("  Ensure the target path exists inside the container.")
        except Exception as e:
             print(f"Warning: Unexpected error creating symlink {symlink_path}: {e}")
    # Return the relative path (just the project name) for the workspace file
    return project_dir_name

def open_in_vscode(crct_root_dir):
    """Open the CRCT root folder in VS Code and provide guidance for container setup."""
    if is_running_in_container():
        print("Already running inside a container. Skipping VS Code launch.")
        return

    print("\nSetup complete! Opening the CRCT folder in VS Code...")

    try:
        # Determine the command based on the OS
        if sys.platform == "win32":
            # On Windows, use 'start code'
            command = ["cmd", "/c", "start", "code", crct_root_dir]
            subprocess.run(command, check=True, cwd=crct_root_dir) # Run from CRCT root
        elif sys.platform == "darwin":
            # On macOS, use 'open -a "Visual Studio Code"'
            command = ["open", "-a", "Visual Studio Code", crct_root_dir]
            subprocess.run(command, check=True)
        else:
            # On Linux/other Unix, use 'code'
            command = ["code", crct_root_dir]
            # Try running 'code' command directly. Add it to PATH if needed.
            subprocess.run(command, check=True, cwd=crct_root_dir) # Run from CRCT root

        print("\nWhen VS Code opens, if prompted, click 'Reopen in Container'.")
        print("This will rebuild and restart the container with your project mounted.")
        print("You can then access your project folder within the container at:")
        print(f"  /workspaces/{os.path.basename(crct_root_dir)}") # Assuming crct_root_dir name matches target

    except FileNotFoundError:
        print("\nError: 'code' command not found. Please ensure VS Code is installed and its command-line tool is in your system's PATH.")
        print("  - On macOS/Linux: Open VS Code, run 'Shell Command: Install \\'code\\' command in PATH' from the Command Palette (Cmd+Shift+P / Ctrl+Shift+P).")
        print("  - On Windows: Ensure the VS Code installation directory is in your system's PATH environment variable during installation.")
        print("\nYou can manually open the CRCT folder in VS Code:")
        print(f"  {crct_root_dir}")
        print("Then, click 'Reopen in Container' when prompted.")
    except subprocess.CalledProcessError as e:
        print(f"\nError launching VS Code: {e}")
        print("\nYou can manually open the CRCT folder in VS Code:")
        print(f"  {crct_root_dir}")
        print("Then, click 'Reopen in Container' when prompted.")
    except Exception as e:
        print(f"\nAn unexpected error occurred while trying to open VS Code: {e}")
        print("\nYou can manually open the CRCT folder in VS Code:")
        print(f"  {crct_root_dir}")
        print("Then, click 'Reopen in Container' when prompted.")

def main():
    print("CRCT Dev Container Setup")
    print("------------------------")
    print("This script will configure the dev container to include your project.")

    # Get CRCT root directory (assuming script is in cline_utils)
    crct_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --- Clean existing mounts? ---
    clean_choice = input("Do you want to remove all existing project mounts from devcontainer.json first? (y/N): ").strip().lower()
    if clean_choice == 'y':
        clean_mounts_in_devcontainer()
        print("-" * 20)


    # --- Get user's project path ---
    project_root_dir = get_project_path()
    project_dir_name = os.path.basename(project_root_dir)
    print(f"Selected project folder: {project_root_dir}")
    print(f"Project directory name: {project_dir_name}")
    print("-" * 20)

    # --- Determine target path in container ---
    container_target_path = get_container_project_path(project_dir_name)
    print(f"Project will be mounted at: {container_target_path}")

    # --- Add mount to devcontainer.json ---
    print("Updating devcontainer.json...")
    add_mount_to_devcontainer(project_root_dir, container_target_path)
    print("-" * 20)

    # --- Update .clinerules ---
    print("Updating .clinerules...")
    update_clinerules(crct_root_dir, project_dir_name)
    print("-" * 20)

    # --- Ensure symlink ---
    print("Ensuring symlink exists...")
    # The symlink should point to the container path
    # Ensure the symlink name is just the project directory name
    relative_symlink_path = ensure_symlink(crct_root_dir, project_dir_name, container_target_path)
    print("-" * 20)

    # --- Final Instructions ---
    print("\n---------------------------------------------------------------------")
    print("Configuration complete!")
    print("The devcontainer.json and .clinerules files have been updated.")
    print(f"A symlink named '{relative_symlink_path}' pointing to your project's container location")
    print(f"('{container_target_path}') should now exist in the CRCT root directory:")
    print(f"  {os.path.join(crct_root_dir, relative_symlink_path)}")
    print("\nImportant Next Steps:")
    print("1. If VS Code is currently open in this CRCT folder, close it.")
    print("2. If you are currently inside the dev container, exit it.")
    # print("3. The script will now attempt to open the CRCT folder in VS Code.") # Removed this line
    # print("4. When prompted by VS Code, click 'Reopen in Container'.") # Removed this line
    # print("   This rebuilds the container with your project mounted.") # Removed this line
    print("---------------------------------------------------------------------")

    # --- Open in VS Code ---
    open_in_vscode(crct_root_dir) # Pass the CRCT root dir


if __name__ == "__main__":
    main()
