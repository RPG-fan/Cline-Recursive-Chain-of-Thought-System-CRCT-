import os

def run_health_check():
    print("=== CRCT Container Health Check ===")
    # Check if running in a container
    in_container = os.path.exists("/.dockerenv") or os.environ.get("REMOTE_CONTAINERS") or os.environ.get("CODESPACES")
    print(f"Running in container: {'Yes' if in_container else 'No'}")

    # Check mounts
    mounts = []
    try:
        with open("/proc/mounts", "r") as f:
            for line in f:
                if "/workspaces/" in line:
                    mounts.append(line.strip())
    except Exception:
        print("Could not read /proc/mounts to check mounts.")

    if mounts:
        print("Mounted workspace folders:")
        for m in mounts:
            print("  " + m)
    else:
        print("No /workspaces/ mounts detected.")

    # Check for Cline extension (basic check)
    ext_dir = "/root/.vscode-server/extensions"
    found_cline = False
    if os.path.isdir(ext_dir):
        for d in os.listdir(ext_dir):
            if "cline" in d.lower():
                found_cline = True
                print(f"Cline extension found: {d}")
    if not found_cline:
        print("Cline extension not found in VS Code extensions directory.")

    # Check for dependency_processor.py
    dep_proc = "/workspaces/CRCT System/cline_utils/dependency_system/dependency_processor.py"
    if os.path.exists(dep_proc):
        print("dependency_processor.py found.")
    else:
        print("dependency_processor.py NOT found!")

    print("Health check complete.")

def main(args):
    run_health_check()
