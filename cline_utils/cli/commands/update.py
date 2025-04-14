import os
import subprocess
import shutil
from datetime import datetime

def backup_config(crct_root):
    backup_dir = os.path.join(crct_root, "crct_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(backup_dir)
    for fname in [".clinerules", "cline_docs", "docs"]:
        src = os.path.join(crct_root, fname)
        if os.path.exists(src):
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(backup_dir, fname))
            else:
                shutil.copy2(src, backup_dir)
    print(f"Backup created at {backup_dir}")

def main(args):
    crct_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Backing up configuration files...")
    backup_config(crct_root)
    print("Checking for CRCT updates (git pull)...")
    try:
        subprocess.run(["git", "pull"], cwd=crct_root, check=True)
        print("CRCT updated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Update failed: {e}")
