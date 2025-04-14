import os
import subprocess

def read_code_roots(clinerules_path):
    roots = []
    in_section = False
    with open(clinerules_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "[CODE_ROOT_DIRECTORIES]":
                in_section = True
                continue
            if in_section:
                if line.strip().startswith("[") and not line.strip().startswith("[CODE_ROOT_DIRECTORIES]"):
                    break
                if line.strip().startswith("-"):
                    path = line.strip()[1:].strip()
                    if path:
                        roots.append(path)
    return roots

def main(args):
    crct_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    clinerules_path = os.path.join(crct_root, ".clinerules")
    if not os.path.exists(clinerules_path):
        print(f".clinerules not found at {clinerules_path}")
        return
    code_roots = read_code_roots(clinerules_path)
    if not code_roots:
        print("No code roots found in .clinerules [CODE_ROOT_DIRECTORIES].")
        return
    dep_proc = os.path.join(crct_root, "cline_utils", "dependency_system", "dependency_processor.py")
    if not os.path.exists(dep_proc):
        print(f"dependency_processor.py not found at {dep_proc}")
        return
    for root in code_roots:
        print(f"Analyzing dependencies for: {root}")
        try:
            subprocess.run(
                ["python3", dep_proc, "--code-root", root],
                cwd=crct_root,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Dependency analysis failed for {root}: {e}")
    print("Dependency initialization complete.")
