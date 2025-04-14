import os

TEMPLATES = {
    "python": {
        "dirs": ["src", "tests"],
        "files": {
            "README.md": "# New Python Project\n",
            "src/__init__.py": "",
            "tests/__init__.py": "",
            "requirements.txt": "",
            "main.py": "#!/usr/bin/env python3\n\nif __name__ == '__main__':\n    print('Hello, world!')\n"
        }
    },
    "js": {
        "dirs": ["src", "tests"],
        "files": {
            "README.md": "# New JavaScript Project\n",
            "src/index.js": "// Entry point\nconsole.log('Hello, world!');\n",
            "tests/index.test.js": "// Tests\n",
            "package.json": '{\n  "name": "new-js-project",\n  "version": "1.0.0"\n}\n'
        }
    }
}

def prompt_project_type():
    print("Select project type:")
    for i, key in enumerate(TEMPLATES.keys(), 1):
        print(f"  {i}) {key}")
    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(TEMPLATES):
            return list(TEMPLATES.keys())[int(choice) - 1]
        print("Invalid choice.")

def main(args):
    print("=== CRCT Project Template Generator ===")
    project_type = prompt_project_type()
    name = input("Project name: ").strip()
    base_path = input("Directory to create project in (default: current): ").strip() or os.getcwd()
    project_path = os.path.join(base_path, name)
    if os.path.exists(project_path):
        print(f"Directory already exists: {project_path}")
        return
    os.makedirs(project_path)
    # Create directories
    for d in TEMPLATES[project_type]["dirs"]:
        os.makedirs(os.path.join(project_path, d))
    # Create files
    for rel_path, content in TEMPLATES[project_type]["files"].items():
        file_path = os.path.join(project_path, rel_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Project '{name}' created at {project_path}")
