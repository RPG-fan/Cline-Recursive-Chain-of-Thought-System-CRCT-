import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from cline_utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cline_utils.dependency_system.utils import path_utils
from cline_utils.dependency_system.utils.config_manager import ConfigManager

# Tree-sitter imports
try:
    import tree_sitter_javascript
    import tree_sitter_python
    import tree_sitter_typescript
    from tree_sitter import Language, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("Warning: tree-sitter not available. Falling back to regex-based analysis.")

# Initialize ConfigManager
config = ConfigManager()

# Configuration
EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt"}

PATTERNS = {
    "TODO": re.compile(r"TODO", re.IGNORECASE),
    "FIXME": re.compile(r"FIXME", re.IGNORECASE),
    "pass": re.compile(r"^\s*pass\s*$", re.MULTILINE),  # stricter pass check
    "NotImplementedError": re.compile(r"NotImplementedError"),
    "in a real": re.compile(r"in a real", re.IGNORECASE),
    "for now": re.compile(r"for now", re.IGNORECASE),
    "simplified": re.compile(r"simplified", re.IGNORECASE),
    "placeholder": re.compile(r"placeholder", re.IGNORECASE),
}

# Patterns to exclude from false positive matches (e.g., sql.Placeholder, include_placeholder=False)
EXCLUSION_PATTERNS = {
    "placeholder": [
        re.compile(r"_placeholder", re.IGNORECASE),  # _placeholder (variable suffix)
        re.compile(r"placeholder_", re.IGNORECASE),  # placeholder_ (variable prefix)
        re.compile(r"sql\.Placeholder", re.IGNORECASE),  # sql.Placeholder()
        re.compile(r"placeholders\s*=", re.IGNORECASE),  # placeholders = ...
        re.compile(r"placeholder\s*=", re.IGNORECASE),  # placeholder=False
        re.compile(r"placeholder\s*:", re.IGNORECASE),  # placeholder: type hint
        re.compile(r"Placeholder\(\)", re.IGNORECASE),  # Placeholder() call
        re.compile(r"\.placeholder\.", re.IGNORECASE),  # .placeholder. (method calls)
        re.compile(r'"placeholder"', re.IGNORECASE),  # "placeholder" (string literal)
    ],
}

OUTPUT_FILE = "code_analysis/issues_report.md"
PYRIGHT_OUTPUT = "pyright_output.json"


def get_parser(lang_name):
    if not TREE_SITTER_AVAILABLE:
        return None

    try:
        parser = Parser()
        if lang_name == "python":
            parser.language = Language(tree_sitter_python.language())
        elif lang_name == "javascript":
            parser.language = Language(tree_sitter_javascript.language())
        elif lang_name == "typescript":
            parser.language = Language(tree_sitter_typescript.language_typescript())
        elif lang_name == "tsx":
            parser.language = Language(tree_sitter_typescript.language_tsx())
        else:
            return None
        return parser
    except Exception as e:
        print(f"Error initializing parser for {lang_name}: {e}")
        return None


def analyze_node(node, issues, filepath, source_code):
    """Recursively analyze tree-sitter nodes."""

    # Python checks
    if node.type in ("function_definition", "async_function_definition"):
        # Check for empty body or pass/docstring only
        body_node = node.child_by_field_name("body")
        if body_node:
            has_raise_not_implemented = False

            # Filter out trivial children (comments, pass, docstrings)
            non_trivial_children = []

            for child in body_node.children:
                if child.type == "comment":
                    continue
                if child.type == "pass_statement":
                    continue
                if child.type == "expression_statement":
                    # Check if it's a docstring (string literal)
                    if child.child_count == 1 and child.children[0].type == "string":
                        continue

                if child.type == "raise_statement":
                    # Check if raising NotImplementedError
                    if "NotImplementedError" in child.text.decode("utf8"):
                        has_raise_not_implemented = True
                        # We count this as "trivial" for the purpose of finding *other* code,
                        # but we flag it specifically later.
                        continue

                non_trivial_children.append(child)

            if not non_trivial_children:
                if has_raise_not_implemented:
                    issues.append(
                        {
                            "type": "Incomplete Implementation",
                            "subtype": "NotImplementedError",
                            "file": str(filepath),
                            "line": node.start_point[0] + 1,
                            "content": node.text.decode("utf8").split("\n")[0] + "...",
                        }
                    )
                else:
                    issues.append(
                        {
                            "type": "Improper Implementation",
                            "subtype": "Empty/Stub Function",
                            "file": str(filepath),
                            "line": node.start_point[0] + 1,
                            "content": node.text.decode("utf8").split("\n")[0] + "...",
                        }
                    )

    elif node.type == "class_definition":
        body_node = node.child_by_field_name("body")
        if body_node:
            non_trivial_children = []
            for child in body_node.children:
                if child.type == "comment":
                    continue
                if child.type == "pass_statement":
                    continue
                if child.type == "expression_statement":
                    # Check if it's a docstring (string literal)
                    if child.child_count == 1 and child.children[0].type == "string":
                        continue
                non_trivial_children.append(child)

            if not non_trivial_children:
                issues.append(
                    {
                        "type": "Improper Implementation",
                        "subtype": "Empty/Stub Class",
                        "file": str(filepath),
                        "line": node.start_point[0] + 1,
                        "content": node.text.decode("utf8").split("\n")[0] + "...",
                    }
                )

    # JS/TS checks
    elif node.type in (
        "function_declaration",
        "method_definition",
        "arrow_function",
        "class_declaration",
    ):
        body_node = node.child_by_field_name("body")
        if body_node and body_node.type == "statement_block":
            # Check if block is empty or only comments
            non_comment_children = [
                c for c in body_node.children if c.type not in ("comment", "{", "}")
            ]
            if not non_comment_children:
                issues.append(
                    {
                        "type": "Improper Implementation",
                        "subtype": "Empty/Stub Function/Class",
                        "file": str(filepath),
                        "line": node.start_point[0] + 1,
                        "content": node.text.decode("utf8").split("\n")[0] + "...",
                    }
                )

    # Recurse
    for child in node.children:
        analyze_node(child, issues, filepath, source_code)


def scan_file(filepath):
    issues = []
    try:
        with open(filepath, "rb") as f:  # Read as binary for tree-sitter
            content = f.read()

        # Regex scanning (always run for comments/patterns)
        try:
            text_content = content.decode("utf-8", errors="ignore")
            lines = text_content.splitlines()
            for i, line in enumerate(lines):
                for label, pattern in PATTERNS.items():
                    # Skip 'pass' and 'NotImplementedError' regex if tree-sitter is active for this file
                    ext = Path(filepath).suffix
                    is_parsed = TREE_SITTER_AVAILABLE and ext in (
                        ".py",
                        ".js",
                        ".ts",
                        ".jsx",
                        ".tsx",
                    )

                    if is_parsed and label in ("pass", "NotImplementedError"):
                        continue

                    if pattern.search(line):
                        # Check if this matches any exclusion pattern for this label
                        excluded = False
                        if label in EXCLUSION_PATTERNS:
                            for excl_pattern in EXCLUSION_PATTERNS[label]:
                                if excl_pattern.search(line):
                                    excluded = True
                                    break

                        if excluded:
                            continue

                        issues.append(
                            {
                                "type": "Incomplete/Improper",
                                "subtype": label,
                                "file": str(filepath),
                                "line": i + 1,
                                "content": line.strip(),
                            }
                        )

                # Fallback for simplistic check if tree-sitter not available
                if not TREE_SITTER_AVAILABLE and "def " in line and "pass" in line:
                    issues.append(
                        {
                            "type": "Improper Implementation",
                            "subtype": "One-line stub",
                            "file": str(filepath),
                            "line": i + 1,
                            "content": line.strip(),
                        }
                    )

        except Exception as e:
            print(f"Error doing regex scan on {filepath}: {e}")

        # Tree-sitter scanning
        if TREE_SITTER_AVAILABLE:
            ext = Path(filepath).suffix
            lang = None
            if ext == ".py":
                lang = "python"
            elif ext == ".js":
                lang = "javascript"
            elif ext == ".ts":
                lang = "typescript"
            elif ext in (".jsx", ".tsx"):
                lang = "tsx"  # simplified

            if lang:
                parser = get_parser(lang)
                if parser:
                    tree = parser.parse(content)
                    analyze_node(tree.root_node, issues, filepath, content)

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return issues


def get_unused_items():
    unused = []
    if os.path.exists(PYRIGHT_OUTPUT):
        try:
            with open(PYRIGHT_OUTPUT, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Pyright output structure varies, assuming standard JSON output
                # We look for "diagnostics" with "message" containing "is not accessed"
                if "generalDiagnostics" in data:
                    for diag in data["generalDiagnostics"]:
                        if "is not accessed" in diag.get("message", ""):
                            unused.append(
                                {
                                    "type": "Unused Item",
                                    "subtype": "Pyright Diagnostic",
                                    "file": diag.get("file", "unknown"),
                                    "line": diag.get("range", {})
                                    .get("start", {})
                                    .get("line", 0)
                                    + 1,
                                    "content": diag.get("message", ""),
                                }
                            )
        except Exception as e:
            print(f"Error parsing pyright output: {e}")
    else:
        print(f"Warning: {PYRIGHT_OUTPUT} not found. Skipping unused item analysis.")
    return unused


def generate_report(issues, unused):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# Code Analysis Issues Report\n\n")

        f.write("## Incomplete & Improper Items\n")
        if issues:
            # Sort by file, subtype, and content for grouping
            issues.sort(
                key=lambda x: (x["file"], x["subtype"], x["content"], x["line"])
            )

            # Group issues by (file, subtype, content)
            grouped = {}
            for issue in issues:
                key = (issue["file"], issue["subtype"], issue["content"])
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(issue["line"])

            # Sort groups by file and first line
            sorted_groups = sorted(grouped.items(), key=lambda x: (x[0][0], x[1][0]))

            for (filepath, subtype, content), lines in sorted_groups:
                if len(lines) == 1:
                    # Single occurrence - original format
                    f.write(f"- **{subtype}** in `{filepath}:{lines[0]}`\n")
                    f.write(f"  ```\n  {content}\n  ```\n")
                else:
                    # Multiple occurrences - condensed format
                    lines_str = ", ".join(str(ln) for ln in sorted(lines))
                    f.write(f"- **{subtype}** in `{filepath}` (lines: {lines_str})\n")
                    f.write(f"  ```\n  {content}\n  ```\n")
        else:
            f.write("No incomplete items found.\n")

        f.write("\n## Unused Items\n")
        if unused:
            for item in unused:
                f.write(f"- **{item['subtype']}** in `{item['file']}:{item['line']}`\n")
                f.write(f"  > {item['content']}\n")
        else:
            f.write("No unused items found (or pyright output missing).\n")


def main():
    all_issues = []

    # Get configuration from ConfigManager
    code_roots = config.get_code_root_directories()
    excluded_paths = config.get_excluded_paths()

    # Run pyright to generate analysis data for unused items
    try:
        print("Running pyright for unused item analysis...")
        with open(PYRIGHT_OUTPUT, "w") as f:
            result = subprocess.run(
                ["pyright", "--outputjson"],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=project_root,
            )
        if result.returncode == 0:
            print("Pyright analysis completed successfully.")
        else:
            print(
                f"Pyright completed with warnings/errors (exit code {result.returncode}). Output file generated."
            )
    except Exception as e:
        print(f"Warning: Unexpected error running pyright: {e}")

    print(f"Scanning code roots: {code_roots}")

    # Walk through directories
    for root_dir in code_roots:
        # Resolve root_dir relative to project root if needed, but ConfigManager usually returns relative to project root or absolute
        # Let's ensure we are walking from the project root + code_root

        # We need to handle if code_roots are relative or absolute.
        # ConfigManager.get_code_root_directories() returns normalized paths, likely relative to project root if they were defined that way in .clinerules

        # Assuming running from project root
        start_dir = root_dir
        if not os.path.exists(start_dir):
            print(f"Warning: Code root {start_dir} does not exist. Skipping.")
            continue

        for root, dirs, files in os.walk(start_dir):
            # Filter directories
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [
                d
                for d in dirs
                if not path_utils.is_path_excluded(
                    os.path.join(root, d), excluded_paths
                )
            ]

            for file in files:
                filepath = os.path.join(root, file)

                if path_utils.is_path_excluded(filepath, excluded_paths):
                    continue

                ext = Path(file).suffix
                if ext not in EXTENSIONS:
                    continue

                all_issues.extend(scan_file(filepath))

    unused_items = get_unused_items()

    generate_report(all_issues, unused_items)
    print(f"Report generated at {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
