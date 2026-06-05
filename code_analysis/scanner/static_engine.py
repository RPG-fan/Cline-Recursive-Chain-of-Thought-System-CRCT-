"""
Static analysis engine for CRCT report generator.
Handles regex-based and tree-sitter AST-based code scanning.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Tree-sitter configuration
_has_tree_sitter = False
_Language: Any = None
_Parser: Any = None
_ts_js: Any = None
_ts_py: Any = None
_ts_ts: Any = None

try:
    import tree_sitter_javascript as _ts_js
    import tree_sitter_python as _ts_py
    import tree_sitter_typescript as _ts_ts
    from tree_sitter import Language as _Language, Parser as _Parser

    _has_tree_sitter = True
except ImportError:
    # This will be reported by the orchestrator if needed
    pass

# ---------------------------------------------------------------------------
# Configuration / Constants
# ---------------------------------------------------------------------------
EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt"}

PATTERNS = {
    "TODO": re.compile(r"TODO", re.IGNORECASE),
    "FIXME": re.compile(r"FIXME", re.IGNORECASE),
    "pass": re.compile(r"^\s*pass\s*$", re.MULTILINE),
    "NotImplementedError": re.compile(r"NotImplementedError"),
    "in a real": re.compile(r"in a real", re.IGNORECASE),
    "for now": re.compile(r"for now", re.IGNORECASE),
    "simplified": re.compile(r"simplified", re.IGNORECASE),
    "placeholder": re.compile(r"placeholder", re.IGNORECASE),
}

EXCLUSION_PATTERNS = {
    "placeholder": [
        re.compile(r"_placeholder", re.IGNORECASE),
        re.compile(r"placeholder_", re.IGNORECASE),
        re.compile(r"sql\.Placeholder", re.IGNORECASE),
        re.compile(r"placeholders\s*=", re.IGNORECASE),
        re.compile(r"placeholder\s*=", re.IGNORECASE),
        re.compile(r"placeholder\s*:", re.IGNORECASE),
        re.compile(r"Placeholder\(\)", re.IGNORECASE),
        re.compile(r"\.placeholder\.", re.IGNORECASE),
        re.compile(r'"placeholder"', re.IGNORECASE),
    ],
}

PYRIGHT_OUTPUT = "pyright_output.json"


# ===========================================================================
# Tree-sitter helpers
# ===========================================================================
def get_parser(lang_name: str) -> Any:
    """Initialize and return a tree-sitter parser for the given language."""
    if not _has_tree_sitter or _Parser is None:
        return None
    try:
        parser = _Parser()
        if lang_name == "python":
            parser.language = _Language(_ts_py.language())
        elif lang_name == "javascript":
            parser.language = _Language(_ts_js.language())
        elif lang_name == "typescript":
            parser.language = _Language(_ts_ts.language_typescript())
        elif lang_name == "tsx":
            parser.language = _Language(_ts_ts.language_tsx())
        else:
            return None
        return parser
    except Exception as e:
        print(f"Error initializing parser for {lang_name}: {e}")
        return None


def analyze_node(
    node: Any,
    issues: List[Dict[str, Any]],
    filepath: Union[str, Path],
    source_code: bytes,
) -> None:
    """Recursively walk tree-sitter nodes; emit raw issue dicts."""
    if node.type in ("function_definition", "async_function_definition"):
        body_node = node.child_by_field_name("body")
        if body_node:
            has_raise_not_implemented = False
            non_trivial_children: List[Any] = []
            for child in body_node.children:
                if child.type == "comment":
                    continue
                if child.type == "pass_statement":
                    continue
                if child.type == "expression_statement":
                    if child.child_count == 1 and child.children[0].type == "string":
                        continue
                if child.type == "raise_statement":
                    if "NotImplementedError" in child.text.decode("utf-8", errors="replace"):
                        has_raise_not_implemented = True
                        continue
                non_trivial_children.append(child)

            if not non_trivial_children:
                kind = (
                    "NotImplementedError"
                    if has_raise_not_implemented
                    else "Empty/Stub Function"
                )
                issues.append(
                    {
                        "type": (
                            "Incomplete Implementation"
                            if has_raise_not_implemented
                            else "Improper Implementation"
                        ),
                        "subtype": kind,
                        "file": str(filepath),
                        "line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "content": node.text.partition(b"\n")[0].rstrip(b"\r").decode("utf-8", errors="replace") + "...",
                    }
                )

    elif node.type == "class_definition":
        body_node = node.child_by_field_name("body")
        if body_node:
            non_trivial_children: List[Any] = []
            for child in body_node.children:
                if child.type == "comment":
                    continue
                if child.type == "pass_statement":
                    continue
                if child.type == "expression_statement":
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
                        "end_line": node.end_point[0] + 1,
                        "content": node.text.partition(b"\n")[0].rstrip(b"\r").decode("utf-8", errors="replace") + "...",
                    }
                )

    elif node.type in (
        "function_declaration",
        "method_definition",
        "arrow_function",
        "class_declaration",
    ):
        body_node = node.child_by_field_name("body")
        if body_node and body_node.type in ("statement_block", "class_body"):
            non_trivial_children = []
            has_throw_not_implemented = False
            for child in body_node.children:
                if child.type in ("comment", "{", "}"):
                    continue
                if child.type == "throw_statement":
                    child_text = child.text.decode("utf-8", errors="replace")
                    if "Not implemented" in child_text or "NotImplementedError" in child_text or "not implemented" in child_text:
                        has_throw_not_implemented = True
                        continue
                non_trivial_children.append(child)
            
            if not non_trivial_children:
                kind = (
                    "NotImplementedError"
                    if has_throw_not_implemented
                    else "Empty/Stub Function/Class"
                )
                issues.append(
                    {
                        "type": (
                            "Incomplete Implementation"
                            if has_throw_not_implemented
                            else "Improper Implementation"
                        ),
                        "subtype": kind,
                        "file": str(filepath),
                        "line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "content": node.text.partition(b"\n")[0].rstrip(b"\r").decode("utf-8", errors="replace") + "...",
                    }
                )

    for child in node.children:
        analyze_node(child, issues, filepath, source_code)


def scan_file(filepath: str) -> List[Dict[str, Any]]:
    """Per-file static scan (regex + tree-sitter)."""
    issues: List[Dict[str, Any]] = []
    try:
        with open(filepath, "rb") as f:
            content = f.read()

        ext = Path(filepath).suffix
        lang = None
        if ext == ".py":
            lang = "python"
        elif ext == ".js":
            lang = "javascript"
        elif ext == ".ts":
            lang = "typescript"
        elif ext in (".jsx", ".tsx"):
            lang = "tsx"

        parser = get_parser(lang) if lang else None
        is_parsed = parser is not None

        try:
            text_content = content.decode("utf-8", errors="ignore")
            lines = text_content.splitlines()
            for i, line in enumerate(lines):
                for label, pattern in PATTERNS.items():
                    if is_parsed and label in ("pass", "NotImplementedError"):
                        continue
                    if pattern.search(line):
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

                if not is_parsed and "def " in line and "pass" in line:
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

        if is_parsed and parser:
            tree = parser.parse(content)
            analyze_node(tree.root_node, issues, filepath, content)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return issues


def get_unused_items(project_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parse pyright output for unused items."""
    unused: List[Dict[str, Any]] = []
    target_path = (
        os.path.join(project_root, PYRIGHT_OUTPUT) if project_root else PYRIGHT_OUTPUT
    )
    if os.path.exists(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
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
        # Warning printed by orchestrator if needed
        pass
    return unused
