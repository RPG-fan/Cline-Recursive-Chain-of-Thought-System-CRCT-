import pytest
from code_analysis.reporting.markdown_formatter import format_markdown

def test_format_markdown_empty(tmp_path):
    output_file = tmp_path / "output.md"
    format_markdown([], [], str(output_file))

    content = output_file.read_text(encoding="utf-8")
    assert "No incomplete items found." in content
    assert "No unused items found (or pyright output missing)." in content
    assert "- Total issues: **0**" in content
    assert "- Unused items (pyright): 0" in content

def test_format_markdown_with_issues(tmp_path):
    issues = [
        {
            "subtype": "TestSubtype",
            "file": "test_file.py",
            "line": 42,
            "content": "def foo(): pass",
            "context": {
                "severity": "high",
                "owning_symbol": {
                    "qualname": "MyClass.foo",
                    "signature": "(x: int)",
                    "kind": "method"
                },
                "type_annotations": "some_type",
                "decorators": ["@staticmethod"],
                "inheritance": {
                    "bases": ["Base1"],
                    "mro": ["MyClass", "Base1", "object"]
                },
                "scope_references": {
                    "globals": ["g1", "g2"],
                    "nonlocals": ["nl1"]
                },
                "closure_dependencies": ["cd1", "cd2"],
                "linked_areas": {
                    "caller_count": 2,
                    "callers": ["caller1", "caller2"]
                },
                "exported": True
            }
        }
    ]
    output_file = tmp_path / "output.md"
    format_markdown(issues, [], str(output_file))

    content = output_file.read_text(encoding="utf-8")

    # Check severity, subtype, file, line
    assert "- **[HIGH] TestSubtype** in `test_file.py:42`" in content

    # Check code content
    assert "  ```\n  def foo(): pass\n  ```\n" in content

    # Check owning symbol
    assert "- **Owning symbol**: `MyClass.foo`(x: int) (method)" in content

    # Check type annotations
    assert "- **Types**: `some_type`" in content

    # Check decorators
    assert "- **Decorators**: ['@staticmethod']" in content

    # Check inheritance
    assert "- **Inheritance**: bases=['Base1'] mro=['MyClass', 'Base1', 'object']" in content

    # Check scope references
    assert "- **Scope refs**: globals=['g1', 'g2'] nonlocals=['nl1']" in content

    # Check closure dependencies
    assert "- **Closure deps**: ['cd1', 'cd2']" in content

    # Check linked areas
    assert "- **Linked areas** (2 files): `caller1`, `caller2`" in content

    # Check exported
    assert "- **Exported**: yes (in `__all__`)" in content

def test_format_markdown_sorting(tmp_path):
    issues = [
        {"subtype": "low2", "file": "b.py", "line": 1, "context": {"severity": "low"}},
        {"subtype": "med1", "file": "a.py", "line": 2, "context": {"severity": "medium"}},
        {"subtype": "high1", "file": "a.py", "line": 1, "context": {"severity": "high"}},
        {"subtype": "low1", "file": "a.py", "line": 1, "context": {"severity": "low"}},
        {"subtype": "crit1", "file": "a.py", "line": 1, "context": {"severity": "critical"}},
        {"subtype": "low_line2", "file": "a.py", "line": 2, "context": {"severity": "low"}},
    ]
    output_file = tmp_path / "output.md"
    format_markdown(issues, [], str(output_file))

    content = output_file.read_text(encoding="utf-8")

    lines = content.split('\n')
    issue_lines = [line for line in lines if line.startswith("- **[")]

    assert len(issue_lines) == 6
    assert "crit1" in issue_lines[0]
    assert "high1" in issue_lines[1]
    assert "med1" in issue_lines[2]
    assert "low1" in issue_lines[3]
    assert "low_line2" in issue_lines[4]
    assert "low2" in issue_lines[5]

    # Check Summary numbers
    assert "- Critical: 1" in content
    assert "- High: 1" in content
    assert "- Medium: 1" in content
    assert "- Low: 3" in content


def test_format_markdown_unused(tmp_path):
    unused = [
        {
            "subtype": "UnusedVar",
            "file": "utils.py",
            "line": 10,
            "content": "x = 42"
        }
    ]
    output_file = tmp_path / "output.md"
    format_markdown([], unused, str(output_file))

    content = output_file.read_text(encoding="utf-8")

    assert "- **UnusedVar** in `utils.py:10`" in content
    assert "> x = 42" in content

def test_format_markdown_missing_context_and_keys(tmp_path):
    issues = [{}]
    output_file = tmp_path / "output.md"
    format_markdown(issues, [], str(output_file))

    content = output_file.read_text(encoding="utf-8")
    assert "- **[LOW] ?** in `?:?`" in content

def test_format_markdown_owning_symbol_no_signature(tmp_path):
    issues = [
        {
            "context": {
                "owning_symbol": {
                    "qualname": "MyClass.foo",
                    "kind": "method"
                }
            }
        }
    ]
    output_file = tmp_path / "output.md"
    format_markdown(issues, [], str(output_file))

    content = output_file.read_text(encoding="utf-8")
    assert "- **Owning symbol**: `MyClass.foo` (method)" in content

def test_format_markdown_inheritance_empty(tmp_path):
    issues = [
        {
            "context": {
                "inheritance": {
                    "bases": [],
                    "mro": []
                }
            }
        }
    ]
    output_file = tmp_path / "output.md"
    format_markdown(issues, [], str(output_file))

    content = output_file.read_text(encoding="utf-8")
    assert "- **Inheritance**:" not in content

def test_format_markdown_scope_references_truncation(tmp_path):
    issues = [
        {
            "context": {
                "scope_references": {
                    "globals": [f"g{i}" for i in range(15)],
                    "nonlocals": [f"nl{i}" for i in range(15)]
                }
            }
        }
    ]
    output_file = tmp_path / "output.md"
    format_markdown(issues, [], str(output_file))

    content = output_file.read_text(encoding="utf-8")

    expected_g = [f"g{i}" for i in range(10)]
    expected_nl = [f"nl{i}" for i in range(10)]

    assert f"- **Scope refs**: globals={expected_g} nonlocals={expected_nl}" in content

    # Asserting that the 11th items are not in the string at all.
    assert "g10" not in content
    assert "nl10" not in content
