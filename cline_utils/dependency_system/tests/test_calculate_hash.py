import pytest
from pathlib import Path
from cline_utils.dependency_system.utils.calculate_hash import (
    calculate_content_hash,
    _normalize_content_for_hashing,
)


def test_line_endings_normalization():
    """Verify that LF, CRLF, and CR line endings result in identical normalized content and hashes."""
    lf_content = "def foo():\n    x = 1\n    return x\n"
    crlf_content = "def foo():\r\n    x = 1\r\n    return x\r\n"
    cr_content = "def foo():\r    x = 1\r    return x\r"

    assert _normalize_content_for_hashing(lf_content) == _normalize_content_for_hashing(crlf_content)
    assert _normalize_content_for_hashing(lf_content) == _normalize_content_for_hashing(cr_content)

    assert calculate_content_hash(lf_content) == calculate_content_hash(crlf_content)
    assert calculate_content_hash(lf_content) == calculate_content_hash(cr_content)


def test_arbitrary_indentation_normalization():
    """Verify that files with arbitrary indentation or spacing normalize to the same hash."""
    base_content = "def foo():\n    x = 1\n    return x"
    indented_spaces_2 = "  def foo():\n    x = 1\n    return x"
    indented_spaces_8 = "        def foo():\n        x = 1\n        return x"
    indented_tabs = "\tdef foo():\n\t\tx = 1\n\t\treturn x"

    # All should strip down to the same code block (without indentation)
    expected_normalized = "def foo():\nx = 1\nreturn x"
    assert _normalize_content_for_hashing(base_content) == expected_normalized
    assert _normalize_content_for_hashing(indented_spaces_2) == expected_normalized
    assert _normalize_content_for_hashing(indented_spaces_8) == expected_normalized
    assert _normalize_content_for_hashing(indented_tabs) == expected_normalized

    hash_base = calculate_content_hash(base_content)
    assert calculate_content_hash(indented_spaces_2) == hash_base
    assert calculate_content_hash(indented_spaces_8) == hash_base
    assert calculate_content_hash(indented_tabs) == hash_base


def test_comment_formatting_normalization():
    """Verify that different comment formats and trailing comment spacing normalize identically."""
    py_content_1 = "x = 1 # some comment"
    py_content_2 = "x = 1      #   some comment"
    py_content_3 = "x = 1#some comment"

    assert _normalize_content_for_hashing(py_content_1, "test.py") == "x = 1 # some comment"
    assert _normalize_content_for_hashing(py_content_2, "test.py") == "x = 1 # some comment"
    assert _normalize_content_for_hashing(py_content_3, "test.py") == "x = 1 # some comment"

    hash_py_1 = calculate_content_hash(py_content_1, "test.py")
    assert calculate_content_hash(py_content_2, "test.py") == hash_py_1
    assert calculate_content_hash(py_content_3, "test.py") == hash_py_1


def test_comments_inside_string_literals():
    """Verify that comment prefixes inside string literals are NOT normalized or stripped."""
    content = 'url = "http://example.com/test#anchor"\n# real comment'
    normalized = _normalize_content_for_hashing(content, "test.py")
    
    assert 'url = "http://example.com/test#anchor"' in normalized
    assert '# real comment' in normalized


def test_html_comment_normalization():
    """Verify HTML/Markdown comment normalization works correctly."""
    md_content_1 = "<!--   some md comment   -->"
    md_content_2 = "<!--some md comment-->"
    md_content_3 = "  <!-- some md comment -->"

    expected = "<!-- some md comment -->"
    assert _normalize_content_for_hashing(md_content_1, "test.md") == expected
    assert _normalize_content_for_hashing(md_content_2, "test.md") == expected
    assert _normalize_content_for_hashing(md_content_3, "test.md") == expected


def test_empty_lines_normalization():
    """Verify that empty lines are completely ignored during hash normalization."""
    content_1 = "x = 1\n\n\ny = 2\n"
    content_2 = "x = 1\ny = 2"

    assert _normalize_content_for_hashing(content_1) == _normalize_content_for_hashing(content_2)
    assert calculate_content_hash(content_1) == calculate_content_hash(content_2)
