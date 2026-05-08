from unittest.mock import patch, mock_open

from code_analysis.scanner.static_engine import scan_file
from code_analysis.scanner.static_engine import get_unused_items

def test_get_unused_items_open_exception(capsys):
    """Test get_unused_items when open raises an exception."""
    with patch("code_analysis.scanner.static_engine.os.path.exists", return_value=True):
        with patch("code_analysis.scanner.static_engine.open", side_effect=Exception("Test open exception")):
            unused = get_unused_items()
            assert unused == []
            captured = capsys.readouterr()
            assert "Error parsing pyright output: Test open exception" in captured.out

def test_get_unused_items_invalid_json(capsys):
    """Test get_unused_items when JSON is invalid."""
    with patch("code_analysis.scanner.static_engine.os.path.exists", return_value=True):
        with patch("code_analysis.scanner.static_engine.open", mock_open(read_data="invalid json")):
            unused = get_unused_items()
            assert unused == []
            captured = capsys.readouterr()
            assert "Error parsing pyright output:" in captured.out
            assert "Expecting value" in captured.out


def test_scan_file_no_tree_sitter_one_line_stub():
    """Test scan_file without tree-sitter when encountering a one-line stub."""
    mock_content = "def test_func(): pass\n"

    with patch("code_analysis.scanner.static_engine.open", mock_open(read_data=mock_content.encode("utf-8"))):
        with patch("code_analysis.scanner.static_engine._has_tree_sitter", False):
            issues = scan_file("test.py")

            # Check if one-line stub was found
            assert len(issues) == 1
            issue = issues[0]
            assert issue["type"] == "Improper Implementation"
            assert issue["subtype"] == "One-line stub"
            assert issue["line"] == 1
            assert issue["content"] == "def test_func(): pass"

def test_scan_file_no_tree_sitter_no_stub():
    """Test scan_file without tree-sitter when encountering normal code without a stub."""
    mock_content = "def test_func():\n    return True\n"

    with patch("code_analysis.scanner.static_engine.open", mock_open(read_data=mock_content.encode("utf-8"))):
        with patch("code_analysis.scanner.static_engine._has_tree_sitter", False):
            issues = scan_file("test.py")

            # Check if one-line stub was NOT found
            found_stub = any(
                issue["type"] == "Improper Implementation" and
                issue["subtype"] == "One-line stub"
                for issue in issues
            )
            assert not found_stub, f"Did not expect one-line stub issue, got {issues}"
