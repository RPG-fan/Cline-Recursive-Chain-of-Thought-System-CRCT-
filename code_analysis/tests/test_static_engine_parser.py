from unittest.mock import patch, MagicMock
from code_analysis.scanner.static_engine import get_parser

def test_get_parser_no_tree_sitter():
    with patch("code_analysis.scanner.static_engine._has_tree_sitter", False):
        assert get_parser("python") is None

def test_get_parser_no_parser_class():
    with patch("code_analysis.scanner.static_engine._has_tree_sitter", True):
        with patch("code_analysis.scanner.static_engine._Parser", None):
            assert get_parser("python") is None

def test_get_parser_python():
    mock_parser_class = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_class.return_value = mock_parser_instance

    mock_language_class = MagicMock()
    mock_language_instance = MagicMock()
    mock_language_class.return_value = mock_language_instance

    mock_ts_py = MagicMock()
    mock_ts_py.language.return_value = "py_lang_obj"

    with patch("code_analysis.scanner.static_engine._has_tree_sitter", True), \
         patch("code_analysis.scanner.static_engine._Parser", mock_parser_class), \
         patch("code_analysis.scanner.static_engine._Language", mock_language_class), \
         patch("code_analysis.scanner.static_engine._ts_py", mock_ts_py):

        parser = get_parser("python")

        assert parser is mock_parser_instance
        mock_ts_py.language.assert_called_once()
        mock_language_class.assert_called_once_with("py_lang_obj")
        assert parser.language == mock_language_instance


def test_get_parser_javascript():
    mock_parser_class = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_class.return_value = mock_parser_instance

    mock_language_class = MagicMock()
    mock_language_instance = MagicMock()
    mock_language_class.return_value = mock_language_instance

    mock_ts_js = MagicMock()
    mock_ts_js.language.return_value = "js_lang_obj"

    with patch("code_analysis.scanner.static_engine._has_tree_sitter", True), \
         patch("code_analysis.scanner.static_engine._Parser", mock_parser_class), \
         patch("code_analysis.scanner.static_engine._Language", mock_language_class), \
         patch("code_analysis.scanner.static_engine._ts_js", mock_ts_js):

        parser = get_parser("javascript")

        assert parser is mock_parser_instance
        mock_ts_js.language.assert_called_once()
        mock_language_class.assert_called_once_with("js_lang_obj")
        assert parser.language == mock_language_instance

def test_get_parser_typescript():
    mock_parser_class = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_class.return_value = mock_parser_instance

    mock_language_class = MagicMock()
    mock_language_instance = MagicMock()
    mock_language_class.return_value = mock_language_instance

    mock_ts_ts = MagicMock()
    mock_ts_ts.language_typescript.return_value = "ts_lang_obj"

    with patch("code_analysis.scanner.static_engine._has_tree_sitter", True), \
         patch("code_analysis.scanner.static_engine._Parser", mock_parser_class), \
         patch("code_analysis.scanner.static_engine._Language", mock_language_class), \
         patch("code_analysis.scanner.static_engine._ts_ts", mock_ts_ts):

        parser = get_parser("typescript")

        assert parser is mock_parser_instance
        mock_ts_ts.language_typescript.assert_called_once()
        mock_language_class.assert_called_once_with("ts_lang_obj")
        assert parser.language == mock_language_instance

def test_get_parser_tsx():
    mock_parser_class = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_class.return_value = mock_parser_instance

    mock_language_class = MagicMock()
    mock_language_instance = MagicMock()
    mock_language_class.return_value = mock_language_instance

    mock_ts_ts = MagicMock()
    mock_ts_ts.language_tsx.return_value = "tsx_lang_obj"

    with patch("code_analysis.scanner.static_engine._has_tree_sitter", True), \
         patch("code_analysis.scanner.static_engine._Parser", mock_parser_class), \
         patch("code_analysis.scanner.static_engine._Language", mock_language_class), \
         patch("code_analysis.scanner.static_engine._ts_ts", mock_ts_ts):

        parser = get_parser("tsx")

        assert parser is mock_parser_instance
        mock_ts_ts.language_tsx.assert_called_once()
        mock_language_class.assert_called_once_with("tsx_lang_obj")
        assert parser.language == mock_language_instance

def test_get_parser_unsupported_language():
    mock_parser_class = MagicMock()
    with patch("code_analysis.scanner.static_engine._has_tree_sitter", True), \
         patch("code_analysis.scanner.static_engine._Parser", mock_parser_class):

        assert get_parser("ruby") is None

def test_get_parser_exception(capsys):
    mock_parser_class = MagicMock(side_effect=Exception("Test parser init exception"))

    with patch("code_analysis.scanner.static_engine._has_tree_sitter", True), \
         patch("code_analysis.scanner.static_engine._Parser", mock_parser_class):

        parser = get_parser("python")

        assert parser is None
        captured = capsys.readouterr()
        assert "Error initializing parser for python: Test parser init exception" in captured.out
