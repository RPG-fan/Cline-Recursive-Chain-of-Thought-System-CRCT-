import builtins
import unittest
import logging
from unittest.mock import MagicMock

from cline_utils.dependency_system.core.exceptions_enhanced import (
    ProjectAnalyzerError,
    ProjectPermissionError,
    FileAnalysisError,
    EncodingError,
    ParsingError,
    TrackerIOError,
    TrackerUpdateError,
    handle_file_analysis_error,
    log_and_reraise,
)


class TestExceptionsEnhanced(unittest.TestCase):
    """Unit tests for exceptions_enhanced.py exception mappings and handling."""

    def test_builtins_permission_error_mapping(self) -> None:
        """Verify that built-ins PermissionError is correctly converted to ProjectPermissionError."""
        # Create a standard built-in PermissionError
        original_error = builtins.PermissionError("OS permission denied")
        file_path = "h:/test/path/file.py"

        # Call handle_file_analysis_error to convert it
        converted = handle_file_analysis_error(file_path, original_error)

        # Assertions
        self.assertIsInstance(converted, ProjectPermissionError)
        from typing import cast
        perm_err = cast(ProjectPermissionError, converted)
        self.assertEqual(perm_err.path, file_path)
        self.assertEqual(perm_err.operation, "read")
        self.assertIn("Permission denied", perm_err.message)

    def test_project_permission_error_properties(self) -> None:
        """Verify the custom properties of ProjectPermissionError."""
        err = ProjectPermissionError("h:/test/path/file.py", "write")
        self.assertIsInstance(err, ProjectAnalyzerError)
        self.assertEqual(err.path, "h:/test/path/file.py")
        self.assertEqual(err.operation, "write")
        self.assertEqual(
            err.message, "Permission denied for h:/test/path/file.py during write"
        )

    def test_unicode_decode_error_mapping(self) -> None:
        """Verify that UnicodeDecodeError is converted to EncodingError."""
        original_error = UnicodeDecodeError(
            "utf-8", b"\xff", 0, 1, "invalid start byte"
        )
        file_path = "h:/test/path/file.py"

        converted = handle_file_analysis_error(file_path, original_error)

        self.assertIsInstance(converted, EncodingError)
        from typing import cast
        enc_err = cast(EncodingError, converted)
        self.assertEqual(enc_err.file_path, file_path)
        self.assertEqual(enc_err.encoding_attempted, "utf-8")

    def test_syntax_error_mapping(self) -> None:
        """Verify that SyntaxError is converted to ParsingError."""
        original_error = SyntaxError(
            "invalid syntax", ("h:/test/path/file.py", 10, 5, "x =")
        )
        file_path = "h:/test/path/file.py"

        converted = handle_file_analysis_error(file_path, original_error)

        self.assertIsInstance(converted, ParsingError)
        from typing import cast
        parse_err = cast(ParsingError, converted)
        self.assertEqual(parse_err.file_path, file_path)
        self.assertEqual(parse_err.line_number, 10)
        self.assertIn("invalid syntax", parse_err.syntax_details)

    def test_log_and_reraise_custom(self) -> None:
        """Verify that log_and_reraise reraises custom ProjectAnalyzerError without wrapping."""
        logger_mock = MagicMock(spec=logging.Logger)
        custom_err = ProjectPermissionError("h:/path.py", "read")

        with self.assertRaises(ProjectPermissionError) as context:
            log_and_reraise(logger_mock, custom_err, context="test_context")

        self.assertEqual(context.exception, custom_err)
        logger_mock.error.assert_called_once_with(
            "Error in test_context: Permission denied for h:/path.py during read"
        )

    def test_log_and_reraise_generic(self) -> None:
        """Verify that log_and_reraise wraps generic exceptions into ProjectAnalyzerError."""
        logger_mock = MagicMock(spec=logging.Logger)
        generic_err = ValueError("Invalid value")

        with self.assertRaises(ProjectAnalyzerError) as context:
            log_and_reraise(logger_mock, generic_err, context="test_context")

        self.assertIsInstance(context.exception, ProjectAnalyzerError)
        self.assertIn("ValueError: Invalid value", context.exception.message)

    def test_tracker_io_error_properties(self) -> None:
        """Verify the custom properties of TrackerIOError."""
        err = TrackerIOError("h:/test/path/tracker.md", "write_tracker")
        self.assertIsInstance(err, ProjectAnalyzerError)
        self.assertEqual(err.tracker_path, "h:/test/path/tracker.md")
        self.assertEqual(err.operation, "write_tracker")
        self.assertEqual(
            err.message,
            "Tracker I/O failed for h:/test/path/tracker.md during write_tracker",
        )

    def test_tracker_update_error_properties(self) -> None:
        """Verify the custom properties of TrackerUpdateError."""
        err = TrackerUpdateError("h:/test/path/tracker.md", "update_row")
        self.assertIsInstance(err, ProjectAnalyzerError)
        self.assertEqual(err.tracker_path, "h:/test/path/tracker.md")
        self.assertEqual(err.operation, "update_row")
        self.assertEqual(
            err.message,
            "Tracker update failed for h:/test/path/tracker.md during update_row",
        )


if __name__ == "__main__":
    unittest.main()
