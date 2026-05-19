import pytest
import os
import sys
import time
from unittest.mock import MagicMock, patch
from io import StringIO

from cline_utils.dependency_system.utils.phase_tracker import PhaseTracker


class TestPhaseTracker:
    def test_init(self):
        tracker = PhaseTracker(total=100, phase_name="TestPhase")
        assert tracker.total == 100
        assert tracker.phase_name == "TestPhase"
        assert tracker.current == 0
        assert tracker.description == ""

    def test_format_time(self):
        tracker = PhaseTracker(100)
        assert tracker._format_time(30) == "30s"
        assert tracker._format_time(65) == "01:05"
        assert tracker._format_time(3665) == "01:01:05"

    def test_context_manager(self):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with PhaseTracker(100) as tracker:
                assert tracker.start_time > 0
                assert tracker.last_update_time == tracker.start_time

            # Verify newline on exit for TTY (mocking isatty=True)
            # Note: StringIO.isatty() returns False by default, so we need to patch it if we want to test TTY behavior

    def test_update(self):
        tracker = PhaseTracker(100)
        tracker._print_progress = MagicMock()

        tracker.update(10, "Processing items")
        assert tracker.current == 10
        assert tracker.description == "Processing items"
        tracker._print_progress.assert_called()

    def test_set_description(self):
        tracker = PhaseTracker(100)
        tracker._print_progress = MagicMock()

        tracker.set_description("New description")
        assert tracker.description == "New description"
        tracker._print_progress.assert_called()

    def test_set_total(self):
        tracker = PhaseTracker(100)
        tracker._print_progress = MagicMock()

        tracker.set_total(200)
        assert tracker.total == 200
        tracker._print_progress.assert_called()

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_progress_non_tty(self, mock_stdout):
        # Mock isatty to return False
        with patch("sys.stdout.isatty", return_value=False):
            tracker = PhaseTracker(100, phase_name="Test")
            tracker.is_tty = False  # Explicitly set for the test logic

            # Should print periodically
            tracker.update(20, "Step 1")
            output = mock_stdout.getvalue()
            assert "[Test] 20/100 (20.0%) - Step 1" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_progress_tty(self, mock_stdout):
        # Mock isatty to return True
        with patch("sys.stdout.isatty", return_value=True):
            with patch(
                "shutil.get_terminal_size", return_value=os.terminal_size((100, 20))
            ):
                tracker = PhaseTracker(100, phase_name="Test")
                tracker.is_tty = True
                tracker.start_time = time.time() - 10  # 10 seconds elapsed

                tracker.update(50, "Halfway")
                output = mock_stdout.getvalue()

                # Check for key components in the output
                assert "\r[Test]" in output
                assert "50.0%" in output
                assert "Halfway" in output
                assert "ETA:" in output

    def test_eta_calculation(self):
        tracker = PhaseTracker(100)
        tracker.start_time = time.time() - 10  # 10 seconds elapsed
        tracker.current = 50  # 50 items done

        # Rate = 5 items/sec
        # Remaining = 50 items
        # ETA should be 10 seconds

        # We can't easily access the internal local variable eta_str without refactoring or capturing stdout
        # But we can verify the logic by calling _print_progress and checking output

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            tracker.is_tty = True
            tracker.term_width = 100
            tracker._print_progress()
            output = mock_stdout.getvalue()
            assert (
                "ETA: 10s" in output or "ETA: 09s" in output or "ETA: 11s" in output
            )  # Allow slight timing drift

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_progress_resizing(self, mock_stdout):
        # Mock isatty to return True
        with patch("sys.stdout.isatty", return_value=True):
            # We mock shutil.get_terminal_size to return different widths sequentially
            mock_sizes = [
                os.terminal_size((100, 20)),
                os.terminal_size((100, 20)),
                os.terminal_size((60, 20)),
            ]
            with patch(
                "shutil.get_terminal_size", side_effect=mock_sizes
            ) as mock_get_size:
                tracker = PhaseTracker(100, phase_name="Test")
                tracker.is_tty = True

                # First update: 100 wide
                tracker.update(50, "First")
                assert tracker.term_width == 100

                # Second update: 60 wide
                tracker.update(60, "Second")
                assert tracker.term_width == 60

                # Assert shutil.get_terminal_size was called at init and then for each update
                assert mock_get_size.call_count >= 2
