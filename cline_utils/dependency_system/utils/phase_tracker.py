import sys
import time
import shutil
from typing import Optional

class PhaseTracker:
    """
    A simple progress tracker that displays a progress bar, current task description,
    and ETA in the terminal. Designed to be used as a context manager.
    """

    def __init__(self, total: int, phase_name: str = "Processing", unit: str = "items"):
        super().__init__()
        self.total = total
        self.phase_name = phase_name
        self.unit = unit
        self.current = 0
        self.start_time = 0.0
        self.last_update_time = 0.0
        self.description = ""
        self._check_tty()

    def _check_tty(self):
        """Check if we are running in a TTY."""
        self.is_tty = sys.stdout.isatty()
        # Fallback width if not TTY or can't determine
        self.term_width = 80
        if self.is_tty:
            try:
                self.term_width = shutil.get_terminal_size((80, 20)).columns
            except Exception:
                pass

    def __enter__(self):
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self._print_progress()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ensure we print a final newline so the cursor moves down
        if self.is_tty:
            sys.stdout.write("\n")
            sys.stdout.flush()
        if exc_type:
            # If an error occurred, we might want to log it or let it propagate
            pass

    def update(self, n: int = 1, description: Optional[str] = None):
        """
        Increment progress by n and optionally update the description.
        """
        self.current += n
        if description is not None:
            self.description = description
        self._print_progress()

    def set_description(self, description: str):
        """Update the current task description without incrementing progress."""
        self.description = description
        self._print_progress()

    def set_total(self, total: int):
        """Update the total count."""
        self.total = total
        self._print_progress()

    def _format_time(self, seconds: float) -> str:
        """Format seconds into MM:SS or HH:MM:SS."""
        if seconds < 60:
            return f"{int(seconds)}s"
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        if minutes < 60:
            return f"{minutes:02d}:{seconds:02d}"
        hours = int(minutes // 60)
        minutes = int(minutes % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _print_progress(self):
        """Render the progress bar to stdout."""
        if not self.is_tty:
            if self.total > 0:
                percent = (self.current / self.total) * 100
                if self.current == self.total or self.current % max(1, self.total // 5) == 0:
                     print(f"[{self.phase_name}] {self.current}/{self.total} ({percent:.1f}%) - {self.description}")
            return

        now = time.time()
        elapsed = now - self.start_time
        
        # Calculate percentage
        percent = 0.0
        if self.total > 0:
            percent = min(100.0, (self.current / self.total) * 100.0)
        
        # Calculate ETA
        eta_str = "??"
        if self.current > 0 and self.total > 0:
            rate = self.current / elapsed if elapsed > 0 else 0
            if rate > 0:
                remaining_items = self.total - self.current
                eta_seconds = remaining_items / rate
                eta_str = self._format_time(eta_seconds)
        
        # Construct bar
        # Available width calculation
        # Format: [Phase] [Bar] Percent% | Desc | ETA
        # Fixed parts approx length:
        # Phase: len(phase_name) + 2
        # Percent: 6 chars " 100.0%"
        # ETA: 10 chars " | ETA: xx"
        # Spacers: 4 chars
        
        prefix = f"[{self.phase_name}] "
        suffix = f" {percent:5.1f}% | ETA: {eta_str}"
        
        # Description is variable, let's truncate it if needed
        # Reserve space for bar (at least 10 chars)
        reserved_width = len(prefix) + len(suffix) + 15 # +15 for spacers and min bar
        available_for_desc = max(10, self.term_width - reserved_width - 20) # -20 for bar length
        
        desc_str = ""
        if self.description:
            desc_str = f" | {self.description}"
            if len(desc_str) > available_for_desc:
                desc_str = desc_str[:available_for_desc-3] + "..."
        
        # Recalculate bar width based on actual desc length
        bar_width = self.term_width - len(prefix) - len(suffix) - len(desc_str) - 3 # -3 for brackets and space
        bar_width = max(5, bar_width)
        
        filled_length = int(bar_width * percent / 100.0)
        bar = "=" * filled_length + "-" * (bar_width - filled_length)
        
        line = f"\r{prefix}[{bar}]{suffix}{desc_str}"
        
        # Pad with spaces to overwrite previous line if it was longer
        if len(line) < self.term_width:
            line += " " * (self.term_width - len(line))
            
        sys.stdout.write(line)
        sys.stdout.flush()
