# utils/path_utils.py

"""
Core module for path utilities.
Handles path normalization, validation, and comparison.
"""

import fnmatch
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

PathMigrationInfo = Dict[str, Tuple[Optional[str], Optional[str]]]

logger = logging.getLogger(__name__)

# Regex to strip invisible/zero-width Unicode characters that can poison filenames.
# Strips: ZWSP (U+200B), ZWNJ (U+200C), ZWJ (U+200D), BOM/ZWNBS (U+FEFF),
# Word Joiner (U+2060), Left-to-Right Mark (U+200E), Right-to-Left Mark (U+200F),
# and other format characters (Unicode category "Cf") except normal whitespace.
#
# Also strips ASCII control characters (0x00-0x1F, 0x7F) that are NOT normal
# whitespace (tab, newline, carriage return are excluded from stripping since
# they should never appear in a path anyway on modern filesystems).
_INVISIBLE_CHARS_RE = re.compile(
    "["
    "\u200b"  # ZERO WIDTH SPACE
    "\u200c"  # ZERO WIDTH NON-JOINER
    "\u200d"  # ZERO WIDTH JOINER
    "\u200e"  # LEFT-TO-RIGHT MARK
    "\u200f"  # RIGHT-TO-LEFT MARK
    "\u2060"  # WORD JOINER
    "\u2061"  # FUNCTION APPLICATION
    "\u2062"  # INVISIBLE TIMES
    "\u2063"  # INVISIBLE SEPARATOR
    "\u2064"  # INVISIBLE PLUS
    "\ufffe"  # NON-CHARACTER
    "\ufeff"  # ZERO WIDTH NO-BREAK SPACE (BOM)
    "\u00ad"  # SOFT HYPHEN
    "\u034f"  # COMBINING GRAPHEME JOINER
    "\u061c"  # ARABIC LETTER MARK
    "\u115f"  # HANGUL CHOSEONG FILLER
    "\u1160"  # HANGUL JUNGSEONG FILLER
    "\u17b4"  # KHMER VOWEL INHERENT AQ
    "\u17b5"  # KHMER VOWEL INHERENT AA
    "\u180e"  # MONGOLIAN VOWEL SEPARATOR
    "\u3164"  # HANGUL FILLER
    "\uffa0"  # HALFWIDTH HANGUL FILLER
    "]+",
    re.UNICODE,
)


def _strip_invisible_chars(path: str) -> str:
    """Remove invisible/zero-width Unicode characters from a path string."""
    return _INVISIBLE_CHARS_RE.sub("", path)


def normalize_path(path: str) -> str:
    """
    Normalize a file path for consistent comparison.

    This function:
    - Makes relative paths absolute
    - Normalizes path separators to forward slashes
    - Uppercases the drive letter on Windows
    - Removes trailing slashes (except for root dirs)
    - Strips invisible Unicode characters (e.g. zero-width spaces)
      that can cause duplicate-filename corruption

    Args:
        path: Path to normalize

    Returns:
        Normalized path
    """
    if not path:
        return ""
    # Strip invisible characters FIRST so they don't propagate
    path = _strip_invisible_chars(path)
    if not path:
        return ""
    if not os.path.isabs(path):
        path = os.path.abspath(path)  # Make absolute based on CWD
    normalized = os.path.normpath(path).replace("\\", "/")
    # Uppercase drive letter on Windows for file operation compatibility
    if (
        os.name == "nt"
        and len(normalized) > 1
        and normalized[1] == ":"
        and normalized[0].isalpha()
    ):
        normalized = normalized[0].upper() + normalized[1:]
    # Remove trailing slash unless it's the root directory
    limit = 3 if os.name == "nt" and ":" in normalized else 1
    if len(normalized) > limit and normalized.endswith("/"):
        normalized = normalized.rstrip("/")

    return normalized


_FILE_TYPE_MAP = {
    ".py": "py",
    ".js": "js",
    ".ts": "js",
    ".jsx": "js",
    ".tsx": "js",
    ".md": "md",
    ".rst": "md",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".svelte": "svelte",
    ".sql": "sql",
    ".csv": "csv",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".txt": "txt",
    ".rs": "rs",
}


def get_file_type(file_path: str) -> str:
    """
    Determines the file type based on its extension.

    Args:
        file_path: The path to the file.

    Returns:
        The file type as a string (e.g., "py", "js", "md", "generic").
    """
    _, ext = os.path.splitext(file_path)
    return _FILE_TYPE_MAP.get(ext.lower(), "generic")


def resolve_relative_path(
    source_dir: str, relative_path: str, default_extension: str = ".js"
) -> str:
    """
    Resolve a relative import path to an absolute path based on the source directory.

    Args:
        source_dir: The directory of the source file (e.g., 'h:/path/to/project').
        relative_path: The relative import path (e.g., './module3' or '../utils/helper').
        default_extension: The file extension to append if none is present (default is '.js').

    Returns:
        The resolved absolute path (e.g., 'h:/path/to/project/module3.js').
    """
    # Combine the source directory and relative path, then normalize it
    resolved = os.path.normpath(os.path.join(source_dir, relative_path))
    if not os.path.splitext(resolved)[1]:
        resolved += default_extension
    return normalize_path(resolved)  # Normalize the final result


def get_relative_path(path: str, base_path: str) -> str:
    """
    Get a path relative to a base path.

    Args:
        path: Path to convert
        base_path: Base path to make relative to

    Returns:
        Relative path
    """
    norm_path = normalize_path(path)
    norm_base = normalize_path(base_path)
    try:
        return os.path.relpath(norm_path, norm_base).replace(
            "\\", "/"
        )  # Ensure forward slashes
    except ValueError:
        return norm_path  # Different drive


def get_project_root() -> str:
    """
    Find the project root directory.

    Returns:
        Path to the project root directory
    """

    def _get_project_root() -> str:
        def _find_root(start_dir: str) -> Optional[str]:
            current_dir = os.path.abspath(start_dir)
            root_indicators = ["project_root.cfg"]
            while True:
                for indicator in root_indicators:
                    if os.path.exists(os.path.join(current_dir, indicator)):
                        return normalize_path(current_dir)
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:
                    break
                current_dir = parent_dir
            return None

        # Try from CWD first
        root = _find_root(os.getcwd())
        if root is not None:
            return root

        # Fallback to the location of this script/module (only if not running under tests)
        import sys

        is_testing = (
            "pytest" in sys.modules
            or "unittest" in sys.modules
            or "vitest" in sys.modules
        )
        if not is_testing:
            root = _find_root(os.path.dirname(os.path.abspath(__file__)))
            if root is not None:
                return root

        # Final fallback to CWD
        return normalize_path(os.path.abspath(os.getcwd()))

    return _get_project_root()


def join_paths(base_path: str, *paths: str) -> str:
    """
    Join paths and normalize the result.

    Args:
        base_path: Base path
        *paths: Additional path components

    Returns:
        Joined and normalized path
    """
    return normalize_path(os.path.join(base_path, *paths))


def is_path_excluded(path: str, excluded_paths: List[str]) -> bool:
    """
    Check if a path should be excluded based on a list of exclusion patterns.
    Matches the path itself or any of its parent/ancestor directories.

    Args:
        path: Path to check
        excluded_paths: List of exclusion patterns

    Returns:
        True if the path should be excluded, False otherwise
    """
    if not excluded_paths:
        return False
    norm_path = normalize_path(path)

    # Generate path and all its ancestor directories
    check_paths = [norm_path]
    current = norm_path
    while True:
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break
        check_paths.append(normalize_path(parent))
        current = parent

    for excluded in excluded_paths:
        norm_excluded = normalize_path(excluded)
        is_wildcard = "*" in norm_excluded or "?" in norm_excluded

        for p in check_paths:
            if is_wildcard:
                if fnmatch.fnmatch(p, norm_excluded):
                    return True
            else:
                if p == norm_excluded:
                    return True
    return False


def is_subpath(path: str, parent_path: str) -> bool:
    """
    Check if a path is a subpath of another path.

    Args:
        path: Path to check
        parent_path: Potential parent path

    Returns:
        True if path is a subpath of parent_path, False otherwise
    """
    norm_path = normalize_path(path)
    norm_parent = normalize_path(parent_path)
    # Ensure parent_path is not empty and path is not identical before checking prefix
    if not norm_parent or norm_path == norm_parent:
        return False
    # Append separator to parent to ensure matching whole directory names
    parent_with_sep = norm_parent + "/"
    return norm_path.startswith(parent_with_sep)


def get_common_path(paths: List[str]) -> str:
    """
    Find the common path prefix for a list of paths.

    Args:
        paths: List of paths

    Returns:
        Common path prefix
    """
    if not paths:
        return ""
    norm_paths = [normalize_path(p) for p in paths]
    try:
        return normalize_path(os.path.commonpath(norm_paths))  # Normalize result
    except ValueError:
        return ""  # Different drive


def is_valid_project_path(path: str) -> bool:
    """
    Check if a path is within the project root directory.

    Args:
        path: Path to check

    Returns:
        True if the path is within the project root, False otherwise
    """

    def _is_valid_project_path(p: str) -> bool:
        project_root = get_project_root()
        norm_p = normalize_path(p)
        # Check if it starts with the root (and separator), or is the root itself
        return norm_p == project_root or norm_p.startswith(project_root + "/")

    return _is_valid_project_path(path)


# EoF
