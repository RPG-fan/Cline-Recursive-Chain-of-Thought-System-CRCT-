# utils/path_utils.py

"""
Core module for path utilities.
Handles path normalization, validation, and comparison.
"""

import os
import re
from typing import List, Optional, Set, Union, Tuple
import logging

logger = logging.getLogger(__name__)

# <<< *** REMOVED outdated constants *** >>>
# HIERARCHICAL_KEY_PATTERN = r'^\d+[A-Z][a-z0-9]*$' # Removed
# KEY_PATTERN = r'\d+|\D+' # Removed

def normalize_path(path: str) -> str:
    """
    Normalize a file path for consistent comparison.

    Args:
        path: Path to normalize

    Returns:
        Normalized path
    """
    from .cache_manager import cached # Keep import near usage if re-enabled

    @cached("path_normalization",
           key_func=lambda p: f"normalize:{p if p else 'empty'}")
    def _normalize_path(p: str) -> str:
        if not p: return ""
        # Ensure absolute path before normpath for consistency, especially with relative inputs
        # Use os.path.abspath cautiously if CWD is not guaranteed to be project root during execution
        # Let's assume paths passed are either absolute or meant to be relative to CWD when called
        # If relative paths need resolving against project_root, do it *before* calling normalize_path
        # However, making it absolute generally prevents unexpected behavior.
        if not os.path.isabs(p):
            p = os.path.abspath(p) # Make absolute based on CWD
        normalized = os.path.normpath(p).replace("\\", "/")
        # Lowercase drive letter on Windows for consistency
        if os.name == 'nt' and re.match(r"^[a-zA-Z]:", normalized):
             normalized = normalized[0].lower() + normalized[1:]
        # Remove trailing slash unless it's the root directory
        if len(normalized) > 1 and normalized.endswith('/'):
             normalized = normalized.rstrip('/')
        elif os.name == 'nt' and len(normalized) > 3 and normalized.endswith('/'): # Handle C:/ case
             normalized = normalized.rstrip('/')

        return normalized

    return _normalize_path(path)


def get_file_type(file_path: str) -> str:
    """
    Determines the file type based on its extension.

    Args:
        file_path: The path to the file.

    Returns:
        The file type as a string (e.g., "py", "js", "md", "generic").
    """

    def _get_file_type(fp: str) -> str:
        _, ext = os.path.splitext(fp)
        ext = ext.lower()
        if ext == ".py": return "py"
        elif ext in (".js", ".ts", ".jsx", ".tsx"): return "js"
        elif ext in (".md", ".rst"): return "md"
        elif ext in (".html", ".htm"): return "html"
        elif ext == ".css": return "css"
        else: return "generic"

    return _get_file_type(file_path)

def resolve_relative_path(source_dir: str, relative_path: str, default_extension: str = '.js') -> str:
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
    if not os.path.splitext(resolved)[1]: resolved += default_extension
    return normalize_path(resolved) # Normalize the final result

def get_relative_path(path: str, base_path: str) -> str:
    """
    Get a path relative to a base path.

    Args:
        path: Path to convert
        base_path: Base path to make relative to

    Returns:
        Relative path
    """
    norm_path = normalize_path(path); norm_base = normalize_path(base_path)
    try: return os.path.relpath(norm_path, norm_base).replace("\\", "/") # Ensure forward slashes
    except ValueError: return norm_path # Different drive

def get_project_root() -> str:
    """
    Find the project root directory.

    Returns:
        Path to the project root directory
    """
    from .cache_manager import cached

    @cached("project_root",
            key_func=lambda: f"project_root:{normalize_path(os.getcwd())}") # Key depends only on starting CWD
    def _get_project_root() -> str:
        current_dir = os.path.abspath(os.getcwd())
        root_indicators = ['.git', '.clinerules', 'pyproject.toml', 'setup.py', 'package.json', 'Cargo.toml', 'CMakeLists.txt', '.clinerules.config.json'] # Added config file
        # Prevent infinite loop on root (e.g., '/')
        while True:
            for indicator in root_indicators:
                if os.path.exists(os.path.join(current_dir, indicator)):
                    return normalize_path(current_dir) # Normalize the found root
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir: # Reached the filesystem root
                break
            current_dir = parent_dir
        # Fallback to original CWD if no indicator found up to root
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

    Args:
        path: Path to check
        excluded_paths: List of exclusion patterns

    Returns:
        True if the path should be excluded, False otherwise
    """
    if not excluded_paths: return False
    norm_path = normalize_path(path)
    for excluded in excluded_paths:
        norm_excluded = normalize_path(excluded) # Normalize exclusion pattern/path
        if '*' in norm_excluded: # Simple wildcard matching
            # Convert basic wildcard to regex - more robust globbing might be needed
            pattern_str = norm_excluded.replace('.', r'\.').replace('*', r'.*')
            # Ensure pattern matches whole path segment or end
            # This regex needs careful testing, might not cover all glob cases
            # Example: exclude '*/temp/*' -> '.*/temp/.*'
            # Example: exclude '*.log' -> '.*\.log' (might match too broadly)
            # Consider using fnmatch or glob for better pattern matching if needed
            try:
                if re.fullmatch(pattern_str, norm_path): # Use fullmatch? Or search?
                    return True
            except re.error as e:
                 logger.warning(f"Invalid regex pattern derived from exclusion '{excluded}': {e}")
        elif norm_path == norm_excluded or is_subpath(norm_path, norm_excluded):
            return True
    return False

def matches_exclusion_patterns(path: str, patterns: List[str]) -> bool:
    """
    Check if a path matches any of the given glob exclusion patterns.
    Supports patterns like "**/bin/**", "**/*.dll", etc.

    Args:
        path: Path to check (normalized)
        patterns: List of glob patterns

    Returns:
        True if the path matches any pattern, False otherwise
    """
    if not patterns:
        return False
    
    import fnmatch
    from pathlib import Path
    
    norm_path = normalize_path(path)
    
    for pattern in patterns:
        try:
            # Handle different pattern types
            if pattern.startswith('**/') and pattern.endswith('/**'):
                # Pattern like "**/bin/**" - matches any path containing the directory
                dir_name = pattern[3:-3]  # Remove "**/" and "/**"
                if f'/{dir_name}/' in norm_path or norm_path.endswith(f'/{dir_name}'):
                    return True
            elif pattern.startswith('**/'):
                # Pattern like "**/*.dll" - matches files at any depth
                if fnmatch.fnmatch(norm_path, pattern) or fnmatch.fnmatch(os.path.basename(norm_path), pattern[3:]):
                    return True
            else:
                # Standard glob pattern
                if fnmatch.fnmatch(norm_path, pattern) or fnmatch.fnmatch(os.path.basename(norm_path), pattern):
                    return True
                    
            # Also try pathlib.Path.match for more robust glob support
            try:
                path_obj = Path(norm_path)
                if path_obj.match(pattern):
                    return True
            except (ValueError, TypeError):
                # pathlib.Path.match can be picky about patterns
                pass
                
        except Exception as e:
            logger.warning(f"Error matching pattern '{pattern}' against path '{norm_path}': {e}")
            continue
    
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
    norm_path = normalize_path(path); norm_parent = normalize_path(parent_path)
    # Ensure parent_path is not empty and path is not identical before checking prefix
    if not norm_parent or norm_path == norm_parent: return False
    # Append separator to parent to ensure matching whole directory names
    parent_with_sep = norm_parent + '/'
    return norm_path.startswith(parent_with_sep)

def get_common_path(paths: List[str]) -> str:
    """
    Find the common path prefix for a list of paths.

    Args:
        paths: List of paths

    Returns:
        Common path prefix
    """
    if not paths: return ""
    norm_paths = [normalize_path(p) for p in paths]
    try: return normalize_path(os.path.commonpath(norm_paths)) # Normalize result
    except ValueError: return "" # Different drive

def is_valid_project_path(path: str) -> bool:
    """
    Check if a path is within the project root directory.

    Args:
        path: Path to check

    Returns:
        True if the path is within the project root, False otherwise
    """
    from .cache_manager import cached

    @cached("valid_project_paths",
           key_func=lambda p: f"valid_project_path:{normalize_path(p)}:{get_project_root()}") # Key depends on path and project root value
    def _is_valid_project_path(p: str) -> bool:
        project_root = get_project_root(); norm_p = normalize_path(p)
        # Check if it starts with the root (and separator), or is the root itself
        return norm_p == project_root or norm_p.startswith(project_root + '/')

    return _is_valid_project_path(path)

# EoF