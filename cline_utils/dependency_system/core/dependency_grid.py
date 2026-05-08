# core/dependency_grid.py

"""
Core module for dependency grid operations.
Handles grid creation, compression, decompression, and dependency management with key-centric design.
Grid structure is defined by an ordered List[KeyInfo], while the grid dictionary uses KeyInfo.key_string.
"""

import re
from typing import Dict, List, Tuple
from collections import defaultdict


# Import only from utils or sibling core modules if necessary
from cline_utils.dependency_system.utils.cache_manager import (
    cached,
    invalidate_dependent_entries,
    clear_all_caches,
)

# Import KeyInfo for type hinting and usage
from .key_manager import (
    KeyInfo,
    sort_key_strings_hierarchically,
    validate_key,
)  # Added KeyInfo

import logging

logger = logging.getLogger(__name__)

# Constants
DIAGONAL_CHAR = "o"
PLACEHOLDER_CHAR = "p"
EMPTY_CHAR = "."

# Compile regex pattern for RLE compression scheme (repeating characters, excluding 'o')
COMPRESSION_PATTERN = re.compile(r"([^o])\1{2,}")


def compress(s: str) -> str:
    """
    Compress a dependency string using Run-Length Encoding (RLE).
    Only compresses sequences of 3 or more repeating characters (excluding 'o').

    Args:
        s: String to compress (e.g., "nnnnnpppdd")
    Returns:
        Compressed string (e.g., "n5p3d2")
    """
    if not s or len(s) <= 3:
        return s
    return COMPRESSION_PATTERN.sub(lambda m: m.group(1) + str(len(m.group())), s)


@cached("grid_decompress", key_func=lambda s: f"decompress:{s}")
def decompress(s: str) -> str:
    """
    Decompress a Run-Length Encoded dependency string with caching.

    Args:
        s: Compressed string (e.g., "n5p3d2")
    Returns:
        Decompressed string (e.g., "nnnnnpppdd")
    """
    if not s or (len(s) <= 3 and not any(c.isdigit() for c in s)):
        return s

    result = []
    i = 0
    while i < len(s):
        if i + 1 < len(s) and s[i + 1].isdigit():
            char = s[i]
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            count = int(s[i + 1 : j])
            result += char * count
            i = j
        else:
            result += s[i]
            i += 1
    return "".join(result)


# --- Grid Creation ---
@cached(
    "initial_grids",
    key_func=lambda key_info_list: f"initial_grid:{':'.join(sort_key_strings_hierarchically([ki.key_string for ki in key_info_list]))}",
)
def create_initial_grid(key_info_list: List[KeyInfo]) -> Dict[str, str]:
    """
    Create an initial dependency grid with placeholders and diagonal markers.
    The grid dictionary is keyed by KeyInfo.key_string.

    Args:
        key_info_list: List of KeyInfo objects defining the grid structure and order.
    Returns:
        Dictionary mapping key_strings to compressed dependency strings.
    """
    if not key_info_list or not all(
        isinstance(ki, KeyInfo) and validate_key(ki.key_string) for ki in key_info_list
    ):
        logger.error(
            f"Invalid key_info_list provided for initial grid: {key_info_list}"
        )
        raise ValueError(
            "All items in key_info_list must be valid KeyInfo objects with valid key_strings"
        )

    grid = {}
    num_keys = len(key_info_list)
    # The key strings used in the grid dict and for row iteration are from key_info_list
    key_strings_ordered = [ki.key_string for ki in key_info_list]

    for i, row_key_str in enumerate(key_strings_ordered):
        row_list_chars = [PLACEHOLDER_CHAR] * num_keys
        row_list_chars[i] = DIAGONAL_CHAR
        grid[row_key_str] = compress("".join(row_list_chars))
    return grid


# --- Character Access Helpers (No changes needed) ---
def _parse_count(s: str, start: int) -> Tuple[int, int]:
    """Helper function to parse the count from a string.
    Args:
        s: The string to parse
        start: The starting index for parsing
    Returns:
        Tuple containing:
        - The parsed count as an integer
        - The index after the count
    """
    j = start
    while j < len(s) and s[j].isdigit():
        j += 1
    return int(s[start:j]), j


def get_char_at(s: str, index: int) -> str:
    """
    Get the character at a specific index in a decompressed string.

    Args:
        s: The compressed string
        index: The index in the decompressed string
    Returns:
        The character at the specified index
    Raises:
        IndexError: If the index is out of range
    """
    decompressed_index = 0
    i = 0
    while i < len(s):
        if i + 1 < len(s) and s[i + 1].isdigit():
            char = s[i]
            count, i = _parse_count(s, i + 1)
            if decompressed_index + count > index:
                return char
            decompressed_index += count
        else:
            if decompressed_index == index:
                return s[i]
            decompressed_index += 1
            i += 1
    raise IndexError("Index out of range")


def set_char_at(s: str, index: int, new_char: str) -> str:
    """Set a character at a specific index and return the compressed string.
    Args:
        s: The compressed string
        index: The index in the decompressed string
        new_char: The new character to set
    Returns:
        The updated compressed string
    Raises:
        ValueError: If new_char is not a single character string
        IndexError: If the index is out of range
    """
    if not isinstance(new_char, str) or len(new_char) != 1:
        logger.error(f"Invalid new_char: {new_char}")
        raise ValueError("new_char must be a single character")
    decompressed = decompress(s)
    if index >= len(decompressed):
        raise IndexError("Index out of range")
    decompressed = decompressed[:index] + new_char + decompressed[index + 1 :]
    return compress(decompressed)


# --- Grid Validation ---
@cached(
    "grid_validation",
    key_func=lambda grid, key_info_list: f"validate_grid:{hash(str(sorted(grid.items())))}:{':'.join(sort_key_strings_hierarchically([ki.key_string for ki in key_info_list]))}",
)
def validate_grid(
    grid: Dict[str, str], key_info_list: List[KeyInfo]
) -> bool:  # MODIFIED: takes List[KeyInfo]
    """
    Validate a dependency grid for consistency with an ordered list of KeyInfo objects.
    Grid keys are KeyInfo.key_string.

    Args:
        grid: Dictionary mapping key_strings to compressed dependency strings.
        key_info_list: Pre-sorted list of KeyInfo objects defining the grid.
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(grid, dict):
        logger.error("Grid validation failed: 'grid' not a dict.")
        return False
    if not isinstance(key_info_list, list) or not all(
        isinstance(ki, KeyInfo) for ki in key_info_list
    ):
        logger.error(
            "Grid validation failed: 'key_info_list' not a list of KeyInfo objects."
        )
        return False

    # Extract ordered key strings from KeyInfo list
    ordered_key_strings = [ki.key_string for ki in key_info_list]

    num_keys = len(ordered_key_strings)
    if num_keys == 0 and not grid:
        return True
    if num_keys == 0 and grid:
        logger.error("Grid validation failed: Grid not empty but key_info_list is.")
        return False

    expected_keys_set = set(ordered_key_strings)
    actual_grid_keys_set = set(grid.keys())

    # 1. Check row keys match expected keys
    missing_rows = expected_keys_set - actual_grid_keys_set
    extra_rows = actual_grid_keys_set - expected_keys_set
    if missing_rows:
        logger.error(
            f"Grid validation failed: Missing rows for key_strings: {sorted(list(missing_rows))}"
        )
        return False
    if extra_rows:
        logger.error(
            f"Grid validation failed: Extra rows found for key_strings: {sorted(list(extra_rows))}"
        )
        return False

    # 2. Check row lengths and diagonal character
    for idx, key_str in enumerate(
        ordered_key_strings
    ):  # Iterate using the order from key_info_list
        compressed_row = grid.get(key_str)
        if compressed_row is None:
            logger.error(
                f"Grid validation failed: Row missing for key_string '{key_str}'."
            )
            return False

        try:
            decompressed = decompress(compressed_row)
        except Exception as e:
            logger.error(
                f"Grid validation failed: Error decompressing row for key_string '{key_str}': {e}"
            )
            return False

        if len(decompressed) != num_keys:
            logger.error(
                f"Grid validation failed: Row for key_string '{key_str}' length incorrect (Exp:{num_keys}, Got:{len(decompressed)})."
            )
            return False

        if decompressed[idx] != DIAGONAL_CHAR:
            logger.error(
                f"Grid validation failed: Row for key_string '{key_str}' has incorrect diagonal character at index {idx} (Expected: '{DIAGONAL_CHAR}', Got: '{decompressed[idx]}'). Row: '{decompressed}'"
            )
            return False

    logger.debug("Grid validation successful.")
    return True


# --- Grid Modification ---
def add_dependency_to_grid(
    grid: Dict[str, str],
    source_key_str: str,
    target_key_str: str,
    key_info_list: List[KeyInfo],
    dep_type: str = ">",
) -> Dict[str, str]:  # MODIFIED
    """
    Add a dependency between two keys in the grid. Grid keys are KeyInfo.key_string.

    Args:
        grid: Dictionary mapping key_strings to compressed dependency strings.
        source_key_str: Source key_string (row).
        target_key_str: Target key_string (column).
        key_info_list: List of KeyInfo objects for index mapping.
        dep_type: Dependency type character.

    Returns:
        Updated grid.
    """
    ordered_key_strings = [ki.key_string for ki in key_info_list]
    if (
        source_key_str not in ordered_key_strings
        or target_key_str not in ordered_key_strings
    ):
        raise ValueError(
            f"Key_strings {source_key_str} or {target_key_str} not in key_info_list"
        )

    source_idx = ordered_key_strings.index(source_key_str)
    target_idx = ordered_key_strings.index(target_key_str)
    if source_idx == target_idx:
        # Diagonal elements ('o') cannot be changed directly.
        # Grid validation ensures they are 'o'.
        # This prevents accidental overwrites and maintains grid integrity.
        raise ValueError(
            f"Cannot directly modify diagonal element for key_string '{source_key_str}'. Self-dependency must be 'o'."
        )

    # Create a copy of the grid to avoid modifying the original
    new_grid = grid.copy()
    # source_key_str is used to get the row from the grid dict
    row = decompress(
        new_grid.get(
            source_key_str, compress(PLACEHOLDER_CHAR * len(ordered_key_strings))
        )
    )
    new_row = row[:target_idx] + dep_type + row[target_idx + 1 :]
    new_grid[source_key_str] = compress(new_row)

    invalidate_dependent_entries(
        "grid_decompress", f"decompress:{new_grid.get(source_key_str)}"
    )
    # For validate_grid cache invalidation, use the key_info_list to form the cache key
    cache_key_validate = f"validate_grid:{hash(str(sorted(new_grid.items())))}:{':'.join(sort_key_strings_hierarchically([ki.key_string for ki in key_info_list]))}"
    invalidate_dependent_entries("grid_validation", cache_key_validate)
    return new_grid


def remove_dependency_from_grid(
    grid: Dict[str, str],
    source_key_str: str,
    target_key_str: str,
    key_info_list: List[KeyInfo],
) -> Dict[str, str]:  # MODIFIED
    """
    Remove a dependency between two keys in the grid. Grid keys are KeyInfo.key_string.

    Args:
        grid: Dictionary mapping key_strings to compressed dependency strings.
        source_key_str: Source key_string (row).
        target_key_str: Target key_string (column).
        key_info_list: List of KeyInfo objects for index mapping.

    Returns:
        Updated grid.
    """
    ordered_key_strings = [ki.key_string for ki in key_info_list]
    if (
        source_key_str not in ordered_key_strings
        or target_key_str not in ordered_key_strings
    ):
        raise ValueError(
            f"Key_strings {source_key_str} or {target_key_str} not in key_info_list"
        )

    source_idx = ordered_key_strings.index(source_key_str)
    target_idx = ordered_key_strings.index(target_key_str)
    if source_idx == target_idx:
        return grid

    new_grid = grid.copy()
    row = decompress(
        new_grid.get(
            source_key_str, compress(PLACEHOLDER_CHAR * len(ordered_key_strings))
        )
    )
    new_row = row[:target_idx] + EMPTY_CHAR + row[target_idx + 1 :]
    new_grid[source_key_str] = compress(new_row)

    invalidate_dependent_entries(
        "grid_decompress", f"decompress:{new_grid[source_key_str]}"
    )
    cache_key_validate = f"validate_grid:{hash(str(sorted(new_grid.items())))}:{':'.join(sort_key_strings_hierarchically([ki.key_string for ki in key_info_list]))}"
    invalidate_dependent_entries("grid_validation", cache_key_validate)
    return new_grid


# --- Dependency Retrieval ---
@cached(
    "grid_dependencies",
    key_func=lambda grid, source_key_str, key_info_list: f"grid_deps:{hash(str(sorted(grid.items())))}:{source_key_str}:{':'.join(sort_key_strings_hierarchically([ki.key_string for ki in key_info_list]))}",
)
def get_dependencies_from_grid(
    grid: Dict[str, str], source_key_str: str, key_info_list: List[KeyInfo]
) -> Dict[str, List[str]]:  # MODIFIED
    """
    Get dependencies for a specific key_string, categorized by relationship type.
    Grid keys are KeyInfo.key_string. Returns related key_strings.

    Args:
        grid: Dictionary mapping key_strings to compressed dependency strings.
        source_key_str: Key_string to get dependencies for.
        key_info_list: List of KeyInfo objects for index mapping and context.

    Returns:
        Dictionary mapping dependency characters ('<', '>', 'x', 'd', 's', 'S', 'p')
        to lists of related key_strings.
    """
    ordered_key_strings = [ki.key_string for ki in key_info_list]
    if source_key_str not in ordered_key_strings:
        raise ValueError(f"Source key_string {source_key_str} not in key_info_list")

    results = defaultdict(set)
    source_idx = ordered_key_strings.index(source_key_str)
    defined_dep_chars = {"<", ">", "x", "d", "s", "S"}

    # The row key for the grid dictionary is source_key_str
    row_key_compressed = grid.get(source_key_str)
    if (
        not row_key_compressed
    ):  # Source key has no row in grid (should not happen if grid is valid)
        logger.warning(
            f"No grid row found for source_key_str '{source_key_str}' during dependency retrieval."
        )
        return {k: list(v) for k, v in results.items()}

    # Iterate through columns using key_info_list for order and target key_strings
    for col_idx, target_ki in enumerate(key_info_list):
        target_key_str = target_ki.key_string
        if source_idx == col_idx:
            continue  # Skip self

        char_outgoing = EMPTY_CHAR
        try:
            char_outgoing = get_char_at(row_key_compressed, col_idx)
        except IndexError:
            logger.warning(
                f"IndexError getting char at col {col_idx} for row '{source_key_str}'. Row len might be wrong."
            )
            pass

        if char_outgoing == "x":
            results["x"].add(target_key_str)
        elif char_outgoing == "d":
            results["d"].add(target_key_str)
        elif char_outgoing == "S":
            results["S"].add(target_key_str)
        elif char_outgoing == "s":
            results["s"].add(target_key_str)
        elif char_outgoing == ">":
            results[">"].add(target_key_str)
        elif char_outgoing == "<":
            results["<"].add(target_key_str)
        elif char_outgoing not in defined_dep_chars:
            if char_outgoing == "p":
                results["p"].add(target_key_str)

    return {k: list(v) for k, v in results.items()}


# --- Grid Formatting ---
def format_grid_for_display(
    grid: Dict[str, str], key_info_list: List[KeyInfo]
) -> str:  # MODIFIED
    """
    Format a grid for display. Uses KeyInfo.key_string for labels.

    Args:
        grid: Dictionary mapping key_strings to compressed dependency strings.
        key_info_list: List of KeyInfo objects in the grid, defining order.

    Returns:
        Formatted string representation of the grid.
    """
    ordered_key_strings = [ki.key_string for ki in key_info_list]
    result = ["X " + " ".join(ordered_key_strings)]
    for key_str in ordered_key_strings:  # Iterate using the order from key_info_list
        # Get row from grid using key_str
        compressed_row_data = grid.get(
            key_str, compress(PLACEHOLDER_CHAR * len(ordered_key_strings))
        )
        result.append(f"{key_str} = {compressed_row_data}")
    return "\n".join(result)


def clear_cache():
    """Clear all function caches via cache_manager."""
    clear_all_caches()


# EoF
