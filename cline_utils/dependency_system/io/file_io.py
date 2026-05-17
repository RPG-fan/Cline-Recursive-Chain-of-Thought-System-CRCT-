import os
from typing import Any, Dict, Optional
from cline_utils.dependency_system.utils.cache_manager import cached


# Standard comment prefixes for various file types
COMMENT_PREFIXES: Dict[str, str] = {
    ".py": "#",
    ".js": "//",
    ".ts": "//",
    ".tsx": "//",
    ".jsx": "//",
    ".cs": "//",
    ".sql": "--",
    ".glsl": "//",
    ".hlsl": "//",
    ".wgsl": "//",
    ".md": "<!--",
}

AUTO_TAG = "[AUTO]"


def _get_read_file_deps(file_path: str, *args: Any, **kwargs: Any) -> list[str]:
    return [file_path]


@cached("file_content_reads", ttl=3600, file_deps=_get_read_file_deps, check_mtime=True)
def read_file_content_safely(file_path: str) -> Optional[str]:
    """Reads a file safely, returning None if an error occurs."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def strip_auto_generated_blocks(
    content: str, file_path: str, preserve_lines: bool = True
) -> str:
    """
    Remove STATION_HEADER and CONNECTION_MAP blocks tagged with [AUTO].
    This stabilizes file content for hashing/embedding.

    Args:
        content: The file content to strip.
        file_path: The path to the file (used for extension-based comment prefix).
        preserve_lines: If True, replaces stripped lines with newlines to maintain line counts.
                         If False, removes the lines entirely (useful for stable hashing).
    """
    if not content:
        return ""

    _, ext = os.path.splitext(file_path)
    prefix = COMMENT_PREFIXES.get(ext.lower(), "#")

    # 1. Strip Station Header (bounded by markers)
    if prefix == "<!--":
        start_marker = "<!-- STATION_HEADER_START"
        end_marker = "STATION_HEADER_END -->"
    else:
        start_marker = f"{prefix} STATION_HEADER_START"
        end_marker = f"{prefix} STATION_HEADER_END"

    lines = content.splitlines(keepends=True)
    new_lines = []
    in_station_block = False

    for line in lines:
        if start_marker in line:
            in_station_block = True
            if preserve_lines:
                new_lines.append("\n")  # Preserve line number
            continue
        if end_marker in line:
            in_station_block = False
            if preserve_lines:
                new_lines.append("\n")  # Preserve line number
            continue
        if in_station_block:
            if preserve_lines:
                new_lines.append("\n")  # Preserve line number
            continue

        # 2. Strip any comment line containing [AUTO]
        trimmed = line.lstrip()
        if trimmed.startswith(prefix) and AUTO_TAG in line:
            if preserve_lines:
                new_lines.append("\n")  # Preserve line number
            continue

        new_lines.append(line)

    return "".join(new_lines)
