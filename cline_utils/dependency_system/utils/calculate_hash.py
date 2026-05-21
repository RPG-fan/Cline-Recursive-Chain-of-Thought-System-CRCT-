import hashlib
import os
import re
from typing import Optional
from cline_utils.dependency_system.io.file_io import strip_auto_generated_blocks

# Standard comment prefixes for supported file extensions
COMMENT_PREFIXES = {
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


def _normalize_content_for_hashing(content: str, file_path: Optional[str] = None) -> str:
    """
    Normalizes carriage returns, comment formatting, indentation, and trailing
    whitespace to produce a stable content state before generating a hash.
    """
    if not content:
        return ""

    # 1. Normalize carriage returns to LF
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Strip auto-generated blocks completely if file_path is provided
    if file_path:
        content = strip_auto_generated_blocks(content, file_path, preserve_lines=False)

    # 3. Determine comment prefix based on file extension
    ext = ""
    if file_path:
        _, ext = os.path.splitext(file_path.lower())
    prefix = COMMENT_PREFIXES.get(ext, "#")

    normalized_lines = []
    for line in content.splitlines():
        # Trim leading and trailing whitespace to eliminate varying indentation styles
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Standardize comment formats and spacing
        if prefix == "<!--":
            # Markdown / HTML comments (e.g., <!-- comment -->)
            m = re.match(r"^(.*?)(?:\s*)(<!--)\s*(.*?)\s*(-->)?$", line)
            if m:
                code_part, pref, comment_part, suffix = m.groups()
                code_part = code_part.strip()
                comment_part = comment_part.strip()
                suffix = " -->" if suffix else ""
                if code_part:
                    line = f"{code_part} {pref} {comment_part}{suffix}"
                else:
                    line = f"{pref} {comment_part}{suffix}"
        else:
            # Single line code comments (e.g., #, //, --)
            escaped_prefix = re.escape(prefix)
            m = re.match(rf"^(.*?)(?:\s*)({escaped_prefix})\s*(.*?)$", line)
            if m:
                code_part, pref, comment_part = m.groups()
                single_quotes = code_part.count("'")
                double_quotes = code_part.count('"')
                # Quote-matching heuristic: only process if the prefix is not within strings
                if single_quotes % 2 == 0 and double_quotes % 2 == 0:
                    code_part = code_part.strip()
                    comment_part = comment_part.strip()
                    if code_part:
                        line = f"{code_part} {pref} {comment_part}"
                    else:
                        line = f"{pref} {comment_part}"

        line = line.strip()
        if line:
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def calculate_content_hash(content: str, file_path: Optional[str] = None) -> str:
    """
    Calculates a stable SHA-256 hash for the given content.
    If file_path is provided, it strips [AUTO] blocks COMPLETELY (no preserve_lines)
    to ensure the hash is stable even when auto-docs are added/removed.

    Args:
        content: Content to hash
        file_path: Optional path to the file to enable [AUTO] stripping.

    Returns:
        Hex digest of the hash
    """
    normalized = _normalize_content_for_hashing(content, file_path)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

