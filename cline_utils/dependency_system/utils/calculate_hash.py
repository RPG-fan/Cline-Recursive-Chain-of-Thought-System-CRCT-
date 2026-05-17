import hashlib
from cline_utils.dependency_system.io.file_io import strip_auto_generated_blocks
from typing import Optional


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
    if not content:
        return hashlib.sha256(b"").hexdigest()

    # Stabilize for hashing by stripping [AUTO] blocks COMPLETELY
    if file_path:
        content = strip_auto_generated_blocks(content, file_path, preserve_lines=False)

    # Normalize line endings to LF for cross-platform hash stability
    content = content.replace("\r\n", "\n").strip()

    return hashlib.sha256(content.encode("utf-8")).hexdigest()
