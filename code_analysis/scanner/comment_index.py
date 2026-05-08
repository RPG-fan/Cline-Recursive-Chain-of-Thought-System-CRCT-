"""comment_index.py -- stateless comment location scanner.

Builds a {norm_path: [{line: int, preview: str, subtype: str|None}, ...]}
index for every non-silenced comment line found in a file or list of files.
Silenced prefixes and block ranges strip infrastructure noise (CONNECTION_MAP,
STATION_HEADER blocks, etc.) so the agent sees only comments worth investigating.

Silenced prefixes and block delimiters are checked against the comment BODY
(after stripping the leading comment marker), making them language-agnostic.
This means the same SILENCED_PREFIXES tuple works for Python (#), SQL (--),
JavaScript (//), etc. without modification.

Public API
----------
scan_file_comments(file_path, silenced_prefixes, preview_len, silence_blocks)
    -> list[dict]  # [{"line": int, "preview": str, "subtype": str|None,
                   #   "tier": str}, ...]

scan_project_comments(file_paths, **kwargs)
    -> dict[str, list[dict]]  # {norm_path: [entries, ...]}

scan_project_root(**kwargs)
    -> dict[str, list[dict]]  # full project scan respecting ConfigManager exclusions
"""

from __future__ import annotations

import os
import sys
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple
import logging


# ---------------------------------------------------------------------------
# CRCT integration
# ---------------------------------------------------------------------------

from cline_utils.dependency_system.utils.path_utils import (
    get_project_root,
    normalize_path,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)

__all__ = [
    "scan_file_comments",
    "scan_project_comments",
    "scan_project_root",
    "SILENCED_PREFIXES",
    "BLOCK_BODY_START",
    "BLOCK_BODY_END",
    "DEFAULT_PREVIEW_LEN",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Line-level silenced prefixes -- checked against the comment BODY (after the
# leading comment marker is stripped), so they are language-agnostic.
# A Python "# --- CONNECTION_MAP: ..." and a SQL "-- --- CONNECTION_MAP: ..."
# both produce body "--- CONNECTION_MAP: ..." and match the same entry.
SILENCED_PREFIXES: Tuple[str, ...] = (
    "--- CONNECTION_MAP:",
    "CONNECTION_MAP:",
)

# Block delimiters -- also matched against the comment body (marker-agnostic).
# Everything from BLOCK_BODY_START through BLOCK_BODY_END (inclusive) is skipped.
BLOCK_BODY_START: str = "--- STATION_HEADER_START"
BLOCK_BODY_END: str = "--- STATION_HEADER_END"

# ---------------------------------------------------------------------------
# Subtype classification
# ---------------------------------------------------------------------------

# Incomplete-code markers: these signal unfinished work and are the primary
# targets for WIP comment triage.
_INCOMPLETE_SUBTYPES: Tuple[str, ...] = (
    "TODO",
    "FIXME",
    "WIP",
    "HACK",
    "XXX",
    "STUB",
    "REFACTOR",
)

# Audit markers: linting suppressions and structural notes an agent may want
# to review for removal or upgrade, but which are not unfinished-work signals.
_AUDIT_SUBTYPES: Tuple[str, ...] = (
    "NOQA",
    "TYPE:",
    "NOTE:",
)

# Tier labels exposed in each entry so callers can filter without knowing
# which tags belong to which category.
_TIER_INCOMPLETE: str = "incomplete"
_TIER_AUDIT: str = "audit"

# Per-extension single-line comment markers.
# Longer markers are tried first in _strip_comment_marker to avoid ambiguous
# prefix matches (e.g. "///" before "//").
_EXT_COMMENT_CHARS: Dict[str, Tuple[str, ...]] = {
    ".py": ("#",),
    ".pyw": ("#",),
    ".js": ("//",),
    ".jsx": ("//",),
    ".ts": ("//",),
    ".tsx": ("//",),
    ".cs": ("//",),
    ".java": ("//",),
    ".cpp": ("//", "///"),
    ".c": ("//",),
    ".h": ("//",),
    ".rs": ("///", "//"),
    ".go": ("//",),
    ".swift": ("//",),
    ".kt": ("//",),
    ".sql": ("--",),
    ".lua": ("--",),
    ".rb": ("#",),
    ".sh": ("#",),
    ".bash": ("#",),
    ".zsh": ("#",),
    ".yaml": ("#",),
    ".yml": ("#",),
    ".toml": ("#",),
    ".ini": ("#", ";"),
    ".cfg": ("#", ";"),
    ".r": ("#",),
    ".glsl": ("//",),
    ".hlsl": ("//",),
    ".md": (
        "<!--",
    ),  # HTML Comments (<!-- comment -->), since Markdown doesn't have a comment syntax
}

# Default preview length (characters of comment body shown).
DEFAULT_PREVIEW_LEN: int = 40

# Default directories to exclude when no ConfigManager is available.
# "cline_utils" is excluded to avoid surfacing CRCT's own infrastructure
# comments in project-level output, which would overwhelm results.
_FALLBACK_EXCLUDED_DIRS: Tuple[str, ...] = (
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    ".vscode",
    ".idea",
    "venv",
    "env",
    ".venv",
    "node_modules",
    "build",
    "dist",
    "cline_utils",
    "cline_docs",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".cache",
)

# Default extensions to exclude when no ConfigManager is available.
_FALLBACK_EXCLUDED_EXTS: Tuple[str, ...] = (
    ".pyc",
    ".pyo",
    ".pyd",
    ".dll",
    ".exe",
    ".so",
    ".log",
    ".tmp",
    ".bak",
    ".DS_Store",
    ".zip",
    ".tar",
    ".tgz",
    ".jar",
    ".embedding",
    ".npy",
    ".env",
    ".clinerules",
    ".clinerules.config.json",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".pdf",
    ".docx",
    ".xlsx",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_comment_chars(file_path: str) -> Tuple[str, ...]:
    """Return the single-line comment marker(s) for the given file extension.

    Falls back to ("#",) for unknown extensions so the scanner never
    silently skips a file due to a missing mapping.
    """
    ext = os.path.splitext(file_path)[1].lower()
    return _EXT_COMMENT_CHARS.get(ext, ("#",))


def _strip_comment_marker(
    stripped: str,
    comment_chars: Tuple[str, ...],
) -> Optional[str]:
    """Strip the leading comment marker from a pre-stripped line.

    Returns the comment body (leading whitespace removed) when the line
    starts with any known marker, or None if it is not a comment line.
    Markers are sorted longest-first to prevent short-prefix false matches
    (e.g. "//" matching before "///").
    """
    for marker in sorted(comment_chars, key=len, reverse=True):
        if stripped.startswith(marker):
            return stripped[len(marker) :].lstrip()
    return None


def _is_silenced_body(body: str, silenced_prefixes: Tuple[str, ...]) -> bool:
    """Return True if the comment body starts with any silenced prefix.

    Prefix-matching on the body (post-marker) makes silencing
    language-agnostic -- the same tuple works for all comment styles.
    """
    for prefix in silenced_prefixes:
        if body.startswith(prefix):
            return True
    return False


def _detect_subtype(body: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (subtype_tag, tier) for the first matching tag found in body.

    Searches the full body so inline tags like "refactor this WIP" are
    captured as well as leading tags like "TODO: fix".

    Returns (None, None) when no tag is found.

    Tier is one of:
        "incomplete"  -- unfinished-work marker (TODO, FIXME, WIP, HACK, XXX, STUB)
        "audit"       -- linting suppression or structural note (NOQA, TYPE:, NOTE:)
    """
    upper = body.upper()

    for tag in _INCOMPLETE_SUBTYPES:
        if tag.upper() in upper:
            return tag, _TIER_INCOMPLETE

    for tag in _AUDIT_SUBTYPES:
        if tag.upper() in upper:
            return tag, _TIER_AUDIT

    return None, None


def _normalize_path_local(path: str) -> str:
    """Normalise a path to forward-slash form using CRCT normalize_path.

    Keys are consistent with the rest of the dependency index.
    """
    return normalize_path(path)


def _collect_project_files(
    project_root: str,
    excluded_dirs: Tuple[str, ...],
    excluded_exts: Tuple[str, ...],
) -> List[str]:
    """Walk project_root and return all scannable source file paths.

    Excluded dirs are pruned in-place during os.walk so the walker never
    descends into them, keeping traversal fast on large projects.
    """
    files: List[str] = []
    excluded_dirs_set = set(excluded_dirs)
    excluded_exts_set = {e.lower() for e in excluded_exts}

    for dirpath, dirnames, filenames in os.walk(project_root, topdown=True):
        # Prune excluded dir names in-place -- os.walk respects this mutation.
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs_set]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in excluded_exts_set:
                files.append(os.path.join(dirpath, fname))

    return files


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_file_comments(
    file_path: str,
    silenced_prefixes: Tuple[str, ...] = SILENCED_PREFIXES,
    preview_len: int = DEFAULT_PREVIEW_LEN,
    silence_blocks: bool = True,
) -> List[Dict[str, Any]]:
    """Return comment entries for all non-silenced comment lines in file_path.

    Each entry is:
        {"line": int, "preview": str, "subtype": str | None, "tier": str | None}

    Lines are 1-indexed. Station header blocks are range-silenced via an
    explicit state machine when silence_blocks is True; CONNECTION_MAP /
    WIP_BEACON lines are body-prefix-silenced. All silencing is
    marker-agnostic.

    The block state machine is fail-closed: a missing BLOCK_BODY_END
    silences everything after the opening delimiter, which is safer than
    surfacing infrastructure noise in the output.

    Args:
        file_path:         Path to the source file.
        silenced_prefixes: Body-level prefixes to suppress (marker-agnostic).
        preview_len:       Number of body characters in the preview string.
        silence_blocks:    Whether to skip content in STATION_HEADER blocks.

    Returns:
        Ordered list of comment entry dicts, one per non-silenced comment line.
    """
    results: List[Dict[str, Any]] = []
    comment_chars = _get_comment_chars(file_path)

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            in_silenced_block = False

            for lineno, raw_line in enumerate(fh, start=1):
                stripped = raw_line.strip()

                # Strip comment marker; skip non-comment lines entirely.
                # in_silenced_block intentionally persists across non-comment
                # lines -- a missing END delimiter is fail-closed by design.
                body = _strip_comment_marker(stripped, comment_chars)
                if body is None:
                    continue

                # --- Block-range silencer (marker-agnostic) -----------------
                # Explicit continue on START so the delimiter line itself is
                # silenced, not just the content between delimiters.
                if silence_blocks and body.startswith(BLOCK_BODY_START):
                    in_silenced_block = True
                    continue  # START line is silenced

                if silence_blocks and in_silenced_block:
                    if body.startswith(BLOCK_BODY_END):
                        in_silenced_block = False
                    continue  # END line and all content between are silenced

                # --- Line-level body-prefix silencer ------------------------
                if _is_silenced_body(body, silenced_prefixes):
                    continue

                subtype, tier = _detect_subtype(body)
                results.append(
                    {
                        "line": lineno,
                        "preview": body[:preview_len],
                        "subtype": subtype,
                        "tier": tier,
                    }
                )

    except OSError as e:
        logger.debug(f"comment_index: skipping {file_path}: {e}")
        pass

    return results


def scan_project_comments(
    file_paths: List[str],
    silenced_prefixes: Tuple[str, ...] = SILENCED_PREFIXES,
    preview_len: int = DEFAULT_PREVIEW_LEN,
    skip_empty: bool = True,
    silence_blocks: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return {norm_path: [entries, ...]} for a given list of files.

    Uses BatchProcessor when available for parallel throughput via
    ThreadPoolExecutor; falls back to a sequential loop otherwise.
    BatchProcessor results are guarded against None (error-resilient items)
    before unpacking. Files with no non-silenced comments are omitted when
    skip_empty is True (default).

    Args:
        file_paths:        Explicit list of paths to scan.
        silenced_prefixes: Forwarded to scan_file_comments.
        preview_len:       Forwarded to scan_file_comments.
        skip_empty:        Omit files with zero entries from the result.
        silence_blocks:    Forwarded to scan_file_comments.

    Returns:
        Mapping from normalised path to list of comment entry dicts.
    """

    def _scan_one(
        path: str,
        silenced_prefixes: tuple[str, ...],
        preview_len: int,
        silence_blocks: bool,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        return (
            path,
            scan_file_comments(
                path,
                silenced_prefixes=silenced_prefixes,
                preview_len=preview_len,
                silence_blocks=silence_blocks,
            ),
        )

    if len(file_paths) > 1:
        processor = BatchProcessor(show_progress=False, phase_name="Comment Index")
        # process_items returns List[Optional[R]] -- None items indicate
        # internal errors and must be guarded before unpacking.
        raw_results: List[Optional[Tuple[str, List[Dict[str, Any]]]]] = (
            processor.process_items(
                file_paths,
                _scan_one,
                silenced_prefixes=silenced_prefixes,
                preview_len=preview_len,
                silence_blocks=silence_blocks,
            )
        )
    else:
        raw_results = [
            _scan_one(
                p,
                silenced_prefixes=silenced_prefixes,
                preview_len=preview_len,
                silence_blocks=silence_blocks,
            )
            for p in file_paths
        ]

    index: Dict[str, List[Dict[str, Any]]] = {}
    for result in raw_results:
        if result is None:
            continue
        path, entries = result
        if skip_empty and not entries:
            continue
        index[_normalize_path_local(path)] = entries

    return index


def scan_project_root(
    project_root: Optional[str] = None,
    silenced_prefixes: Tuple[str, ...] = SILENCED_PREFIXES,
    preview_len: int = DEFAULT_PREVIEW_LEN,
    skip_empty: bool = True,
    silence_blocks: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Scan the full project, respecting ConfigManager / .clinerules.config.json exclusions.

    When called with no arguments this is the default full-project scan:
    - project_root defaults to get_project_root() (CRCT) or cwd.
    - excluded_dirs and excluded_extensions come from ConfigManager when
      available, with individual attribute fallbacks for partial failures,
      and module-level defaults as the final safety net.
    - cline_utils is in the fallback excluded list to avoid surfacing CRCT's
      own infrastructure comments in project-level output.

    Args:
        project_root:      Override the project root directory.
        silenced_prefixes: Forwarded to scan_file_comments.
        preview_len:       Forwarded to scan_file_comments.
        skip_empty:        Omit files with zero entries from the result.
        silence_blocks:    Forwarded to scan_file_comments.

    Returns:
        Full project comment index keyed by normalised path.
    """
    # Resolve root --------------------------------------------------------
    if project_root is None:
        project_root = get_project_root()

    # Load exclusions with individual attribute fallbacks -----------------
    # Each getter is guarded separately so a missing or renamed method on a
    # future ConfigManager version doesn't bring down the whole scan.
    excluded_dirs: Tuple[str, ...]
    excluded_exts: Tuple[str, ...]

    try:
        cfg = ConfigManager()
    except Exception:
        cfg = None

    if cfg is not None:
        try:
            excluded_dirs = tuple(cfg.get_excluded_dirs())
        except AttributeError:
            excluded_dirs = _FALLBACK_EXCLUDED_DIRS

        try:
            excluded_exts = tuple(cfg.get_excluded_extensions())
        except AttributeError:
            excluded_exts = _FALLBACK_EXCLUDED_EXTS
    else:
        excluded_dirs = _FALLBACK_EXCLUDED_DIRS
        excluded_exts = _FALLBACK_EXCLUDED_EXTS

    file_paths = _collect_project_files(project_root, excluded_dirs, excluded_exts)
    return scan_project_comments(
        file_paths,
        silenced_prefixes=silenced_prefixes,
        preview_len=preview_len,
        skip_empty=skip_empty,
        silence_blocks=silence_blocks,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Scan source files and emit a JSON comment index. "
            "With no FILE arguments, performs a full project scan respecting "
            "ConfigManager / .clinerules.config.json exclusions."
        )
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help=(
            "Source files to scan. Pass '-' to read paths from stdin "
            "(one per line). Omit entirely for a full project scan."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=None,
        metavar="DIR",
        help="Project root for full-project scan (default: auto-detect).",
    )
    parser.add_argument(
        "--preview-len",
        type=int,
        default=DEFAULT_PREVIEW_LEN,
        metavar="N",
        help=f"Preview character count (default: {DEFAULT_PREVIEW_LEN}).",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include files with zero non-silenced comments in output.",
    )
    parser.add_argument(
        "--subtype",
        metavar="TAG",
        help=(
            "Filter output to entries matching a specific subtype tag "
            "(e.g. TODO, FIXME, NOQA). Case-insensitive."
        ),
    )
    parser.add_argument(
        "--tier",
        choices=[_TIER_INCOMPLETE, _TIER_AUDIT],
        help=(
            f"Filter output by tier: '{_TIER_INCOMPLETE}' for unfinished-work "
            f"markers, '{_TIER_AUDIT}' for linting suppression / structural notes."
        ),
    )
    parser.add_argument(
        "--include-infra",
        action="store_true",
        help=(
            "Disable all silenced prefixes and block ranges. "
            "Surfaces CONNECTION_MAP and STATION_HEADER entries -- "
            "useful for auditing infrastructure comment placement."
        ),
    )
    args = parser.parse_args()

    # Resolve silenced prefixes based on --include-infra flag
    effective_silenced: Tuple[str, ...] = (
        () if args.include_infra else SILENCED_PREFIXES
    )

    # Resolve file list: explicit args, stdin (-), or full project scan.
    if args.files:
        if args.files == ["-"]:
            file_paths = [line.strip() for line in sys.stdin if line.strip()]
        else:
            file_paths = args.files
        index = scan_project_comments(
            file_paths,
            silenced_prefixes=effective_silenced,
            preview_len=args.preview_len,
            skip_empty=not args.include_empty,
            silence_blocks=not args.include_infra,
        )
    else:
        index = scan_project_root(
            project_root=args.project_root,
            silenced_prefixes=effective_silenced,
            preview_len=args.preview_len,
            skip_empty=not args.include_empty,
            silence_blocks=not args.include_infra,
        )

    # Optional subtype filter
    if args.subtype:
        tag = args.subtype.upper()
        index = {
            path: [e for e in entries if e.get("subtype") == tag]
            for path, entries in index.items()
        }
        if not args.include_empty:
            index = {p: e for p, e in index.items() if e}

    if args.tier:
        index = {
            path: [e for e in entries if e.get("tier") == args.tier]
            for path, entries in index.items()
        }
        if not args.include_empty:
            index = {p: e for p, e in index.items() if e}

    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    _main()
