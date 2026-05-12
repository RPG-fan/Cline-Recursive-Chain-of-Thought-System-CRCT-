"""
populate_comments.py
====================
Injects and refreshes [AUTO] Station Headers and CONNECTION_MAP comments
into project source files, consuming data already prepared by
TrackerBatchCollector rather than performing its own CRCT discovery.

PUBLIC API
----------
populate_comments_for_batch(
    project_root       : Path,
    updates            : List[TrackerUpdate],
    symbol_map         : Dict[str, Any],
    *,
    dry_run            : bool = True,
    verbose            : bool = False,
) -> List[Dict[str, Any]]

OWNERSHIP CONTRACT
------------------
- Every [AUTO] field is owned by this module and overwritten on each run.
- Agents own only [FILL: ...] prose fields (ROLE, LAYER).
- Prose fields are extracted before a block is removed and re-injected
  into the replacement block unchanged.

NOTES
-----
- No CRCT discovery, path scanning, or JSON loading happens here.
- All dependency context comes from TrackerUpdate objects.
- backup_and_write() creates a .comment_backup/ snapshot before each
  source-file write. The tracker_batch_collector handles tracker backups.
"""

import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from cline_utils.dependency_system.core.dependency_grid import (
    DIAGONAL_CHAR,
    EMPTY_CHAR,
    PLACEHOLDER_CHAR,
    decompress,
)
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.io.tracker_io import (
    get_mini_tracker_path,
    get_tracker_path,
    is_path_in_doc_roots,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import normalize_path

logger = logging.getLogger(__name__)

__all__ = ["populate_comments_for_batch"]

# ── Constants ──────────────────────────────────────────────────────────────────

AUTO_TAG = "[AUTO]"
FILL_TAG = "[FILL:"

SUPPORTED_EXTENSIONS: Set[str] = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".cs",
    ".sql",
    ".glsl",
    ".hlsl",
    ".wgsl",
    ".md",
}

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

# Dependency characters that are meaningful to surface in a CONNECTION_MAP.
# Diagonal ("o") and empty/placeholder states are excluded.
_SKIP_CHARS: Set[str] = {DIAGONAL_CHAR, EMPTY_CHAR, PLACEHOLDER_CHAR, "n"}


# ── Symbol-level relevance helpers ─────────────────────────────────────────────


def _extract_names(data: Any, keys: Optional[List[str]] = None) -> Set[str]:
    """
    Safely extract unique names from various symbol data formats.
    Handles:
    - List[str]: direct names
    - List[Dict]: names from keys (default: 'name', 'target_name', 'path')
    - Dict: keys as names (for Runtime exports)
    """
    if keys is None:
        keys = ["name", "target_name", "base_class_name", "path", "url", "class_name"]

    names: Set[str] = set()
    if not data:
        return names

    if isinstance(data, dict):
        # Handle dict-based exports or similar (Runtime format)
        for k in cast(Dict[Any, Any], data).keys():
            names.add(str(k))
    elif isinstance(data, list):
        for item in cast(List[Any], data):
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, dict):
                d_item = cast(Dict[Any, Any], item)
                for k in keys:
                    val = d_item.get(k)
                    if val is not None:
                        names.add(str(val))
                        break
            elif item is not None and not isinstance(item, (list, dict)):
                # Fallback for other scalar types (int, etc.) if they appear as names
                names.add(str(item))
    return names


def _collect_symbol_references(
    item: Dict[str, Any], symbol_type: str
) -> Optional[Set[str]]:
    """
    Collect all external name references from a function or class symbol entry.

    Returns None if no scope data exists (data inconclusive — caller should
    skip filtering).  Returns an empty set if scope data exists but the
    symbol genuinely references nothing external.
    """
    has_scope_data = "scope_references" in item

    if symbol_type == "class":
        for method in cast(List[Dict[str, Any]], item.get("methods", [])):
            if method and "scope_references" in method:
                has_scope_data = True
                break

    if not has_scope_data:
        return None  # Inconclusive — no runtime scope data available

    refs: Set[str] = set()

    scope = cast(Dict[str, Any], item.get("scope_references", {}))
    if scope:
        refs.update(_extract_names(scope.get("globals", [])))
        refs.update(_extract_names(scope.get("nonlocals", [])))

    refs.update(_extract_names(item.get("attribute_accesses", [])))
    refs.update(_extract_names(item.get("closure_dependencies", [])))

    # For classes: aggregate references from inheritance and methods
    if symbol_type == "class":
        inheritance = item.get("inheritance", {})
        if isinstance(inheritance, dict):
            # Runtime format: {"bases": ["Base1", "Base2"]}
            d_inh = cast(Dict[Any, Any], inheritance)
            refs.update(_extract_names(d_inh.get("bases", [])))
        elif isinstance(inheritance, list):
            # AST format: [{"class_name": "MyClass", "base_class_name": "Base1"}]
            refs.update(_extract_names(inheritance))

        for method in cast(List[Dict[str, Any]], item.get("methods", [])):
            if not method:
                continue
            m_scope = cast(Dict[str, Any], method.get("scope_references", {}))
            if m_scope:
                refs.update(_extract_names(m_scope.get("globals", [])))
                refs.update(_extract_names(m_scope.get("nonlocals", [])))
            refs.update(_extract_names(method.get("attribute_accesses", [])))
            refs.update(_extract_names(method.get("closure_dependencies", [])))

    return refs


def _collect_file_exports(file_symbol_data: Dict[str, Any]) -> Set[str]:
    """
    Collect all names exported or defined by a file from its symbol map entry.

    Includes function names, class names (and their method names),
    globals_defined, and explicit exports.
    """
    names: Set[str] = set()

    # Functions
    names.update(_extract_names(file_symbol_data.get("functions", [])))

    # Classes and their methods
    classes = file_symbol_data.get("classes", [])
    if isinstance(classes, list):
        for cls in cast(List[Any], classes):
            if isinstance(cls, dict):
                d_cls = cast(Dict[Any, Any], cls)
                names.add(str(d_cls.get("name", "")))
                names.update(_extract_names(d_cls.get("methods", [])))

    # Global variables/assignments
    names.update(_extract_names(file_symbol_data.get("globals_defined", [])))

    # Explicit exports (handles both List[Dict] from AST and Dict from Runtime)
    names.update(_extract_names(file_symbol_data.get("exports", [])))

    # Filter out empty strings
    if "" in names:
        names.remove("")

    return names


# ── Language helpers ───────────────────────────────────────────────────────────


def get_comment_prefix(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return COMMENT_PREFIXES.get(ext, "#")


def _is_commentable(file_path: str) -> bool:
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False
    prefix = COMMENT_PREFIXES.get(ext, "")
    return bool(prefix)


# ── Prose preservation ─────────────────────────────────────────────────────────

_PROSE_FIELDS = ("ROLE", "LAYER")
_FILL_FIELD_RE = re.compile(
    r"^([#/\-]{1,2}|<!--)?\s*(" + "|".join(_PROSE_FIELDS) + r"):\s*(.+)$"
)


def extract_prose_fields(block_lines: List[str]) -> Dict[str, str]:
    """
    Extract agent-authored [FILL] prose from an existing STATION_HEADER block.
    Returns {field_name: value_string} for any non-[FILL] values found,
    so the next write can restore them.
    """
    prose: Dict[str, str] = {}
    for line in block_lines:
        stripped = line.strip()
        m = _FILL_FIELD_RE.match(stripped)
        if m:
            field, value = m.group(2), m.group(3).strip()
            if not value.startswith(FILL_TAG):
                prose[field] = value
    return prose


# ── Block locating ─────────────────────────────────────────────────────────────


def get_station_markers(prefix: str) -> Tuple[str, str]:
    """Get the consistent start and end markers for a STATION_HEADER block."""
    start = f"{prefix} --- STATION_HEADER_START --- {AUTO_TAG}"
    if prefix == "<!--":
        end = f"--- STATION_HEADER_END --- {AUTO_TAG} -->"
    else:
        end = f"{prefix} --- STATION_HEADER_END --- {AUTO_TAG}"
    return start, end


def find_block_range(
    lines: List[str], start_marker: str, end_marker: str
) -> Optional[Tuple[int, int]]:
    """Return (start_idx, end_idx) inclusive for a marker-bounded block, or None."""
    start_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i
        elif end_marker in line and start_idx is not None:
            return (start_idx, i)
    return None


def find_definition_line(
    lines: List[str], item_name: str, symbol_type: str, hint_line: int, ext: str
) -> int:
    """
    Find the 0-indexed line of a function/class definition.
    Searches a window around hint_line first, falls back to full-file scan.
    hint_line is 1-indexed (from symbol map).
    """
    hint_0 = max(0, hint_line - 1)

    if ext == ".py":
        patterns = [f"def {item_name}", f"class {item_name}"]
    elif ext in (".js", ".ts", ".tsx", ".jsx"):
        patterns = [
            f"function {item_name}",
            f"class {item_name}",
            f"const {item_name}",
            f"let {item_name}",
            f"var {item_name}",
        ]
    elif ext == ".cs":
        patterns = [
            f"class {item_name}",
            f"struct {item_name}",
            f"interface {item_name}",
            f"enum {item_name}",
            f"void {item_name}",
            f"Task {item_name}",
        ]
    elif ext == ".sql":
        patterns = [
            f"CREATE TABLE {item_name}",
            f"CREATE VIEW {item_name}",
            f"CREATE PROCEDURE {item_name}",
            f"CREATE FUNCTION {item_name}",
        ]
    else:
        patterns = [item_name]

    window_start = max(0, hint_0 - 5)
    window_end = min(len(lines), hint_0 + 50)
    for i in range(window_start, window_end):
        if any(p in lines[i] for p in patterns):
            return i

    for i, line in enumerate(lines):
        if any(p in line for p in patterns):
            return i

    return hint_0


def get_item_line(item: Dict[str, Any]) -> int:
    """Safely extract the 1-indexed start line from a symbol-map entry."""
    val = item.get("line", 0)
    if isinstance(val, list):
        try:
            return int(cast(Any, val[0]))
        except (ValueError, TypeError, IndexError):
            return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


# ── Station Header builder ─────────────────────────────────────────────────────


def build_station_header(
    rel_path: str,
    crct_key: str,
    tracker_ref: str,
    preserved_prose: Dict[str, str],
    prefix: str,
) -> str:
    """
    Build a complete STATION_HEADER block string for insertion.
    - [AUTO] fields are always regenerated from CRCT data.
    - ROLE, LAYER are restored from preserved_prose if available,
      otherwise a [FILL: ...] placeholder is emitted.
    """
    p = f"{prefix} " if prefix != "<!--" else ""

    start_marker, end_marker = get_station_markers(prefix)

    role = preserved_prose.get(
        "ROLE", f"{FILL_TAG} describe this file's responsibility]"
    )
    layer = preserved_prose.get(
        "LAYER", f"{FILL_TAG} e.g. Service | Utility | Controller | Model]"
    )
    tracker_ref = preserved_prose.get(
        "TRACKER_REF", tracker_ref or f"{FILL_TAG} path to mini-tracker]"
    )

    return (
        f"{start_marker}\n"
        f"{p}ROLE:    {role}\n"
        f"{p}LAYER:   {layer}\n"
        f"{p}CRCT_KEY:   {crct_key} {AUTO_TAG}\n"
        f"{p}TRACKER_REF: {tracker_ref} {AUTO_TAG}\n"
        f"{end_marker}\n"
    )


# ── Connection map builder ─────────────────────────────────────────────────────


def build_connection_map(
    symbol_name: str,
    source_key: str,
    key_info_list: List[KeyInfo],
    grid_row: str,
    prefix: str,
    relevant_keys: Optional[Set[str]] = None,
) -> str:
    """
    Build a single-line CONNECTION_MAP comment for one symbol.

    Uses the tracker-native grid row for the file that owns symbol_name.
    Rail format: "KEY1<char>, KEY2<char>, ..." or "none".
    Sorted by key string for stable, deterministic output.

    Args:
        symbol_name:   Function or class name (used in the comment label).
        source_key:    CRCT key string for the file containing this symbol.
        key_info_list: Ordered list of KeyInfo objects for this tracker.
        grid_row:      Compressed dependency row for source_key.
        prefix:        Comment prefix for this file's language.
        relevant_keys: If provided, only include target keys in this set.
                       None means include all (no filtering).
    """
    decompressed = decompress(grid_row)
    entries: List[Tuple[str, str]] = []

    for col_idx, dep_char in enumerate(decompressed):
        if dep_char in _SKIP_CHARS:
            continue
        if col_idx >= len(key_info_list):
            break
        tgt_key = key_info_list[col_idx].key_string
        if tgt_key != source_key:  # skip self (should already be DIAGONAL_CHAR)
            if relevant_keys is not None and tgt_key not in relevant_keys:
                continue
            entries.append((tgt_key, dep_char))

    if entries:
        entries.sort(key=lambda x: x[0])
        rail = ", ".join(f"{k}{c}" for k, c in entries)
    else:
        rail = "none"

    p = f"{prefix} " if prefix and not prefix.endswith(" ") else prefix
    suffix = " -->" if prefix == "<!--" else ""
    return f"{p}--- CONNECTION_MAP: {rail} --- {symbol_name} {AUTO_TAG}{suffix}"


# ── File write ─────────────────────────────────────────────────────────────────


def backup_and_write(file_path: Path, new_content: str, project_root: Path) -> None:
    """
    Write new_content to file_path after creating a timestamped backup.
    Skips the write if backup creation fails to protect the original.
    """
    backup_dir = project_root / ".comment_backup"
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{file_path.name}.{timestamp}.bak"
    try:
        shutil.copy2(file_path, backup_path)
    except Exception as exc:
        logger.warning(
            f"Could not create backup for {file_path}: {exc}. Skipping write."
        )
        return
    file_path.write_text(new_content, encoding="utf-8")


# ── Core per-file processor ────────────────────────────────────────────────────


def process_file(
    file_path: Path,
    source_key: str,
    key_info_list: List[KeyInfo],
    grid_row: str,
    tracker_ref: str,
    entry_from: List[str],
    exits_to: List[str],
    symbol_data: Dict[str, Any],  # {"functions": [...], "classes": [...]}
    project_root: Path,
    dry_run: bool,
    verbose: bool,
    full_symbol_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Inject or refresh Station Header and CONNECTION_MAP comments in one file.

    - Overwrites every [AUTO] field with fresh CRCT data from the TrackerUpdate.
    - Preserves agent-authored [FILL] prose (ROLE, LAYER).
    - Processes symbols in reverse line order to avoid index shifting.
    - Filters CONNECTION_MAP entries per-symbol by cross-referencing each
      symbol's scope references against target files' exports in the
      project_symbol_map.
    - Returns a result dict for batch reporting.

    Args:
        file_path:       Absolute path to the source file.
        source_key:      CRCT key string for this file.
        key_info_list:   Ordered KeyInfo list from the owning TrackerUpdate.
        grid_row:        Compressed dependency row for source_key.
        tracker_ref:     Tracker file path for TRACKER_REF field.
        entry_from:      Relative paths of files that depend on this file.
        exits_to:        Relative paths of files this file depends on.
        symbol_data:     Symbol map entry for this file (functions + classes).
        project_root:    Absolute project root Path.
        dry_run:         If True, no files are modified.
        verbose:         If True, log per-symbol activity.
        full_symbol_map: Full project symbol map for per-symbol filtering.
    """
    result: Dict[str, Any] = {
        "path": str(file_path),
        "station_added": False,
        "maps_added": 0,
        "maps_updated": 0,
        "skipped": False,
        "error": None,
    }

    if not _is_commentable(str(file_path)):
        result["skipped"] = True
        return result

    ext = file_path.suffix.lower()
    prefix = get_comment_prefix(str(file_path))

    station_start, station_end = get_station_markers(prefix)

    try:
        source = file_path.read_text(encoding="utf-8")
        rel_path = os.path.relpath(file_path, project_root).replace("\\", "/")
    except Exception as exc:
        result["error"] = f"Read error: {exc}"
        return result

    new_source = source
    scaffolds: List[str] = []

    # ── 1. Station Header ──────────────────────────────────────────────────
    src_lines = new_source.splitlines(keepends=True)
    existing = find_block_range(src_lines, station_start, station_end)
    preserved_prose: Dict[str, str] = {}

    if existing is not None:
        s, e = existing
        preserved_prose = extract_prose_fields(src_lines[s : e + 1])
        src_lines = src_lines[:s] + src_lines[e + 1 :]
        new_source = "".join(src_lines)
        if verbose:
            logger.debug(
                f"[{rel_path}] Refreshing Station Header "
                f"(preserved prose: {list(preserved_prose.keys()) or 'none'})"
            )

    header = build_station_header(
        rel_path,
        source_key,
        tracker_ref,
        preserved_prose,
        prefix,
    )

    src_lines = new_source.splitlines(keepends=True)
    insert_idx = 0

    # 1.1 Placement logic: Tags take precedence, then docstrings
    tags_end_marker = "---TAGS_END---"
    tags_found = False
    for i, line in enumerate(src_lines):
        if tags_end_marker in line:
            insert_idx = i + 1
            tags_found = True
            break

    if not tags_found:
        # Insert after module docstring if present
        if src_lines and src_lines[0].startswith(('"""', "'''")):
            quote = src_lines[0][:3]
            if src_lines[0].strip().endswith(quote) and len(src_lines[0].strip()) > 3:
                insert_idx = 1
            else:
                for i in range(1, min(50, len(src_lines))):
                    if src_lines[i].strip().endswith(quote):
                        insert_idx = i + 1
                        break

    src_lines.insert(insert_idx, header)
    new_source = "".join(src_lines)
    result["station_added"] = True
    scaffolds.append("Station Header")

    # ── 2. Connection Maps ─────────────────────────────────────────────────
    functions = cast(List[Dict[str, Any]], symbol_data.get("functions", []))
    classes = cast(List[Dict[str, Any]], symbol_data.get("classes", []))
    items = [(f, "function") for f in functions] + [(c, "class") for c in classes]

    # Reverse order so insertions don't shift subsequent line indices
    items.sort(key=lambda x: get_item_line(x[0]), reverse=True)

    maps_added = 0
    maps_updated = 0
    conn_marker = "--- CONNECTION_MAP:"

    # Pre-compute target file exports for per-symbol relevance filtering.
    # Keyed by norm_path so lookup is shared across symbols in this file.
    _target_exports_cache: Dict[str, Optional[Set[str]]] = {}
    if full_symbol_map:
        decompressed_full = decompress(grid_row)
        for col_idx, dep_char in enumerate(decompressed_full):
            if dep_char in _SKIP_CHARS or col_idx >= len(key_info_list):
                continue
            tgt_ki = key_info_list[col_idx]
            if tgt_ki.key_string == source_key:
                continue
            if tgt_ki.norm_path not in _target_exports_cache:
                tgt_data = full_symbol_map.get(tgt_ki.norm_path)
                _target_exports_cache[tgt_ki.norm_path] = (
                    _collect_file_exports(tgt_data) if tgt_data else None
                )

    for item, symbol_type in items:
        name = str(item.get("name", "")).strip()
        hint_line = get_item_line(item)
        if not name or not hint_line:
            continue

        src_lines = new_source.splitlines(keepends=True)

        # Step A: remove any existing map for this symbol
        was_present = False
        for i, line in enumerate(src_lines):
            if conn_marker in line and f"--- {name} {AUTO_TAG}" in line:
                if prefix in line:
                    src_lines.pop(i)
                    new_source = "".join(src_lines)
                    was_present = True
                    if verbose:
                        logger.debug(f"[{rel_path}] Refreshing CONNECTION_MAP: {name}")
                    break

        # Step B: determine which keys are relevant to THIS symbol
        relevant_keys: Optional[Set[str]] = None
        if full_symbol_map:
            symbol_refs = _collect_symbol_references(item, symbol_type)
            if symbol_refs is not None:
                # We have scope data — filter by overlap
                relevant_keys = set()
                decompressed_check = decompress(grid_row)
                for col_idx, dep_char in enumerate(decompressed_check):
                    if dep_char in _SKIP_CHARS or col_idx >= len(key_info_list):
                        continue
                    tgt_ki = key_info_list[col_idx]
                    if tgt_ki.key_string == source_key:
                        continue
                    tgt_exports = _target_exports_cache.get(tgt_ki.norm_path)
                    if tgt_exports is None:
                        # No symbol data for target → conservatively include
                        relevant_keys.add(tgt_ki.key_string)
                    elif symbol_refs & tgt_exports:
                        relevant_keys.add(tgt_ki.key_string)
                if verbose:
                    total_deps = (
                        sum(1 for c in decompress(grid_row) if c not in _SKIP_CHARS) - 1
                    )  # exclude self
                    logger.debug(
                        f"[{rel_path}] {name}: {len(relevant_keys)}/{total_deps}"
                        f" keys relevant"
                    )
            else:
                # symbol_refs is None → no scope data → no filtering
                relevant_keys = None

        # Step C: build and inject new map
        conn_map = build_connection_map(
            name,
            source_key,
            key_info_list,
            grid_row,
            prefix,
            relevant_keys=relevant_keys,
        )
        src_lines = new_source.splitlines(keepends=True)
        def_idx = find_definition_line(src_lines, name, symbol_type, hint_line, ext)

        src_lines.insert(def_idx, conn_map + "\n")
        new_source = "".join(src_lines)

        if was_present:
            maps_updated += 1
        else:
            maps_added += 1
        scaffolds.append(f"{'~' if was_present else '+'} CONNECTION_MAP: {name}")

    result["maps_added"] = maps_added
    result["maps_updated"] = maps_updated

    if verbose and len(scaffolds) > 1:
        for s in scaffolds[1:]:
            logger.debug(f"  {s}")

    if not dry_run and scaffolds:
        backup_and_write(file_path, new_source, project_root)

    return result


# ── Key/path resolution helpers ────────────────────────────────────────────────


def _build_dep_lists(
    norm_path: str,
    source_key: str,
    key_info_list: List[KeyInfo],
    grid_row: str,
    project_root: Path,
    global_map: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Derive entry_from and exits_to relative path lists from the tracker grid row.
    If global_map and runtime cache are available, uses global dependencies.
    """
    from cline_utils.dependency_system.utils.tracker_utils import (
        runtime_aggregation_cache,
    )

    OUTBOUND = {"<", "d", "x", "p", "s", "S"}
    INBOUND = {">", "x"}

    entry_from: List[str] = []
    exits_to: List[str] = []

    if runtime_aggregation_cache and global_map:
        key_to_path = {ki.key_string: ki.norm_path for ki in global_map.values()}
        seen_entry: Set[str] = set()
        seen_exits: Set[str] = set()

        for (r_k, c_k), (char, _) in runtime_aggregation_cache.items():
            if char in _SKIP_CHARS:
                continue

            if r_k == source_key and c_k != source_key:
                tgt_path = key_to_path.get(c_k)
                if tgt_path:
                    tgt_rel = os.path.relpath(tgt_path, project_root).replace("\\", "/")
                    if char in OUTBOUND:
                        seen_exits.add(tgt_rel)
                    if char in INBOUND:
                        seen_entry.add(tgt_rel)

            if c_k == source_key and r_k != source_key:
                tgt_path = key_to_path.get(r_k)
                if tgt_path:
                    tgt_rel = os.path.relpath(tgt_path, project_root).replace("\\", "/")
                    if char in OUTBOUND:
                        seen_entry.add(tgt_rel)
                    if char in INBOUND:
                        seen_exits.add(tgt_rel)

        entry_from = list(seen_entry)
        exits_to = list(seen_exits)

    else:
        decompressed = decompress(grid_row)
        for col_idx, dep_char in enumerate(decompressed):
            if dep_char in _SKIP_CHARS:
                continue
            if col_idx >= len(key_info_list):
                break
            tgt_ki = key_info_list[col_idx]
            if tgt_ki.key_string == source_key:
                continue

            tgt_path = tgt_ki.norm_path
            tgt_rel = os.path.relpath(tgt_path, project_root).replace("\\", "/")

            if dep_char in OUTBOUND:
                exits_to.append(tgt_rel)
            if dep_char in INBOUND:
                entry_from.append(tgt_rel)

    return sorted(entry_from), sorted(exits_to)


# ── Batch entry point ──────────────────────────────────────────────────────────


def populate_comments_for_batch(
    project_root: Path,
    updates: List[Any],  # List[TrackerUpdate]
    symbol_map: Dict[str, Any],  # norm_path -> {functions:[...], classes:[...]}
    *,
    dry_run: bool = True,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Main entry point. Called by TrackerBatchCollector.commit_all() after all
    tracker writes succeed.

    For each TrackerUpdate, iterates over its key_info_list, finds the matching
    source file in symbol_map, and calls process_file() with tracker-native data.

    Args:
        project_root: Absolute project root Path.
        updates:      List of TrackerUpdate objects from the completed batch.
        symbol_map:   Project symbol map keyed by normalized absolute path.
                      Values have {"functions": [...], "classes": [...]}.
        dry_run:      If True, log changes but do not write files.
        verbose:      If True, emit per-symbol debug logging.

    Returns:
        List of per-file result dicts from process_file().
    """
    all_results: List[Dict[str, Any]] = []
    processed_paths: Set[str] = set()

    # Pre-calculate common tracker paths and doc roots
    config = ConfigManager()
    norm_proj_root = normalize_path(str(project_root))
    d_roots = {
        normalize_path(os.path.join(norm_proj_root, p))
        for p in config.get_doc_directories()
    }
    doc_tracker_path = get_tracker_path(norm_proj_root, "doc")
    main_tracker_path = get_tracker_path(norm_proj_root, "main")

    for update in updates:
        if not update.key_info_list or not update.grid_rows:
            continue

        for row_idx, ki in enumerate(update.key_info_list):
            if ki.is_directory:
                continue

            norm_path = ki.norm_path
            file_path = Path(norm_path)

            if not file_path.exists() or not _is_commentable(norm_path):
                continue

            if norm_path in processed_paths:
                continue
            processed_paths.add(norm_path)

            symbol_data = symbol_map.get(norm_path)
            if not symbol_data:
                continue

            source_key = ki.key_string
            grid_row = update.grid_rows[row_idx]
            key_info_list = update.key_info_list

            # Resolve the home tracker for THIS file specifically
            if is_path_in_doc_roots(norm_path, d_roots):
                tracker_ref = doc_tracker_path
            elif ki.parent_path:
                tracker_ref = get_mini_tracker_path(ki.parent_path)
            else:
                tracker_ref = main_tracker_path

            entry_from, exits_to = _build_dep_lists(
                norm_path,
                source_key,
                key_info_list,
                grid_row,
                project_root,
                update.path_to_key_info,
            )

            result = process_file(
                file_path=file_path,
                source_key=source_key,
                key_info_list=key_info_list,
                grid_row=grid_row,
                tracker_ref=tracker_ref,
                entry_from=entry_from,
                exits_to=exits_to,
                symbol_data=symbol_data,
                project_root=project_root,
                dry_run=dry_run,
                verbose=verbose,
                full_symbol_map=symbol_map,
            )
            all_results.append(result)

    return all_results


# ── Batch reporting ────────────────────────────────────────────────────────────


def report_batch_results(results: List[Dict[str, Any]], dry_run: bool) -> None:
    """Log a structured summary of a populate_comments_for_batch() run."""
    total = len(results)
    stations = sum(1 for r in results if r.get("station_added"))
    maps_added = sum(int(r.get("maps_added", 0)) for r in results)
    maps_upd = sum(int(r.get("maps_updated", 0)) for r in results)
    skipped = sum(1 for r in results if r.get("skipped"))
    errors = [r for r in results if r.get("error") and not r.get("skipped")]

    mode = "DRY-RUN" if dry_run else "WRITE"
    verb = "would write" if dry_run else "written"

    logger.info(f"populate_comments [{mode}] -- {total} files processed")
    logger.info(f"  Station Headers : {stations} ({verb})")
    logger.info(f"  CONNECTION_MAPs : {maps_added} new, {maps_upd} refreshed ({verb})")
    logger.info(f"  Skipped         : {skipped}")
    if errors:
        logger.warning(f"  Errors          : {len(errors)}")
        for e in errors:
            logger.warning(f"    {e.get('path', '?')}: {e.get('error', '?')}")
