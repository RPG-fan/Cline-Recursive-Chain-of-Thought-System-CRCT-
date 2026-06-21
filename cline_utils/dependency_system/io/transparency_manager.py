import contextlib
import json
import logging
import os
import re
import threading
import time
from typing import Dict, Any, Optional, Tuple, List, Set, cast

try:
    import msvcrt

    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False
from cline_utils.dependency_system.utils.calculate_hash import calculate_content_hash
from cline_utils.dependency_system.utils.cache_manager import (
    normalize_path_cached as normalize_path,
)
from cline_utils.dependency_system.io.file_io import read_file_content_safely
from cline_utils.dependency_system.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

_CONNECTION_MAP_LINE_RE = re.compile(
    r"CONNECTION_MAP:\s*(?P<rail>.*?)\s*---\s*(?P<source_symbol>.*?)\s+\[AUTO\]"
)
_CONNECTION_MAP_ENTRY_RE = re.compile(
    r"(?P<target_key>[A-Za-z0-9#]+)\((?P<targets>[^)]*)\)\s*\{(?P<dep_char>[^}]+)\}"
)

# Registry location
from cline_utils.dependency_system.core import resolve_state_path

_core_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "core")
REGISTRY_PATH = resolve_state_path("transparency_registry.json", _core_dir)


def _coerce_line(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        line = int(value)
        return line if line > 0 else None
    except (TypeError, ValueError):
        return None


def _find_source_line(lines: List[str], map_line_index: int, source_symbol: str) -> int:
    """
    Find the source definition line that follows a CONNECTION_MAP line.

    Falls back to the comment line itself if no nearby definition is visible.
    """
    escaped = re.escape(source_symbol)
    compiled_patterns = [
        # Keywords: def, class, function, const, let, var
        re.compile(rf"\b(def|class|function|const|let|var)\s+{escaped}\b"),
        # Arrow function: name = (...) => or name: (...) =>
        re.compile(
            rf"\b{escaped}\s*[:=]\s*(async\s*)?(?:\([^)]*\)|[a-zA-Z0-9_$]+)\s*=>"
        ),
        # Dynamic Class/Object Method: myMethod(...) or async myMethod(...) or static, public, private, protected modifiers
        re.compile(
            rf"\b(?:async\s+|static\s+|public\s+|private\s+|protected\s+)*{escaped}\s*\("
        ),
    ]

    fallback_patterns = (
        f"def {source_symbol}",
        f"class {source_symbol}",
        f"function {source_symbol}",
        f"const {source_symbol}",
        f"let {source_symbol}",
        f"var {source_symbol}",
    )
    for idx in range(map_line_index + 1, min(len(lines), map_line_index + 12)):
        line_content = lines[idx]
        if any(pat.search(line_content) for pat in compiled_patterns):
            return idx + 1
        if any(pattern in line_content for pattern in fallback_patterns):
            return idx + 1
    return map_line_index + 1


def _parse_connection_map_text(content: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        match = _CONNECTION_MAP_LINE_RE.search(line)
        if not match:
            continue

        rail = match.group("rail").strip()
        if not rail or rail == "none":
            continue

        source_symbol = match.group("source_symbol").strip()
        source_line = _find_source_line(lines, idx, source_symbol)

        for entry_match in _CONNECTION_MAP_ENTRY_RE.finditer(rail):
            target_key = entry_match.group("target_key")
            dep_char = entry_match.group("dep_char").strip()
            target_blob = entry_match.group("targets").strip()
            for target in target_blob.split("|"):
                if not target:
                    continue
                if ":" in target:
                    target_symbol, line_text = target.rsplit(":", 1)
                    target_line = _coerce_line(line_text) if line_text != "?" else None
                else:
                    target_symbol = target
                    target_line = None
                records.append(
                    {
                        "source_symbol": source_symbol,
                        "source_line": source_line,
                        "target_key": target_key,
                        "target_symbol": target_symbol,
                        "target_line": target_line,
                        "dep_char": dep_char,
                    }
                )
    return records


def _adjust_connection_record_lines(
    records: List[Dict[str, Any]], removed_line_numbers: List[int]
) -> List[Dict[str, Any]]:
    adjusted: List[Dict[str, Any]] = []
    sorted_removed = sorted(removed_line_numbers)
    for record in records:
        next_record = dict(record)
        source_line = _coerce_line(next_record.get("source_line"))
        if source_line is not None:
            removed_before = sum(
                1 for line_no in sorted_removed if line_no < source_line
            )
            next_record["source_line"] = source_line - removed_before
        adjusted.append(next_record)
    return adjusted


def extract_connection_map_metadata(
    content: str, transparency_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Parse precise CONNECTION_MAP comments into structured dependency records.

    If transparency metadata already carries parsed connection maps or virtual
    CONNECTION_MAPS content, that metadata is used first; otherwise current file
    content is parsed directly. This helper is read-only.
    """
    if transparency_metadata:
        direct = transparency_metadata.get("connection_maps")
        if isinstance(direct, list):
            direct_list = cast(List[Any], direct)
            return [
                cast(Dict[str, Any], item)
                for item in direct_list
                if isinstance(item, dict)
            ]

        sections = transparency_metadata.get("sections", {})
        if isinstance(sections, dict):
            sections_dict = cast(Dict[str, Any], sections)
            section = sections_dict.get("CONNECTION_MAPS")
            if isinstance(section, dict):
                section_dict = cast(Dict[str, Any], section)
                content_val = section_dict.get("content", "")
                if isinstance(content_val, str):
                    return _parse_connection_map_text(content_val)

    return _parse_connection_map_text(content)


def overlay_connection_maps(
    content: str, transparency_metadata: Optional[Dict[str, Any]]
) -> str:
    """
    Return content with virtualized CONNECTION_MAP comments reinserted.

    The source file stays clean on disk; callers that need the automated layer can
    opt into this overlay before analysis or context packaging.
    """
    if not transparency_metadata or "CONNECTION_MAP:" in content:
        return content

    raw_lines = transparency_metadata.get("connection_map_lines")
    if not isinstance(raw_lines, list):
        return content

    raw_lines_list = cast(List[Any], raw_lines)
    lines = content.splitlines(keepends=True)
    inserts: List[Tuple[int, str]] = []
    for item in raw_lines_list:
        if not isinstance(item, dict):
            continue
        item_dict = cast(Dict[str, Any], item)
        line_no = _coerce_line(item_dict.get("line"))
        line_content = item_dict.get("content")
        if line_no is None or not isinstance(line_content, str):
            continue
        if not line_content.endswith(("\n", "\r")):
            line_content += "\n"
        inserts.append((line_no, line_content))

    for line_no, line_content in sorted(
        inserts, key=lambda item: item[0], reverse=True
    ):
        index = min(len(lines), max(0, line_no - 1))
        lines.insert(index, line_content)

    return "".join(lines)


class TransparencyLock:
    def __init__(self, lock_path: str, timeout: float = 10.0):
        super().__init__()
        self.lock_path = lock_path
        self.timeout = timeout
        self.fd = None

    def acquire(self) -> None:
        start_time = time.time()
        while True:
            try:
                self.fd = open(self.lock_path, "a", encoding="utf-8")
                break
            except PermissionError:
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(
                        f"Timed out trying to open lock file: {self.lock_path}"
                    )
                time.sleep(0.05)

        while True:
            try:
                if _HAS_MSVCRT:
                    self.fd.seek(0)
                    msvcrt.locking(self.fd.fileno(), msvcrt.LK_NBLCK, 1)
                elif _HAS_FCNTL:
                    fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, BlockingIOError):
                if time.time() - start_time > self.timeout:
                    try:
                        self.fd.close()
                    except Exception:
                        pass
                    self.fd = None
                    raise TimeoutError(f"Timed out acquiring lock on: {self.lock_path}")
                time.sleep(0.05)

    def release(self) -> None:
        if self.fd:
            try:
                if _HAS_MSVCRT:
                    self.fd.seek(0)
                    msvcrt.locking(self.fd.fileno(), msvcrt.LK_UNLCK, 1)
                elif _HAS_FCNTL:
                    fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            finally:
                try:
                    self.fd.close()
                except Exception:
                    pass
                self.fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class TransparencyManager:
    """
    Manages the "invisible" transparency layer for documentation section markers.
    Maps file paths to line-number based section definitions stored externally.
    """

    def __init__(self, registry_path: str = REGISTRY_PATH):
        super().__init__()
        self.registry_path = registry_path
        self._lock = threading.RLock()
        self._registry: Dict[str, Any] = {"files": {}}
        self._in_update = 0
        self._last_loaded_mtime = 0.0
        self._load()

    def _load(self) -> None:
        """Loads the transparency registry from disk. Call under _lock."""
        with self._lock:
            if os.path.exists(self.registry_path):
                try:
                    mtime = os.path.getmtime(self.registry_path)
                    with open(self.registry_path, "r", encoding="utf-8") as f:
                        self._registry = json.load(f)
                    self._last_loaded_mtime = mtime
                except Exception as e:
                    logger.error(f"Failed to load transparency registry: {e}")
                    self._registry = {"files": {}}
                    self._last_loaded_mtime = 0.0
            else:
                self._registry = {"files": {}}
                self._last_loaded_mtime = 0.0

    def _reload_if_stale(self) -> None:
        """Reloads the registry if it has been modified on disk by another process. Call under _lock."""
        with self._lock:
            if os.path.exists(self.registry_path):
                try:
                    mtime = os.path.getmtime(self.registry_path)
                    if mtime > self._last_loaded_mtime:
                        self._load()
                except Exception as e:
                    logger.debug(f"Failed to check/reload registry: {e}")

    def _save_under_lock(self) -> None:
        """Saves the transparency registry to disk atomically. Call under lock."""
        temp_path = self.registry_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, indent=2)

            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    if os.path.exists(self.registry_path):
                        os.replace(temp_path, self.registry_path)
                    else:
                        os.rename(temp_path, self.registry_path)
                    success = True
                    break
                except (PermissionError, OSError) as ex:
                    if attempt < max_retries - 1:
                        sleep_time = 0.05 * (2**attempt)
                        logger.warning(
                            f"Atomic save retry {attempt+1}/{max_retries} due to transient lock: {ex}. Retrying in {sleep_time}s."
                        )
                        time.sleep(sleep_time)
                    else:
                        raise ex

            if not success:
                raise RuntimeError(
                    "Failed to rename temporary file after maximum retries"
                )

            try:
                self._last_loaded_mtime = os.path.getmtime(self.registry_path)
            except Exception:
                pass

        except Exception as e:
            logger.error(
                f"Failed to save transparency registry atomically: {e}. Attempting direct fallback write."
            )
            try:
                with open(self.registry_path, "w", encoding="utf-8") as f:
                    json.dump(self._registry, f, indent=2)
                logger.info("Direct fallback write to transparency registry succeeded.")
                try:
                    self._last_loaded_mtime = os.path.getmtime(self.registry_path)
                except Exception:
                    pass
            except Exception as fallback_err:
                logger.error(
                    f"Direct fallback write to transparency registry also failed: {fallback_err}"
                )
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

    def _save(self) -> None:
        """Saves the transparency registry to disk atomically with retry and fallback."""
        lock_path = self.registry_path + ".lock"
        with self._lock:
            with TransparencyLock(lock_path):
                self._save_under_lock()

    @contextlib.contextmanager
    def update_context(self):
        """
        Context manager to handle file locking, reloading the registry from disk,
        allowing modifications to self._registry, and saving it back atomically.
        Supports reentrant calls.
        """
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        lock_path = self.registry_path + ".lock"

        with self._lock:
            if self._in_update == 0:
                with TransparencyLock(lock_path):
                    self._load()
                    self._in_update += 1
                    try:
                        yield
                    finally:
                        self._in_update -= 1
                        self._save_under_lock()
            else:
                self._in_update += 1
                try:
                    yield
                finally:
                    self._in_update -= 1

    def get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Retrieves transparency metadata for a specific file, excluding locked ones."""
        with self._lock:
            self._reload_if_stale()
            norm_path = normalize_path(file_path)
            entry = self._registry["files"].get(norm_path)
            if entry and entry.get("locked"):
                return None
            return entry

    def get_raw_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Retrieves raw transparency metadata for a file, even if locked."""
        with self._lock:
            self._reload_if_stale()
            norm_path = normalize_path(file_path)
            return self._registry["files"].get(norm_path)

    def lock_entry(self, file_path: str) -> None:
        """Flags the transparency metadata for a file as locked/pending realignment."""
        with self.update_context():
            norm_path = normalize_path(file_path)
            if norm_path in self._registry["files"]:
                self._registry["files"][norm_path]["locked"] = True
                logger.warning(
                    f"Placed a drift lock on transparency entry for {file_path} (pending next project analysis)"
                )

    def update_file_metadata(
        self, file_path: str, sections: Dict[str, Any], content: str
    ) -> None:
        """
        Updates the registry for a file, recording section mappings and current checksum.

        Args:
            file_path: Absolute or relative path to the file.
            sections: Dictionary mapping section names (e.g. 'TAGS') to metadata.
            content: Current file content to generate checksum.
        """
        with self.update_context():
            norm_path = normalize_path(file_path)
            checksum = calculate_content_hash(content, file_path)
            existing = self._registry["files"].get(norm_path, {})

            # Clean content lines count
            total_lines = len(content.splitlines())

            # Generate fresh floating markers
            floating_markers = self._generate_floating_markers(content)

            entry = {
                "checksum": checksum,
                "last_modified": (
                    os.path.getmtime(file_path)
                    if os.path.exists(file_path)
                    else time.time()
                ),
                "total_lines": total_lines,
                "floating_markers": floating_markers,
                "sections": sections,
            }
            if isinstance(existing, dict):
                for key in ("connection_maps", "connection_map_lines"):
                    if key in existing:
                        entry[key] = existing[key]

            self._registry["files"][norm_path] = entry

    def _generate_floating_markers(self, content: str) -> List[Dict[str, Any]]:
        """Generates 3 floating markers at 25%, 50%, 75% of the total lines of the clean content."""
        lines = content.splitlines()
        N = len(lines)
        if N == 0:
            return []

        percentages = [25, 50, 75]
        markers: List[Dict[str, Any]] = []
        for pct in percentages:
            target_idx = round((pct / 100.0) * (N - 1))
            actual_idx, anchor_text = self._choose_floating_anchor(lines, target_idx)
            markers.append(
                {
                    "percentage": pct,
                    "original_line": actual_idx + 1,
                    "anchor": anchor_text,
                }
            )
        return markers

    def _choose_floating_anchor(
        self, lines: List[str], target_idx: int
    ) -> Tuple[int, str]:
        """Finds a good, non-empty anchor line near target_idx (0-indexed)."""
        if not lines:
            return 0, ""

        max_search = min(10, len(lines))
        for dist in range(max_search + 1):
            for sign in (1, -1) if dist > 0 else (0,):
                idx = target_idx + dist * sign
                if 0 <= idx < len(lines):
                    content = lines[idx].strip()
                    if content:  # Non-empty and not just whitespace
                        return idx, lines[idx].strip()

        idx = max(0, min(len(lines) - 1, target_idx))
        return idx, lines[idx].strip()

    def _is_excluded(self, norm_path: str, file_path: str, action: str) -> bool:
        config = ConfigManager()
        from cline_utils.dependency_system.utils.path_utils import get_project_root

        project_root = get_project_root()
        excluded_paths = config.get_excluded_paths()
        excluded_dirs = config.get_excluded_dirs()

        if norm_path in excluded_paths:
            logger.warning(f"Refusing to {action} EXCLUDED file: {file_path}")
            return True

        for directory in excluded_dirs:
            abs_directory = normalize_path(os.path.join(project_root, directory))
            if norm_path.startswith(abs_directory):
                logger.warning(
                    f"Refusing to {action} file in EXCLUDED directory ({directory}): {file_path}"
                )
                return True

        return False

    def _write_connection_map_metadata(
        self,
        file_path: str,
        content: str,
        connection_maps: Optional[List[Dict[str, Any]]],
        connection_map_lines: Optional[List[Dict[str, Any]]],
        *,
        defer_save: bool = False,
    ) -> None:
        with self.update_context():
            norm_path = normalize_path(file_path)
            existing = self._registry["files"].get(norm_path, {})
            existing_dict = (
                cast(Dict[str, Any], existing) if isinstance(existing, dict) else {}
            )
            sections = existing_dict.get("sections", {})
            if not isinstance(sections, dict):
                sections = {}
            entry: Dict[str, Any] = {
                "checksum": calculate_content_hash(content, file_path),
                "last_modified": (
                    os.path.getmtime(file_path)
                    if os.path.exists(file_path)
                    else time.time()
                ),
                "sections": cast(Dict[str, Any], sections),
            }
            if connection_maps:
                entry["connection_maps"] = connection_maps
            if connection_map_lines:
                entry["connection_map_lines"] = connection_map_lines
            self._registry["files"][norm_path] = entry

    def virtualize_connection_maps(
        self, file_path: str, *, clear_if_absent: bool = False, defer_save: bool = False
    ) -> bool:
        """
        Move visible AUTO CONNECTION_MAP comments into transparency metadata.

        Args:
            file_path: Absolute or relative path to the file.
            clear_if_absent: Clear stale hidden connection metadata when a fresh
                populate pass leaves no visible CONNECTION_MAP comments behind.
            defer_save: If True, defer saving the registry to disk.

        Returns:
            True if file content or metadata changed.
        """
        with self.update_context():
            norm_path = normalize_path(file_path)
            if self._is_excluded(
                norm_path, file_path, "virtualize connection maps for"
            ):
                return False
            if not os.path.exists(file_path):
                return False

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                raw_lines: List[Dict[str, Any]] = []
                indices_to_remove: Set[int] = set()
                comments_removed = 0

                # Pre-load symbol map to check active symbols
                symbol_map = {}
                try:
                    from cline_utils.dependency_system.analysis.dependency_suggester import (
                        load_project_symbol_map,
                    )

                    symbol_map = load_project_symbol_map()
                except Exception as e:
                    logger.debug(
                        f"Could not load project symbol map in virtualize_connection_maps: {e}"
                    )

                active_symbols = set()
                has_symbol_data = False
                if symbol_map:
                    file_symbol_data = symbol_map.get(norm_path)
                    if file_symbol_data is not None:
                        has_symbol_data = True
                        for func in file_symbol_data.get("functions", []):
                            if isinstance(func, dict) and "name" in func:
                                active_symbols.add(func["name"])
                        for cls in file_symbol_data.get("classes", []):
                            if isinstance(cls, dict) and "name" in cls:
                                active_symbols.add(cls["name"])
                                for method in cls.get("methods", []):
                                    if isinstance(method, dict) and "name" in method:
                                        active_symbols.add(method["name"])

                for idx, line in enumerate(lines):
                    match = _CONNECTION_MAP_LINE_RE.search(line)
                    if match:
                        source_symbol = match.group("source_symbol").strip()
                        rail = match.group("rail").strip()

                        # Pruning conditions:
                        # 1. Contains "file:?"
                        # 2. Not in active symbols (if symbol data is available for this file)
                        is_stale = "file:?" in rail
                        if not is_stale and has_symbol_data:
                            if source_symbol not in active_symbols:
                                is_stale = True

                        if is_stale:
                            logger.info(
                                f"Pruning stale CONNECTION_MAP for '{source_symbol}' in {file_path}"
                            )
                            indices_to_remove.add(idx)
                            comments_removed += 1
                            continue

                        clean_line_no = idx + 1 - comments_removed
                        raw_lines.append(
                            {"line": clean_line_no, "content": line.rstrip("\r\n")}
                        )
                        indices_to_remove.add(idx)
                        comments_removed += 1

                if not raw_lines:
                    existing = self.get_file_metadata(file_path)
                    if (
                        clear_if_absent
                        and isinstance(existing, dict)
                        and (
                            "connection_maps" in existing
                            or "connection_map_lines" in existing
                        )
                    ):
                        content = "".join(lines)
                        self._write_connection_map_metadata(
                            file_path, content, None, None, defer_save=defer_save
                        )
                        logger.info(
                            f"Cleared stale virtual CONNECTION_MAP metadata for {file_path}"
                        )
                        return True
                    # If we pruned all comments and none remain, ensure we clean file on disk
                    if indices_to_remove:
                        clean_lines = [
                            line
                            for idx, line in enumerate(lines)
                            if idx not in indices_to_remove
                        ]
                        clean_content = "".join(clean_lines)
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(clean_content)
                        self._write_connection_map_metadata(
                            file_path, clean_content, None, None, defer_save=defer_save
                        )
                        return True
                    return False

                original_content = "".join(lines)
                clean_lines = [
                    line
                    for idx, line in enumerate(lines)
                    if idx not in indices_to_remove
                ]
                clean_content = "".join(clean_lines)
                removed_line_numbers = [idx + 1 for idx in sorted(indices_to_remove)]
                records = _adjust_connection_record_lines(
                    _parse_connection_map_text(original_content), removed_line_numbers
                )

                # Filter records to keep only those corresponding to the active (non-pruned) symbols
                kept_symbols = set()
                for rl in raw_lines:
                    m = _CONNECTION_MAP_LINE_RE.search(rl["content"])
                    if m:
                        kept_symbols.add(m.group("source_symbol").strip())
                records = [r for r in records if r.get("source_symbol") in kept_symbols]

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(clean_content)

                self._write_connection_map_metadata(
                    file_path, clean_content, records, raw_lines, defer_save=defer_save
                )

                logger.debug(f"Virtualized CONNECTION_MAP comments for {file_path}")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to virtualize CONNECTION_MAP comments for {file_path}: {e}"
                )
                return False

    def bulk_virtualize_connection_maps(
        self, file_paths: List[str], *, clear_if_absent: bool = False
    ) -> int:
        """
        Perform virtualization of connection maps on multiple files in bulk,
        saving the registry exactly once at the end to avoid expensive disk I/O.
        """
        with self.update_context():
            virtualized = 0
            for path in file_paths:
                if self.virtualize_connection_maps(
                    path, clear_if_absent=clear_if_absent, defer_save=True
                ):
                    virtualized += 1
            return virtualized

    def bulk_prune_stale_virtual_maps(
        self, files_processed: Set[str], files_with_maps: Set[str]
    ) -> None:
        """
        Clears stale virtual connection map metadata from the transparency registry in bulk.
        For any file in files_processed that is NOT in files_with_maps, removes 'connection_maps'
        and 'connection_map_lines' to ensure they don't linger as ghost/stale metadata.
        """
        with self.update_context():
            for file_path in files_processed:
                if file_path in files_with_maps:
                    continue

                norm_path = normalize_path(file_path)
                existing = self._registry["files"].get(norm_path)
                if isinstance(existing, dict):
                    for key in ("connection_maps", "connection_map_lines"):
                        if key in existing:
                            del existing[key]

    def bulk_restore_markers(self, file_paths: List[str]) -> Dict[str, bool]:
        """
        Restores physical markers to multiple files in bulk,
        saving the registry exactly once at the end.
        """
        results: Dict[str, bool] = {}
        with self.update_context():
            for path in file_paths:
                results[path] = self.restore_markers(path)
        return results

    def bulk_remove_markers(self, file_paths: List[str]) -> Dict[str, bool]:
        """
        Removes physical markers from multiple files in bulk,
        saving the registry exactly once at the end.
        """
        results: Dict[str, bool] = {}
        with self.update_context():
            for path in file_paths:
                results[path] = self.remove_markers(path)
        return results

    def check_drift(self, file_path: str, current_content: str) -> bool:
        """
        Checks if the file content has drifted from the recorded checksum.
        Returns True if drift is detected.
        """
        with self._lock:
            metadata = self.get_file_metadata(file_path)
            if not metadata:
                return False

            current_checksum = calculate_content_hash(current_content, file_path)
            return current_checksum != metadata.get("checksum")

    def restore_markers(self, file_path: str) -> bool:
        """
        Restores physical markers to a file based on transparency metadata.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            True if markers were restored, False otherwise.
        """
        with self.update_context():
            norm_path = normalize_path(file_path)

            # Safety Check: Do not operate on excluded files or directories
            config = ConfigManager()
            from cline_utils.dependency_system.utils.path_utils import get_project_root

            project_root = get_project_root()

            excluded_paths = config.get_excluded_paths()
            excluded_dirs = config.get_excluded_dirs()

            if norm_path in excluded_paths:
                logger.warning(
                    f"Refusing to restore markers to EXCLUDED file: {file_path}"
                )
                return False

            for d in excluded_dirs:
                abs_d = normalize_path(os.path.join(project_root, d))
                if norm_path.startswith(abs_d):
                    logger.warning(
                        f"Refusing to restore markers to file in EXCLUDED directory ({d}): {file_path}"
                    )
                    return False

            metadata = self.get_file_metadata(file_path)

            if not metadata:
                logger.debug(
                    f"No transparency metadata found for {file_path}. Cannot restore."
                )
                return False

            if not os.path.exists(file_path):
                logger.error(f"File {file_path} not found. Cannot restore markers.")
                return False

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                sections = metadata.get("sections", {})
                if not sections:
                    logger.debug(f"No sections defined in metadata for {file_path}.")
                    return False

                content = "".join(lines)
                # Drift Check: If content hash doesn't match, line numbers might be wrong!
                is_drifted = self.check_drift(file_path, content)
                if is_drifted:
                    logger.debug(
                        f"Transparency drift detected for {file_path}! Attempting instant auto-recovery..."
                    )
                    recovered_metadata = self.recover_alignment(file_path, content)
                    if recovered_metadata:
                        logger.debug(
                            f"Successfully recovered transparency alignment for {file_path} before restoring markers!"
                        )
                        metadata = recovered_metadata
                        sections = metadata.get("sections", {})
                        is_drifted = False  # Clear drifted status since we've recovered
                    else:
                        logger.error(
                            f"Could not recover transparency alignment for {file_path}. Entry has been invalidated. Cannot restore markers."
                        )
                        return False

                # Explode sections into flat individual marker insertion tasks
                insertions: List[Dict[str, Any]] = []
                for name, val in sections.items():
                    # Skip if this specific marker is already present in clean content
                    if f"---{name}_START---" in content:
                        logger.debug(
                            f"Marker {name} already present in {file_path}. Skipping."
                        )
                        continue

                    if isinstance(val, dict):
                        v_dict = cast(Dict[str, Any], val)
                        if "start_line" in v_dict:
                            sl = v_dict["start_line"]
                            if isinstance(sl, int):
                                start_idx = sl - 1
                                if name == "TAGS":
                                    tags_content = str(v_dict.get("content", ""))
                                    if tags_content:
                                        insertions.append(
                                            {
                                                "idx": start_idx,
                                                "priority": 500,
                                                "content": "---TAGS_END---\n",
                                            }
                                        )
                                        insertions.append(
                                            {
                                                "idx": start_idx,
                                                "priority": 400,
                                                "content": tags_content + "\n",
                                            }
                                        )
                                        insertions.append(
                                            {
                                                "idx": start_idx,
                                                "priority": 300,
                                                "content": "---TAGS_START---\n",
                                            }
                                        )
                        elif "range" in v_dict:
                            rv = v_dict["range"]
                            if isinstance(rv, list) and len(rv) == 2:
                                start_idx = int(rv[0]) - 1
                                end_idx = int(rv[1]) - 1

                                anchors = v_dict.get("anchors")
                                if isinstance(anchors, list) and len(anchors) == 2:
                                    start_anchor = str(anchors[0])
                                    end_anchor = str(anchors[1])

                                    # Perform verification check on start_idx and end_idx
                                    start_matched = (
                                        0 <= start_idx < len(lines)
                                        and lines[start_idx].strip()
                                        == start_anchor.strip()
                                    )
                                    end_matched = (
                                        0 <= (end_idx - 1) < len(lines)
                                        and lines[end_idx - 1].strip()
                                        == end_anchor.strip()
                                    )

                                    if not start_matched or not end_matched:
                                        new_start = self._find_anchor(
                                            lines, start_anchor, start_idx
                                        )
                                        new_end = self._find_anchor(
                                            lines, end_anchor, end_idx
                                        )
                                        if (
                                            new_start is not None
                                            and new_end is not None
                                        ):
                                            start_idx = new_start
                                            end_idx = new_end + 1
                                            logger.info(
                                                f"Re-aligned section {name} via anchors verification check to lines {start_idx+1}-{end_idx+1}"
                                            )

                                range_len = end_idx - start_idx
                                insertions.append(
                                    {
                                        "idx": end_idx,
                                        "priority": 1000 + range_len,
                                        "content": f"---{name}_END---\n",
                                    }
                                )
                                insertions.append(
                                    {
                                        "idx": start_idx,
                                        "priority": 100 - range_len,
                                        "content": f"---{name}_START---\n",
                                    }
                                )
                    elif isinstance(val, list) and len(val) == 2:
                        start_idx = int(val[0]) - 1
                        end_idx = int(val[1]) - 1
                        range_len = end_idx - start_idx
                        insertions.append(
                            {
                                "idx": end_idx,
                                "priority": 1000 + range_len,
                                "content": f"---{name}_END---\n",
                            }
                        )
                        insertions.append(
                            {
                                "idx": start_idx,
                                "priority": 100 - range_len,
                                "content": f"---{name}_START---\n",
                            }
                        )

                # Sort descending by (idx, priority) so insertions happen from bottom-up
                insertions.sort(key=lambda x: (x["idx"], x["priority"]), reverse=True)

                # Perform the insertions in order
                for ins in insertions:
                    try:
                        idx = min(len(lines), max(0, ins["idx"]))
                        lines.insert(idx, ins["content"])
                    except Exception as e_inner:
                        logger.error(
                            f"Error inserting marker in {file_path}: {e_inner}"
                        )
                        continue

                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

                logger.info(f"Restored markers for {file_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to restore markers for {file_path}: {e}")
                return False

    def find_anchor(
        self, lines: List[str], anchor: str, original_idx: int
    ) -> Optional[int]:
        """Public interface to search for an anchor line, prioritizing proximity to original_idx."""
        return self._find_anchor(lines, anchor, original_idx)

    def _find_anchor(
        self, lines: List[str], anchor: str, original_idx: int
    ) -> Optional[int]:
        """Searches for an anchor line, prioritizing proximity to original_idx."""
        if not anchor:
            return None

        stripped_anchor = anchor.strip()
        if not stripped_anchor:
            return None

        # 1. Check original position first
        if original_idx < len(lines) and lines[original_idx].strip() == stripped_anchor:
            return original_idx

        # 2. Search outwards from original position
        max_dist = max(original_idx, len(lines) - original_idx)
        for dist in range(1, max_dist + 1):
            # Check after
            idx = original_idx + dist
            if idx < len(lines) and lines[idx].strip() == stripped_anchor:
                return idx
            # Check before
            idx = original_idx - dist
            if 0 <= idx < len(lines) and lines[idx].strip() == stripped_anchor:
                return idx

        return None

    def remove_markers(self, file_path: str) -> bool:
        """
        Removes physical markers from a file and moves them to the registry.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            True if markers were removed and registered, False otherwise.
        """
        with self.update_context():
            norm_path = normalize_path(file_path)

            # Safety Check: Do not operate on excluded files or directories
            config = ConfigManager()
            from cline_utils.dependency_system.utils.path_utils import get_project_root

            project_root = get_project_root()

            excluded_paths = config.get_excluded_paths()
            excluded_dirs = config.get_excluded_dirs()

            if norm_path in excluded_paths:
                logger.warning(
                    f"Refusing to remove markers from EXCLUDED file: {file_path}"
                )
                return False

            for d in excluded_dirs:
                abs_d = normalize_path(os.path.join(project_root, d))
                if norm_path.startswith(abs_d):
                    logger.warning(
                        f"Refusing to remove markers from file in EXCLUDED directory ({d}): {file_path}"
                    )
                    return False

            if not os.path.exists(file_path):
                return False

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # 1. Identify all markers and sections
                sections: Dict[str, Tuple[int, int]] = {}
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith("---") and stripped.endswith("_START---"):
                        section_name = stripped[3:-9]
                        # Find end
                        for j in range(i + 1, len(lines)):
                            if lines[j].strip() == f"---{section_name}_END---":
                                sections[section_name] = (i, j)
                                break

                if not sections:
                    logger.debug(f"No markers found in {file_path}")
                    return False

                # 2. Identify indices to remove
                indices_to_remove: Set[int] = set()
                for range_pair in sections.values():
                    i_val, j_val = range_pair
                    indices_to_remove.add(int(i_val))
                    indices_to_remove.add(int(j_val))

                # Special handling for TAGS: also remove content
                tags_content: Optional[str] = None
                if "TAGS" in sections:
                    tags_range = sections["TAGS"]
                    s_idx, e_idx = tags_range
                    for idx in range(s_idx + 1, e_idx):
                        indices_to_remove.add(idx)
                    tags_content = "".join(lines[s_idx + 1 : e_idx]).strip()

                # 3. Calculate new content and adjusted indices
                sorted_remove = sorted(list(indices_to_remove))
                new_lines = [
                    l for i, l in enumerate(lines) if i not in indices_to_remove
                ]
                new_content = "".join(new_lines)

                adjusted_sections: Dict[str, Any] = {}
                for name, range_indices in sections.items():
                    start_idx, end_idx = range_indices
                    # Calculate new indices by subtracting removed lines strictly BEFORE them
                    m_before_start = sum(1 for m in sorted_remove if m < start_idx)
                    m_before_end = sum(1 for m in sorted_remove if m < end_idx)

                    new_start: int = start_idx - m_before_start
                    new_end: int = end_idx - m_before_end

                    if name == "TAGS" and tags_content is not None:
                        adjusted_sections["TAGS"] = {
                            "content": tags_content,
                            "start_line": new_start + 1,
                        }
                    else:
                        # Record anchors (content of first and last lines of the section)
                        # to allow re-alignment if the file is edited while clean.
                        start_anchor: str = (
                            new_lines[new_start].strip()
                            if new_start < len(new_lines)
                            else ""
                        )
                        # end_idx in clean file is new_end. Last line of content is new_end - 1.
                        end_anchor: str = (
                            new_lines[new_end - 1].strip()
                            if new_end > 0 and new_end <= len(new_lines)
                            else ""
                        )

                        adjusted_sections[name] = {
                            "range": [new_start + 1, new_end + 1],
                            "anchors": [start_anchor, end_anchor],
                        }

                # 4. Save file first, then update registry
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

                self.update_file_metadata(file_path, adjusted_sections, new_content)

                logger.info(f"Removed and virtualized markers for {file_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to remove markers for {file_path}: {e}")
                return False

    def invalidate_entry(self, file_path: str) -> None:
        """Invalidates/deletes the transparency metadata for a file."""
        with self.update_context():
            norm_path = normalize_path(file_path)
            if norm_path in self._registry["files"]:
                del self._registry["files"][norm_path]
                logger.warning(
                    f"Invalidated and removed drifted transparency entry for {file_path}"
                )

    def recover_alignment(
        self, file_path: str, current_content: str
    ) -> Optional[Dict[str, Any]]:
        """
        Attempts to automatically recover transparency alignment for a drifted file.
        If successful, updates the registry with new line numbers, fresh checksum,
        and fresh floating markers.
        If unrecoverable, places a lock on the entry that is dependent upon the next
        analyze project run to realign it.
        """
        with self.update_context():
            norm_path = normalize_path(file_path)
            metadata = self.get_raw_file_metadata(file_path)
            if not metadata:
                return None

            lines = current_content.splitlines()
            N_new = len(lines)
            if N_new == 0:
                logger.warning(f"Unrecoverable drift: File is empty for {file_path}")
                self.lock_entry(file_path)
                return None

            N_orig = _coerce_line(metadata.get("total_lines"))
            if N_orig is None:
                max_line = 0
                sections_for_derivation = metadata.get("sections", {})
                if isinstance(sections_for_derivation, dict):
                    for name, val in sections_for_derivation.items():
                        if isinstance(val, dict):
                            d_val = cast(Dict[str, Any], val)
                            s_line = _coerce_line(d_val.get("start_line"))
                            if s_line is not None:
                                max_line = max(max_line, s_line)
                            rng = d_val.get("range")
                            if isinstance(rng, (list, tuple)):
                                rng_list = cast(List[Any], rng)
                                if len(rng_list) == 2:
                                    end_line = _coerce_line(rng_list[1])
                                    if end_line is not None:
                                        max_line = max(max_line, end_line)
                        elif isinstance(val, list):
                            val_list = cast(List[Any], val)
                            if len(val_list) == 2:
                                end_line = _coerce_line(val_list[1])
                                if end_line is not None:
                                    max_line = max(max_line, end_line)
                connection_map_lines = metadata.get("connection_map_lines", [])
                if isinstance(connection_map_lines, list):
                    for item in connection_map_lines:
                        if isinstance(item, dict):
                            c_line = _coerce_line(item.get("line"))
                            if c_line is not None:
                                max_line = max(max_line, c_line)
                N_orig = max_line if max_line > 0 else N_new

            # 1. Establish the points for triangulation/interpolation: (original_line, shift)
            points = [(1, 0)]  # Start of file (0% freebie)

            floating_markers = metadata.get("floating_markers", [])
            if isinstance(floating_markers, list):
                for marker in floating_markers:
                    if not isinstance(marker, dict):
                        continue
                    orig_l = _coerce_line(marker.get("original_line"))
                    anchor_text = marker.get("anchor")
                    if orig_l is not None and isinstance(anchor_text, str):
                        found_idx = self._find_anchor(lines, anchor_text, orig_l - 1)
                        if found_idx is not None:
                            new_l = found_idx + 1
                            points.append((orig_l, new_l - orig_l))

            # End of file (100% freebie)
            if N_orig > 1:
                points.append((N_orig, N_new - N_orig))

            points.sort(key=lambda x: x[0])

            # Helper for triangulation
            def get_interpolated_shift(x: int) -> int:
                if not points:
                    return 0
                if x <= points[0][0]:
                    return points[0][1]
                if x >= points[-1][0]:
                    return points[-1][1]

                for i in range(len(points) - 1):
                    x_a, s_a = points[i]
                    x_b, s_b = points[i + 1]
                    if x_a <= x <= x_b:
                        if x_b == x_a:
                            return s_a
                        fraction = (x - x_a) / (x_b - x_a)
                        return round(s_a + (s_b - s_a) * fraction)
                return 0

            # 2. Recover sections
            sections = metadata.get("sections", {})
            recovered_sections: Dict[str, Any] = {}
            anchor_mismatches = 0
            total_range_sections = 0

            for name, val in sections.items():
                if isinstance(val, dict):
                    v_dict = cast(Dict[str, Any], val)
                    if "range" in v_dict:
                        total_range_sections += 1
                        rng = v_dict.get("range")
                        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                            logger.warning(
                                f"Unrecoverable drift: malformed dict range for section {name} in {file_path}"
                            )
                            self.lock_entry(file_path)
                            return None

                        orig_start_line = _coerce_line(rng[0])
                        orig_end_line = _coerce_line(rng[1])
                        if orig_start_line is None or orig_end_line is None:
                            logger.warning(
                                f"Unrecoverable drift: invalid dict range [{rng[0]!r}, {rng[1]!r}] for section {name} in {file_path}"
                            )
                            self.lock_entry(file_path)
                            return None

                        projected_start = orig_start_line + get_interpolated_shift(
                            orig_start_line
                        )
                        projected_end = orig_end_line + get_interpolated_shift(
                            orig_end_line
                        )

                        projected_start = max(1, min(N_new, projected_start))
                        projected_end = max(1, min(N_new + 1, projected_end))

                        anchors = v_dict.get("anchors")
                        recovered_start = projected_start
                        recovered_end = projected_end

                        if isinstance(anchors, list):
                            anchors_list = cast(List[Any], anchors)
                            start_anchor: str = ""
                            end_anchor: str = ""
                            if len(anchors_list) == 2:
                                start_anchor = str(anchors_list[0])
                                end_anchor = str(anchors_list[1])
                            else:
                                anchor_mismatches += 2

                            new_start = self._find_anchor(
                                lines, start_anchor, projected_start - 1
                            )
                            new_end = self._find_anchor(
                                lines, end_anchor, projected_end - 1
                            )

                            if new_start is not None:
                                recovered_start = new_start + 1
                            else:
                                anchor_mismatches += 1

                            if new_end is not None:
                                recovered_end = new_end + 2
                            else:
                                anchor_mismatches += 1
                        else:
                            anchor_mismatches += 2

                        if (
                            recovered_start > recovered_end
                            or recovered_start < 1
                            or recovered_end > N_new + 1
                        ):
                            logger.warning(
                                f"Unrecoverable drift: invalid recovered range [{recovered_start}, {recovered_end}] "
                                f"for section {name} in {file_path}"
                            )
                            self.lock_entry(file_path)
                            return None

                        recovered_sections[name] = {
                            "range": [recovered_start, recovered_end],
                            "anchors": anchors,
                        }
                    elif "start_line" in v_dict:
                        orig_start = _coerce_line(v_dict.get("start_line"))
                        if orig_start is None:
                            logger.warning(
                                f"Unrecoverable drift: invalid start_line {v_dict.get('start_line')!r} for section {name} in {file_path}"
                            )
                            self.lock_entry(file_path)
                            return None
                        projected_start = orig_start + get_interpolated_shift(
                            orig_start
                        )
                        projected_start = max(1, min(N_new, projected_start))
                        recovered_sections[name] = {
                            "content": v_dict.get("content", ""),
                            "start_line": projected_start,
                        }
                elif isinstance(val, list):
                    val_list = cast(List[Any], val)
                    if len(val_list) != 2:
                        logger.warning(
                            f"Unrecoverable drift: malformed list range for section {name} in {file_path}"
                        )
                        self.lock_entry(file_path)
                        return None

                    orig_start_line = _coerce_line(val_list[0])
                    orig_end_line = _coerce_line(val_list[1])
                    if orig_start_line is None or orig_end_line is None:
                        logger.warning(
                            f"Unrecoverable drift: invalid list range [{val_list[0]!r}, {val_list[1]!r}] for section {name} in {file_path}"
                        )
                        self.lock_entry(file_path)
                        return None

                    projected_start = orig_start_line + get_interpolated_shift(
                        orig_start_line
                    )
                    projected_end = orig_end_line + get_interpolated_shift(
                        orig_end_line
                    )
                    projected_start = max(1, min(N_new, projected_start))
                    projected_end = max(1, min(N_new + 1, projected_end))

                    if projected_start > projected_end:
                        logger.warning(
                            f"Unrecoverable drift: invalid projected range [{projected_start}, {projected_end}] "
                            f"for list section {name} in {file_path}"
                        )
                        self.lock_entry(file_path)
                        return None

                    recovered_sections[name] = [projected_start, projected_end]

            if (
                total_range_sections > 0
                and anchor_mismatches == 2 * total_range_sections
            ):
                logger.warning(
                    f"Unrecoverable drift: failed to match any anchor in {file_path}"
                )
                self.lock_entry(file_path)
                return None

            # 3. Update connection maps in registry first so they get carried over
            recovered_connection_maps: List[Dict[str, Any]] = []
            connection_maps = metadata.get("connection_maps", [])
            if isinstance(connection_maps, list):
                for record in connection_maps:
                    if not isinstance(record, dict):
                        continue
                    new_record = dict(record)
                    orig_sl = _coerce_line(record.get("source_line"))
                    if orig_sl is not None:
                        new_record["source_line"] = orig_sl + get_interpolated_shift(
                            orig_sl
                        )
                    recovered_connection_maps.append(new_record)

            recovered_connection_map_lines: List[Dict[str, Any]] = []
            connection_map_lines = metadata.get("connection_map_lines", [])
            if isinstance(connection_map_lines, list):
                for item in connection_map_lines:
                    if not isinstance(item, dict):
                        continue
                    new_item = dict(item)
                    orig_l = _coerce_line(item.get("line"))
                    if orig_l is not None:
                        new_item["line"] = orig_l + get_interpolated_shift(orig_l)
                    recovered_connection_map_lines.append(new_item)

            self._registry["files"][norm_path][
                "connection_maps"
            ] = recovered_connection_maps
            self._registry["files"][norm_path][
                "connection_map_lines"
            ] = recovered_connection_map_lines

            # 4. Save and return updated metadata
            self.update_file_metadata(file_path, recovered_sections, current_content)
            return self.get_file_metadata(file_path)

    def realign_locked_entries(self, project_symbol_map: Dict[str, Any]) -> int:
        """
        Scans all files in the registry for locked entries and attempts to realign them
        using precise symbol line numbers from the newly regenerated project symbol map.

        Returns:
            Number of successfully realigned files.
        """
        realigned_count = 0
        with self.update_context():
            locked_paths = [
                path
                for path, entry in self._registry["files"].items()
                if entry.get("locked")
            ]
            if not locked_paths:
                return 0

            logger.info(
                f"Attempting to realign {len(locked_paths)} locked transparency entries..."
            )

            for norm_path in locked_paths:
                entry = self._registry["files"][norm_path]
                file_path = norm_path  # The path is absolute and normalized

                if not os.path.exists(file_path):
                    logger.warning(
                        f"Locked file {file_path} no longer exists. Invalidating entry."
                    )
                    del self._registry["files"][norm_path]
                    continue

                # Read current file content
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    logger.error(f"Failed to read locked file {file_path}: {e}")
                    continue

                lines = content.splitlines()
                N_new = len(lines)
                if N_new == 0:
                    logger.warning(
                        f"Locked file {file_path} is empty. Invalidating entry."
                    )
                    del self._registry["files"][norm_path]
                    continue

                symbol_data = project_symbol_map.get(norm_path)
                if not symbol_data:
                    logger.warning(
                        f"No symbol data found for locked file {file_path} in symbol map. Leaving locked."
                    )
                    continue

                sections = entry.get("sections", {})
                new_sections = {}
                success = True

                # Determine if it's a markdown or code file
                is_md = file_path.lower().endswith(".md")

                if is_md:
                    # Retrieve headers from symbol data or fallback to searching the file
                    headers_list = symbol_data.get("headers", [])

                    # 1. Resolve start lines for range-based sections
                    range_sections: List[Dict[str, Any]] = []
                    for name, sec_val in sections.items():
                        if isinstance(sec_val, dict) and "range" in sec_val:
                            anchors = cast(Dict[str, Any], sec_val).get("anchors", [])
                            if not isinstance(anchors, list):
                                continue
                            anchors_list = cast(List[Any], anchors)
                            if len(anchors_list) < 1:
                                continue
                            start_anchor = str(anchors_list[0])

                            # Search in symbol map headers
                            line_idx = None
                            for h_val in headers_list:
                                if isinstance(h_val, dict):
                                    h = cast(Dict[str, Any], h_val)
                                    h_name = h.get("name")
                                    if (
                                        h_name == start_anchor
                                        or h_name == start_anchor.lstrip("#").strip()
                                    ):
                                        line_idx = h.get("line")
                                        if line_idx is not None:
                                            line_idx -= 1  # convert to 0-indexed
                                            break

                            # Fallback: search the file content directly
                            if line_idx is None:
                                for idx, line in enumerate(lines):
                                    if (
                                        start_anchor in line
                                        or start_anchor.lstrip("#").strip() in line
                                    ):
                                        line_idx = idx
                                        break

                            if line_idx is not None:
                                range_sections.append(
                                    {
                                        "name": name,
                                        "start_idx": line_idx,
                                        "anchors": anchors,
                                    }
                                )
                            else:
                                logger.warning(
                                    f"Could not find start anchor '{start_anchor}' for section {name} in {file_path}"
                                )
                                success = False
                                break
                        elif isinstance(sec_val, dict) and "start_line" in sec_val:
                            # Keep TAGS or other start_line sections
                            new_sections[name] = dict(cast(Dict[str, Any], sec_val))

                    if not success or not range_sections:
                        logger.warning(
                            f"Failed to resolve headers/anchors for {file_path}. Keeping locked."
                        )
                        continue

                    # Sort sections by start index to establish ranges
                    range_sections.sort(key=lambda x: x["start_idx"])

                    # 2. Assign end ranges and construct updated sections
                    for i, sec in enumerate(range_sections):
                        name = sec["name"]
                        start_idx = sec["start_idx"]

                        if i < len(range_sections) - 1:
                            end_idx = range_sections[i + 1]["start_idx"] - 1
                        else:
                            end_idx = N_new - 1

                        # Ensure validity
                        if start_idx > end_idx or start_idx < 0 or end_idx >= N_new:
                            logger.warning(
                                f"Invalid resolved range [{start_idx + 1}, {end_idx + 1}] for section {name} in {file_path}"
                            )
                            success = False
                            break

                        # Update end anchor to be the content of the new end line
                        sec_anchors = cast(List[Any], sec.get("anchors", []))
                        start_anchor = sec_anchors[0]
                        end_anchor = lines[end_idx].strip()

                        new_sections[name] = {
                            "range": [start_idx + 1, end_idx + 1],
                            "anchors": [start_anchor, end_anchor],
                        }

                    if not success:
                        continue

                    # Carry over sections
                    entry["sections"] = new_sections

                # 3. Realign connection map lines for code/all files
                if "connection_map_lines" in entry:
                    connection_map_lines = cast(
                        List[Dict[str, Any]], entry.get("connection_map_lines", [])
                    )
                    new_conn_map_lines: List[Dict[str, Any]] = []
                    for item_val in connection_map_lines:
                        item = cast(Dict[str, Any], item_val)
                        symbol_name = item.get("symbol")
                        if symbol_name:
                            resolved_line = None
                            # Search in symbol map
                            for sym_type in ("functions", "classes", "globals_defined"):
                                for sym in cast(
                                    List[Any], symbol_data.get(sym_type, [])
                                ):
                                    if isinstance(sym, dict):
                                        sym_dict = cast(Dict[str, Any], sym)
                                        if sym_dict.get("name") == symbol_name:
                                            resolved_line = sym_dict.get("line")
                                            if resolved_line is not None:
                                                break
                                if resolved_line is not None:
                                    break

                            # Fallback: search in file lines
                            if resolved_line is None:
                                for idx, line in enumerate(lines):
                                    if symbol_name in line:
                                        resolved_line = idx + 1
                                        break

                            if resolved_line is not None:
                                item["line"] = resolved_line
                        new_conn_map_lines.append(item)
                    entry["connection_map_lines"] = new_conn_map_lines

                # 4. Clear lock, update total lines, checksum, last modified, and floating markers
                entry["total_lines"] = N_new
                entry["checksum"] = calculate_content_hash(content, file_path)
                entry["last_modified"] = os.path.getmtime(file_path)
                entry["floating_markers"] = self._generate_floating_markers(content)
                if "locked" in entry:
                    del entry["locked"]

                self._registry["files"][norm_path] = entry
                realigned_count += 1
                logger.info(
                    f"Successfully realigned drifted transparency entry for {file_path} using project symbol map!"
                )

        return realigned_count


# Global instance for shared use
_manager_instance: Optional[TransparencyManager] = None


def get_transparency_manager() -> TransparencyManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = TransparencyManager()
    return _manager_instance


def read_file_transparently(
    file_path: str,
    include_connection_maps: bool = False,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Reads a file and retrieves its transparency metadata (virtual markers).

    Returns:
        A tuple of (content, transparency_metadata).
    """
    content = read_file_content_safely(file_path)
    if content is None:
        return None, None

    manager = get_transparency_manager()
    metadata = manager.get_file_metadata(file_path)

    # Check for drift
    if metadata and manager.check_drift(file_path, content):
        logger.debug(
            f"Transparency drift detected for {file_path}. Attempting instant auto-recovery..."
        )
        try:
            recovered_metadata = manager.recover_alignment(file_path, content)
        except Exception as recovery_err:
            logger.error(
                f"Transparency recovery failed for {file_path}: {recovery_err}",
                exc_info=True,
            )
            try:
                manager.lock_entry(file_path)
            except Exception:
                logger.error(
                    f"Failed to lock transparency entry after recovery failure for {file_path}",
                    exc_info=True,
                )
            metadata = None
        else:
            if recovered_metadata:
                logger.debug(
                    f"Successfully recovered transparency alignment for {file_path}!"
                )
                metadata = recovered_metadata
            else:
                logger.error(
                    f"Could not recover transparency alignment for {file_path}. Entry has been invalidated."
                )
                metadata = None

    if include_connection_maps:
        content = overlay_connection_maps(content, metadata)

    return content, metadata
