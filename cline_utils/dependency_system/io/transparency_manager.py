import json
import logging
import os
import re
import time
from typing import Dict, Any, Optional, Tuple, List, Set, cast
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
REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "core", "transparency_registry.json"
)


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
    patterns = (
        f"def {source_symbol}",
        f"class {source_symbol}",
        f"function {source_symbol}",
        f"const {source_symbol}",
        f"let {source_symbol}",
        f"var {source_symbol}",
    )
    for idx in range(map_line_index + 1, min(len(lines), map_line_index + 12)):
        if any(pattern in lines[idx] for pattern in patterns):
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
            return [
                cast(Dict[str, Any], item) for item in direct if isinstance(item, dict)
            ]

        sections = transparency_metadata.get("sections", {})
        if isinstance(sections, dict):
            section = cast(Dict[Any, Any], sections).get("CONNECTION_MAPS")
            if isinstance(section, dict) and "content" in section:
                return _parse_connection_map_text(str(section.get("content", "")))

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

    lines = content.splitlines(keepends=True)
    inserts: List[Tuple[int, str]] = []
    for item in raw_lines:
        if not isinstance(item, dict):
            continue
        line_no = _coerce_line(item.get("line"))
        line_content = item.get("content")
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


class TransparencyManager:
    """
    Manages the "invisible" transparency layer for documentation section markers.
    Maps file paths to line-number based section definitions stored externally.
    """

    def __init__(self, registry_path: str = REGISTRY_PATH):
        super().__init__()
        self.registry_path = registry_path
        self._registry: Dict[str, Any] = {"files": {}}
        self._load()

    def _load(self) -> None:
        """Loads the transparency registry from disk."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self._registry = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load transparency registry: {e}")
                self._registry = {"files": {}}
        else:
            self._registry = {"files": {}}

    def _save(self) -> None:
        """Saves the transparency registry to disk atomically."""
        temp_path = self.registry_path + ".tmp"
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, indent=2)

            # Atomic rename (on Windows this may require os.replace)
            if os.path.exists(self.registry_path):
                os.replace(temp_path, self.registry_path)
            else:
                os.rename(temp_path, self.registry_path)
        except Exception as e:
            logger.error(f"Failed to save transparency registry atomically: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Retrieves transparency metadata for a specific file."""
        norm_path = normalize_path(file_path)
        return self._registry["files"].get(norm_path)

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
        norm_path = normalize_path(file_path)
        checksum = calculate_content_hash(content, file_path)
        existing = self._registry["files"].get(norm_path, {})

        entry = {
            "checksum": checksum,
            "last_modified": (
                os.path.getmtime(file_path)
                if os.path.exists(file_path)
                else time.time()
            ),
            "sections": sections,
        }
        if isinstance(existing, dict):
            for key in ("connection_maps", "connection_map_lines"):
                if key in existing:
                    entry[key] = existing[key]

        self._registry["files"][norm_path] = entry
        self._save()

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
    ) -> None:
        norm_path = normalize_path(file_path)
        existing = self._registry["files"].get(norm_path, {})
        sections = existing.get("sections", {}) if isinstance(existing, dict) else {}
        entry = {
            "checksum": calculate_content_hash(content, file_path),
            "last_modified": (
                os.path.getmtime(file_path)
                if os.path.exists(file_path)
                else time.time()
            ),
            "sections": sections,
        }
        if connection_maps:
            entry["connection_maps"] = connection_maps
        if connection_map_lines:
            entry["connection_map_lines"] = connection_map_lines
        self._registry["files"][norm_path] = entry
        self._save()

    def virtualize_connection_maps(
        self, file_path: str, *, clear_if_absent: bool = False
    ) -> bool:
        """
        Move visible AUTO CONNECTION_MAP comments into transparency metadata.

        Args:
            file_path: Absolute or relative path to the file.
            clear_if_absent: Clear stale hidden connection metadata when a fresh
                populate pass leaves no visible CONNECTION_MAP comments behind.

        Returns:
            True if file content or metadata changed.
        """
        norm_path = normalize_path(file_path)
        if self._is_excluded(norm_path, file_path, "virtualize connection maps for"):
            return False
        if not os.path.exists(file_path):
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            raw_lines: List[Dict[str, Any]] = []
            indices_to_remove: Set[int] = set()
            for idx, line in enumerate(lines):
                if _CONNECTION_MAP_LINE_RE.search(line):
                    raw_lines.append({"line": idx + 1, "content": line.rstrip("\r\n")})
                    indices_to_remove.add(idx)

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
                    self._write_connection_map_metadata(file_path, content, None, None)
                    logger.info(
                        f"Cleared stale virtual CONNECTION_MAP metadata for {file_path}"
                    )
                    return True
                return False

            original_content = "".join(lines)
            clean_lines = [
                line for idx, line in enumerate(lines) if idx not in indices_to_remove
            ]
            clean_content = "".join(clean_lines)
            removed_line_numbers = [idx + 1 for idx in sorted(indices_to_remove)]
            records = _adjust_connection_record_lines(
                _parse_connection_map_text(original_content), removed_line_numbers
            )

            self._write_connection_map_metadata(
                file_path, clean_content, records, raw_lines
            )

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(clean_content)

            logger.info(f"Virtualized CONNECTION_MAP comments for {file_path}")
            return True
        except Exception as e:
            logger.error(
                f"Failed to virtualize CONNECTION_MAP comments for {file_path}: {e}"
            )
            return False

    def check_drift(self, file_path: str, current_content: str) -> bool:
        """
        Checks if the file content has drifted from the recorded checksum.
        Returns True if drift is detected.
        """
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
        norm_path = normalize_path(file_path)

        # Safety Check: Do not operate on excluded files or directories
        config = ConfigManager()
        from cline_utils.dependency_system.utils.path_utils import get_project_root

        project_root = get_project_root()

        excluded_paths = config.get_excluded_paths()
        excluded_dirs = config.get_excluded_dirs()

        if norm_path in excluded_paths:
            logger.warning(f"Refusing to restore markers to EXCLUDED file: {file_path}")
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
                logger.warning(
                    f"Transparency drift detected for {file_path}! "
                    "File content has changed since markers were removed. "
                    "Restoration line numbers may be inaccurate."
                )
                # We still proceed, but the warning is important.

            # Sections can be:
            # 1. List [start, end] (1-indexed)
            # 2. Dict {"content": "...", "start_line": n} (virtual content like TAGS)

            # Items to restore: (start_line, type, name, data)
            sorted_items: List[Tuple[int, str, str, Any]] = []
            for name, val in sections.items():
                if isinstance(val, dict):
                    v_dict = cast(Dict[str, Any], val)
                    if "start_line" in v_dict:
                        sl = v_dict["start_line"]
                        if isinstance(sl, int):
                            sorted_items.append((sl, "virtual", str(name), v_dict))
                    elif "range" in v_dict:
                        rv = v_dict["range"]
                        if isinstance(rv, list) and len(rv) >= 1:
                            sorted_items.append(
                                (int(rv[0]), "range", str(name), v_dict)
                            )
                elif isinstance(val, list):
                    v_list = cast(List[Any], val)
                    if len(v_list) == 2:
                        sorted_items.append(
                            (int(v_list[0]), "range", str(name), v_list)
                        )

            # Sort by start line DESCENDING so insertions don't affect earlier indices
            sorted_items.sort(key=lambda x: x[0], reverse=True)

            for start_line, item_type, name, data in sorted_items:
                start_idx: int = int(start_line) - 1
                end_idx = 0  # Will be calculated below

                try:
                    # Skip if this specific marker is already present
                    if f"---{name}_START---" in content:
                        logger.debug(
                            f"Marker {name} already present in {file_path}. Skipping."
                        )
                        continue

                    if item_type == "range":
                        # Calculate default end_idx
                        if isinstance(data, dict):
                            d_dict = cast(Dict[str, Any], data)
                            orig_end_line: int = int(d_dict["range"][1])
                        else:
                            orig_end_line = int(data[1])

                        end_idx: int = orig_end_line - 1

                        # Try to re-align using anchors if drifted
                        if is_drifted:
                            anchors = None
                            if isinstance(data, dict):
                                anchors = cast(Dict[str, Any], data).get("anchors")

                            if isinstance(anchors, list) and len(anchors) == 2:
                                a_list = cast(List[Any], anchors)
                                start_anchor: str = str(a_list[0])
                                end_anchor: str = str(a_list[1])
                                # Search for anchors in a window around the original positions
                                new_start = self._find_anchor(
                                    lines, start_anchor, start_idx
                                )
                                new_end = self._find_anchor(lines, end_anchor, end_idx)

                                if new_start is not None and new_end is not None:
                                    start_idx = new_start
                                    end_idx = (
                                        new_end + 1
                                    )  # End marker goes AFTER the anchor line
                                    logger.info(
                                        f"Re-aligned section {name} via anchors to lines {start_idx+1}-{end_idx+1}"
                                    )

                    if item_type == "virtual" and name == "TAGS":
                        tags_content: str = ""
                        if isinstance(data, dict):
                            tags_content = str(
                                cast(Dict[str, Any], data).get("content", "")
                            )

                        if tags_content:
                            # Clamp indices to current file size
                            idx = min(len(lines), max(0, start_idx))
                            # Insert TAGS block (End, then Content, then Start)
                            lines.insert(idx, "---TAGS_END---\n")
                            lines.insert(idx, tags_content + "\n")
                            lines.insert(idx, "---TAGS_START---\n")

                    elif item_type == "range":
                        # Clamp indices to current file size
                        s_idx = min(len(lines), max(0, start_idx))
                        e_idx = min(len(lines), max(0, int(end_idx)))

                        # Insert END marker first, then START marker (End first to avoid shifting Start)
                        # Wait, if we are at the same line, order matters.
                        # But since we are iterating DESCENDING, we should be fine if we do End then Start.
                        lines.insert(e_idx, f"---{name}_END---\n")
                        lines.insert(s_idx, f"---{name}_START---\n")
                except Exception as e_inner:
                    logger.error(
                        f"Error restoring section {name} ({type}) in {file_path}: {e_inner}"
                    )
                    continue

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            logger.info(f"Restored markers for {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore markers for {file_path}: {e}")
            return False

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
            new_lines = [l for i, l in enumerate(lines) if i not in indices_to_remove]
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

            # 4. Update registry and save file
            self.update_file_metadata(file_path, adjusted_sections, new_content)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            logger.info(f"Removed and virtualized markers for {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove markers for {file_path}: {e}")
            return False


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
        import logging

        logging.getLogger(__name__).warning(
            f"Transparency drift detected for {file_path}. Metadata may be inaccurate."
        )

    if include_connection_maps:
        content = overlay_connection_maps(content, metadata)

    return content, metadata
