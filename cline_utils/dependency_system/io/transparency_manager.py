import os
import json
import logging
import time
from typing import Dict, Any, Optional, Tuple, List, Set, cast
from cline_utils.dependency_system.utils.calculate_hash import calculate_content_hash
from cline_utils.dependency_system.utils.cache_manager import (
    normalize_path_cached as normalize_path,
)
from cline_utils.dependency_system.io.file_io import read_file_content_safely
from cline_utils.dependency_system.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Registry location
REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "core", "transparency_registry.json"
)


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

        self._registry["files"][norm_path] = {
            "checksum": checksum,
            "last_modified": (
                os.path.getmtime(file_path)
                if os.path.exists(file_path)
                else time.time()
            ),
            "sections": sections,
        }
        self._save()

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
                logger.warning(f"Refusing to restore markers to file in EXCLUDED directory ({d}): {file_path}")
                return False

        metadata = self.get_file_metadata(file_path)
        
        if not metadata:
            logger.debug(f"No transparency metadata found for {file_path}. Cannot restore.")
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
                            sorted_items.append((int(rv[0]), "range", str(name), v_dict))
                elif isinstance(val, list):
                    v_list = cast(List[Any], val)
                    if len(v_list) == 2:
                        sorted_items.append((int(v_list[0]), "range", str(name), v_list))
            
            # Sort by start line DESCENDING so insertions don't affect earlier indices
            sorted_items.sort(key=lambda x: x[0], reverse=True)
            
            for start_line, item_type, name, data in sorted_items:
                start_idx: int = int(start_line) - 1
                end_idx = 0 # Will be calculated below
                
                try:
                    # Skip if this specific marker is already present
                    if f"---{name}_START---" in content:
                        logger.debug(f"Marker {name} already present in {file_path}. Skipping.")
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
                                new_start = self._find_anchor(lines, start_anchor, start_idx)
                                new_end = self._find_anchor(lines, end_anchor, end_idx)
                                
                                if new_start is not None and new_end is not None:
                                    start_idx = new_start
                                    end_idx = new_end + 1 # End marker goes AFTER the anchor line
                                    logger.info(f"Re-aligned section {name} via anchors to lines {start_idx+1}-{end_idx+1}")
                    
                    if item_type == "virtual" and name == "TAGS":
                        tags_content: str = ""
                        if isinstance(data, dict):
                            tags_content = str(cast(Dict[str, Any], data).get("content", ""))
                        
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
                    logger.error(f"Error restoring section {name} ({type}) in {file_path}: {e_inner}")
                    continue
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
                
            logger.info(f"Restored markers for {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore markers for {file_path}: {e}")
            return False

    def _find_anchor(self, lines: List[str], anchor: str, original_idx: int) -> Optional[int]:
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
            logger.warning(f"Refusing to remove markers from EXCLUDED file: {file_path}")
            return False
            
        for d in excluded_dirs:
            abs_d = normalize_path(os.path.join(project_root, d))
            if norm_path.startswith(abs_d):
                logger.warning(f"Refusing to remove markers from file in EXCLUDED directory ({d}): {file_path}")
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
                        "start_line": new_start + 1
                    }
                else:
                    # Record anchors (content of first and last lines of the section)
                    # to allow re-alignment if the file is edited while clean.
                    start_anchor: str = new_lines[new_start].strip() if new_start < len(new_lines) else ""
                    # end_idx in clean file is new_end. Last line of content is new_end - 1.
                    end_anchor: str = new_lines[new_end - 1].strip() if new_end > 0 and new_end <= len(new_lines) else ""
                    
                    adjusted_sections[name] = {
                        "range": [new_start + 1, new_end + 1],
                        "anchors": [start_anchor, end_anchor]
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

    return content, metadata
