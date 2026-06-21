# context_packager.py

"""
Constructs token-optimized context packages based on dependency tiers and SES fallbacks.
Decouples token window management and dynamic LLM routing.
"""

import json
import logging
import os
import re
from typing import Dict, List, Set, Tuple, Any, Optional, TypedDict


class InsertionDict(TypedDict):
    idx: int
    priority: int
    content: str


from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import (
    get_project_root,
    normalize_path,
)
from cline_utils.dependency_system.core.key_manager import (
    KeyInfo,
    load_global_key_map,
    load_old_global_key_map,
)
from cline_utils.dependency_system.io.tracker_io import build_path_migration_map
from cline_utils.dependency_system.utils.tracker_utils import (
    find_all_tracker_paths,
    aggregate_all_dependencies,
)
from cline_utils.dependency_system.io.transparency_manager import (
    get_transparency_manager,
    overlay_connection_maps,
)
from cline_utils.dependency_system.analysis.embedding_manager import (
    generate_symbol_essence_string,
)
from cline_utils.dependency_system.analysis.dependency_suggester import (
    load_project_symbol_map,
)

logger = logging.getLogger(__name__)


def _load_token_metadata(project_root: str) -> Dict[str, Dict[str, int]]:
    """Loads token counts from metadata.json."""
    metadata_path = os.path.join(
        project_root,
        "cline_utils",
        "dependency_system",
        "analysis",
        "embeddings",
        "metadata.json",
    )
    token_map: Dict[str, Dict[str, int]] = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                keys = data.get("keys", {})
                for key_data in keys.values():
                    path = key_data.get("path")
                    if not path:
                        continue
                    path = normalize_path(path)

                    ses = key_data.get("ses_tokens")
                    full = key_data.get("full_tokens")

                    if ses is None and "tokens" in key_data:
                        ses = key_data["tokens"]

                    if full is None:
                        full = ses

                    if ses is not None:
                        token_map[path] = {
                            "ses_tokens": int(ses),
                            "full_tokens": int(full or ses),
                        }
        except Exception as e:
            logger.warning(f"Failed to load token metadata: {e}")
    return token_map


class ContextPackager:
    """
    Constructs token-optimized context packages based on dependency tiers and SES fallbacks.
    """

    # Character Tier mappings (conforming to exact CRCT definitions)
    TIER_MAP = {
        "x": 1,  # Mutual Requirement
        "<": 2,  # Inbound / Row depends on column
        ">": 2,  # Outbound / Column depends on row
        "d": 3,  # Documentation
        "S": 4,  # Strong Semantic dependency (.07+)
        "s": 5,  # Weak Semantic dependency (.06-.07)
    }

    def __init__(self, project_root: Optional[str] = None):
        super().__init__()
        self.project_root = project_root if project_root else get_project_root()
        self.config_mgr = ConfigManager()
        self._token_meta_cache: Optional[Dict[str, Dict[str, int]]] = None
        self._symbol_map_cache: Optional[Dict[str, Dict[str, Any]]] = None

    def _get_token_metadata(self) -> Dict[str, Dict[str, int]]:
        """Loads and caches token metadata."""
        if self._token_meta_cache is None:
            self._token_meta_cache = _load_token_metadata(self.project_root)
        return self._token_meta_cache

    def _get_symbol_map(self) -> Dict[str, Dict[str, Any]]:
        """Loads and caches the project symbol map."""
        if self._symbol_map_cache is None:
            self._symbol_map_cache = load_project_symbol_map()
        return self._symbol_map_cache

    def get_file_tokens(self, file_path: str) -> Tuple[int, int]:
        """Returns (full_tokens, ses_tokens) for a given normalized path."""
        norm_path = normalize_path(file_path)
        meta = self._get_token_metadata().get(norm_path, {})
        full = meta.get("full_tokens", 1000)
        ses = meta.get("ses_tokens", 250)
        return full, ses

    def get_overlaid_content(
        self, file_path: str, include_connection_maps: bool = True
    ) -> str:
        """Retrieves file content with transparency overlay maps/tags reinserted in memory."""
        norm_path = normalize_path(file_path)
        if not os.path.exists(norm_path):
            return f"// File not found: {norm_path}"

        try:
            with open(norm_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {norm_path}: {e}")
            return f"// Failed to read file: {norm_path}"

        tm = get_transparency_manager()
        meta = tm.get_file_metadata(norm_path)
        if not meta:
            return content

        # 1. Overlay connection maps
        if include_connection_maps:
            content = overlay_connection_maps(content, meta)

        # 2. Overlay sections in memory without physical file writes
        sections: Dict[str, Any] = meta.get("sections", {})
        if not sections:
            return content

        lines = content.splitlines(keepends=True)
        insertions: List[InsertionDict] = []

        is_drifted = tm.check_drift(norm_path, content)
        if is_drifted:
            recovered = tm.recover_alignment(norm_path, content)
            if recovered:
                meta = recovered
                sections = meta.get("sections", {})
                is_drifted = False

        for name, val in sections.items():
            if f"---{name}_START---" in content:
                continue

            if isinstance(val, dict):
                val_dict: Dict[str, Any] = val
                if "start_line" in val_dict:
                    sl = val_dict.get("start_line")
                    if isinstance(sl, int):
                        start_idx = sl - 1
                        if name == "TAGS":
                            tags_content = str(val_dict.get("content", ""))
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
                elif "range" in val_dict:
                    rv = val_dict.get("range")
                    rv_list: List[Any] = rv if isinstance(rv, list) else []
                    if len(rv_list) == 2:
                        start_idx = int(rv_list[0]) - 1
                        end_idx = int(rv_list[1]) - 1

                        anchors = val_dict.get("anchors")
                        anchors_list: List[Any] = (
                            anchors if isinstance(anchors, list) else []
                        )
                        if len(anchors_list) == 2:
                            start_anchor = str(anchors_list[0])
                            end_anchor = str(anchors_list[1])

                            # Perform verification check on start_idx and end_idx
                            start_matched = (
                                0 <= start_idx < len(lines)
                                and lines[start_idx].strip() == start_anchor.strip()
                            )
                            end_matched = (
                                0 <= (end_idx - 1) < len(lines)
                                and lines[end_idx - 1].strip() == end_anchor.strip()
                            )

                            if not start_matched or not end_matched:
                                new_start = tm.find_anchor(
                                    lines, start_anchor, start_idx
                                )
                                new_end = tm.find_anchor(lines, end_anchor, end_idx)
                                if new_start is not None and new_end is not None:
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

        insertions.sort(key=lambda x: (x["idx"], x["priority"]), reverse=True)
        for ins in insertions:
            idx = min(len(lines), max(0, ins["idx"]))
            lines.insert(idx, ins["content"])

        return "".join(lines)

    def get_ses_content(
        self, file_path: str, include_connection_maps: bool = True
    ) -> str:
        """Retrieves lightweight Structural/Execution/Signature (SES) content using Static Engine."""
        norm_path = normalize_path(file_path)
        symbol_map = self._get_symbol_map()
        if norm_path in symbol_map:
            try:
                return generate_symbol_essence_string(
                    norm_path, symbol_map[norm_path], symbol_map=symbol_map
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate symbol essence for {norm_path}: {e}"
                )

        # Fallback to basic overlay if static generator fails or not in map
        return self.get_overlaid_content(
            norm_path, include_connection_maps=include_connection_maps
        )

    def _extract_sliced_block(self, file_path: str, start_line_1: int) -> str:
        """
        Extracts the function/class/method definition starting at start_line_1.
        """
        norm_path = normalize_path(file_path)
        if not os.path.exists(norm_path):
            return ""
        try:
            with open(norm_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read file {norm_path} for slicing: {e}")
            return ""

        if not lines:
            return ""

        start_idx = max(0, min(len(lines) - 1, start_line_1 - 1))
        ext = os.path.splitext(norm_path)[1].lower()

        # Scan upwards for decorators in Python/JS/TS/CS/Java
        if ext in (".py", ".js", ".ts", ".tsx", ".jsx", ".cs", ".java"):
            while start_idx > 0:
                prev_line = lines[start_idx - 1].strip()
                if prev_line.startswith("@"):
                    start_idx -= 1
                else:
                    break

        end_idx = start_idx + 1

        if ext == ".py":
            # Python: indentation-based scanning
            def_idx = start_line_1 - 1
            if def_idx >= len(lines):
                def_idx = start_idx

            # Find the end of the def/class header (ends with :)
            body_start_idx = def_idx
            for i in range(def_idx, len(lines)):
                stripped = lines[i].strip()
                if ":" in stripped:
                    clean = stripped.split("#")[0].strip()
                    if clean.endswith(":"):
                        body_start_idx = i + 1
                        break

            def_line = lines[def_idx]
            indent_len = len(def_line) - len(def_line.lstrip())

            end_idx = len(lines)
            for i in range(body_start_idx, len(lines)):
                line = lines[i]
                if not line.strip():
                    continue
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= indent_len:
                    if not line.strip().startswith("#"):
                        end_idx = i
                        break
        elif ext in (
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".cs",
            ".java",
            ".glsl",
            ".hlsl",
            ".wgsl",
        ):
            # Brace-balancing scanning
            open_braces = 0
            has_opened = False
            end_idx = len(lines)
            for i in range(start_idx, len(lines)):
                line = lines[i]
                for char in line:
                    if char == "{":
                        open_braces += 1
                        has_opened = True
                    elif char == "}":
                        open_braces -= 1
                if has_opened and open_braces <= 0:
                    end_idx = i + 1
                    break
        elif ext == ".sql":
            # SQL: scan until semicolon or next CREATE statement
            end_idx = len(lines)
            for i in range(start_idx + 1, len(lines)):
                line = lines[i]
                if ";" in line:
                    end_idx = i + 1
                    break
                if re.match(r"^\s*CREATE\b", line, re.IGNORECASE):
                    end_idx = i
                    break
        else:
            # Default: 50 lines
            end_idx = min(len(lines), start_idx + 50)

        return "".join(lines[start_idx:end_idx])

    def _extract_sliced_block_from_overlaid(
        self, file_path: str, symbol_name: str, hint_line_1: int, inc_conn: bool
    ) -> str:
        """
        Loads overlaid content, finds the symbol definition, and extracts the block.
        """
        content = self.get_overlaid_content(file_path, include_connection_maps=inc_conn)
        lines = content.splitlines(keepends=True)
        if not lines:
            return ""

        ext = os.path.splitext(file_path)[1].lower()
        escaped = re.escape(symbol_name)

        patterns = []
        if ext == ".py":
            patterns = [re.compile(rf"\b(def|class)\s+{escaped}\b")]
        elif ext in (".js", ".ts", ".tsx", ".jsx"):
            patterns = [
                re.compile(rf"\b(function|class|const|let|var)\s+{escaped}\b"),
                re.compile(
                    rf"\b{escaped}\s*[:=]\s*(async\s*)?(?:\([^)]*\)|[a-zA-Z0-9_$]+)\s*=>"
                ),
                re.compile(
                    rf"\b(?:async\s+|static\s+|public\s+|private\s+|protected\s+)*{escaped}\s*\("
                ),
            ]
        elif ext == ".cs":
            patterns = [
                re.compile(rf"\b(class|struct|interface|enum|void|Task)\s+{escaped}\b")
            ]
        elif ext == ".sql":
            patterns = [
                re.compile(
                    rf"\b(CREATE\s+(TABLE|VIEW|PROCEDURE|FUNCTION))\s+{escaped}\b",
                    re.IGNORECASE,
                )
            ]
        else:
            patterns = [re.compile(rf"\b{escaped}\b")]

        # Search window first
        hint_0 = max(0, min(len(lines) - 1, hint_line_1 - 1))
        window_start = max(0, hint_0 - 50)
        window_end = min(len(lines), hint_0 + 150)

        def_idx = -1
        for i in range(window_start, window_end):
            if any(pat.search(lines[i]) for pat in patterns):
                def_idx = i
                break

        if def_idx == -1:
            # Full scan
            for i, line in enumerate(lines):
                if any(pat.search(line) for pat in patterns):
                    def_idx = i
                    break

        if def_idx == -1:
            # Fallback
            def_idx = hint_0

        start_idx = def_idx
        # Scan upwards for decorators
        if ext in (".py", ".js", ".ts", ".tsx", ".jsx", ".cs", ".java"):
            while start_idx > 0:
                prev_line = lines[start_idx - 1].strip()
                if prev_line.startswith("@"):
                    start_idx -= 1
                else:
                    break

        end_idx = start_idx + 1

        if ext == ".py":
            # Find body start after header closes (ends with :)
            body_start_idx = def_idx
            for i in range(def_idx, len(lines)):
                stripped = lines[i].strip()
                if ":" in stripped:
                    clean = stripped.split("#")[0].strip()
                    if clean.endswith(":"):
                        body_start_idx = i + 1
                        break

            def_line = lines[def_idx]
            indent_len = len(def_line) - len(def_line.lstrip())

            end_idx = len(lines)
            for i in range(body_start_idx, len(lines)):
                line = lines[i]
                if not line.strip():
                    continue
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= indent_len:
                    if not line.strip().startswith("#"):
                        end_idx = i
                        break
        elif ext in (
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".cs",
            ".java",
            ".glsl",
            ".hlsl",
            ".wgsl",
        ):
            open_braces = 0
            has_opened = False
            end_idx = len(lines)
            for i in range(start_idx, len(lines)):
                line = lines[i]
                for char in line:
                    if char == "{":
                        open_braces += 1
                        has_opened = True
                    elif char == "}":
                        open_braces -= 1
                if has_opened and open_braces <= 0:
                    end_idx = i + 1
                    break
        elif ext == ".sql":
            end_idx = len(lines)
            for i in range(start_idx + 1, len(lines)):
                line = lines[i]
                if ";" in line:
                    end_idx = i + 1
                    break
                if re.match(r"^\s*CREATE\b", line, re.IGNORECASE):
                    end_idx = i
                    break
        else:
            end_idx = min(len(lines), start_idx + 50)

        return "".join(lines[start_idx:end_idx])

    def build_package(
        self,
        target_keys: List[str],
        max_tokens: Optional[int] = None,
        routing_mode: str = "auto",
        include_connection_maps: Optional[bool] = None,
    ) -> str:
        """
        Builds a token-budgeted targeted context package centering around target_keys.
        """
        # Resolve config limits
        local_max = (
            self.config_mgr.get_context_packager_setting("local_max_tokens", 20000)
            or 20000
        )
        cloud_max = (
            self.config_mgr.get_context_packager_setting("cloud_max_tokens", 100000)
            or 100000
        )
        default_mode = self.config_mgr.get_context_packager_setting(
            "default_mode", "auto"
        )
        config_inc_conn = self.config_mgr.get_context_packager_setting(
            "include_connection_maps", False
        )

        mode = routing_mode if routing_mode != "auto" else default_mode
        inc_conn = (
            include_connection_maps
            if include_connection_maps is not None
            else config_inc_conn
        )

        if max_tokens is None:
            max_tokens_val = int(local_max) if mode == "local" else int(cloud_max)
        else:
            if mode == "local" and max_tokens > local_max:
                logger.warning(
                    f"Local routing requested but max_tokens > {local_max}. Clamping."
                )
                max_tokens_val = int(local_max)
            else:
                max_tokens_val = int(max_tokens)

        global_map = load_global_key_map()
        if not global_map:
            raise ValueError(
                "Global key map not found. Please run analyze-project first."
            )

        key_to_path = {ki.key_string: p for p, ki in global_map.items()}

        def is_directory_key(key: str) -> bool:
            base_key = key.split("#")[0]
            if not base_key:
                return True
            if not base_key[-1].isdigit():
                return True
            path = key_to_path.get(key)
            if path and path in global_map and global_map[path].is_directory:
                return True
            return False

        # 1. Resolve Target Files
        target_paths: List[str] = []
        target_kis: List[KeyInfo] = []
        for key in target_keys:
            if is_directory_key(key):
                logger.warning(
                    f"Omitting target key '{key}' because it is a directory key."
                )
                continue
            path = key_to_path.get(key)
            if not path:
                base_key = key.split("#")[0]
                matching = [
                    ki
                    for ki in global_map.values()
                    if ki.key_string.split("#")[0] == base_key
                ]
                if matching:
                    matching.sort(key=lambda k: k.norm_path)
                    ki = matching[0]
                    path = ki.norm_path
                    logger.warning(
                        f"Ambiguous target key '{key}' resolved to '{ki.key_string}'"
                    )

            if path:
                ki = global_map[path]
                if ki.is_directory or is_directory_key(ki.key_string):
                    logger.warning(
                        f"Omitting target path '{path}' because it is a directory key."
                    )
                    continue
                target_paths.append(path)
                target_kis.append(ki)
            else:
                logger.error(f"Target key '{key}' could not be resolved.")

        if not target_paths:
            return "# Error: No valid target keys resolved."

        # Scan transparency manager for specific references to/from target files
        tm = get_transparency_manager()
        referenced_slices: Dict[str, Set[Tuple[str, int]]] = {}
        target_keys_set = {ki.key_string for ki in target_kis}
        norm_target_paths = {normalize_path(p) for p in target_paths}

        # Outbound references from target files
        for t_path in target_paths:
            meta = tm.get_file_metadata(t_path)
            if meta and "connection_maps" in meta:
                for m in meta["connection_maps"]:
                    tgt_key = m.get("target_key")
                    tgt_sym = m.get("target_symbol")
                    tgt_line = m.get("target_line")
                    if tgt_key and tgt_sym and tgt_line is not None:
                        if tgt_sym != "file":
                            tgt_path = key_to_path.get(tgt_key)
                            if (
                                tgt_path
                                and normalize_path(tgt_path) not in norm_target_paths
                            ):
                                normalized_dep_path = normalize_path(tgt_path)
                                referenced_slices.setdefault(
                                    normalized_dep_path, set()
                                ).add((tgt_sym, int(tgt_line)))

        # Inbound references to target files
        registry_files = tm._registry.get("files", {})
        for file_path, file_meta in registry_files.items():
            norm_file_path = normalize_path(file_path)
            if norm_file_path in norm_target_paths:
                continue
            if isinstance(file_meta, dict) and "connection_maps" in file_meta:
                for m in file_meta["connection_maps"]:
                    tgt_key = m.get("target_key")
                    if tgt_key in target_keys_set:
                        src_sym = m.get("source_symbol")
                        src_line = m.get("source_line")
                        if src_sym and src_line is not None:
                            if src_sym != "file":
                                referenced_slices.setdefault(norm_file_path, set()).add(
                                    (src_sym, int(src_line))
                                )

        current_tokens = 0
        package_components: List[str] = []

        package_components.append("# TARGET CORE LOGIC")
        package_components.append(
            "This section contains full file logic for the primary targets.\n"
        )

        # Pack target files completely first
        for path, ki in zip(target_paths, target_kis):
            full_t, _ = self.get_file_tokens(path)
            content = self.get_overlaid_content(path, include_connection_maps=inc_conn)
            token_count = full_t if full_t > 0 else len(content) // 4
            current_tokens += token_count

            rel_path = os.path.relpath(path, self.project_root)
            package_components.append(f"## [{ki.key_string}] {rel_path} (Full Content)")
            package_components.append(f"```python\n{content}\n```\n")

        # Retrieve Compiled Dependency Graph
        all_tp = find_all_tracker_paths(self.config_mgr, self.project_root)
        old_global_map = load_old_global_key_map()
        path_migration_info = build_path_migration_map(old_global_map, global_map)

        agg_deps = aggregate_all_dependencies(
            tracker_paths=all_tp,
            path_migration_info=path_migration_info,
            current_global_path_to_key_info=global_map,
            show_progress=False,
        )

        tiers: Dict[int, Set[str]] = {1: set(), 2: set(), 3: set(), 4: set(), 5: set()}
        target_key_strings = {ki.key_string for ki in target_kis}
        for (src_gi, tgt_gi), (char, _) in agg_deps.items():
            if src_gi in target_key_strings and tgt_gi not in target_key_strings:
                if is_directory_key(tgt_gi):
                    continue
                if char in self.TIER_MAP:
                    tier_level = self.TIER_MAP[char]
                    tiers[tier_level].add(tgt_gi)

        # Pack sliced dependency files
        for tier in sorted(tiers.keys()):
            if not tiers[tier]:
                continue

            # Sort dependency keys by full token count (smallest first) to maximize representation
            sorted_deps = sorted(
                list(tiers[tier]),
                key=lambda k: (
                    self.get_file_tokens(key_to_path.get(k, ""))[0]
                    if key_to_path.get(k)
                    else 99999
                ),
            )
            filtered_deps = [
                k
                for k in sorted_deps
                if key_to_path.get(k)
                and normalize_path(key_to_path[k]) in referenced_slices
            ]

            if not filtered_deps:
                continue

            tier_desc = {
                1: "Tier 1: Mutual dependencies (x)",
                2: "Tier 2: Inbound/Outbound imports (<, >)",
                3: "Tier 3: Documentation relationships (d)",
                4: "Tier 4: Strong semantic boundaries (S)",
                5: "Tier 5: Weak semantic relationships (s)",
            }[tier]

            package_components.append(f"# {tier_desc}")
            package_components.append(
                "Includes only referenced symbols and definitions to stay within target token ceilings.\n"
            )

            for dep_key in filtered_deps:
                path = key_to_path[dep_key]
                norm_dep_path = normalize_path(path)
                slices = sorted(
                    list(referenced_slices[norm_dep_path]), key=lambda x: x[1]
                )

                content_parts = []
                for sym, line_no in slices:
                    block = self._extract_sliced_block_from_overlaid(
                        path, sym, line_no, inc_conn
                    )
                    if block:
                        content_parts.append(
                            f"# --- Symbol: {sym} (Line {line_no}) ---\n{block}"
                        )

                if not content_parts:
                    continue

                content = "\n\n".join(content_parts)
                # Roughly estimate sliced tokens (e.g. chars // 4)
                token_count = len(content) // 4
                rel_path = os.path.relpath(path, self.project_root)

                if current_tokens + token_count <= max_tokens_val:
                    current_tokens += token_count
                    package_components.append(
                        f"## [{dep_key}] {rel_path} (Sliced Symbols: {', '.join(s[0] for s in slices)})"
                    )
                    package_components.append(f"```python\n{content}\n```\n")
                else:
                    package_components.append(
                        f"> Context ceiling of {max_tokens_val} tokens reached. Truncating further outer ring dependencies."
                    )
                    logger.info(
                        f"Context package budget capped at ~{current_tokens} tokens."
                    )
                    return "\n".join(package_components)

        logger.info(
            f"Context package assembled successfully at ~{current_tokens} tokens."
        )
        return "\n".join(package_components)
