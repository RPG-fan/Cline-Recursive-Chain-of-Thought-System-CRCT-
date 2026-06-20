# context_packager.py

"""
Constructs token-optimized context packages based on dependency tiers and SES fallbacks.
Decouples token window management and dynamic LLM routing.
"""

import json
import logging
import os
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

    def get_overlaid_content(self, file_path: str) -> str:
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
                        anchors_list: List[Any] = anchors if isinstance(anchors, list) else []
                        if len(anchors_list) == 2:
                            start_anchor = str(anchors_list[0])
                            end_anchor = str(anchors_list[1])

                            # Perform verification check on start_idx and end_idx
                            start_matched = (0 <= start_idx < len(lines) and lines[start_idx].strip() == start_anchor.strip())
                            end_matched = (0 <= (end_idx - 1) < len(lines) and lines[end_idx - 1].strip() == end_anchor.strip())

                            if not start_matched or not end_matched:
                                new_start = tm.find_anchor(lines, start_anchor, start_idx)
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

    def get_ses_content(self, file_path: str) -> str:
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
        return self.get_overlaid_content(norm_path)

    def build_package(
        self,
        target_keys: List[str],
        max_tokens: Optional[int] = None,
        routing_mode: str = "auto",
    ) -> str:
        """
        Builds a token-budgeted targeted context package centering around target_keys.
        """
        # Resolve config limits
        local_max = self.config_mgr.get_context_packager_setting(
            "local_max_tokens", 20000
        ) or 20000
        cloud_max = self.config_mgr.get_context_packager_setting(
            "cloud_max_tokens", 100000
        ) or 100000
        default_mode = self.config_mgr.get_context_packager_setting(
            "default_mode", "auto"
        )

        mode = routing_mode if routing_mode != "auto" else default_mode

        if max_tokens is None:
            max_tokens_val = int(local_max) if mode == "local" else int(cloud_max)
        else:
            # Clamp limits if mode requests it
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

        # Invert global map to map key strings (including GI suffixes) to paths
        key_to_path = {ki.key_string: p for p, ki in global_map.items()}

        def is_directory_key(key: str) -> bool:
            base_key = key.split("#")[0]
            if not base_key:
                return True
            # Directory keys lack a trailing number (end with a letter)
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
                logger.warning(f"Omitting target key '{key}' because it is a directory key.")
                continue
            path = key_to_path.get(key)
            if not path:
                # Try finding without instance suffix
                base_key = key.split("#")[0]
                matching = [
                    ki
                    for ki in global_map.values()
                    if ki.key_string.split("#")[0] == base_key
                ]
                if matching:
                    # Sort and pick first as fallback
                    matching.sort(key=lambda k: k.norm_path)
                    ki = matching[0]
                    path = ki.norm_path
                    logger.warning(
                        f"Ambiguous target key '{key}' resolved to '{ki.key_string}'"
                    )

            if path:
                ki = global_map[path]
                if ki.is_directory or is_directory_key(ki.key_string):
                    logger.warning(f"Omitting target path '{path}' because it is a directory key.")
                    continue
                target_paths.append(path)
                target_kis.append(ki)
            else:
                logger.error(f"Target key '{key}' could not be resolved.")

        if not target_paths:
            return "# Error: No valid target keys resolved."

        # Estimate budget allocations
        current_tokens = 0
        package_components: List[str] = []

        package_components.append("# TARGET CORE LOGIC")
        package_components.append(
            "This section contains full file logic for the primary targets.\n"
        )

        # 2. Pack target files completely first
        for path, ki in zip(target_paths, target_kis):
            full_t, _ = self.get_file_tokens(path)
            content = self.get_overlaid_content(path)
            # Rough estimation of token count from character count if not found in metadata
            token_count = full_t if full_t > 0 else len(content) // 4
            current_tokens += token_count

            rel_path = os.path.relpath(path, self.project_root)
            package_components.append(f"## [{ki.key_string}] {rel_path} (Full Content)")
            package_components.append(f"```python\n{content}\n```\n")

        # 3. Retrieve Compiled Dependency Graph
        all_tp = find_all_tracker_paths(self.config_mgr, self.project_root)
        old_global_map = load_old_global_key_map()
        path_migration_info = build_path_migration_map(old_global_map, global_map)

        agg_deps = aggregate_all_dependencies(
            tracker_paths=all_tp,
            path_migration_info=path_migration_info,
            current_global_path_to_key_info=global_map,
            show_progress=False,
        )

        # Categorize dependencies into Tiers
        # Tiers 1-5 maps: Tier -> Set of target keys
        tiers: Dict[int, Set[str]] = {1: set(), 2: set(), 3: set(), 4: set(), 5: set()}

        target_key_strings = {ki.key_string for ki in target_kis}
        for (src_gi, tgt_gi), (char, _) in agg_deps.items():
            if src_gi in target_key_strings and tgt_gi not in target_key_strings:
                if is_directory_key(tgt_gi):
                    continue
                if char in self.TIER_MAP:
                    tier_level = self.TIER_MAP[char]
                    tiers[tier_level].add(tgt_gi)

        # 4. Greedy Knapsack Allocation with SES Fallback
        for tier in sorted(tiers.keys()):
            if not tiers[tier]:
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
                "Pruned dependencies from this tier to stay within target token ceilings.\n"
            )

            # Sort dependency keys by full token count (smallest first) to maximize representation
            sorted_deps = sorted(
                list(tiers[tier]),
                key=lambda k: (
                    self.get_file_tokens(key_to_path.get(k, ""))[0]
                    if key_to_path.get(k)
                    else 99999
                ),
            )

            for dep_key in sorted_deps:
                if is_directory_key(dep_key):
                    continue
                path = key_to_path.get(dep_key)
                if not path:
                    continue
                if global_map[path].is_directory:
                    continue

                full_tokens, ses_tokens = self.get_file_tokens(path)
                rel_path = os.path.relpath(path, self.project_root)

                # Try Full Content First
                if current_tokens + full_tokens <= max_tokens_val:
                    current_tokens += full_tokens
                    content = self.get_overlaid_content(path)
                    package_components.append(
                        f"## [{dep_key}] {rel_path} (Full Content)"
                    )
                    package_components.append(f"```python\n{content}\n```\n")

                # Fallback to SES Signature/Execution Content
                elif current_tokens + ses_tokens <= max_tokens_val:
                    current_tokens += ses_tokens
                    content = self.get_ses_content(path)
                    package_components.append(
                        f"## [{dep_key}] {rel_path} (SES Signatures Only)"
                    )
                    package_components.append(f"```python\n{content}\n```\n")

                # Context threshold crossed; drop remaining outer ring elements
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
