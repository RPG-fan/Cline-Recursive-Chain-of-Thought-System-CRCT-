# cline_utils/dependency_system/utils/viz/mermaid_builder.py

"""
Logic for constructing the Mermaid DSL string for visualizing project dependencies.
"""

import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from cline_utils.dependency_system.utils.tracker_utils import (
    aggregate_all_dependencies,
    get_key_global_instance_string,
    resolve_key_global_instance_to_ki,
)

from cline_utils.dependency_system.core.key_manager import (
    KeyInfo,
    sort_key_strings_hierarchically,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import (
    get_project_root,
    normalize_path,
    PathMigrationInfo,
)
from .layout_config import (
    CLASS_DEFS,
    DEP_CHAR_TO_STYLE,
    LINK_STYLE,
    SUBGRAPH_FILL,
    SUBGRAPH_FOCUS_STROKE,
    SUBGRAPH_STROKE,
    SUBGRAPH_TEXT_COLOR,
)

logger = logging.getLogger(__name__)


def _is_direct_parent_child_key_relationship(
    key1_gi_str: str,
    key2_gi_str: str,
    global_path_to_key_info_map: Dict[str, KeyInfo],
) -> bool:
    """Checks if one key's path is the direct parent_path of the other."""
    key1_info = resolve_key_global_instance_to_ki(
        key1_gi_str, global_path_to_key_info_map
    )
    key2_info = resolve_key_global_instance_to_ki(
        key2_gi_str, global_path_to_key_info_map
    )

    if not key1_info or not key2_info:
        return False
    if key2_info.parent_path and key1_info.norm_path == key2_info.parent_path:
        return True
    if key1_info.parent_path and key2_info.norm_path == key1_info.parent_path:
        return True
    return False


def _get_safe_mermaid_id(
    key_gi: str, safe_id_map: Dict[str, str], safe_id_counts: Dict[str, int]
) -> str:
    """Generates a Mermaid-safe identifier for a key."""
    if key_gi in safe_id_map:
        return safe_id_map[key_gi]
    base = re.sub(r"[^a-zA-Z0-9_]", "_", key_gi)
    if base not in safe_id_counts:
        safe_id_counts[base] = 0
        safe_id_map[key_gi] = base
        return base
    safe_id_counts[base] += 1
    safe_id = f"{base}_{safe_id_counts[base]}"
    safe_id_map[key_gi] = safe_id
    return safe_id


def _dir_has_renderable_descendant(
    dir_norm_path: str,
    renderable_dirs: Set[str],
    parent_norm_path_to_child_key_infos: Dict[Optional[str], List[KeyInfo]],
    ki_to_gi: Dict[str, str],
    nodes_to_render_gi: Set[str],
    global_path_to_key_info_map: Dict[str, KeyInfo],
    all_kis_for_hierarchy: Dict[str, KeyInfo],
) -> bool:
    """Check if a directory contains any files that are marked for rendering."""
    gi_dir = ki_to_gi.get(dir_norm_path)
    if not gi_dir:
        return False
    if gi_dir in renderable_dirs:
        return True
    children = parent_norm_path_to_child_key_infos.get(dir_norm_path, [])
    for child in children:
        child_gi = get_key_global_instance_string(child, global_path_to_key_info_map)
        if not child_gi or child_gi not in all_kis_for_hierarchy:
            continue
        if not child.is_directory and child_gi in nodes_to_render_gi:
            renderable_dirs.add(gi_dir)
            return True
        if child.is_directory and _dir_has_renderable_descendant(
            child.norm_path,
            renderable_dirs,
            parent_norm_path_to_child_key_infos,
            ki_to_gi,
            nodes_to_render_gi,
            global_path_to_key_info_map,
            all_kis_for_hierarchy,
        ):
            renderable_dirs.add(gi_dir)
            return True
    return False


def build_mermaid_string(
    focus_keys_list_input: List[str],
    global_path_to_key_info_map: Dict[str, KeyInfo],
    path_migration_info: PathMigrationInfo,
    all_tracker_paths_list: List[str],
    config_manager_instance: ConfigManager,
    pre_aggregated_links: Optional[Dict[Tuple[str, str], Tuple[str, set[str]]]] = None,
) -> str:
    """Constructs the Mermaid DSL string."""
    # Resolve focus keys
    resolved_focus_keys_gi: List[str] = []
    if focus_keys_list_input:
        from cline_utils.dependency_system.utils.tracker_utils import (
            get_globally_resolved_key_info_for_cli,
        )

        for key_input_str in focus_keys_list_input:
            parts = key_input_str.split("#")
            base_key = parts[0]
            instance_num = (
                int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            )
            resolved_ki = get_globally_resolved_key_info_for_cli(
                base_key, instance_num, global_path_to_key_info_map, "focus"
            )
            if not resolved_ki:
                # get_globally_resolved_key_info_for_cli already logs/prints the error
                return (
                    f"Error: Focus key '{key_input_str}' not found in global key map."
                )

            gi_str = get_key_global_instance_string(
                resolved_ki, global_path_to_key_info_map
            )
            if gi_str:
                resolved_focus_keys_gi.append(gi_str)
            else:
                return f"Error: Could not resolve global instance for focus key '{key_input_str}'."

    # Aggregate links
    aggregated_links_with_origins: Dict[Tuple[str, str], Tuple[str, Set[str]]]
    if pre_aggregated_links is not None:
        aggregated_links_with_origins = pre_aggregated_links
    else:
        aggregated_links_with_origins = aggregate_all_dependencies(
            set(all_tracker_paths_list),
            path_migration_info,
            global_path_to_key_info_map,
        )

    consolidated_directed_links_gi: Dict[Tuple[str, str], str] = {
        link_gi_tuple: char_and_origins[0]
        for link_gi_tuple, char_and_origins in aggregated_links_with_origins.items()
    }

    # Scope determination
    focus_keys_valid_gi: Set[str] = set(resolved_focus_keys_gi)
    keys_in_module_scope_gi: Set[str] = set()
    is_module_view = False
    module_path_prefix_for_scope = ""

    if len(resolved_focus_keys_gi) == 1:
        focus_key_gi_str = resolved_focus_keys_gi[0]
        focus_info = resolve_key_global_instance_to_ki(
            focus_key_gi_str, global_path_to_key_info_map
        )
        if focus_info and focus_info.is_directory:
            is_module_view = True
            module_path_prefix_for_scope = focus_info.norm_path + (
                "/" if not focus_info.norm_path.endswith("/") else ""
            )
            for ki_val in global_path_to_key_info_map.values():
                if (
                    ki_val.norm_path == focus_info.norm_path
                    or (
                        ki_val.parent_path
                        and normalize_path(ki_val.parent_path) == focus_info.norm_path
                    )
                    or ki_val.norm_path.startswith(module_path_prefix_for_scope)
                ):
                    gi_str_module_item = get_key_global_instance_string(
                        ki_val, global_path_to_key_info_map
                    )
                    if gi_str_module_item:
                        keys_in_module_scope_gi.add(gi_str_module_item)

    # Edge Preparation
    intermediate_edges_gi: List[Tuple[str, str, str]] = []
    processed_pairs_for_intermediate_gi: Set[Tuple[str, ...]] = set()
    non_n_links_gi: Dict[Tuple[str, str], str] = {
        (s_gi, t_gi): char
        for (s_gi, t_gi), char in consolidated_directed_links_gi.items()
        if char != "n"
    }

    for (source_gi, target_gi), forward_char in sorted(non_n_links_gi.items()):
        pair_tuple_gi = tuple(sorted((source_gi, target_gi)))
        if pair_tuple_gi in processed_pairs_for_intermediate_gi:
            continue
        reverse_char = non_n_links_gi.get((target_gi, source_gi))
        if forward_char == "x" or reverse_char == "x":
            intermediate_edges_gi.append((source_gi, target_gi, "x"))
        elif forward_char == "<" and reverse_char == ">":
            intermediate_edges_gi.append((source_gi, target_gi, "<"))
        elif forward_char == ">" and reverse_char == "<":
            intermediate_edges_gi.append((target_gi, source_gi, "<"))
        elif forward_char == ">":
            intermediate_edges_gi.append((target_gi, source_gi, "<"))
        elif forward_char == "<":
            intermediate_edges_gi.append((source_gi, target_gi, "<"))
        elif forward_char:
            intermediate_edges_gi.append((source_gi, target_gi, forward_char))
        processed_pairs_for_intermediate_gi.add(pair_tuple_gi)

    # Edge Filtering
    edges_within_scope_gi: List[Tuple[str, str, str]] = []
    relevant_keys_for_nodes_gi: Set[str] = set()
    if is_module_view:
        relevant_keys_for_nodes_gi.update(keys_in_module_scope_gi)
        for k1_gi, k2_gi, char_val in intermediate_edges_gi:
            if (k1_gi in keys_in_module_scope_gi) or (k2_gi in keys_in_module_scope_gi):
                edges_within_scope_gi.append((k1_gi, k2_gi, char_val))
                relevant_keys_for_nodes_gi.add(k1_gi)
                relevant_keys_for_nodes_gi.add(k2_gi)
    elif focus_keys_valid_gi:
        relevant_keys_for_nodes_gi.update(focus_keys_valid_gi)
        for k1_gi, k2_gi, char_val in intermediate_edges_gi:
            if k1_gi in focus_keys_valid_gi or k2_gi in focus_keys_valid_gi:
                edges_within_scope_gi.append((k1_gi, k2_gi, char_val))
                relevant_keys_for_nodes_gi.add(k1_gi)
                relevant_keys_for_nodes_gi.add(k2_gi)
    else:
        edges_within_scope_gi = intermediate_edges_gi
        relevant_keys_for_nodes_gi = {
            k_gi
            for edge_tuple_gi in edges_within_scope_gi
            for k_gi in edge_tuple_gi[:2]
        }

    final_edges_to_draw_gi: List[Tuple[str, str, str]] = []
    for k1_gi, k2_gi, char_val in edges_within_scope_gi:
        if char_val in ("p", "s"):
            continue
        info1 = resolve_key_global_instance_to_ki(k1_gi, global_path_to_key_info_map)
        info2 = resolve_key_global_instance_to_ki(k2_gi, global_path_to_key_info_map)
        if not info1 or not info2:
            continue
        if char_val == "x" and _is_direct_parent_child_key_relationship(
            k1_gi, k2_gi, global_path_to_key_info_map
        ):
            continue
        if char_val != "d" and info1.is_directory != info2.is_directory:
            continue
        final_edges_to_draw_gi.append((k1_gi, k2_gi, char_val))

    nodes_to_render_gi: Set[str] = {
        k_gi for edge_tuple_gi in final_edges_to_draw_gi for k_gi in edge_tuple_gi[:2]
    }
    if focus_keys_valid_gi:
        nodes_to_render_gi.update(focus_keys_valid_gi)

    if not nodes_to_render_gi:
        return "flowchart TB\n\n// No relevant data to visualize."

    # Hierarchy and Subgraphs
    parent_norm_path_to_child_key_infos: Dict[Optional[str], List[KeyInfo]] = (
        defaultdict(list)
    )
    all_kis_for_hierarchy: Dict[str, KeyInfo] = {}
    queue: List[str] = list(nodes_to_render_gi)
    visited: Set[str] = set()

    while queue:
        k_gi = queue.pop(0)
        if k_gi in visited:
            continue
        visited.add(k_gi)
        ki = resolve_key_global_instance_to_ki(k_gi, global_path_to_key_info_map)
        if not ki:
            continue
        all_kis_for_hierarchy[k_gi] = ki
        if ki.parent_path:
            parent_ki = global_path_to_key_info_map.get(ki.parent_path)
            if parent_ki:
                p_gi = get_key_global_instance_string(
                    parent_ki, global_path_to_key_info_map
                )
                if p_gi and p_gi not in visited:
                    queue.append(p_gi)

    for ki_h in all_kis_for_hierarchy.values():
        parent_norm_path_to_child_key_infos[ki_h.parent_path].append(ki_h)

    ki_to_gi = {ki_v.norm_path: gi_k for gi_k, ki_v in all_kis_for_hierarchy.items()}
    renderable_dirs: Set[str] = set()

    # Node styling
    project_root = get_project_root()
    try:
        from cline_utils.dependency_system.utils.template_generator import get_item_type
    except ImportError:

        def get_item_type(
            item_path: str, config: ConfigManager, project_root: str
        ) -> Optional[str]:
            return "doc" if str(item_path).endswith(".md") else "file"

    def get_node_class(ki_obj: KeyInfo) -> str:
        if ki_obj.is_directory:
            return "module"
        it = get_item_type(ki_obj.norm_path, config_manager_instance, project_root)
        return "doc" if it == "doc" else "file"

    global_key_counts: Dict[str, int] = defaultdict(int)
    for ki_c in global_path_to_key_info_map.values():
        global_key_counts[ki_c.key_string] += 1

    mermaid_parts = ["flowchart TB"]
    mermaid_parts.extend(["  " + cd for cd in CLASS_DEFS])
    mermaid_parts.append(f"  {LINK_STYLE}")

    mermaid_rendered_node_ids: Set[str] = set()
    dir_gi_to_sg_id: Dict[str, str] = {}
    sg_counter = 0
    safe_id_map: Dict[str, str] = {}
    safe_id_counts: Dict[str, int] = {}

    def generate_structure_recursive(parent_path: Optional[str], indent: str):
        nonlocal sg_counter
        children: List[KeyInfo] = sorted(
            parent_norm_path_to_child_key_infos.get(parent_path, []),
            key=lambda k: (
                sort_key_strings_hierarchically([k.key_string])[0],
                str(k.norm_path),
            ),
        )
        for child_ki in children:
            c_gi = get_key_global_instance_string(child_ki, global_path_to_key_info_map)
            if not c_gi or c_gi not in all_kis_for_hierarchy:
                continue

            m_id = _get_safe_mermaid_id(c_gi, safe_id_map, safe_id_counts)
            node_label = (
                child_ki.key_string
                if global_key_counts[child_ki.key_string] <= 1
                else c_gi
            )

            if child_ki.is_directory:
                if not _dir_has_renderable_descendant(
                    child_ki.norm_path,
                    renderable_dirs,
                    parent_norm_path_to_child_key_infos,
                    ki_to_gi,
                    nodes_to_render_gi,
                    global_path_to_key_info_map,
                    all_kis_for_hierarchy,
                ):
                    continue
                sg_counter += 1
                sg_id = f"sg_{re.sub(r'[^a-zA-Z0-9_]', '_', child_ki.key_string)}_{sg_counter}"
                dir_gi_to_sg_id[c_gi] = sg_id
                title = f"<font color='{SUBGRAPH_TEXT_COLOR}'>{child_ki.key_string}<br>{os.path.basename(child_ki.norm_path)}"
                mermaid_parts.append(f'{indent}subgraph {sg_id} ["{title}"]')
                mermaid_rendered_node_ids.add(c_gi)
                mermaid_parts.append(
                    f"{indent}  style {sg_id} fill:{SUBGRAPH_FILL},stroke:{SUBGRAPH_STROKE},stroke-width:4px"
                )
                if c_gi in resolved_focus_keys_gi:
                    mermaid_parts.append(
                        f"{indent}  style {sg_id} stroke-width:5px,stroke:{SUBGRAPH_FOCUS_STROKE}"
                    )
                generate_structure_recursive(child_ki.norm_path, indent + "  ")
                mermaid_parts.append(f"{indent}end")
            elif c_gi in nodes_to_render_gi and c_gi not in mermaid_rendered_node_ids:
                label_text = f"{node_label}<br>{os.path.basename(child_ki.norm_path)}"
                mermaid_parts.append(f'{indent}{m_id}["{label_text}"]')
                mermaid_parts.append(f"{indent}class {m_id} {get_node_class(child_ki)}")
                if c_gi in focus_keys_valid_gi:
                    mermaid_parts.append(f"{indent}class {m_id} focusNode")
                mermaid_rendered_node_ids.add(c_gi)

    generate_structure_recursive(None, "  ")

    # Edges
    mermaid_parts.append("\n  %% -- Dependencies --")
    for k1_gi, k2_gi, dep_char in sorted(final_edges_to_draw_gi):
        if (
            k1_gi not in mermaid_rendered_node_ids
            or k2_gi not in mermaid_rendered_node_ids
        ):
            continue
        id1 = dir_gi_to_sg_id.get(
            k1_gi, _get_safe_mermaid_id(k1_gi, safe_id_map, safe_id_counts)
        )
        id2 = dir_gi_to_sg_id.get(
            k2_gi, _get_safe_mermaid_id(k2_gi, safe_id_map, safe_id_counts)
        )
        arrow, label = DEP_CHAR_TO_STYLE.get(dep_char, ("-->", dep_char))
        if label:
            mermaid_parts.append(f'  {id1} {arrow}|"{label}"| {id2}')
        else:
            mermaid_parts.append(f"  {id1} {arrow} {id2}")

    return "\n".join(mermaid_parts)
