"""Build renderer-neutral dependency diagram inputs."""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from cline_utils.dependency_system.core.key_manager import (
    KeyInfo,
    sort_key_strings_hierarchically,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import (
    PathMigrationInfo,
    get_project_root,
    normalize_path,
)
from cline_utils.dependency_system.utils.tracker_utils import (
    aggregate_all_dependencies,
    get_key_global_instance_string,
    resolve_key_global_instance_to_ki,
)

from .models import DiagramEdge, DiagramInput, DiagramNode

logger = logging.getLogger(__name__)


def _is_direct_parent_child_key_relationship(
    key1_gi_str: str,
    key2_gi_str: str,
    global_path_to_key_info_map: Dict[str, KeyInfo],
) -> bool:
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


def _resolve_focus_keys(
    focus_keys_list_input: List[str],
    global_path_to_key_info_map: Dict[str, KeyInfo],
) -> Tuple[List[str], Optional[str]]:
    resolved_focus_keys_gi: List[str] = []
    if not focus_keys_list_input:
        return resolved_focus_keys_gi, None

    from cline_utils.dependency_system.utils.tracker_utils import (
        get_globally_resolved_key_info_for_cli,
    )

    for key_input_str in focus_keys_list_input:
        parts = key_input_str.split("#")
        base_key = parts[0]
        instance_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        resolved_ki = get_globally_resolved_key_info_for_cli(
            base_key, instance_num, global_path_to_key_info_map, "focus"
        )
        if not resolved_ki:
            return [], f"Error: Focus key '{key_input_str}' not found in global key map."

        gi_str = get_key_global_instance_string(
            resolved_ki, global_path_to_key_info_map
        )
        if not gi_str:
            return (
                [],
                f"Error: Could not resolve global instance for focus key '{key_input_str}'.",
            )
        resolved_focus_keys_gi.append(gi_str)

    return resolved_focus_keys_gi, None


def _get_node_class(
    ki_obj: KeyInfo, config_manager_instance: ConfigManager, project_root: str
) -> str:
    if ki_obj.is_directory:
        return "module"
    try:
        from cline_utils.dependency_system.utils.template_generator import get_item_type
    except ImportError:
        return "doc" if str(ki_obj.norm_path).endswith(".md") else "file"

    item_type = get_item_type(ki_obj.norm_path, config_manager_instance, project_root)
    return "doc" if item_type == "doc" else "file"


def _sort_child_gis(gis: List[str], nodes: Dict[str, DiagramNode]) -> List[str]:
    return sorted(
        gis,
        key=lambda gi: (
            sort_key_strings_hierarchically([nodes[gi].key_string])[0],
            nodes[gi].norm_path,
        ),
    )


def build_diagram_model(
    focus_keys_list_input: List[str],
    global_path_to_key_info_map: Dict[str, KeyInfo],
    path_migration_info: PathMigrationInfo,
    all_tracker_paths_list: List[str],
    config_manager_instance: ConfigManager,
    pre_aggregated_links: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
) -> DiagramInput:
    """Builds a renderer-neutral dependency diagram model."""
    resolved_focus_keys_gi, focus_error = _resolve_focus_keys(
        focus_keys_list_input, global_path_to_key_info_map
    )
    if focus_error:
        return DiagramInput(nodes={}, edges=[], error=focus_error)

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

    edges_within_scope_gi: List[Tuple[str, str, str]] = []
    if is_module_view:
        for k1_gi, k2_gi, char_val in intermediate_edges_gi:
            if (k1_gi in keys_in_module_scope_gi) or (k2_gi in keys_in_module_scope_gi):
                edges_within_scope_gi.append((k1_gi, k2_gi, char_val))
    elif focus_keys_valid_gi:
        for k1_gi, k2_gi, char_val in intermediate_edges_gi:
            if k1_gi in focus_keys_valid_gi or k2_gi in focus_keys_valid_gi:
                edges_within_scope_gi.append((k1_gi, k2_gi, char_val))
    else:
        edges_within_scope_gi = intermediate_edges_gi

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
    nodes_to_render_gi.update(focus_keys_valid_gi)

    if not nodes_to_render_gi:
        return DiagramInput(nodes={}, edges=[], no_data=True)

    all_kis_for_hierarchy: Dict[str, KeyInfo] = {}
    queue: deque[str] = deque(sorted(nodes_to_render_gi))
    visited: Set[str] = set()
    while queue:
        k_gi = queue.popleft()
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

    global_key_counts: Dict[str, int] = defaultdict(int)
    for ki_c in global_path_to_key_info_map.values():
        global_key_counts[ki_c.key_string] += 1

    project_root = get_project_root()
    path_to_gi = {ki.norm_path: gi for gi, ki in all_kis_for_hierarchy.items()}
    nodes: Dict[str, DiagramNode] = {}
    for gi, ki in sorted(all_kis_for_hierarchy.items()):
        parent_gi = path_to_gi.get(ki.parent_path) if ki.parent_path else None
        display_key = ki.key_string if global_key_counts[ki.key_string] <= 1 else gi
        basename = os.path.basename(ki.norm_path)
        label = f"{display_key}\n{basename}" if basename else display_key
        nodes[gi] = DiagramNode(
            gi_str=gi,
            key_string=ki.key_string,
            norm_path=ki.norm_path,
            parent_gi=parent_gi,
            tier=ki.tier,
            is_directory=ki.is_directory,
            label=label,
            node_class=_get_node_class(ki, config_manager_instance, project_root),
            is_focus=gi in focus_keys_valid_gi,
        )

    children_by_parent: Dict[Optional[str], List[str]] = defaultdict(list)
    for gi, node in nodes.items():
        children_by_parent[node.parent_gi].append(gi)
    sorted_children_by_parent = {
        parent: _sort_child_gis(children, nodes)
        for parent, children in children_by_parent.items()
    }
    root_ids = sorted_children_by_parent.get(None, [])

    edges = [
        DiagramEdge(source_gi=source, target_gi=target, dep_char=dep_char)
        for source, target, dep_char in sorted(final_edges_to_draw_gi)
        if source in nodes and target in nodes
    ]

    return DiagramInput(
        nodes=nodes,
        edges=edges,
        root_ids=root_ids,
        children_by_parent=sorted_children_by_parent,
        focus_gis=resolved_focus_keys_gi,
        is_module_view=is_module_view,
    )
