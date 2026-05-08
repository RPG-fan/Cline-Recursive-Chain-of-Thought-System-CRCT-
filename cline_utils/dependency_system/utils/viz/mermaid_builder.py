"""Mermaid DSL adapter for dependency diagrams."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import PathMigrationInfo

from .diagram_model import build_diagram_model
from .layout_config import (
    CLASS_DEFS,
    DEP_CHAR_TO_STYLE,
    LINK_STYLE,
    SUBGRAPH_FILL,
    SUBGRAPH_FOCUS_STROKE,
    SUBGRAPH_STROKE,
    SUBGRAPH_TEXT_COLOR,
)
from .models import DiagramInput


def _get_safe_mermaid_id(
    key_gi: str, safe_id_map: Dict[str, str], safe_id_counts: Dict[str, int]
) -> str:
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


def build_mermaid_from_diagram(diagram: DiagramInput) -> str:
    """Builds Mermaid syntax from the shared renderer-neutral diagram model."""
    if diagram.error:
        return diagram.error
    if diagram.no_data:
        return "flowchart TB\n\n// No relevant data to visualize."

    mermaid_parts = ["flowchart TB"]
    mermaid_parts.extend(["  " + class_def for class_def in CLASS_DEFS])
    mermaid_parts.append(f"  {LINK_STYLE}")

    safe_id_map: Dict[str, str] = {}
    safe_id_counts: Dict[str, int] = {}
    rendered_ids: Set[str] = set()
    dir_gi_to_sg_id: Dict[str, str] = {}
    sg_counter = 0

    def render_children(parent_gi: Optional[str], indent: str) -> None:
        nonlocal sg_counter
        for child_gi in diagram.children_by_parent.get(parent_gi, []):
            node = diagram.nodes[child_gi]
            if node.is_directory:
                child_ids = diagram.children_by_parent.get(child_gi, [])
                if not child_ids:
                    continue
                sg_counter += 1
                sg_id = f"sg_{re.sub(r'[^a-zA-Z0-9_]', '_', node.key_string)}_{sg_counter}"
                dir_gi_to_sg_id[child_gi] = sg_id
                title = (
                    f"<font color='{SUBGRAPH_TEXT_COLOR}'>"
                    f"{node.label.replace(chr(10), '<br>')}"
                )
                mermaid_parts.append(f'{indent}subgraph {sg_id} ["{title}"]')
                rendered_ids.add(child_gi)
                mermaid_parts.append(
                    f"{indent}  style {sg_id} fill:{SUBGRAPH_FILL},"
                    f"stroke:{SUBGRAPH_STROKE},stroke-width:4px"
                )
                if node.is_focus:
                    mermaid_parts.append(
                        f"{indent}  style {sg_id} stroke-width:5px,"
                        f"stroke:{SUBGRAPH_FOCUS_STROKE}"
                    )
                render_children(child_gi, indent + "  ")
                mermaid_parts.append(f"{indent}end")
                continue

            if child_gi in rendered_ids:
                continue
            m_id = _get_safe_mermaid_id(child_gi, safe_id_map, safe_id_counts)
            label_text = node.label.replace("\n", "<br>")
            mermaid_parts.append(f'{indent}{m_id}["{label_text}"]')
            mermaid_parts.append(f"{indent}class {m_id} {node.node_class}")
            if node.is_focus:
                mermaid_parts.append(f"{indent}class {m_id} focusNode")
            rendered_ids.add(child_gi)

    render_children(None, "  ")

    mermaid_parts.append("\n  %% -- Dependencies --")
    for edge in sorted(
        diagram.edges, key=lambda item: (item.source_gi, item.target_gi, item.dep_char)
    ):
        if edge.source_gi not in rendered_ids or edge.target_gi not in rendered_ids:
            continue
        id1 = dir_gi_to_sg_id.get(
            edge.source_gi,
            _get_safe_mermaid_id(edge.source_gi, safe_id_map, safe_id_counts),
        )
        id2 = dir_gi_to_sg_id.get(
            edge.target_gi,
            _get_safe_mermaid_id(edge.target_gi, safe_id_map, safe_id_counts),
        )
        arrow, label = DEP_CHAR_TO_STYLE.get(edge.dep_char, ("-->", edge.dep_char))
        if label:
            mermaid_parts.append(f'  {id1} {arrow}|"{label}"| {id2}')
        else:
            mermaid_parts.append(f"  {id1} {arrow} {id2}")

    return "\n".join(mermaid_parts)


def build_mermaid_string(
    focus_keys_list_input: List[str],
    global_path_to_key_info_map: Dict[str, KeyInfo],
    path_migration_info: PathMigrationInfo,
    all_tracker_paths_list: List[str],
    config_manager_instance: ConfigManager,
    pre_aggregated_links: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
) -> str:
    """Constructs the Mermaid DSL string through the shared diagram model."""
    diagram = build_diagram_model(
        focus_keys_list_input=focus_keys_list_input,
        global_path_to_key_info_map=global_path_to_key_info_map,
        path_migration_info=path_migration_info,
        all_tracker_paths_list=all_tracker_paths_list,
        config_manager_instance=config_manager_instance,
        pre_aggregated_links=pre_aggregated_links,
    )
    return build_mermaid_from_diagram(diagram)
