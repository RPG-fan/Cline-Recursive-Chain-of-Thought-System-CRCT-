from __future__ import annotations

from typing import Dict, Set, Tuple

from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.viz import (
    NativeLayoutConfig,
    build_diagram_model,
    build_mermaid_string,
    compute_native_layout,
    render_svg_pass,
    route_edges_and_pipes,
)
from cline_utils.dependency_system.utils.viz.native_layout import (
    layout_cache_key,
    pipe_stroke_width,
)


AggregatedLinks = Dict[Tuple[str, str], Tuple[str, Set[str]]]


def _global_map() -> Dict[str, KeyInfo]:
    return {
        "h:/repo/src": KeyInfo("1A", "h:/repo/src", None, 1, True),
        "h:/repo/src/a.py": KeyInfo("1A1", "h:/repo/src/a.py", "h:/repo/src", 1, False),
        "h:/repo/src/b.py": KeyInfo("1A2", "h:/repo/src/b.py", "h:/repo/src", 1, False),
        "h:/repo/src/c.md": KeyInfo("1A3", "h:/repo/src/c.md", "h:/repo/src", 1, False),
        "h:/repo/src/d.py": KeyInfo("1A4", "h:/repo/src/d.py", "h:/repo/src", 1, False),
        "h:/repo/engine": KeyInfo("1B", "h:/repo/engine", None, 1, True),
        "h:/repo/engine/e.py": KeyInfo(
            "1B1", "h:/repo/engine/e.py", "h:/repo/engine", 1, False
        ),
    }


def _config() -> ConfigManager:
    return ConfigManager()


from cline_utils.dependency_system.analysis.project_analyzer import PathMigrationInfo


def _migration_info(global_map: Dict[str, KeyInfo]) -> PathMigrationInfo:
    return {
        path: (info.key_string, info.key_string) for path, info in global_map.items()
    }


def _links() -> AggregatedLinks:
    return {
        ("1A1", "1A2"): ("<", {"fixture"}),
        ("1A2", "1A4"): ("<", {"fixture"}),
        ("1A3", "1A4"): ("<", {"fixture"}),
        ("1B1", "1A4"): ("<", {"fixture"}),
        ("1A1", "1A3"): ("p", {"fixture"}),
        ("1A", "1A1"): ("x", {"fixture"}),
        ("1A", "1A3"): ("d", {"fixture"}),
    }


def test_build_diagram_model_preserves_scope_and_filters_edges():
    global_map = _global_map()
    diagram = build_diagram_model(
        focus_keys_list_input=[],
        global_path_to_key_info_map=global_map,
        path_migration_info=_migration_info(global_map),
        all_tracker_paths_list=[],
        config_manager_instance=_config(),
        pre_aggregated_links=_links(),
    )

    assert not diagram.error
    assert not diagram.no_data
    assert "1A" in diagram.nodes
    assert "1A1" in diagram.nodes
    assert ("1A1", "1A3", "p") not in {
        (edge.source_gi, edge.target_gi, edge.dep_char) for edge in diagram.edges
    }
    assert ("1A", "1A1", "x") not in {
        (edge.source_gi, edge.target_gi, edge.dep_char) for edge in diagram.edges
    }
    assert ("1A", "1A3", "d") in {
        (edge.source_gi, edge.target_gi, edge.dep_char) for edge in diagram.edges
    }


def test_build_diagram_model_module_view_keeps_external_neighbors():
    global_map = _global_map()
    diagram = build_diagram_model(
        focus_keys_list_input=["1A"],
        global_path_to_key_info_map=global_map,
        path_migration_info=_migration_info(global_map),
        all_tracker_paths_list=[],
        config_manager_instance=_config(),
        pre_aggregated_links=_links(),
    )

    assert diagram.is_module_view
    assert "1B1" in diagram.nodes
    assert diagram.nodes["1A"].is_focus


def test_mermaid_builder_still_emits_subgraphs_and_dependencies():
    global_map = _global_map()
    mermaid = build_mermaid_string(
        focus_keys_list_input=[],
        global_path_to_key_info_map=global_map,
        path_migration_info=_migration_info(global_map),
        all_tracker_paths_list=[],
        config_manager_instance=_config(),
        pre_aggregated_links=_links(),
    )

    assert mermaid.startswith("flowchart TB")
    assert "subgraph sg_1A_1" in mermaid
    assert "%% -- Dependencies --" in mermaid
    assert '"needs"' in mermaid


def test_no_relevant_data_matches_mermaid_compatibility_message():
    global_map = _global_map()
    mermaid = build_mermaid_string(
        focus_keys_list_input=[],
        global_path_to_key_info_map=global_map,
        path_migration_info=_migration_info(global_map),
        all_tracker_paths_list=[],
        config_manager_instance=_config(),
        pre_aggregated_links={},
    )

    assert "// No relevant data to visualize." in mermaid


def test_native_layout_is_deterministic_and_groups_enclose_children():
    global_map = _global_map()
    config = NativeLayoutConfig(pipe_threshold=3)
    diagram = build_diagram_model(
        focus_keys_list_input=[],
        global_path_to_key_info_map=global_map,
        path_migration_info=_migration_info(global_map),
        all_tracker_paths_list=[],
        config_manager_instance=_config(),
        pre_aggregated_links=_links(),
    )

    layout_a = route_edges_and_pipes(compute_native_layout(diagram, config), config)
    layout_b = route_edges_and_pipes(compute_native_layout(diagram, config), config)

    assert layout_a.to_json_dict() == layout_b.to_json_dict()
    assert layout_cache_key(diagram, config) == layout_cache_key(diagram, config)
    group = layout_a.groups["1A"]
    child = layout_a.nodes["1A1"]
    assert group.x < child.x
    assert group.y < child.y
    assert group.x + group.width > child.x + child.width
    assert group.y + group.height > child.y + child.height


def test_native_pipe_aggregation_and_svg_stages():
    global_map = _global_map()
    config = NativeLayoutConfig(pipe_threshold=3)
    diagram = build_diagram_model(
        focus_keys_list_input=[],
        global_path_to_key_info_map=global_map,
        path_migration_info=_migration_info(global_map),
        all_tracker_paths_list=[],
        config_manager_instance=_config(),
        pre_aggregated_links=_links(),
    )
    layout = route_edges_and_pipes(compute_native_layout(diagram, config), config)

    assert len(layout.pipes) == 1
    pipe = layout.pipes[0]
    assert pipe.target_gi == "1A4"
    assert pipe.dep_char == "<"
    assert pipe.count == 3
    assert pipe.stroke_width == pipe_stroke_width(3, config)
    assert all(
        edge.is_pipe_member for edge in layout.edges if edge.pipe_id == pipe.pipe_id
    )

    structure_svg = render_svg_pass(
        layout, NativeLayoutConfig(pipe_threshold=3, render_stage=1)
    )
    full_svg = render_svg_pass(
        layout, NativeLayoutConfig(pipe_threshold=3, render_stage=5)
    )

    assert 'id="groups"' in structure_svg
    assert 'id="pipe-spines"' not in structure_svg
    assert 'id="pipe-spines"' in full_svg
    assert 'id="pipe-feeders"' in full_svg
    assert 'id="edges-direct"' in full_svg
    assert 'id="edges-aux"' in full_svg
    assert 'data-count="3"' in full_svg
