# cline_utils/dependency_system/utils/visualize_dependencies.py

"""Orchestrator for dependency visualization backends."""

import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from .path_utils import PathMigrationInfo, get_project_root
from .viz.diagram_model import build_diagram_model
from .viz.mermaid_builder import build_mermaid_from_diagram, build_mermaid_string
from .viz.models import NativeLayoutConfig
from .viz.native_cache import write_native_layout_cache
from .viz.native_layout import compute_native_layout, route_edges_and_pipes
from .viz.native_renderer import render_svg_pass, write_native_svg
from .viz.renderer import render_mermaid_to_image
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.core.key_manager import KeyInfo

logger = logging.getLogger(__name__)


def _diagram_base_filename(focus_keys_list_input: List[str]) -> str:
    if not focus_keys_list_input:
        return "project_overview_dependencies"
    if len(focus_keys_list_input) == 1:
        sanitized_key = re.sub(r"[^\w\-.#]", "_", focus_keys_list_input[0])
        return f"module_{sanitized_key}_dependencies"
    sanitized_keys = [re.sub(r"[^\w\-.#]", "_", k) for k in focus_keys_list_input]
    joined_keys = "_".join(sanitized_keys)
    if len(joined_keys) > 50:
        joined_keys = joined_keys[:50] + "_etc"
    return f"custom_view_{len(focus_keys_list_input)}_keys_{joined_keys}_dependencies"


def generate_dependency_diagram(
    focus_keys_list_input: List[str],
    global_path_to_key_info_map: Dict[str, KeyInfo],
    path_migration_info: PathMigrationInfo,
    all_tracker_paths_list: List[str],
    config_manager_instance: ConfigManager,
    pre_aggregated_links: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
    backend: str = "mermaid",
    render: bool = True,
    native_config: Optional[NativeLayoutConfig] = None,
) -> Optional[str]:
    """Generates a dependency diagram through the requested backend."""
    backend = backend.lower()
    if backend not in {"mermaid", "native"}:
        raise ValueError(f"Unsupported dependency diagram backend: {backend}")

    diagram = build_diagram_model(
        focus_keys_list_input=focus_keys_list_input,
        global_path_to_key_info_map=global_path_to_key_info_map,
        path_migration_info=path_migration_info,
        all_tracker_paths_list=all_tracker_paths_list,
        config_manager_instance=config_manager_instance,
        pre_aggregated_links=pre_aggregated_links,
    )

    if backend == "mermaid":
        output = build_mermaid_from_diagram(diagram)
        if render and "Error:" not in output and "No relevant data" not in output:
            output_dir = os.path.join(
                get_project_root(), "cline_docs", "dependency_diagrams"
            )
            output_image_file = os.path.join(
                output_dir, f"{_diagram_base_filename(focus_keys_list_input)}.svg"
            )
            render_mermaid_to_image(output, output_image_file)
        return output

    if diagram.error:
        return diagram.error
    if diagram.no_data:
        return "<svg><!-- No relevant data to visualize. --></svg>"

    native_config = native_config or NativeLayoutConfig()
    layout = compute_native_layout(diagram, native_config)
    layout = route_edges_and_pipes(layout, native_config)
    output = render_svg_pass(layout, native_config)

    if render:
        output_dir = os.path.join(
            get_project_root(), "cline_docs", "dependency_diagrams"
        )
        base_filename = _diagram_base_filename(focus_keys_list_input)
        output_image_file = os.path.join(output_dir, f"{base_filename}.svg")
        write_native_svg(layout, output_image_file, native_config)
        cache_file = os.path.join(output_dir, f"{base_filename}.layout.json")
        write_native_layout_cache(cache_file, diagram, layout, native_config)

    return output


def generate_mermaid_diagram(
    focus_keys_list_input: List[str],
    global_path_to_key_info_map: Dict[str, KeyInfo],
    path_migration_info: PathMigrationInfo,
    all_tracker_paths_list: List[str],
    config_manager_instance: ConfigManager,
    pre_aggregated_links: Optional[Dict[Tuple[str, str], Tuple[str, Set[str]]]] = None,
    render: bool = True,
) -> Optional[str]:
    """
    Generates a Mermaid diagram and optionally renders it to an image.
    This is the main entry point for the dependency visualization system.
    """
    mermaid_syntax = build_mermaid_string(
        focus_keys_list_input=focus_keys_list_input,
        global_path_to_key_info_map=global_path_to_key_info_map,
        path_migration_info=path_migration_info,
        all_tracker_paths_list=all_tracker_paths_list,
        config_manager_instance=config_manager_instance,
        pre_aggregated_links=pre_aggregated_links,
    )

    if (
        render
        and "Error:" not in mermaid_syntax
        and "No relevant data" not in mermaid_syntax
    ):
        output_dir = os.path.join(
            get_project_root(), "cline_docs", "dependency_diagrams"
        )
        output_image_file = os.path.join(
            output_dir, f"{_diagram_base_filename(focus_keys_list_input)}.svg"
        )
        render_mermaid_to_image(mermaid_syntax, output_image_file)

    return mermaid_syntax


# Backward compatibility for any direct renderer calls if they exist
__all__ = [
    "generate_dependency_diagram",
    "generate_mermaid_diagram",
    "render_mermaid_to_image",
]
