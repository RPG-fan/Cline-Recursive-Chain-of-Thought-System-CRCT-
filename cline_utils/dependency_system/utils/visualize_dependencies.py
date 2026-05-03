# cline_utils/dependency_system/utils/visualize_dependencies.py

"""
Orchestrator for dependency visualization.
Splits logic into mermaid_builder (DSL), renderer (mmdc), and layout_config (styles).
"""

import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from .path_utils import PathMigrationInfo, get_project_root
from .viz.mermaid_builder import build_mermaid_string
from .viz.renderer import render_mermaid_to_image
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.core.key_manager import KeyInfo

logger = logging.getLogger(__name__)


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

        if not focus_keys_list_input:
            base_filename = "project_overview_dependencies"
        elif len(focus_keys_list_input) == 1:
            sanitized_key = re.sub(r"[^\w\-.#]", "_", focus_keys_list_input[0])
            base_filename = f"module_{sanitized_key}_dependencies"
        else:
            sanitized_keys = [
                re.sub(r"[^\w\-.#]", "_", k) for k in focus_keys_list_input
            ]
            joined_keys = "_".join(sanitized_keys)
            if len(joined_keys) > 50:
                joined_keys = joined_keys[:50] + "_etc"
            base_filename = f"custom_view_{len(focus_keys_list_input)}_keys_{joined_keys}_dependencies"

        output_image_file = os.path.join(output_dir, f"{base_filename}.svg")
        render_mermaid_to_image(mermaid_syntax, output_image_file)

    return mermaid_syntax


# Backward compatibility for any direct renderer calls if they exist
__all__ = ["generate_mermaid_diagram", "render_mermaid_to_image"]
