# Dependency Visualization Package (viz)

The `cline_utils.dependency_system.utils.viz` package is a modularized overhaul of the dependency visualization system. It separates data processing from rendering, allowing for easier extensions and more robust diagram generation.

## 1. Sub-Package Structure

### `mermaid_builder.py` (DSL Construction)
- **Role**: Converts project dependency graphs into Mermaid DSL.
- **Responsibility**: Handles node grouping (subgraphs), edge types (solid vs. dashed), and recursive relationship expansion.
- **Key Function**: `build_mermaid_string(focus_keys_list_input, global_path_to_key_info_map, path_migration_info, all_tracker_paths_list, config_manager_instance, pre_aggregated_links=None)`

### `renderer.py` (Image Rendering)
- **Role**: Converts Mermaid DSL into visual image files.
- **Responsibility**: Manages the local Node.js environment to call `@mermaid-js/mermaid-cli` (`mmdc`). Dynamically locates the `mmdc` executable via `npm config get prefix` and falls back to PATH resolution.
- **Key Function**: `render_mermaid_to_image(mermaid_syntax, output_file_path)`

### `layout_config.py` (Styles & Layouts)
- **Role**: Defines the visual "look and feel" of the diagrams.
- **Responsibility**: Contains Mermaid themes, layout configuration, and CSS styling for nodes.
- **Key Elements**: `MERMAID_CONFIG`, `PUPPETEER_CONFIG`, `CLASS_DEFS`, `DEP_CHAR_TO_STYLE`, `SUBGRAPH_FILL`, `SUBGRAPH_STROKE`, `LINK_STYLE`.

---

## 2. How to Use

### Via Orchestrator (CLI)
Most users will interact with the visualization system via the main dependency processor:

```bash
# Generate a Mermaid diagram for the entire project
python -m cline_utils.dependency_system.dependency_processor visualize-dependencies

# Generate a focused diagram for specific keys
python -m cline_utils.dependency_system.dependency_processor visualize-dependencies --key 1A1 2B#3

# Specify output file path
python -m cline_utils.dependency_system.dependency_processor visualize-dependencies --output exports/graph.svg
```

Diagrams are saved as `.svg` files by default to `cline_docs/dependency_diagrams/`.

### In Code
You can leverage the modular package directly in your scripts via the `generate_mermaid_diagram` orchestrator function in `visualize_dependencies.py`, or by calling the sub-package modules directly:

```python
from cline_utils.dependency_system.utils.viz.mermaid_builder import build_mermaid_string
from cline_utils.dependency_system.utils.viz.renderer import render_mermaid_to_image

# 1. Build the DSL
dsl = build_mermaid_string(focus_keys, global_map, migration_info, tracker_paths, config)

# 2. Render to SVG via mmdc
render_mermaid_to_image(dsl, "output/project_map.svg")
```

---

## 3. Configuration

Visual settings are managed via constants in `layout_config.py` to ensure performance and stability. The system supports:
- **Nested Subgraphs**: Automatically groups files by their directory structure.
- **Focus Highlighting**: Nodes passed as "focus keys" are rendered with distinct styling.
- **Layout Engine**: Uses the `dagre` algorithm by default (configured in `MERMAID_CONFIG`). An `elk` configuration block is also defined in `layout_config.py` for reference but is not dynamically selected at runtime.

---

## 4. Extension Guide

The modular structure makes it easy to add new features:
1.  **New Styling**: Modify `CLASS_DEFS`, `SUBGRAPH_FILL`, or `DEP_CHAR_TO_STYLE` in `layout_config.py`.
2.  **New Rendering Logic**: Update `renderer.py` to support different CLI flags for `mmdc`.
3.  **New DSL Logic**: Update `mermaid_builder.py` to support different node types or relationship logic.

---

## 5. Troubleshooting
-   **"mmdc command not found"**: Ensure you have installed the Mermaid CLI globally or in your project: `npm install -g @mermaid-js/mermaid-cli`.
-   **Empty Diagrams**: This usually means no dependencies were found. Run `analyze-project` first to build the trackers.
-   **Timeout**: Large diagrams can take time to render. The system has a 900s timeout. If it still fails, try focusing on specific keys to reduce complexity.
