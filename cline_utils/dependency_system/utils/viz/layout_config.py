# cline_utils/dependency_system/utils/viz/layout_config.py

"""
Specialized Mermaid and ELK configurations, including color schemes and rendering limits.
"""

# Neon/Purple color scheme and styling constants
SUBGRAPH_TEXT_COLOR = "#D1C4E9"
SUBGRAPH_FILL = "#282828"
SUBGRAPH_STROKE = "#39FF14"
SUBGRAPH_FOCUS_STROKE = "#00FF00"

CLASS_DEFS = [
    "classDef module fill:#f9f,stroke:#333,stroke-width:2px,color:#333,font-weight:bold;",
    "classDef file fill:#D1C4E9,stroke:#666,stroke-width:1px,color:#333;",  # Mild purple fill
    "classDef doc fill:#D1C4E9,stroke:#666,stroke-width:1px,color:#333;",
    "classDef focusNode stroke:#007bff,stroke-width:3px;",
]

LINK_STYLE = "linkStyle default stroke:#CCCCCC,stroke-width:1px,stroke-linecap:round,stroke-linejoin:round;"

# Mermaid rendering limits and flowchart options
MERMAID_CONFIG = {
    "maxEdges": 3500,
    "maxTextSize": 200000,
    "flowchart": {
        "useMaxWidth": False,
        "useMaxHeight": False,
        "defaultRenderer": "dagre",
        "dagre": {"nodeSpacing": 1, "rankSpacing": 10000},
        "elk": {
            "algorithm": "layered",
            "org.eclipse.elk.direction": "DOWN",
            "org.eclipse.elk.edgeRouting": "ORTHOGONAL",
            "org.eclipse.elk.layered.spacing.nodeNodeBetweenLayers": ".05",
            "org.eclipse.elk.spacing.nodeNode": ".03",
            "nodePlacementStrategy": "LINEAR_SEGMENTS",
            "org.eclipse.elk.layered.unnecessaryBendpoints": False,
            "org.eclipse.elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
            "org.eclipse.elk.layered.cycleBreakingStrategy": "GREEDY",
            "elk.mergeEdges": True,
        },
    },
}

# Puppeteer configuration for headless rendering
PUPPETEER_CONFIG = {
    "protocolTimeout": 600000,
    "args": [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-software-rasterizer",
        "--disable-dev-shm-usage",
        "--disable-features=VizDisplayCompositor",
    ],
}

DEP_CHAR_TO_STYLE = {
    "<": ("-->", "needs"),
    ">": ("-->", "needs"),
    "x": ("<-->", "both"),
    "d": ("-.->", "docs"),
    "s": ("-.->", "s"),
    "S": ("==>", "S"),
}
