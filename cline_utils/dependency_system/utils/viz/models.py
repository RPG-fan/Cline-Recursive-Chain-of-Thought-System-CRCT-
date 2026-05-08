"""Renderer-neutral diagram and native layout models."""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class DiagramNode:
    gi_str: str
    key_string: str
    norm_path: str
    parent_gi: Optional[str]
    tier: int
    is_directory: bool
    label: str
    node_class: str
    is_focus: bool = False


@dataclass(frozen=True)
class DiagramEdge:
    source_gi: str
    target_gi: str
    dep_char: str


@dataclass(frozen=True)
class DiagramInput:
    nodes: Dict[str, DiagramNode]
    edges: List[DiagramEdge]
    root_ids: List[str] = field(default_factory=lambda: [])
    children_by_parent: Dict[Optional[str], List[str]] = field(default_factory=lambda: {})
    focus_gis: List[str] = field(default_factory=lambda: [])
    is_module_view: bool = False
    error: Optional[str] = None
    no_data: bool = False

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "nodes": {gi: asdict(node) for gi, node in self.nodes.items()},
            "edges": [asdict(edge) for edge in self.edges],
            "root_ids": list(self.root_ids),
            "children_by_parent": {
                ("__root__" if parent is None else parent): list(children)
                for parent, children in self.children_by_parent.items()
            },
            "focus_gis": list(self.focus_gis),
            "is_module_view": self.is_module_view,
            "error": self.error,
            "no_data": self.no_data,
        }


@dataclass
class LayoutNode:
    gi_str: str
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    label: str = ""
    is_directory: bool = False
    node_class: str = "default"
    parent_gi: Optional[str] = None
    tier: int = 0
    is_focus: bool = False


@dataclass
class LayoutGroup:
    gi_str: str
    child_ids: List[str] = field(default_factory=lambda: [])
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    label: str = ""
    parent_gi: Optional[str] = None
    is_focus: bool = False


@dataclass
class LayoutEdge:
    source_gi: str
    target_gi: str
    dep_char: str
    route_points: List[Point] = field(default_factory=lambda: [])
    is_pipe_member: bool = False
    pipe_id: Optional[str] = None


@dataclass
class LayoutPipe:
    pipe_id: str
    source_gis: List[str]
    target_gi: str
    dep_char: str
    route_points: List[Point] = field(default_factory=lambda: [])
    feeder_routes: List[List[Point]] = field(default_factory=lambda: [])
    stroke_width: float = 0.0
    count_badge_pos: Point = (0.0, 0.0)
    count: int = 0
    color: str = "#8fd16a"


@dataclass
class LayoutModel:
    nodes: Dict[str, LayoutNode]
    groups: Dict[str, LayoutGroup]
    edges: List[LayoutEdge]
    pipes: List[LayoutPipe] = field(default_factory=lambda: [])
    canvas_width: float = 0.0
    canvas_height: float = 0.0

    def to_json_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NativeLayoutConfig:
    pipe_threshold: int = 3
    node_min_width: float = 120.0
    node_height: float = 40.0
    label_char_width: float = 7.0
    label_padding_x: float = 24.0
    group_padding_x: float = 20.0
    group_padding_y: float = 16.0
    group_header_height: float = 28.0
    column_gap: float = 100.0
    row_gap: float = 28.0
    tier_gap: float = 80.0
    canvas_padding: float = 32.0
    feeder_stub_length: float = 24.0
    pipe_base_width: float = 2.0
    pipe_max_width: float = 18.0
    pipe_log_scale: float = 4.0
    max_nodes_per_column: int = 9
    enable_hex_stagger: bool = True
    hex_stagger_ratio: float = 0.5
    use_oklch_pipe_colors: bool = True
    enable_edge_avoidance: bool = True
    edge_avoidance_padding: float = 8.0
    render_stage: int = 5
    background: str = "transparent"
    group_fill: str = "#20242a"
    group_stroke: str = "#5ca3ff"
    node_fill: str = "#f6f3ff"
    doc_fill: str = "#fff6d6"
    focus_stroke: str = "#007bff"
    edge_stroke: str = "#6f7b88"
    text_color: str = "#20242a"
    group_text_color: str = "#e9f0f8"

    def to_json_dict(self) -> Dict[str, object]:
        return asdict(self)
