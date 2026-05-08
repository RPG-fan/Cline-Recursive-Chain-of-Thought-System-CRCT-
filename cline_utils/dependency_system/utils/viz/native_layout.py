"""Deterministic native layout and routing for dependency diagrams."""



import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    DiagramInput,
    LayoutEdge,
    LayoutGroup,
    LayoutModel,
    LayoutNode,
    LayoutPipe,
    NativeLayoutConfig,
    Point,
)


ROOT_ID = "__root__"


def _module_key(key_string: str) -> str:
    match = re.match(r"\d+([A-Z])", key_string)
    return match.group(1) if match else "_"


def _node_width(label: str, config: NativeLayoutConfig) -> float:
    longest = max((len(part) for part in label.splitlines()), default=0)
    return max(
        config.node_min_width,
        longest * config.label_char_width + config.label_padding_x,
    )


def _sort_key(node_id: str, diagram: DiagramInput) -> Tuple[str, str]:
    node = diagram.nodes[node_id]
    return (node.key_string, node.norm_path)


def _center(node: LayoutNode | LayoutGroup) -> Point:
    return (node.x + node.width / 2.0, node.y + node.height / 2.0)


def _right_port(node: LayoutNode | LayoutGroup) -> Point:
    return (node.x + node.width, node.y + node.height / 2.0)


def _left_port(node: LayoutNode | LayoutGroup) -> Point:
    return (node.x, node.y + node.height / 2.0)


def _top_port(node: LayoutNode | LayoutGroup) -> Point:
    return (node.x + node.width / 2.0, node.y)


def _bottom_port(node: LayoutNode | LayoutGroup) -> Point:
    return (node.x + node.width / 2.0, node.y + node.height)


def _dynamic_stub_offsets(
    obstacles: List[Tuple[float, float, float, float]],
    padding: float,
    axis: str,
) -> List[float]:
    """Build stub offsets scaled to actual obstacle dimensions on the given axis."""
    if not obstacles:
        return [float(v) for v in range(10, 70, 10)]
    if axis == "x":
        max_dim = max(ow for _, _, ow, _ in obstacles)
    else:
        max_dim = max(oh for _, _, _, oh in obstacles)
    ceiling = max(70.0, max_dim + padding * 2 + 40.0)
    step = max(10.0, ceiling / 20.0)
    return [round(i * step, 1) for i in range(1, 22)]


def _perimeter_fallback(
    start: Point,
    end: Point,
    obstacles: List[Tuple[float, float, float, float]],
    padding: float,
    prefer_horizontal: bool,
) -> List[Point]:
    """Guaranteed collision-free route around the bounding box of all obstacles."""
    if not obstacles:
        if prefer_horizontal:
            mid = (start[0] + end[0]) / 2.0
            return [start, (mid, start[1]), (mid, end[1]), end]
        mid = (start[1] + end[1]) / 2.0
        return [start, (start[0], mid), (end[0], mid), end]

    margin = padding + 40.0

    # Bounding box that includes start, end, and every obstacle
    all_x = [start[0], end[0]]
    all_y = [start[1], end[1]]
    for ox, oy, ow, oh in obstacles:
        all_x.extend((ox, ox + ow))
        all_y.extend((oy, oy + oh))

    bbox_l = min(all_x) - margin
    bbox_r = max(all_x) + margin
    bbox_t = min(all_y) - margin
    bbox_b = max(all_y) + margin

    if prefer_horizontal:
        # Pick detour Y above or below the entire obstacle field
        avg_y = (start[1] + end[1]) / 2.0
        detour_y = bbox_t if abs(avg_y - bbox_t) < abs(avg_y - bbox_b) else bbox_b

        # Exit X: nearest horizontal edge of bbox from start
        exit_x = bbox_l if abs(start[0] - bbox_l) < abs(start[0] - bbox_r) else bbox_r
        # Entry X: nearest horizontal edge of bbox from end
        entry_x = bbox_l if abs(end[0] - bbox_l) < abs(end[0] - bbox_r) else bbox_r

        return [
            start,
            (exit_x, start[1]),
            (exit_x, detour_y),
            (entry_x, detour_y),
            (entry_x, end[1]),
            end,
        ]

    # Vertical-dominant
    avg_x = (start[0] + end[0]) / 2.0
    detour_x = bbox_l if abs(avg_x - bbox_l) < abs(avg_x - bbox_r) else bbox_r

    exit_y = bbox_t if abs(start[1] - bbox_t) < abs(start[1] - bbox_b) else bbox_b
    entry_y = bbox_t if abs(end[1] - bbox_t) < abs(end[1] - bbox_b) else bbox_b

    return [
        start,
        (start[0], exit_y),
        (detour_x, exit_y),
        (detour_x, entry_y),
        (end[0], entry_y),
        end,
    ]


def _route_between(
    source: LayoutNode | LayoutGroup,
    target: LayoutNode | LayoutGroup,
    layout: LayoutModel,
    config: NativeLayoutConfig,
) -> List[Point]:
    sx, sy = _center(source)
    tx, ty = _center(target)

    # 1. Collect obstacles (nodes and non-ancestor groups)
    passable = {source.gi_str, target.gi_str}
    passable.update(_get_ancestors(source.gi_str, layout))
    passable.update(_get_ancestors(target.gi_str, layout))

    padding = config.edge_avoidance_padding

    obstacles: List[Tuple[float, float, float, float]] = []
    if config.enable_edge_avoidance:
        for node in layout.nodes.values():
            if node.gi_str not in passable:
                obstacles.append((node.x, node.y, node.width, node.height))
        for group in layout.groups.values():
            if group.gi_str not in passable and group.gi_str != "__root__":
                obstacles.append((group.x, group.y, group.width, group.height))

    def is_route_blocked(route: List[Point]) -> bool:
        for i in range(len(route) - 1):
            if _is_segment_blocked(route[i], route[i+1], obstacles, padding):
                return True
        return False

    # Adaptive step size tied to actual layout density
    z_step = max(10.0, config.column_gap * 0.2)
    z_max_steps = 200

    # 2. Try default rectilinear route — horizontal-dominant
    if abs(tx - sx) >= abs(ty - sy):
        start = _right_port(source) if tx >= sx else _left_port(source)
        end = _left_port(target) if tx >= sx else _right_port(target)
        lane_x = (start[0] + end[0]) / 2.0

        if config.enable_edge_avoidance:
            # A. Try 3-segment Z-route with adaptive step
            for step in range(0, z_max_steps):
                for sign in [1, -1]:
                    if step == 0 and sign == -1:
                        continue
                    test_x = lane_x + step * z_step * sign
                    route = [start, (test_x, start[1]), (test_x, end[1]), end]
                    if not is_route_blocked(route):
                        return route

            # B. Try 5-segment Y-detour with dynamic stub offsets
            start_dir = 1 if tx >= sx else -1
            end_dir = -1 if tx >= sx else 1

            y_offsets: List[float] = []
            y_step = max(20.0, config.row_gap)
            for step in range(1, 150):
                y_offsets.append(step * y_step)
                y_offsets.append(-step * y_step)

            x_stubs = _dynamic_stub_offsets(obstacles, padding, "x")

            for y_off in y_offsets:
                test_y = start[1] + y_off
                for s_x_off in x_stubs:
                    stub_x = start[0] + s_x_off * start_dir
                    if _is_segment_blocked(start, (stub_x, start[1]), obstacles, padding):
                        continue
                    if _is_segment_blocked((stub_x, start[1]), (stub_x, test_y), obstacles, padding):
                        continue

                    for t_x_off in x_stubs:
                        target_stub_x = end[0] + t_x_off * end_dir

                        route = [
                            start,
                            (stub_x, start[1]),
                            (stub_x, test_y),
                            (target_stub_x, test_y),
                            (target_stub_x, end[1]),
                            end
                        ]
                        if not is_route_blocked(route):
                            return route

            # C. Guaranteed-clear perimeter fallback
            return _perimeter_fallback(start, end, obstacles, padding, True)

        return [start, (lane_x, start[1]), (lane_x, end[1]), end]

    # 3. Vertical-dominant
    start = _bottom_port(source) if ty >= sy else _top_port(source)
    end = _top_port(target) if ty >= sy else _bottom_port(target)
    lane_y = (start[1] + end[1]) / 2.0

    if config.enable_edge_avoidance:
        # A. Try 3-segment Z-route with adaptive step
        for step in range(0, z_max_steps):
            for sign in [1, -1]:
                if step == 0 and sign == -1:
                    continue
                test_y = lane_y + step * z_step * sign
                route = [start, (start[0], test_y), (end[0], test_y), end]
                if not is_route_blocked(route):
                    return route

        # B. Try 5-segment X-detour with dynamic stub offsets
        start_dir = 1 if ty >= sy else -1
        end_dir = -1 if ty >= sy else 1

        x_offsets: List[float] = []
        x_step = max(20.0, config.column_gap * 0.3)
        for step in range(1, 150):
            x_offsets.append(step * x_step)
            x_offsets.append(-step * x_step)

        y_stubs = _dynamic_stub_offsets(obstacles, padding, "y")

        for x_off in x_offsets:
            test_x = start[0] + x_off
            for s_y_off in y_stubs:
                stub_y = start[1] + s_y_off * start_dir
                if _is_segment_blocked(start, (start[0], stub_y), obstacles, padding):
                    continue
                if _is_segment_blocked((start[0], stub_y), (test_x, stub_y), obstacles, padding):
                    continue

                for t_y_off in y_stubs:
                    target_stub_y = end[1] + t_y_off * end_dir

                    route = [
                        start,
                        (start[0], stub_y),
                        (test_x, stub_y),
                        (test_x, target_stub_y),
                        (end[0], target_stub_y),
                        end
                    ]
                    if not is_route_blocked(route):
                        return route

        # C. Guaranteed-clear perimeter fallback
        return _perimeter_fallback(start, end, obstacles, padding, False)

    return [start, (start[0], lane_y), (end[0], lane_y), end]


def _get_ancestors(gi: str, layout: LayoutModel) -> Set[str]:
    ancestors: Set[str] = set()
    curr: Optional[str] = gi
    while curr:
        node = layout.nodes.get(curr) or layout.groups.get(curr)
        if not node or not node.parent_gi:
            break
        ancestors.add(node.parent_gi)
        curr = node.parent_gi
    return ancestors


def _is_segment_blocked(
    p1: Point,
    p2: Point,
    obstacles: List[Tuple[float, float, float, float]],
    padding: float,
) -> bool:
    x1, y1 = p1
    x2, y2 = p2

    # Rectilinear only
    is_horizontal = abs(y1 - y2) < 0.01

    for ox, oy, ow, oh in obstacles:
        # Inflate obstacle by padding
        rx1, ry1 = ox - padding, oy - padding
        rx2, ry2 = ox + ow + padding, oy + oh + padding

        if is_horizontal:
            # y must be within [ry1, ry2]
            if ry1 <= y1 <= ry2:
                # x range [min(x1,x2), max(x1,x2)] must overlap [rx1, rx2]
                if max(x1, x2) >= rx1 and min(x1, x2) <= rx2:
                    return True
        else:
            # x must be within [rx1, rx2]
            if rx1 <= x1 <= rx2:
                # y range [min(y1,y2), max(y1,y2)] must overlap [ry1, ry2]
                if max(y1, y2) >= ry1 and min(y1, y2) <= ry2:
                    return True
    return False


def compute_native_layout(
    diagram: DiagramInput, config: NativeLayoutConfig | None = None
) -> LayoutModel:
    """Computes deterministic node and group geometry from the hierarchy first."""
    config = config or NativeLayoutConfig()
    nodes: Dict[str, LayoutNode] = {}
    groups: Dict[str, LayoutGroup] = {}
    size_cache: Dict[str, Tuple[float, float]] = {}

    def measure(node_id: str) -> Tuple[float, float]:
        if node_id in size_cache:
            return size_cache[node_id]

        node = diagram.nodes[node_id]
        if not node.is_directory:
            size_cache[node_id] = (_node_width(node.label, config), config.node_height)
            return size_cache[node_id]

        child_ids = diagram.children_by_parent.get(node_id, [])
        if not child_ids:
            width = _node_width(node.label, config)
            height = config.node_height + config.group_header_height
            size_cache[node_id] = (width, height)
            return size_cache[node_id]

        column_ids = _partition_children(child_ids, diagram, config)
        column_widths: List[float] = []
        column_heights: List[float] = []
        for col_idx, column in enumerate(column_ids):
            widths: List[float] = []
            heights: List[float] = []
            for child_id in column:
                child_width, child_height = measure(child_id)
                widths.append(child_width)
                heights.append(child_height)
            column_widths.append(max(widths) if widths else config.node_min_width)
            col_height = sum(heights) + config.row_gap * max(0, len(heights) - 1)
            if config.enable_hex_stagger and col_idx % 2 == 1:
                col_height += (
                    config.node_height + config.row_gap
                ) * config.hex_stagger_ratio
            column_heights.append(col_height)

        width = (
            config.group_padding_x * 2.0
            + sum(column_widths)
            + config.column_gap * max(0, len(column_widths) - 1)
        )
        height = (
            config.group_header_height
            + config.group_padding_y * 2.0
            + max(column_heights, default=config.node_height)
        )
        size_cache[node_id] = (width, height)
        return size_cache[node_id]

    def place(node_id: str, x: float, y: float) -> None:
        node = diagram.nodes[node_id]
        width, height = size_cache[node_id]
        if not node.is_directory:
            nodes[node_id] = LayoutNode(
                gi_str=node_id,
                x=x,
                y=y,
                width=width,
                height=height,
                label=node.label,
                is_directory=False,
                node_class=node.node_class,
                parent_gi=node.parent_gi,
                tier=node.tier,
                is_focus=node.is_focus,
            )
            return

        child_ids = diagram.children_by_parent.get(node_id, [])
        groups[node_id] = LayoutGroup(
            gi_str=node_id,
            child_ids=list(child_ids),
            x=x,
            y=y,
            width=width,
            height=height,
            label=node.label.replace("\n", " / "),
            parent_gi=node.parent_gi,
            is_focus=node.is_focus,
        )
        if not child_ids:
            return

        inner_x = x + config.group_padding_x
        inner_y = y + config.group_header_height + config.group_padding_y
        for col_idx, column in enumerate(
            _partition_children(child_ids, diagram, config)
        ):
            column_width = max(size_cache[child_id][0] for child_id in column)
            stagger_y = 0.0
            if config.enable_hex_stagger and col_idx % 2 == 1:
                stagger_y = (
                    config.node_height + config.row_gap
                ) * config.hex_stagger_ratio
            cursor_y = inner_y + stagger_y
            for child_id in column:
                child_width, child_height = size_cache[child_id]
                child_x = inner_x + (column_width - child_width) / 2.0
                place(child_id, child_x, cursor_y)
                cursor_y += child_height + config.row_gap
            inner_x += column_width + config.column_gap

    root_columns = _partition_children(diagram.root_ids, diagram, config)
    root_x = config.canvas_padding
    max_root_height = 0.0
    for col_idx, column in enumerate(root_columns):
        column_width = 0.0
        stagger_y = 0.0
        if config.enable_hex_stagger and col_idx % 2 == 1:
            stagger_y = (config.node_height + config.row_gap) * config.hex_stagger_ratio
        cursor_y = config.canvas_padding + stagger_y
        for root_id in column:
            measure(root_id)
            root_width, root_height = size_cache[root_id]
            column_width = max(column_width, root_width)
            place(root_id, root_x + (column_width - root_width) / 2.0, cursor_y)
            cursor_y += root_height + config.tier_gap
        max_root_height = max(max_root_height, cursor_y)
        root_x += column_width + config.column_gap

    resolved_edges = [
        LayoutEdge(edge.source_gi, edge.target_gi, edge.dep_char)
        for edge in diagram.edges
    ]
    canvas_width = max(root_x - config.column_gap + config.canvas_padding, 1.0)
    canvas_height = max(max_root_height + config.canvas_padding, 1.0)

    return LayoutModel(
        nodes=nodes,
        groups=groups,
        edges=resolved_edges,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )


def _partition_children(
    child_ids: List[str], diagram: DiagramInput, config: NativeLayoutConfig
) -> List[List[str]]:
    columns: Dict[str, List[str]] = defaultdict(list)
    for child_id in child_ids:
        columns[_module_key(diagram.nodes[child_id].key_string)].append(child_id)
    ordered_columns: List[List[str]] = []
    for module in sorted(columns):
        module_nodes = sorted(
            columns[module], key=lambda item: _sort_key(item, diagram)
        )
        num_nodes = len(module_nodes)
        if num_nodes > config.max_nodes_per_column:
            num_cols = math.ceil(num_nodes / config.max_nodes_per_column)
            nodes_per_col = math.ceil(num_nodes / num_cols)
            for i in range(0, num_nodes, nodes_per_col):
                ordered_columns.append(module_nodes[i : i + nodes_per_col])
        else:
            ordered_columns.append(module_nodes)
    return ordered_columns


def route_edges_and_pipes(
    layout: LayoutModel, config: NativeLayoutConfig | None = None
) -> LayoutModel:
    """Routes individual edges and fan-in pipes using file anchors."""
    config = config or NativeLayoutConfig()
    buckets: Dict[Tuple[str, str], List[LayoutEdge]] = defaultdict(list)
    for edge in layout.edges:
        buckets[(edge.target_gi, edge.dep_char)].append(edge)

    pipes: List[LayoutPipe] = []
    for (target_gi, dep_char), bucket in sorted(buckets.items()):
        target_anchor = _resolve_anchor(target_gi, layout)
        if target_anchor is None or len(bucket) < config.pipe_threshold:
            continue

        source_anchor_ids = [
            _resolve_anchor_id(edge.source_gi, layout) for edge in bucket
        ]
        source_anchor_ids = sorted(
            {anchor_id for anchor_id in source_anchor_ids if anchor_id is not None}
        )
        if len(source_anchor_ids) < config.pipe_threshold:
            continue

        pipe_id = (
            f"pipe-{_stable_token(target_gi + dep_char + ''.join(source_anchor_ids))}"
        )
        merge_point = _merge_point_for_sources(
            source_anchor_ids, target_anchor, layout, config
        )
        target_port = (
            _left_port(target_anchor)
            if merge_point[0] <= _center(target_anchor)[0]
            else _right_port(target_anchor)
        )
        lane_x = (merge_point[0] + target_port[0]) / 2.0

        if config.enable_edge_avoidance:
            # Optimize lane_x for pipes
            source_nodes = [layout.nodes[sid] for sid in source_anchor_ids]
            # Collect ALL ancestors for ALL source nodes
            passable = {target_anchor.gi_str}
            for snode in source_nodes:
                passable.add(snode.gi_str)
                passable.update(_get_ancestors(snode.gi_str, layout))
            passable.update(_get_ancestors(target_anchor.gi_str, layout))

            padding = config.edge_avoidance_padding

            # Simple scan for pipe lane
            obstacles: List[Tuple[float, float, float, float]] = []
            for node in layout.nodes.values():
                if node.gi_str not in passable:
                    obstacles.append((node.x, node.y, node.width, node.height))
            for group in layout.groups.values():
                if group.gi_str not in passable and group.gi_str != "__root__":
                    obstacles.append((group.x, group.y, group.width, group.height))

            def is_route_blocked(route: List[Point]) -> bool:
                for i in range(len(route) - 1):
                    if _is_segment_blocked(route[i], route[i+1], obstacles, padding):
                        return True
                return False

            found = False
            route: List[Point] = []

            z_step = max(10.0, config.column_gap * 0.2)

            # A. Try 3-segment Z-route with adaptive step
            for step in range(0, 200):
                for sign in [1, -1]:
                    if step == 0 and sign == -1:
                        continue
                    test_x = lane_x + step * z_step * sign
                    test_route = [
                        merge_point,
                        (test_x, merge_point[1]),
                        (test_x, target_port[1]),
                        target_port,
                    ]
                    if not is_route_blocked(test_route):
                        route = test_route
                        found = True
                        break
                if found:
                    break

            if not found:
                # B. Try 5-segment Y-detour with dynamic stubs
                y_offsets: List[float] = []
                y_step = max(20.0, config.row_gap)
                for step in range(1, 150):
                    y_offsets.append(step * y_step)
                    y_offsets.append(-step * y_step)

                x_stubs = _dynamic_stub_offsets(obstacles, padding, "x")

                start_dir = 1 if target_port[0] >= merge_point[0] else -1
                end_dir = -1 if target_port[0] >= merge_point[0] else 1

                for y_off in y_offsets:
                    test_y = merge_point[1] + y_off
                    for s_x_off in x_stubs:
                        stub_x = merge_point[0] + s_x_off * start_dir
                        if _is_segment_blocked(merge_point, (stub_x, merge_point[1]), obstacles, padding):
                            continue
                        if _is_segment_blocked((stub_x, merge_point[1]), (stub_x, test_y), obstacles, padding):
                            continue

                        for t_x_off in x_stubs:
                            target_stub_x = target_port[0] + t_x_off * end_dir

                            test_route = [
                                merge_point,
                                (stub_x, merge_point[1]),
                                (stub_x, test_y),
                                (target_stub_x, test_y),
                                (target_stub_x, target_port[1]),
                                target_port
                            ]
                            if not is_route_blocked(test_route):
                                route = test_route
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

            if not found:
                # C. Guaranteed-clear perimeter fallback
                route = _perimeter_fallback(
                    merge_point, target_port, obstacles, padding, True
                )
        else:
            route = [
                merge_point,
                (lane_x, merge_point[1]),
                (lane_x, target_port[1]),
                target_port,
            ]

        badge_pos = route[len(route) // 2]

        # 4. Generate feeder routes with avoidance
        feeder_routes: List[List[Point]] = []
        for source_anchor_id in source_anchor_ids:
            source_anchor = layout.nodes[source_anchor_id]
            # Use the same Z-route logic for feeders
            # We treat the merge_point as a "virtual node" for the router
            virtual_target = LayoutNode(
                gi_str="virtual-merge",
                x=merge_point[0] - 1,
                y=merge_point[1] - 1,
                width=2,
                height=2,
                label="",
                is_directory=False,
                node_class="",
                parent_gi="",
                tier=0,
                is_focus=False,
            )
            feeder_route = _route_between(source_anchor, virtual_target, layout, config)
            # Ensure the last point is exactly the merge_point
            feeder_route[-1] = merge_point
            feeder_routes.append(feeder_route)

        pipes.append(
            LayoutPipe(
                pipe_id=pipe_id,
                source_gis=source_anchor_ids,
                target_gi=target_anchor.gi_str,
                dep_char=dep_char,
                route_points=route,
                feeder_routes=feeder_routes,
                stroke_width=pipe_stroke_width(len(source_anchor_ids), config),
                count_badge_pos=badge_pos,
                count=len(source_anchor_ids),
                color=pipe_color(len(source_anchor_ids)),
            )
        )

        pipe_members = set(source_anchor_ids)
        for edge in bucket:
            anchor_id = _resolve_anchor_id(edge.source_gi, layout)
            if anchor_id in pipe_members:
                edge.is_pipe_member = True
                edge.pipe_id = pipe_id

    for edge in layout.edges:
        if edge.is_pipe_member:
            continue
        source_anchor = _resolve_anchor(edge.source_gi, layout)
        target_anchor = _resolve_anchor(edge.target_gi, layout)
        if source_anchor is not None and target_anchor is not None:
            edge.route_points = _route_between(
                source_anchor, target_anchor, layout, config
            )

    layout.pipes = pipes
    return layout


def _merge_point_for_sources(
    source_anchor_ids: List[str],
    target_anchor: LayoutNode,
    layout: LayoutModel,
    config: NativeLayoutConfig,
) -> Point:
    source_nodes = [
        layout.nodes[source_anchor_id] for source_anchor_id in source_anchor_ids
    ]
    if not source_nodes:
        return (config.canvas_padding, config.canvas_padding)

    source_parent_ids = {node.parent_gi for node in source_nodes}
    if len(source_parent_ids) == 1:
        parent_id = next(iter(source_parent_ids))
        group = layout.groups.get(parent_id) if parent_id else None
        if group is not None:
            avg_y = sum(_center(node)[1] for node in source_nodes) / len(source_nodes)
            if group.x + group.width <= target_anchor.x:
                return (group.x + group.width + config.feeder_stub_length, avg_y)
            return (group.x - config.feeder_stub_length, avg_y)

    if _center(target_anchor)[0] >= sum(
        _center(node)[0] for node in source_nodes
    ) / len(source_nodes):
        max_x = max(node.x + node.width for node in source_nodes)
        return (
            max_x + config.feeder_stub_length,
            sum(_center(node)[1] for node in source_nodes) / len(source_nodes),
        )

    min_x = min(node.x for node in source_nodes)
    return (
        min_x - config.feeder_stub_length,
        sum(_center(node)[1] for node in source_nodes) / len(source_nodes),
    )


def _resolve_anchor_id(gi: str, layout: LayoutModel) -> Optional[str]:
    if gi in layout.nodes:
        return gi
    if gi not in layout.groups:
        return None
    file_ids = _descendant_file_ids(gi, layout)
    if not file_ids:
        return None
    return sorted(file_ids)[0]


def _resolve_anchor(gi: str, layout: LayoutModel) -> Optional[LayoutNode]:
    anchor_id = _resolve_anchor_id(gi, layout)
    return layout.nodes.get(anchor_id) if anchor_id is not None else None


def _descendant_file_ids(group_id: str, layout: LayoutModel) -> List[str]:
    group = layout.groups.get(group_id)
    if group is None:
        return []
    file_ids: List[str] = []
    for child_id in group.child_ids:
        if child_id in layout.nodes:
            file_ids.append(child_id)
        elif child_id in layout.groups:
            file_ids.extend(_descendant_file_ids(child_id, layout))
    return file_ids


def pipe_stroke_width(count: int, config: NativeLayoutConfig | None = None) -> float:
    config = config or NativeLayoutConfig()
    if count <= 1:
        return config.pipe_base_width
    raw = config.pipe_base_width + config.pipe_log_scale * math.log2(count)
    return min(raw, config.pipe_max_width)


def pipe_color(count: int) -> str:
    stops = [
        (1, (143, 209, 106)),
        (4, (220, 211, 83)),
        (9, (236, 163, 67)),
        (21, (220, 88, 54)),
        (65, (160, 34, 42)),
    ]
    if count <= stops[0][0]:
        return _rgb_hex(stops[0][1])
    for idx in range(1, len(stops)):
        prev_count, prev_rgb = stops[idx - 1]
        next_count, next_rgb = stops[idx]
        if count <= next_count:
            denom = math.log2(next_count) - math.log2(prev_count)
            ratio = (
                1.0
                if denom == 0
                else (math.log2(count) - math.log2(prev_count)) / denom
            )
            rgb = (
                round(prev_rgb[0] + (next_rgb[0] - prev_rgb[0]) * ratio),
                round(prev_rgb[1] + (next_rgb[1] - prev_rgb[1]) * ratio),
                round(prev_rgb[2] + (next_rgb[2] - prev_rgb[2]) * ratio),
            )
            return _rgb_hex(rgb)
    return _rgb_hex(stops[-1][1])


def _rgb_hex(rgb: Tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _stable_token(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def layout_cache_key(diagram: DiagramInput, config: NativeLayoutConfig) -> str:
    payload = {
        "diagram": diagram.to_json_dict(),
        "config": config.to_json_dict(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
