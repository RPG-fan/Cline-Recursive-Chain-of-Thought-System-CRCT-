from cline_utils.dependency_system.utils.viz.models import (
    DiagramInput,
    DiagramNode,
    DiagramEdge,
    NativeLayoutConfig,
)
from cline_utils.dependency_system.utils.viz.native_layout import (
    compute_native_layout,
    route_edges_and_pipes,
)


def test_edge_avoidance_rerouting():
    # Setup:
    # A (0,0) -> B (400, 0)
    # C (200, -20) is in the middle
    nodes = {
        "A": DiagramNode("A", "1A1", "a.py", None, 0, False, "Node A", "default"),
        "B": DiagramNode("B", "1A2", "b.py", None, 0, False, "Node B", "default"),
        "C": DiagramNode("C", "1B1", "c.py", None, 1, False, "Node C", "default"),
    }
    edges = [DiagramEdge("A", "B", "R")]
    diagram = DiagramInput(
        nodes=nodes,
        edges=edges,
        children_by_parent={None: ["A", "B", "C"]},
        root_ids=["A", "B", "C"],
    )

    config = NativeLayoutConfig(
        enable_edge_avoidance=True,
        edge_avoidance_padding=10.0,
        column_gap=200.0,
        tier_gap=200.0,
    )

    # Run layout
    layout = compute_native_layout(diagram, config)
    layout = route_edges_and_pipes(layout, config)

    edge = layout.edges[0]
    route = edge.route_points

    # Default lane_x would be middle of A and B
    node_a = layout.nodes["A"]
    node_b = layout.nodes["B"]
    expected_default_lane_x = (node_a.x + node_a.width + node_b.x) / 2.0

    # Find lane_x in route (points 1 and 2 should have same X)
    actual_lane_x = route[1][0]

    # Node C should be at some X that overlaps expected_default_lane_x
    node_c = layout.nodes["C"]
    print(f"Node A: {node_a.x}, {node_a.y} W:{node_a.width}")
    print(f"Node B: {node_b.x}, {node_b.y} W:{node_b.width}")
    print(f"Node C: {node_c.x}, {node_c.y} W:{node_c.width}")
    print(f"Default Lane X: {expected_default_lane_x}")
    print(f"Actual Lane X: {actual_lane_x}")

    # Assert that actual_lane_x is NOT within Node C's X range (plus padding)
    c_left = node_c.x - config.edge_avoidance_padding
    c_right = node_c.x + node_c.width + config.edge_avoidance_padding

    # Wait, the vertical segment at lane_x only blocks if it overlaps Node C's Y range too.
    # In my test, I should ensure they overlap in Y.

    assert not (
        c_left <= actual_lane_x <= c_right
    ), f"Lane X {actual_lane_x} should avoid Node C [{c_left}, {c_right}]"


def test_selective_routing_group_avoidance():
    # A -> B
    # G3 is a sibling group in the way
    nodes = {
        "A": DiagramNode("A", "1A1", "a.py", "G1", 0, False, "A", "default"),
        "B": DiagramNode("B", "1A2", "b.py", "G2", 0, False, "B", "default"),
        "C": DiagramNode("C", "1A3", "c.py", "G3", 0, False, "C", "default"),
    }
    # Hierarchy
    # root -> G1, G2, G3
    # G1 -> A
    # G2 -> B
    # G3 -> C
    children = {None: ["G1", "G2", "G3"], "G1": ["A"], "G2": ["B"], "G3": ["C"]}
    # Add groups to nodes as directories
    nodes["G1"] = DiagramNode("G1", "1A", "g1", None, 0, True, "G1", "default")
    nodes["G2"] = DiagramNode("G2", "2A", "g2", None, 0, True, "G2", "default")
    nodes["G3"] = DiagramNode("G3", "1B", "g3", None, 1, True, "G3", "default")

    edges = [DiagramEdge("A", "B", "R")]
    diagram = DiagramInput(
        nodes=nodes,
        edges=edges,
        children_by_parent=children,
        root_ids=["G1", "G2", "G3"],
    )

    config = NativeLayoutConfig(enable_edge_avoidance=True)
    layout = compute_native_layout(diagram, config)
    layout = route_edges_and_pipes(layout, config)

    edge = layout.edges[0]
    route = edge.route_points

    node_g3 = layout.groups["G3"]
    lane_x = route[1][0]

    g3_left = node_g3.x - config.edge_avoidance_padding
    g3_right = node_g3.x + node_g3.width + config.edge_avoidance_padding

    print(f"G3: {node_g3.x} to {node_g3.x + node_g3.width}")
    print(f"Lane X: {lane_x}")

    assert not (g3_left <= lane_x <= g3_right), f"Lane X {lane_x} should avoid Group G3"


def test_pipe_routing_avoidance():
    # 3 edges A->B, D->B, E->B to trigger pipe
    # C is an obstacle in the middle
    nodes = {
        "A": DiagramNode("A", "1A1", "a.py", None, 0, False, "A", "default"),
        "B": DiagramNode("B", "1A2", "b.py", None, 0, False, "B", "default"),
        "C": DiagramNode("C", "1B1", "c.py", None, 1, False, "C", "default"),
        "D": DiagramNode("D", "1A3", "d.py", None, 0, False, "D", "default"),
        "E": DiagramNode("E", "1A4", "e.py", None, 0, False, "E", "default"),
    }
    edges = [
        DiagramEdge("A", "B", "R"),
        DiagramEdge("D", "B", "R"),
        DiagramEdge("E", "B", "R"),
    ]
    diagram = DiagramInput(
        nodes=nodes,
        edges=edges,
        children_by_parent={None: ["A", "B", "C", "D", "E"]},
        root_ids=["A", "B", "C", "D", "E"],
    )

    config = NativeLayoutConfig(enable_edge_avoidance=True, pipe_threshold=3)
    layout = compute_native_layout(diagram, config)
    layout = route_edges_and_pipes(layout, config)

    assert len(layout.pipes) == 1
    pipe = layout.pipes[0]
    lane_x = pipe.route_points[1][0]

    node_c = layout.nodes["C"]
    c_left = node_c.x - config.edge_avoidance_padding
    c_right = node_c.x + node_c.width + config.edge_avoidance_padding

    print(f"C: {node_c.x} to {node_c.x + node_c.width}")
    print(f"Pipe Lane X: {lane_x}")

    assert not (
        c_left <= lane_x <= c_right
    ), f"Pipe Lane X {lane_x} should avoid Node C"
