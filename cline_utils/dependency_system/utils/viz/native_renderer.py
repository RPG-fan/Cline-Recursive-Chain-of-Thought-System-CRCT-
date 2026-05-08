"""Plain SVG renderer for native dependency layouts."""

from __future__ import annotations

import html
import os
from typing import Iterable, List

from .models import LayoutModel, NativeLayoutConfig, Point


def render_svg_pass(
    layout: LayoutModel, config: NativeLayoutConfig | None = None
) -> str:
    """Renders a staged native SVG string from fixed layout geometry."""
    config = config or NativeLayoutConfig()
    width = max(layout.canvas_width, 1.0)
    height = max(layout.canvas_height, 1.0)
    parts: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.1f}" '
            f'height="{height:.1f}" viewBox="0 0 {width:.1f} {height:.1f}" '
            'role="img" aria-label="Native dependency diagram">'
        ),
        "<defs>",
        (
            '<filter id="native-pipe-halo" x="-20%" y="-20%" width="140%" height="140%">'
            '<feMorphology operator="dilate" radius="1.5" in="SourceAlpha" result="halo"/>'
            '<feFlood flood-color="#ffffff" flood-opacity="0.72" result="halo-color"/>'
            '<feComposite in="halo-color" in2="halo" operator="in" result="halo-layer"/>'
            '<feMerge><feMergeNode in="halo-layer"/><feMergeNode in="SourceGraphic"/></feMerge>'
            "</filter>"
        ),
        "</defs>",
    ]
    if config.background != "transparent":
        parts.append(
            f'<rect width="100%" height="100%" fill="{html.escape(config.background)}"/>'
        )

    parts.append('<g id="groups" data-layer="groups">')
    for group in sorted(layout.groups.values(), key=lambda item: (item.y, item.x, item.gi_str)):
        stroke = config.focus_stroke if group.is_focus else config.group_stroke
        parts.append(
            f'<rect id="{_id("group", group.gi_str)}" data-gi="{_attr(group.gi_str)}" '
            f'x="{group.x:.1f}" y="{group.y:.1f}" width="{group.width:.1f}" '
            f'height="{group.height:.1f}" rx="6" fill="{config.group_fill}" '
            f'stroke="{stroke}" stroke-width="2"/>'
        )
        parts.append(
            f'<text data-gi="{_attr(group.gi_str)}" x="{group.x + 10:.1f}" '
            f'y="{group.y + 18:.1f}" fill="{config.group_text_color}" '
            'font-family="Arial, sans-serif" font-size="12">'
            f"{_text(group.label)}</text>"
        )
    parts.append("</g>")

    parts.append('<g id="nodes" data-layer="nodes">')
    for node in sorted(layout.nodes.values(), key=lambda item: (item.y, item.x, item.gi_str)):
        fill = config.doc_fill if node.node_class == "doc" else config.node_fill
        stroke = config.focus_stroke if node.is_focus else "#59636e"
        parts.append(
            f'<rect id="{_id("node", node.gi_str)}" data-gi="{_attr(node.gi_str)}" '
            f'data-node-class="{_attr(node.node_class)}" x="{node.x:.1f}" y="{node.y:.1f}" '
            f'width="{node.width:.1f}" height="{node.height:.1f}" rx="5" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
    parts.append("</g>")

    parts.append('<g id="labels" data-layer="labels">')
    for node in sorted(layout.nodes.values(), key=lambda item: (item.y, item.x, item.gi_str)):
        lines = node.label.splitlines() or [node.gi_str]
        first_y = node.y + 16.0 if len(lines) > 1 else node.y + node.height / 2.0 + 4.0
        text_parts = [
            f'<text data-gi="{_attr(node.gi_str)}" x="{node.x + node.width / 2.0:.1f}" '
            f'y="{first_y:.1f}" text-anchor="middle" fill="{config.text_color}" '
            'font-family="Arial, sans-serif" font-size="11">'
        ]
        for idx, line in enumerate(lines):
            dy = 0 if idx == 0 else 13
            text_parts.append(f'<tspan x="{node.x + node.width / 2.0:.1f}" dy="{dy}">{_text(line)}</tspan>')
        text_parts.append("</text>")
        parts.append("".join(text_parts))
    parts.append("</g>")

    if config.render_stage >= 2:
        parts.append('<g id="pipe-spines" data-layer="pipe-spines">')
        for pipe in layout.pipes:
            parts.append(
                f'<path id="{_attr(pipe.pipe_id)}" data-pipe-id="{_attr(pipe.pipe_id)}" '
                f'data-target-gi="{_attr(pipe.target_gi)}" data-dep-char="{_attr(pipe.dep_char)}" '
                f'data-count="{pipe.count}" d="{_path(pipe.route_points)}" fill="none" '
                f'stroke="{pipe.color}" stroke-width="{pipe.stroke_width:.1f}" '
                'stroke-linecap="round" stroke-linejoin="round" filter="url(#native-pipe-halo)"/>'
            )
        parts.append("</g>")

    if config.render_stage >= 3:
        parts.append('<g id="pipe-feeders" data-layer="pipe-feeders">')
        for pipe in layout.pipes:
            for idx, route in enumerate(pipe.feeder_routes):
                source_gi = pipe.source_gis[idx] if idx < len(pipe.source_gis) else ""
                parts.append(
                    f'<path data-pipe-id="{_attr(pipe.pipe_id)}" data-source-gi="{_attr(source_gi)}" '
                    f'data-target-gi="{_attr(pipe.target_gi)}" data-dep-char="{_attr(pipe.dep_char)}" '
                    f'd="{_path(route)}" fill="none" stroke="{pipe.color}" stroke-width="1.4" '
                    'stroke-linecap="round" stroke-linejoin="round" opacity="0.75"/>'
                )
        parts.append("</g>")

    if config.render_stage >= 4:
        parts.append('<g id="edges-direct" data-layer="edges-direct">')
        for edge in layout.edges:
            if edge.is_pipe_member or _is_aux(edge.dep_char):
                continue
            parts.append(_edge_path(edge.route_points, edge.source_gi, edge.target_gi, edge.dep_char, config))
        parts.append("</g>")

    if config.render_stage >= 5:
        parts.append('<g id="edges-aux" data-layer="edges-aux">')
        for edge in layout.edges:
            if edge.is_pipe_member or not _is_aux(edge.dep_char):
                continue
            parts.append(_edge_path(edge.route_points, edge.source_gi, edge.target_gi, edge.dep_char, config))
        parts.append("</g>")

    parts.append('<g id="badges" data-layer="badges">')
    if config.render_stage >= 2:
        for pipe in layout.pipes:
            label = f"x{pipe.count} {pipe.dep_char}"
            x, y = pipe.count_badge_pos
            badge_width = max(36, len(label) * 7 + 10)
            parts.append(
                f'<g data-pipe-id="{_attr(pipe.pipe_id)}" data-count="{pipe.count}">'
                f'<rect x="{x - badge_width / 2.0:.1f}" y="{y - 10:.1f}" '
                f'width="{badge_width:.1f}" height="20" rx="5" fill="#1d242c" '
                'stroke="#ffffff" stroke-opacity="0.7"/>'
                f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
                'fill="#ffffff" font-family="Arial, sans-serif" font-size="11">'
                f"{_text(label)}</text></g>"
            )
    parts.append("</g>")

    parts.append("</svg>")
    return "\n".join(parts)


def write_native_svg(
    layout: LayoutModel,
    output_file_path: str,
    config: NativeLayoutConfig | None = None,
) -> None:
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as svg_file:
        svg_file.write(render_svg_pass(layout, config))


def _edge_path(
    points: Iterable[Point],
    source_gi: str,
    target_gi: str,
    dep_char: str,
    config: NativeLayoutConfig,
) -> str:
    dash = ' stroke-dasharray="6 4"' if dep_char in {"d", "p", "s", "S"} else ""
    opacity = "0.48" if dep_char in {"p", "s"} else "0.82"
    return (
        f'<path data-source-gi="{_attr(source_gi)}" data-target-gi="{_attr(target_gi)}" '
        f'data-dep-char="{_attr(dep_char)}" d="{_path(points)}" fill="none" '
        f'stroke="{config.edge_stroke}" stroke-width="1.4" opacity="{opacity}" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash}/>'
    )


def _path(points: Iterable[Point]) -> str:
    point_list = list(points)
    if not point_list:
        return ""
    head = point_list[0]
    tail = " ".join(f"L {x:.1f} {y:.1f}" for x, y in point_list[1:])
    return f"M {head[0]:.1f} {head[1]:.1f} {tail}".strip()


def _is_aux(dep_char: str) -> bool:
    return dep_char in {"d", "p", "s", "S"}


def _id(prefix: str, raw: str) -> str:
    return f"{prefix}-{''.join(ch if ch.isalnum() else '-' for ch in raw)}"


def _attr(value: str) -> str:
    return html.escape(str(value), quote=True)


def _text(value: str) -> str:
    return html.escape(str(value), quote=False)
