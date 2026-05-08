"""JSON serialization helpers for native diagram layout caches."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .models import DiagramInput, LayoutModel, NativeLayoutConfig
from .native_layout import layout_cache_key


def native_cache_payload(
    diagram: DiagramInput, layout: LayoutModel, config: NativeLayoutConfig
) -> Dict[str, Any]:
    return {
        "cache_key": layout_cache_key(diagram, config),
        "diagram": diagram.to_json_dict(),
        "layout": layout.to_json_dict(),
        "config": config.to_json_dict(),
    }


def write_native_layout_cache(
    path: str, diagram: DiagramInput, layout: LayoutModel, config: NativeLayoutConfig
) -> str:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    payload = native_cache_payload(diagram, layout, config)
    with open(path, "w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, indent=2, sort_keys=True)
    return str(payload["cache_key"])


def read_native_layout_cache(path: str, cache_key: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as cache_file:
        payload = json.load(cache_file)
    if payload.get("cache_key") != cache_key:
        return None
    return payload
