"""
JSON exporter for CRCT report generator.
Generates machine-readable JSON from enriched issue data.
"""

import json
import os
from typing import Any, Dict, List


def export_json(
    issues: List[Dict[str, Any]], unused: List[Dict[str, Any]], output_path: str
) -> None:
    """Generate the machine-readable JSON report."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"issues": issues, "unused": unused}, f, indent=2, ensure_ascii=False
            )
    except Exception as e:
        print(f"[report] Failed writing JSON report: {e}")
