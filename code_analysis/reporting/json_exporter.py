"""
JSON exporter for CRCT report generator.
Generates machine-readable JSON from enriched issue data.
"""

import json
import os
from typing import Any, Dict, List, Optional


def export_json(
    issues: List[Dict[str, Any]],
    unused: List[Dict[str, Any]],
    output_path: str,
    comment_index: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> None:
    """Generate the machine-readable JSON report."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report: Dict[str, Any] = {"issues": issues, "unused": unused}
        if comment_index is not None:
            report["comment_index"] = comment_index

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[report] Failed writing JSON report: {e}")
