"""
Worklog exporter for CRCT report generator.
Generates a structured strategy work pool ledger for WIP-tracked items.
"""

import os
from typing import Any, Dict, List, Optional


def export_worklog(
    worklog_items: List[Dict[str, Any]],
    project_root: str,
    output_path: str,
) -> None:
    """Generate the strategy work pool markdown ledger for WIP items.

    WIP-tracked comments represent reviewed work-in-progress markers that
    have been deliberately placed by developers. They are curated separately
    from the technical debt backlog for Strategy phase consumption.

    Args:
        worklog_items: List of issue dicts where type == "Worklog"
        project_root: Absolute path to the project root (for relpath)
        output_path: Full path to the output file
    """
    try:
        md: List[str] = []
        md.append("# Strategy Work Pool")
        md.append("")
        md.append(
            "_Curated ledger for WIP-tracked items reviewed and marked for "
            "Strategy phase consumption._"
        )
        md.append("")
        md.append("| # | File | Line | Content | Module |")
        md.append("|---|------|------|---------|--------|")

        if not worklog_items:
            md.append("| | _No WIP items found._ | | | |")
        else:
            for idx, item in enumerate(worklog_items, start=1):
                file_path = item.get("file", "")
                try:
                    rel_file = os.path.relpath(file_path, project_root).replace(
                        "\\", "/"
                    )
                except ValueError:
                    rel_file = file_path.replace("\\", "/")

                line = item.get("line", 0)
                content = item.get("content", "").strip().replace("|", "\\|")

                # Derive module name from the first directory component of the
                # relative path (e.g. "src/agents/foo.py" -> "agents")
                parts = [p for p in rel_file.split("/") if p]
                module = parts[1] if len(parts) > 2 else (parts[0] if parts else "?")

                md.append(f"| {idx} | `{rel_file}` | {line} | `{content}` | {module} |")

        md.append("")
        md.append("---")
        md.append(
            "_Generated automatically by CRCT report generator. "
            f"Updated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}_"
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        print(f"[worklog] Generated strategy work pool at {output_path}")

    except Exception as e:
        print(f"[worklog] Failed generating strategy work pool: {e}")
