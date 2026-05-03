"""
Enhanced CRCT report_generator (Orchestrator)
==============================================

Decomposed monolithic script into sub-modules:
- scanner.static_engine: Regex and tree-sitter scanning.
- scanner.runtime_bridge: RuntimeIndex and metadata enrichment.
- reporting.markdown_formatter: Markdown report generation.
- reporting.json_exporter: JSON report generation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Project-root bootstrap
# ---------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cline_utils.dependency_system.utils import path_utils  # noqa: E402
from cline_utils.dependency_system.utils.config_manager import (
    ConfigManager,
)  # noqa: E402

from code_analysis.scanner.static_engine import (
    EXTENSIONS,
    PYRIGHT_OUTPUT,
    get_unused_items,
    scan_file,
)
from code_analysis.scanner.runtime_bridge import (
    RuntimeIndex,
    enrich_issue,
    load_runtime_data,
    maybe_run_runtime_inspector,
    runtime_only_findings,
)
from code_analysis.reporting.markdown_formatter import format_markdown
from code_analysis.reporting.json_exporter import export_json

config = ConfigManager()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_FILE = os.path.join(project_root, "code_analysis", "issues_report.md")
OUTPUT_JSON_FILE = os.path.join(project_root, "code_analysis", "issues_report.json")


def main():
    all_issues: List[Dict[str, Any]] = []

    code_roots = config.get_code_root_directories()
    excluded_paths = config.get_excluded_paths()

    # ---- pyright ----
    try:
        print("Running pyright for unused item analysis...")
        # Redirect output to PYRIGHT_OUTPUT in the current working directory or project root
        pyright_output_path = os.path.join(project_root, PYRIGHT_OUTPUT)
        with open(pyright_output_path, "w") as f:
            result = subprocess.run(
                ["pyright", "--outputjson"],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=project_root,
            )
        if result.returncode == 0:
            print("Pyright analysis completed successfully.")
        else:
            print(
                f"Pyright completed with warnings/errors (exit code {result.returncode}). "
                f"Output file generated."
            )
    except Exception as e:
        print(f"Warning: Unexpected error running pyright: {e}")

    # ---- runtime data ----
    maybe_run_runtime_inspector(project_root)
    runtime_data = load_runtime_data(project_root)
    idx = RuntimeIndex(runtime_data)

    # ---- static walk ----
    print(f"Scanning code roots: {code_roots}")
    for root_dir_name in code_roots:
        # Code roots from config are often relative to project root
        start_dir = os.path.join(project_root, root_dir_name)
        if not os.path.exists(start_dir):
            # Try as absolute path
            start_dir = root_dir_name
            if not os.path.exists(start_dir):
                print(f"Warning: Code root {start_dir} does not exist. Skipping.")
                continue

        for root, dirs, files in os.walk(start_dir):
            # Filter directories in-place
            dirs[:] = [
                d
                for d in dirs
                if not path_utils.is_path_excluded(
                    os.path.join(root, d), excluded_paths
                )
            ]
            for file in files:
                filepath = os.path.join(root, file)
                if path_utils.is_path_excluded(filepath, excluded_paths):
                    continue
                if Path(file).suffix not in EXTENSIONS:
                    continue
                all_issues.extend(scan_file(filepath))

    # ---- runtime-only findings ----
    if runtime_data:
        rt_findings = runtime_only_findings(idx)
        print(f"[runtime] Added {len(rt_findings)} runtime-only finding(s).")
        all_issues.extend(rt_findings)

    # ---- enrich every issue with runtime context ----
    if runtime_data:
        enriched: List[Dict[str, Any]] = []
        for it in all_issues:
            it_enriched = enrich_issue(it, idx)
            if not it_enriched.get("_suppress"):
                enriched.append(it_enriched)
        all_issues = enriched
    else:
        # Still attach a basic severity if no runtime data
        from code_analysis.scanner.runtime_bridge import score_severity

        for it in all_issues:
            it.setdefault("context", {"severity": score_severity(it, {})})

    # ---- de-dup: same (file, line, subtype) collapses, keep richest ctx ----
    seen: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for it in all_issues:
        key = (it.get("file", ""), it.get("line", 0), it.get("subtype", ""))
        cur = seen.get(key)
        if cur is None or len((it.get("context") or {})) > len(
            (cur.get("context") or {})
        ):
            seen[key] = it
    all_issues = list(seen.values())

    unused_items = get_unused_items()

    # ---- Generate Reports ----
    export_json(all_issues, unused_items, OUTPUT_JSON_FILE)
    format_markdown(all_issues, unused_items, OUTPUT_FILE)

    print(f"Report generated at {OUTPUT_FILE}")
    print(f"JSON report at {OUTPUT_JSON_FILE}")


if __name__ == "__main__":
    main()
