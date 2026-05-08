"""
Markdown report formatter for CRCT.
Generates human-readable Markdown from enriched issue data.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, cast

_SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def format_markdown(
    issues: List[Dict[str, Any]],
    unused: List[Dict[str, Any]],
    output_path: str,
    comment_index: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> None:
    """Generate the human-readable Markdown report."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Code Analysis Issues Report\n\n")
        f.write("_Static analysis enhanced with runtime inspector data._\n\n")

        # Summary
        by_sev: Dict[str, int] = {}
        for s in _SEV_ORDER:
            by_sev[s] = 0

        for it in issues:
            it_ctx: Dict[str, Any] = it.get("context") or {}
            sev = it_ctx.get("severity", "low")
            by_sev[sev] += 1

        f.write("## Summary\n")
        f.write(f"- Total issues: **{len(issues)}**\n")
        for s in ("critical", "high", "medium", "low"):
            if by_sev.get(s):
                f.write(f"- {s.title()}: {by_sev[s]}\n")
        f.write(f"- Unused items (pyright): {len(unused)}\n\n")

        f.write("## Incomplete & Improper Items\n")
        if not issues:
            f.write("No incomplete items found.\n")
        else:

            def _sort_key(x: Dict[str, Any]) -> Tuple[int, str, int]:
                x_ctx = cast(Dict[str, Any], x.get("context") or {})
                x_sev = cast(str, x_ctx.get("severity", "low"))
                return (
                    _SEV_ORDER.get(x_sev, 9),
                    cast(str, x.get("file", "")),
                    cast(int, x.get("line", 0)),
                )

            issues_sorted = sorted(issues, key=_sort_key)
            for it in issues_sorted:
                ctx: Dict[str, Any] = it.get("context") or {}
                sev = ctx.get("severity", "low")
                f.write(
                    f"- **[{sev.upper()}] {it.get('subtype','?')}** "
                    f"in `{it.get('file','?')}:{it.get('line','?')}`\n"
                )
                content = it.get("content", "")
                if content:
                    f.write(f"  ```\n  {content}\n  ```\n")

                owning: Optional[Dict[str, Any]] = ctx.get("owning_symbol")
                if owning:
                    sig: str = owning.get("signature") or ""
                    f.write(
                        f"  - **Owning symbol**: `{owning.get('qualname')}`{sig} "
                        f"({owning.get('kind')})\n"
                    )
                if ctx.get("type_annotations"):
                    f.write(f"  - **Types**: `{ctx['type_annotations']}`\n")
                if ctx.get("decorators"):
                    f.write(f"  - **Decorators**: {ctx['decorators']}\n")
                if ctx.get("inheritance"):
                    inh = ctx["inheritance"]
                    if inh.get("bases") or inh.get("mro"):
                        f.write(
                            f"  - **Inheritance**: bases={inh.get('bases', [])} "
                            f"mro={inh.get('mro', [])}\n"
                        )
                if ctx.get("scope_references"):
                    sr: Dict[str, Any] = ctx["scope_references"]
                    g_list: List[str] = sr.get("globals") or []
                    nl_list: List[str] = sr.get("nonlocals") or []
                    g = g_list[:10]
                    nl = nl_list[:10]
                    if g or nl:
                        f.write(f"  - **Scope refs**: globals={g} nonlocals={nl}\n")
                if ctx.get("closure_dependencies"):
                    f.write(f"  - **Closure deps**: {ctx['closure_dependencies']}\n")
                la: Dict[str, Any] = cast(Dict[str, Any], ctx.get("linked_areas") or {})
                callers_attr = la.get("callers")
                if callers_attr:
                    callers_list: List[str] = cast(List[str], callers_attr)
                    f.write(
                        f"  - **Linked areas** ({la.get('caller_count')} files): "
                        + ", ".join(f"`{c}`" for c in callers_list)
                        + "\n"
                    )
                if ctx.get("exported") is True:
                    f.write("  - **Exported**: yes (in `__all__`)\n")

        f.write("\n## Unused Items\n")
        if unused:
            for item in unused:
                f.write(f"- **{item['subtype']}** in `{item['file']}:{item['line']}`\n")
                f.write(f"  > {item['content']}\n")
        else:
            f.write("No unused items found (or pyright output missing).\n")

        # ---- Comment Index ----
        if comment_index:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(output_path), "..")
            )
            f.write("\n---\n## Comment Index\n")
            f.write(
                "_Infrastructure comments (STATION_HEADER, CONNECTION_MAP) are silenced. "
                "Subtypes: `incomplete` = unfinished work marker, `audit` = lint suppression / structural note._\n"
            )
            for norm_path, entries in sorted(comment_index.items()):
                if not entries:
                    continue
                # Use short relative path for readability
                try:
                    display = os.path.relpath(norm_path, project_root).replace(
                        "\\", "/"
                    )
                except ValueError:
                    # Fallback if paths are on different drives (Windows)
                    display = norm_path.replace("\\", "/")

                f.write(
                    f"\n<details><summary><code>{display}</code> ({len(entries)} comments)</summary>\n"
                )
                f.write("\n| Line | Preview | Subtype | Tier |")
                f.write("\n|------|---------|---------|------|")
                for entry in entries:
                    subtype = entry.get("subtype") or ""
                    tier = entry.get("tier") or ""
                    preview = entry.get("preview", "").replace("|", "\\|")
                    f.write(f"\n| {entry['line']} | `{preview}` | {subtype} | {tier} |")
                f.write("\n</details>\n")
