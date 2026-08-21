<!--
Instructions: Fill in the placeholders below to create changelog_index.md — the central access
point for the project's changelog DIRECTORY ({memory_dir}/changelog/). The changelog is a set of
per-component files, never a single monolith. This index is the ToC + append-rule authority.
*Do NOT include these comments in the created file.*
-->

# Changelog Index — {ProjectName}

> **Model:** The changelog is a DIRECTORY, not a single file. Each major project area/component owns one dedicated `.md` file in this directory; this index is the table of contents and append-rule authority. Component files cross-link related changes in other component files with markdown links. Do NOT recreate a single monolithic changelog file.

## Append Rules

1. New entries are appended to the file for the affected COMPONENT as a `### <YYYY-MM-DD>` block placed ABOVE that file's current newest date heading (newest-first within each file). Multiple changes on one date share one dated block with multiple `- Type:` bullets.
2. Entry format: `- Type: Feature|Fix|Bugfix|Decision|...` → description (task/plan references, symbol anchors where useful) → `- Files Affected:` → `- Verification:` → `- Status:`. Adapt fields to the project's needs but keep them consistent across all component files.
3. Test-suite additions fold into their feature entry's Files-Added/Verification lines unless the tests ARE the deliverable.
4. When a change relates to work recorded in ANOTHER component's file, add a markdown link in your entry, and optionally a reciprocal pointer there.
5. After appending, update the "Latest entry" column below for the touched file.
6. A new major area/component emerging over time → create its `.md` in this directory, add a row to the table, and record its routing keyword in §Component Routing.
7. {Project-specific rule if any — e.g., "CRCT-system operations are NOT changelog entries" or scope boundaries.}

## Component Routing

{Ordered first-match keywords mapping new/misplaced content to files, so appends stay consistent — e.g.: `database|schema|seed` → database.md · `agent*` → agents.md · `ui` → ui.md · unmatched → misc.md. Keep the list short; ambiguity resolves to the most-specific match.}

## Table of Contents

| File | Scope | Latest entry |
|------|-------|--------------|
| [{component_file_1}.md]({component_file_1}.md) | {one-line scope} | {YYYY-MM-DD} |
| [{component_file_2}.md]({component_file_2}.md) | {one-line scope} | {YYYY-MM-DD} |
| ... | ... | ... |

## Notes

{Optional: migration provenance (if converted from a monolithic file), historical ordering caveats, or archival pointers. Delete this section if empty.}
