---
name: comment-skill
description: >
  LLM-native comment workflow for transforming code comments into agent-navigable
  infrastructure. Use this skill whenever an agent needs to: audit and remove comment
  clutter (Tear Down phase), install Station Headers / Connection Maps / WIP Beacons /
  Goto Pointers (Build Up phase), maintain or update stale comments after renames or
  refactors, or run the CRCT pre-population script to scaffold comment blocks from
  dependency data. Trigger on any task involving reading, writing, auditing, or
  structuring code comments — including phrases like "add comments", "document this
  file", "update the comments", "comment audit", "install WIP beacon", "station header",
  "goto pointer", "populate comments", or any request to make a codebase more
  navigable for agents.
---

# LLM-Native Comment Workflow — Core Rules

**This is the core rules file and plugin registry.** It defines universal comment
categories, tear-down heuristics, and the maintenance protocol. All plugins extend
these rules — they never remove core categories.

**Plugin load order** (most-specific wins on conflicts):
1. Core rules (this file) — universal baseline
2. CRCT plugin (`plugins/comment-skill-crct.md`) — if `.clinerules/` exists at project root
3. File-type plugin — selected by current file extension (Plugin Registry below)
4. WIP plugin (`plugins/comment-skill-wip.md`) — if WIP Beacons are present in the active file

---

## The Core Metaphor

A codebase is a railway network. Comments are the **rails and station signs**, not the
scenery. An LLM agent is the train — it does not need to see the whole city to get from
Station A to Station B. It needs clearly marked tracks, legible stop names, and transfer
instructions when it needs to change lines.

---

## Phase 1 — Tear Down (Comment Audit)

Remove comment clutter — annotations that add noise without navigational or explanatory
value. These waste LLM context tokens and reduce signal-to-noise ratio.

### Clutter Categories (Remove These)

| Category | Example | Why It's Clutter |
|---|---|---|
| Obvious narration | `// increment i` above `i++` | Restates what the code already says |
| Commented-out dead code | `// old_function()` left in place | Use version control instead |
| Placeholder noise | `// TODO: fix this later` (no specifics) | Not actionable, not directional |
| Boilerplate headers | `// Created by: John / Date: 2021` | Belongs in git log |
| Redundant type annotations | `// returns a string` when `-> str` exists | Already in signature |
| Narrative prose journals | Multi-paragraph inline decision history | Belongs in commit messages or ADRs |
| Self-evident dividers | `// ---- FUNCTIONS ----` with no context | No navigational value |

### Clutter Detection — Three-Question Test

Before removing any comment, answer all three questions:
1. **Does this tell an LLM something the code itself cannot?**
2. **Does this serve as a navigation pointer, intention marker, or connection map?**
3. **Would an autonomous agent make a wrong decision without this comment?**

If all three answers are "no" → remove the comment.

### Tear Down Rules

- Run Tear Down and Build Up together, not independently mid-task. Never tear down during an active WIP phase until Build Up is ready to replace.
- Commented-out code blocks → delete entirely. Code that matters lives in version control.
- Vague `TODO`/`FIXME` → either escalate to a proper WIP Beacon (Phase 2) or delete.
- Preserve any comment that provides partial but actionable context for navigation or intent — refine in Phase 2, do not delete.

---

## Phase 2 — Build Up (LLM-Native Comment Construction)

Four comment categories. Every file needs at minimum a **Station Header**.

```
STATION HEADER     ← What this file/module IS            (file-level, one per file)
CONNECTION MAP     ← Crossroads signpost: who connects   (function/class level, 1 line)
WIP BEACON         ← What is IN PROGRESS, INTENDED       (active work zones)
GOTO POINTER       ← "For X, see {file}:{anchor}"        (inline, as needed)
```

---

### Category 1 — Station Header

**Placement:** Top of every file (code or documentation).

**Purpose:** Orients the agent to the file's role in 3–5 structured lines. A cold-start
agent with no prior context should understand the file's job, layer, and primary
connections from this block alone.

**Template (Code):**
```
# ============================================================
# ROLE:    Issues, validates, and revokes JWT access tokens.
#          Central checkpoint for all authenticated API requests.
# LAYER:   Service — sits between route handlers and the DB layer.
# ============================================================
```

**Template (Documentation):**
```
<!--
ROLE:    Describes the end-to-end data pipeline from ingestion to storage.
         Reference document for understanding system-wide data movement.
WIP: Section 3 (Transform Layer) is placeholder — awaiting finalization
     of the ETL module in src/pipeline/transform.py
-->
```

**Rules:**
- `ROLE` = 1–2 sentences in plain language. Write as if the agent has zero prior context.
- `LAYER` optional; useful in layered architectures (routes → services → repos → DB).
- In CRCT projects, `CRCT_KEY` and `TRACKER_REF` are added here by the pre-population
  script. If the pre-population script fails, manually add CRCT_KEY and TRACKER_REF based on the dependency data. See `plugins/comment-skill-crct.md`.

---

### Category 2 — Connection Map (and Rail Network)

**Placement:** Immediately above each function, class, or major doc section.

**Purpose:** A single-line crossroads signpost showing who connects to this unit and how.
The agent can see the actual code body — the Connection Map's only job is to identify the
*external connection points* so the agent knows where to route next without reading the
whole file.

**Hard limit: 1 line.** This is not negotiable. Connection detail lives in the CRCT
tracker grids and `project_symbol_map.json`. The comment is the signpost, not the map.

**Format:**
```
# --- CONNECTION_MAP: {key}{char}, {key}{char}, ... --- {symbol_name} [AUTO]
```

**Where:**
- `{key}` = CRCT hierarchical key of the connected file (from `global_key_map.json`)
- `{char}` = dependency character (see table below)
- `{symbol_name}` = the name of the function, class, or section this applies to
- `[AUTO]` = machine-generated; re-run via `TrackerBatchCollector` or `populate_comments.py` to refresh

**Dependency Characters:**
| Char | Meaning |
|---|---|
| `<` | This unit depends on the keyed file (outbound) |
| `>` | The keyed file depends on this unit (inbound/caller) |
| `x` | Mutual dependency (bidirectional) |
| `d` | Documentation/reference dependency |
| `o` | Self dependency (diagonal only) |
| `n` | Verified no dependency |
| `p` | Placeholder (unverified — run `analyze-project` to resolve) |
| `s` | Semantic dependency (weak, similarity 0.06–0.07) |
| `S` | Semantic dependency (strong, similarity 0.07+) |

**Examples:**
```python
# --- CONNECTION_MAP: 1Ab2 >, 3Ba1 <, 2C3 x --- process_payment [AUTO]
# --- CONNECTION_MAP: 1A1 d --- TokenManager [AUTO]
# --- CONNECTION_MAP: none --- _internal_helper [AUTO]
```

**Rules:**
- One CONNECTION_MAP line per function/class/section. No multi-line expansions.
- `none` when the unit has no external connections (pure internal logic).
- `p` entries indicate CRCT has not yet verified the dependency — trigger
  `dependency_processor` to resolve before marking production-ready.
- Keys are sourced from `global_key_map.json`. If no CRCT project is active,
  use abbreviated file path aliases instead (e.g., `auth/token_manager.py >`).
- The agent reads this line to decide whether to follow a connection. It does not
  need the full call graph — that lives in the CRCT tracker. The signpost points
  to the next station; the agent decides whether to board.

---

### Category 3 — WIP Beacon

**Placement:** Immediately before the function, class, or block under active development.

**Purpose:** Mission briefing for an agent picking up mid-task. Contains enough intent,
status, and specific next steps that no external feedback is required to continue the work.

**This is the densest comment category.** WIP Beacons are intentionally more verbose than
other categories because they carry the entire operational context for an incomplete task.
Keep them tight and specific — but do not abbreviate at the cost of agent correctness.

**Template:**
```
# ┌────────────────────────────────────────────────────────────────────
# │ WIP: {symbol_name}
# │ INTENT:   {What is being built and why — 1–3 sentences}
# │ STATUS:   {What exists now — scaffold / partial / blocked}
# │ NEXT:     1. {Specific, executable step with file refs where needed}
# │           2. {Next step}
# │           3. {Next step}
# │ REQUIRES: {Config keys, env vars, upstream tasks that must exist first}
# │ [Additional Context — optional]
# │ AVOID:    {Ruled-out approaches — prevents dead-end re-exploration}
# │ BLOCKS:   {PR #, feature, or downstream task gated on this}
# └────────────────────────────────────────────────────────────────────
```

**WIP Beacon Fields:**

| Field | Required | Purpose |
|---|---|---|
| `INTENT` | ✅ | What is being built. The "why." |
| `STATUS` | ✅ | What exists now. Scaffold vs. working. |
| `NEXT` | ✅ | Ordered, specific steps. Agent-executable without external input. |
| `REQUIRES` | ✅ | Dependencies, config, or environment prereqs. |
| `Additional Context` | Optional | Section containing AVOID and BLOCKS fields for additional guidance. |

**Rules:**
- `NEXT` steps must be **specific enough to execute**. "Add rate limiting" is not a step.
  "Increment Redis key `rate:{user_id}:{epoch_minute}`, throw `RateLimitExceededError`
  if count > `RATE_LIMIT_MAX` from `config/settings.py`" is a step.
- Balance verbosity and structure: Use prose in INTENT and STATUS fields; keep NEXT steps structured and executable.
- Fields in Additional Context (AVOID, BLOCKS) are underused but critical — document rejected approaches and blockers explicitly so agents don't re-explore dead ends or miss dependencies.
- When WIP work is complete, **remove the beacon entirely**. No "DONE" markers.
  Done state lives in git history.
- Three or more active WIP Beacons in one file requires splitting the file or installing a higher-level WIP summary in the Station Header.
- In CRCT projects, add `CRCT_PHASE` and `HDTA_TASK` fields. See
  `plugins/comment-skill-crct.md`.

---

### Category 4 — Goto Pointer

**Placement:** Inline, at the exact point where an agent would need to follow a connection.

**Purpose:** Directional signpost. Points to the specific file and anchor where more
context lives. Cheap to write, high-value for navigation.

**Template (Code):**
```python
# → For token validation logic: auth/token_manager.py:TokenManager.verify()
# → Config reference: config/jwt_config.py (JWT_SECRET, JWT_EXPIRY)
# → Error types: errors/auth_errors.py (TokenExpiredError, InvalidTokenError)
```

**Template (Documentation):**
```html
<!-- → For API endpoint details: docs/api/auth-endpoints.md#POST-login -->
<!-- → For database schema: docs/architecture/db-schema.md:users_table -->
```

**Rules:**
- Format: `→ [context label]: [file_path]:[anchor]`
- Anchors: function/class names for code, heading slugs for docs.
- Use liberally — one Goto Pointer per external reference is the target density.
- Goto Pointers supplement Connection Maps; they are not replacements.

---

## Cross-Cutting Rules

### Tone and Register

Comments are written **to an LLM agent**, not a human developer:

- **Be explicit, not clever.** Agents do not infer from idiom or convention.
- **Use consistent vocabulary.** One term per concept throughout the file.
- **Avoid pronouns without clear referents.** "It handles the error" →
  "The `AuthGuard` middleware handles the error."
- **Structured > prose** outside WIP Beacons. Labeled fields parse better than sentences.

### Density Guidelines

| File Type | Station Header | Connection Map | WIP Beacon | Goto Pointers |
|---|---|---|---|---|
| Core service / business logic | Required | Required (all public symbols) | Required if WIP | Liberally |
| Utility / helper module | Required | Recommended (complex symbols only) | Required if WIP | As needed |
| Route / controller | Required | Recommended | Required if WIP | Required (to service layer) |
| Configuration file | Required | Not needed | Required if WIP | Required (to consumers) |
| Documentation file | Required | Recommended (major sections) | Required if WIP | Required (all cross-refs) |
| Test file | Optional | Not needed | Required if WIP | Goto to the unit under test |

### Maintenance Protocol

Comments decay. A WIP Beacon pointing to a renamed Redis key is worse than no comment.

**Audit triggers — run a comment audit when:**
- **File rename or move** → Update all Goto Pointers and Station Header paths.
- **Function signature change** → The CONNECTION_MAP line is `[AUTO]`; re-run
  via `TrackerBatchCollector` (triggered by commit) or `populate_comments.py`. Goto Pointers to that function need manual update.
- **WIP task completed** → Remove the WIP Beacon. No "DONE" markers.
- **Dependency removed** → Remove from `DEPENDS ON` and re-run populate script.
- **PR merge** → Audit WIP Beacons in touched files. Remove resolved ones.

Recommended: include comment audit as a step in the PR review checklist.
Audit question: *"Would an agent reading this file now get accurate directions?"*

---

## Agent Traversal Protocols

### Cold-Start Protocol (no prior project context)

1. Read `README.md` or project root Station Header → system overview.
2. Read Station Header of the entry-point file for the task.
3. Follow `ENTRY FROM` / `EXITS TO` chains to map relevant paths without reading bodies.
4. Read CONNECTION_MAP lines for symbols directly relevant to the task.
   If a CRCT key is present, resolve it via `global_key_map.json` or ask CRCT's
   `dependency_processor` to show the full connection detail.
5. Read WIP Beacons in the task area to understand current intent and constraints.
6. Execute, following Goto Pointers as cross-references for external dependencies.

### Targeted Task Protocol (specific symbol or file)

1. Read Station Header first — establish layer and connections before touching code.
2. Read all WIP Beacons in the file — understand what is and isn't done.
3. Read CONNECTION_MAP for symbols directly touched by the task.
4. Follow Goto Pointers only when the referenced file is directly relevant.
5. Do not traverse beyond two hops without re-reading the originating WIP Beacon.

### CONNECTION_MAP Resolution Protocol

When an agent encounters a CONNECTION_MAP line with CRCT keys it needs to resolve:
1. Look up key in `cline_utils/dependency_system/core/global_key_map.json`
2. For full dependency detail: run `dependency_processor show-deps {file_path}`
3. For tracker-level view: read the relevant `{module}_module.md` tracker

The CONNECTION_MAP line is a *pointer to the railway switch*, not the full timetable.
Use CRCT tooling to get the timetable when needed.

---

## Plugin Registry

Read the relevant plugin before writing comments for any specific file type or context.
All plugins are in `plugins/` relative to this SKILL.md.

| Plugin File | Scope | Priority | Status |
|---|---|---|---|
| `comment-skill-crct.md` | All files in a CRCT-managed project | P0 | ✅ v2 |
| `comment-skill-python.md` | `.py` files | P1 | v1 (needs CONNECTION_MAP update) |
| `comment-skill-csharp.md` | `.cs` files (Unity) | P1 | v1 (needs CONNECTION_MAP update) |
| `comment-skill-javascript.md` | `.js`, `.ts`, `.tsx`, `.jsx` | P1 | v1 (needs CONNECTION_MAP update) |
| `comment-skill-docs.md` | `.md`, `.rst`, wiki pages | P1 | v1 (needs CONNECTION_MAP update) |
| `comment-skill-config.md` | `.yaml`, `.toml`, `.json`, `.env` | P2 | v1 |
| `comment-skill-wip.md` | All files (WIP Beacon deep-dive) | P2 | v1 |
| `comment-skill-testing.md` | Test files (`test_*.py`, `*.spec.ts`) | P2 | v1 |
| `comment-skill-glsl.md` | `.glsl`, `.hlsl`, `.wgsl` | P3 | v1 |
| `comment-skill-sql.md` | `.sql`, migration files | P3 | v1 |

**CRCT detection:** If `.clinerules/` directory exists at project root, load
`comment-skill-crct.md` alongside the file-type plugin.

**Conflict resolution:** Most specific plugin wins.
File-type plugin > CRCT plugin > Core.

**Plugin update status:** All v1 file-type plugins need their Connection Map sections
updated to the single-line `CONNECTION_MAP` format. Core rules here are the canonical
reference. Until a plugin is updated, defer to the core Connection Map rules on format.

**For file types not covered by existing plugins:** Apply core rules using the appropriate comment syntax for the language (e.g., // for C-style, # for Python). Use generic comment markers if needed.

---

## Tools

`cline_utils/dependency_system/utils/populate_comments.py` — Pre-population utility.
Integrated with `TrackerBatchCollector`; automatically scans and updates Station
Headers and single-line CONNECTION_MAP entries in source files during the batch
commit phase. Consumes globally aggregated dependency data.

`code_analysis/report_generator.py` (via `code_analysis/scanner/comment_index.py`) —
Reporting and audit tool. Scans for `[AUTO]` tagged comments project-wide,
validates them against current tracker data, and generates consolidated reports.

```
Usage (Populate via Batch Collector):
  # Triggered automatically when using TrackerBatchCollector.commit_all()
  # (e.g., after running dependency_processor analyze-project)

Usage (Audit/Report):
  python code_analysis/report_generator.py --comment-index
```

---

## Quick Reference

```
STATION HEADER  (file top)
  ROLE [/ LAYER]

CONNECTION MAP  (above function/class — ONE LINE)
  # --- CONNECTION_MAP: {key}{char}, {key}{char} --- {symbol_name} [AUTO]
  Chars: < outbound  > inbound  x mutual  d doc  n none  p placeholder
         s semantic-weak  S semantic-strong

WIP BEACON  (active work zone)
  WIP / INTENT / STATUS / NEXT / REQUIRES [/ AVOID / BLOCKS]

GOTO POINTER  (inline)
  → [label]: [file]:[anchor]

REMOVE IF:
  narrates obvious code · dead code · vague TODO ·
  boilerplate date/author · redundant type annotation

AUDIT TRIGGERS:
  rename · signature change · WIP complete ·
  dependency removed · PR merge
```
