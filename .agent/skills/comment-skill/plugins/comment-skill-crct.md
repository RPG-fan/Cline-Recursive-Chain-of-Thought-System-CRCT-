## comment-skill-crct.md — PLUGIN HEADER

EXTENDS: core (SKILL.md)
SCOPE:   All source files in a project managed by CRCT
         (detected by presence of `.clinerules/` directory at project root)
OVERRIDES:
  - Connection Map: enforces single-line `CONNECTION_MAP` format using CRCT keys.
    The core already defines this format; this plugin makes it mandatory and
    specifies the exact key encoding sourced from CRCT's key system.
ADDS:
  - CRCT-specific Station Header fields: `CRCT_KEY`, `TRACKER_REF`, `HDTA_REF`
  - CRCT-specific WIP Beacon fields: `CRCT_PHASE`, `HDTA_TASK`
  - Pre-population script data mapping
  - Phase-aware comment priority table
  - CONNECTION_MAP key encoding reference (sourced from `key_manager.py`)

---

## CRCT Key Encoding Reference

CRCT assigns each file a **hierarchical key** via `key_manager.py`. The encoding is:

```
{tier}{DirLetter}[{subdirLetter}[{fileNumber}]][#{instance}]
```

| Part | Type | Meaning | Example |
|---|---|---|---|
| `tier` | integer ≥ 1 | Nesting depth at time of key generation | `1`, `2` |
| `DirLetter` | `A`–`Z` | Top-level directory index | `A`, `B` |
| `subdirLetter` | `a`–`z` | Subdirectory index within the directory | `a`, `b` |
| `fileNumber` | integer ≥ 1 | File index within its parent | `1`, `2`, `10` |
| `#instance` | `#1`, `#2`… | Disambiguator when base key is non-unique globally | `#1`, `#2` |

**Examples:**
- `1A` → top-level directory, first root
- `1A1` → first file directly in `1A`
- `1Aa` → first subdirectory of `1A`
- `1Aa1` → first file in subdirectory `1Aa`
- `2A` → a promoted directory (subdirectory of a subdirectory — tier increments)
- `1Ab3#1` → third file in second subdir of `1A`, instance #1 (disambiguated globally)

**The key is stable within a CRCT run.** If files are renamed or moved, `analyze-project`
must be re-run to regenerate `global_key_map.json` and refresh all `[AUTO]` CONNECTION_MAP
lines via `cline_utils/dependency_system/utils/populate_comments.py`.

**Source:** `cline_utils/dependency_system/core/key_manager.py` → `generate_keys()`,
`HIERARCHICAL_KEY_PATTERN = r"[1-9]\d*[A-Z](?:[a-z](?:[1-9]\d*)?|[1-9]\d*)?"`

---

## CRCT Data Sources

| Data Source | Location | Used For |
|---|---|---|
| `project_symbol_map.json` | `cline_utils/dependency_system/core/` | `ENTRY FROM`, `EXITS TO`, `DEPENDS ON`, CONNECTION_MAP targets |
| `global_key_map.json` | `cline_utils/dependency_system/core/` | Resolving file path ↔ CRCT key for CONNECTION_MAP |
| `tracker_map.json` | `cline_utils/dependency_system/core/` | `TRACKER_REF` field in Station Header |
| `module_relationship_tracker.md` | `cline_docs/` | Module-level dependency characters for CONNECTION_MAP |
| `{module}_module.md` trackers | (paths in tracker_map.json) | File-level dependency characters; HDTA Domain docs |
| `doc_tracker.md` | `cline_docs/` | `d` dependency characters for documentation files |
| `.clinerules/default-rules.md` | `.clinerules/` | `CRCT_PHASE` field in WIP Beacons |
| `progress.md` | `cline_docs/` | Identifying WIP zones before installing WIP Beacons |

---

## CRCT-Specific Comment Fields

### Station Header — CRCT Additions

```
# ============================================================
# ROLE:    [FILL: 1-2 sentence description]
# LAYER:   [FILL: architectural layer]
# CRCT_KEY:   1Ab2                                      [AUTO]
# TRACKER_REF: src/auth/auth_module.md                  [AUTO]
# HDTA_REF:   cline_docs/implementation_plan_auth.md:token-management
# ============================================================
```

**New fields (required in CRCT-managed projects):**

| Field | Source | Auto? | Description |
|---|---|---|---|
| `CRCT_KEY` | `global_key_map.json` | ✅ | The file's alphanumeric key as CRCT knows it |
| `TRACKER_REF` | `tracker_map.json` | ✅ | Path to this file's mini-tracker / HDTA Domain Module doc |
| `HDTA_REF` | HDTA plan files | ❌ (agent fills) | Path and anchor to the HDTA implementation plan or task governing this file |

### WIP Beacon — CRCT Additions

```
# ┌────────────────────────────────────────────────────────────────────
# │ WIP: issue_token()
# │ INTENT:        [prose]
# │ STATUS:        [prose]
# │ NEXT:          1. [step]
# │                2. [step]
# │ REQUIRES:      [prereqs]
# │ CRCT_PHASE:    Execution                             [AUTO]
# │ HDTA_TASK:     cline_docs/tasks/auth/task_rate_limiting.md
# └────────────────────────────────────────────────────────────────────
```

**New fields:**

| Field | Source | Auto? | Description |
|---|---|---|---|
| `CRCT_PHASE` | `.clinerules/default-rules.md` | ✅ | Current CRCT phase |
| `HDTA_TASK` | HDTA task files | ❌ (agent fills) | Path to the specific CRCT task file governing this WIP item |

### CONNECTION_MAP — CRCT Canonical Format

In CRCT projects the CONNECTION_MAP line is the **only** Connection Map format. No
multi-line Connection Maps in any file within a CRCT-managed project.

```python
# --- CONNECTION_MAP: {key}{char}, {key}{char}, ... --- {symbol_name} [AUTO]
```

**Parser contract:**
- Start sentinel: `# --- CONNECTION_MAP:`  (or language-equivalent comment prefix)
- End sentinel: `--- [AUTO]`
- Everything between the sentinels is a comma-separated list of `{key}{char}` pairs
- The `{symbol_name}` after the last `---` is informational only — not parsed

**Building a CONNECTION_MAP entry:**
1. For each external dependency of the symbol, look up the target file's key in
   `global_key_map.json`.
2. Look up the dependency character in the relevant tracker grid (module or doc tracker).
3. Format: `{key}{char}` with no space between key and character.
4. Join all pairs with `, `.
5. If no external connections exist: `none`.

**Example with annotation (annotation is for reading this doc — not written in source):**
```python
# --- CONNECTION_MAP: 1Ab2 >, 3Ba1 <, 2C3#1 x --- process_payment [AUTO]
#                     ^^^^  ^  ^^^^  ^  ^^^^^  ^
#                     key   |  key   |  key    |
#                           |        |         └── mutual dependency
#                           |        └── outbound (process_payment depends on 3Ba1)
#                           └── inbound (1Ab2 depends on/calls process_payment)
```

**Dependency character reference** (from tracker grids and `dependency_processor`):
| Char | Meaning |
|---|---|
| `<` | This symbol depends on the keyed file (outbound) |
| `>` | The keyed file depends on this symbol (inbound/caller) |
| `x` | Mutual dependency (bidirectional) |
| `d` | Documentation/reference dependency |
| `o` | Self dependency (diagonal only) |
| `n` | Verified no dependency |
| `p` | Placeholder — run `analyze-project` to verify |
| `s` | Semantic dependency (weak, cosine similarity 0.06–0.07) |
| `S` | Semantic dependency (strong, cosine similarity 0.07+) |

---

## CRCT Phase Awareness

Read `.clinerules/default-rules.md` to determine current phase before commenting.

| CRCT Phase | Comment Priority | Primary Action |
|---|---|---|
| **Set-up/Maintenance** | Station Headers + CONNECTION_MAP lines | Run `populate_comments.py` (utility); install `[AUTO]` scaffolds |
| **Strategy** | WIP Beacons | Install `INTENT` + `NEXT` fields from HDTA implementation plans; add `HDTA_TASK` refs |
| **Execution** | All categories, live updates | Agents update WIP Beacons as tasks progress; Goto Pointers at specific call sites |
| **Cleanup/Consolidation** | Tear Down | Remove completed WIP Beacons; re-run populate script after renames; audit stale `[AUTO]` entries |

---

## Pre-Population Script Integration

`cline_utils/dependency_system/utils/populate_comments.py` writes `[AUTO]` fields from
CRCT data. After running, the agent only fills `[FILL: ...]` prose fields — not
the structural skeleton.

### Data Mapping: symbol_map.json + global_key_map.json → Comment Fields

| Source | Field | Comment Field | Category |
|---|---|---|---|
| `file_path` | Direct | `STATION` | Station Header |
| `imports[*]` (internal only) | Filter stdlib/pip | `DEPENDS ON` | Station Header |
| Inbound callers (cross-file call scan) | Deduplicated | `ENTRY FROM` | Station Header |
| `calls[*].target_file` | Deduplicated | `EXITS TO` | Station Header |
| `global_key_map.json` lookup | By file path | `CRCT_KEY` | Station Header |
| `tracker_map.json` lookup | By file path | `TRACKER_REF` | Station Header |
| `.clinerules/default-rules.md` | Current phase | `CRCT_PHASE` | WIP Beacon |
| `functions[*].name` + tracker grid | Key + dep char | CONNECTION_MAP entry | Connection Map |
| `classes[*].name` + tracker grid | Key + dep char | CONNECTION_MAP entry | Connection Map |

### `[AUTO]` Tag Contract

- Every machine-generated comment line ends with `[AUTO]`.
- The populate script never overwrites lines **without** `[AUTO]`.
- `[FILL: ...]` marks prose fields an LLM agent must complete.
- Re-running the populate script refreshes all `[AUTO]` lines. Human/LLM prose is preserved.

### Populate Script Trigger Points

Updates to `[AUTO]` fields are primarily automated through the **CRCT framework**
via the `TrackerBatchCollector` commit lifecycle. Whenever trackers are written,
the comment population hook automatically refreshes the relevant source files.

Manual refresh can be performed by running:
`python cline_utils/dependency_system/utils/populate_comments.py --write`
- Pre-commit hook (use `--dry-run` to flag stale `[AUTO]` entries without writing)

### Resolving CONNECTION_MAP Keys at Runtime

When an agent needs to look up a key from a CONNECTION_MAP line:
```
dependency_processor show-deps {file_path}   ← full dependency detail for a file
dependency_processor show-key {key}          ← resolve key → file path
```
Or directly: read `cline_utils/dependency_system/core/global_key_map.json`.

---

## CRCT Field Rules

- `CRCT_KEY`, `TRACKER_REF`, CONNECTION_MAP lines are **required** in any file within
  a CRCT-managed project. They are optional in the core rules but mandatory here.
- `[AUTO]` fields are owned by the populate script. Do not hand-edit them.
  If a key or dep character is wrong, re-run `analyze-project`.
- `HDTA_REF` and `HDTA_TASK` are agent-filled. The populate script installs a
  `[FILL: ...]` placeholder; the agent resolves it using the HDTA plan files.
- If no scaffold exists (first pass, populate script not yet run), the agent may write
  the full Station Header manually, adding CRCT fields from `global_key_map.json` and
  `tracker_map.json`. Mark all manually sourced CRCT data with `[AUTO]` so the populate
  script can take ownership on next run.

---

## CRCT Integration Notes for File-Type Plugins

CRCT fields are **additive** — they append to whatever the file-type plugin defines.
The file-type plugin governs syntax and structure; this plugin governs CRCT-specific
content.

Example: Python Station Header = core template syntax + `CRCT_KEY` + `TRACKER_REF` +
`HDTA_REF` (from this plugin).

Connection Map: file-type plugins that define their own Connection Map templates should
be updated to use the single-line `CONNECTION_MAP` format. Until updated, defer to
this plugin's format for any file within a CRCT project (this plugin takes precedence).

→ For the pre-population script: `cline_utils/dependency_system/utils/populate_comments.py`
→ For core comment categories: `SKILL.md`
→ For key encoding source: `cline_utils/dependency_system/core/key_manager.py`
