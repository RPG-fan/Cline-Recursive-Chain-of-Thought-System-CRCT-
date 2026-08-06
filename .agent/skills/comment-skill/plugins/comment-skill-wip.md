## comment-skill-wip.md — PLUGIN HEADER

EXTENDS: core (SKILL.md)
SCOPE:   All files containing WIP Beacons; all active development zones
OVERRIDES: none — all core rules apply
ADDS:    Extended fields (APPROACH, AVOID, CRCT_PHASE, HDTA_TASK, BLOCKS, REQUIRES);
         Multi-agent handoff patterns; high-density WIP file management

---

## WIP Beacon: The Mission Briefing

A WIP Beacon is not a `TODO`. It is an **active mission briefing** for an autonomous
agent. It declares intent, documents state, and provides a specific, executable
roadmap for the next agent to pick up.

### Extended Field Reference

| Field | Required | Description |
|---|---|---|
| `INTENT` | ✅ | The high-level "Why". What is the goal of this change? |
| `STATUS` | ✅ | The "What exists". Is it a scaffold, buggy, or partially working? |
| `NEXT` | ✅ | The "How". Ordered, numbered steps that are specific enough to execute. |
| `REQUIRES` | ✅ | Prerequisites. Config keys, environment state, or preceding tasks. |
| `APPROACH` | Optional | The "Style". Preferred algorithm, design pattern, or library choice. |
| `AVOID` | Optional | Dead ends. Rejected approaches to prevent agent back-tracking. |
| `BLOCKS` | Optional | Downstream impact. PRs or features waiting on this WIP. |
| `CRCT_PHASE` | (CRCT) | Auto-populated by the `populate_comments.py` utility. |
| `HDTA_TASK` | (CRCT) | Path to the specific task file in `cline_docs/tasks/`. |

---

## Multi-Agent Handoff Pattern

WIP Beacons are the primary mechanism for handoff between agents (or between a human
and an agent).

### 1. The Handoff Signal
When an agent reaches its token limit or finishes a sub-task, it MUST update the
`STATUS` and `NEXT` fields of any active WIP Beacons.

### 2. The Resume Protocol
The incoming agent's first action in a file is to read the Station Header and then
**all WIP Beacons**.
- If `NEXT` steps are clear → start at step 1.
- If `NEXT` steps are ambiguous → refine the WIP Beacon before touching code.

---

## High-Density WIP Management

- **The Three-Beacon Limit**: If a file has 3+ WIP Beacons, it is "In Flight" and
  dangerously complex. Create a **Master WIP Beacon** at the top of the file (below
  the Station Header) that summarizes the file-wide state and links to the local
  beacons.
- **WIP Completion**: When a task is finished, **DELETE** the WIP Beacon block. Do not
  convert it to `DONE:`. Success is documented in the git commit.
- **NEXT-Item Re-Anchor Rule (MANDATORY)**: Removing a `# WIP` (or `# │ WIP:`) beacon whose
  `NEXT` field lists remaining/uncompleted items REQUIRES re-anchoring each remaining item
  as its own proper WIP beacon at its respective code site BEFORE the referencing parent beacon
  is deleted — recording them only in plan files or activeContext is information loss.
  - *Remediation pattern*: Audit every removed beacon's `NEXT` list; for each uncompleted item,
    create a new `# │ WIP: <topic> — <follow-up title>` block (`INTENT`, `STATUS`, `NEXT`,
    `REQUIRES`, `CRCT_PHASE`, `HDTA_TASK`) at the relevant code site.
- **Grep Syntax & Box-Drawing**: Always use whitespace- and box-drawing-tolerant searches
  (e.g., `#\s*│?\s*WIP:`) so box-drawing `# │ WIP:` beacons are never missed by regex scans.

---

## Examples

### Example 1: Complex logic handoff

```python
# ┌────────────────────────────────────────────────────────────────────
# │ WIP: calculate_damage_roll()
# │
# │ INTENT:   Integrate critical hit multipliers and elemental resistances
# │           into the core combat damage pipeline.
# │
# │ STATUS:   Basic dice roll + strength bonus is working. Critical hit
# │           detection (d20 == 20) is implemented but multiplier is hardcoded.
# │
# │ NEXT:     1. Import `ELEMENTAL_MODIFIERS` from `data/combat_tables.py`
# │           2. Implement `_apply_resistances(base_dmg, target_element)`
# │           3. Replace hardcoded `x2` crit with `weapon.crit_multiplier`
# │           4. Update unit tests in `tests/combat/test_damage.py`
# │
# │ APPROACH: Use a strategy pattern for elemental types to keep the core
# │           function under 100 lines.
# │ AVOID:    Do not modify the `Target` object directly; return a `DamageResult`
# │           dataclass instead.
# │
# │ BLOCKS:   Feature/BossCombat (Branch: boss-ai-v2)
# └────────────────────────────────────────────────────────────────────
```

### Example 2: Cross-file dependency WIP

```javascript
/*
  ┌────────────────────────────────────────────────────────────────────
  │ WIP: AuthProvider Component
  │
  │ INTENT:   Migrate from local state to persistent session cookies
  │           for auth persistence across page refreshes.
  │
  │ STATUS:   Cookie setting logic added to `login()` function.
  │           `useEffect` on mount correctly reads the cookie.
  │           MISSING: Logout logic does not clear the cookie yet.
  │
  │ NEXT:     1. Add `deleteCookie('session_id')` to the `logout` handler.
  │           2. Verify cross-tab logout (SessionStorage event listener).
  │
  │ REQUIRES: `utils/cookie_manager.ts` must be exported and tested.
  │ HDTA_TASK: cline_docs/tasks/auth/task_session_persistence.md
  └────────────────────────────────────────────────────────────────────
*/
```

---

## CRCT Integration

In CRCT-managed projects:

1.  **Strategy Pre-Execution Beacon Pass**: Before transitioning from Strategy to Execution,
    the Strategy Dispatcher runs Step 8.7 (`.clinerules/strategy_dispatcher_plugin.md`). Strategy
    Workers instantiate or update `# WIP` (or `# │ WIP:`) beacons at target code sites for all
    planned `Execution_*` tasks and Implementation Plans for the cycle.
2.  **Phase Alignment**: Ensure `CRCT_PHASE` matches `.clinerules/default-rules.md` (e.g., `Execution`).
3.  **Task Linking**: Always include `HDTA_TASK` pointing to `tasks/exec/Execution_*.md` (or strategy task file).
4.  **Execution Lifecycle & Re-Anchoring**: During Execution (`.clinerules/execution_plugin.md`),
    completed beacons are deleted (not kept as `DONE:`), and any parent beacon removed with remaining `NEXT`
    items triggers the **NEXT-Item Re-Anchor Rule** to instantiate new beacons at code sites before deletion.

→ For core rules: SKILL.md
→ For CRCT fields: plugins/comment-skill-crct.md
