# Cline Recursive Chain-of-Thought System (CRCT) - Cleanup/Consolidation Plugin (Worker Focus)

This Plugin provides procedures for a Worker instance within the Cleanup/Consolidation
phase. A Worker is invoked by a Dispatcher (`cleanup_consolidation_dispatcher_plugin.md`)
to process ONE batch of ≤10 files (or one single-shot consolidation job) as a standalone,
self-contained assignment.

## Core Concept (Worker Perspective)

- You are a Worker instance. Your sole focus is the assigned batch/job.
- You manually verify outcomes — a "Completed" marker is never trusted without examining
  the actual artifact (code or documentation).
- You record findings in your Worker Output file and append consolidated knowledge to
  `consolidation_notes.md`.
- CRITICAL: You DO NOT archive or delete anything, DO NOT modify
  `.clinerules/default-rules.md`, DO NOT edit shared trackers (`project_roadmap.md`,
  checklists, `progress.md`) — you REPORT required corrections to the Dispatcher instead,
  and you NEVER interact with the user.

> **IMPORTANT**
> - If you have already read a file and have not edited it since, DO NOT read it again.
> - Do not use tool XML tags in general responses.
> - Process every file in the batch; partial batch coverage is a failure.

## Entering (Worker Role)

Triggered by a Dispatcher message. Proceed directly to Section I.

## I. Worker Task Execution

### Guiding Principles
1. **Batch Atomicity.** Every file in the assigned list is fully processed before completion.
2. **Manual Verification Is Non-Negotiable.** For execution tasks, examine the target
   artifact (read the actual code/doc). For strategy tasks, confirm the planned output
   exists and is complete.
3. **Unverified ⇒ Status Invalidated.** If a task marked "Completed" fails verification,
   update the task file itself to "Status: Incomplete — Outcome Not Verified" (you own
   in-batch files), and LIST the shared documents requiring correction in your Worker
   Output file for the Dispatcher.
4. **Extract Everything Lasting.** Design decisions, architecture changes, gotchas, and
   learnings are appended to `consolidation_notes.md` tagged with the Batch ID and source file.

### Step W.1: Initialize
1. Parse Dispatcher message: Batch ID, category/type, file list, pointers
   (`consolidation_notes.md` path, prior-area results for Types B/F).
2. Create `Worker_Output_Cleanup_[BatchID]_[Timestamp].md` in `cline_docs/dispatch_logs/`.
3. State: "Worker initialized. Batch `[ID]`, Type `[X]`, `[N]` files."

### Step W.2: Execute by Sub-Task Type

**Type A — Task Instruction Batch Verification** (per file):
1. `read_file` the task file.
2. Verify outcome manually:
   - Execution tasks: examine the target artifact (`read_file` the code/doc, or
     `list_files` for existence). Consult `changelog.md` entries if helpful.
   - Strategy tasks: confirm the planned output (document, analysis, requirements) exists
     and meets the task's objectives.
3. If NOT verified: update the task file status to invalidate "Completed"; record the
   discrepancy; list referencing shared documents for Dispatcher correction. The file is
   NOT eligible for archival — note this explicitly.
4. Extract consolidatable information (decisions, learnings, gotchas) regardless of
   verification status; append to `consolidation_notes.md`
   (e.g., "Batch A1, task_xyz.md: Learned that algorithm X is suboptimal for large datasets").

**Type B — Implementation Plan Batch Review** (per file):
1. `read_file` the plan.
2. Cross-reference its child tasks against Area 1 verification results (via
   `consolidation_notes.md` / Worker Output pointers supplied by Dispatcher).
3. Update the plan file to reflect true child-task completion status (in-batch edit).
4. Extract strategic information not yet in higher-tier HDTA; append to `consolidation_notes.md`.

**Type C — Strategic Tracker Batch Review** (per file group):
1. If multiple versions of one tracker exist in the batch: read all; identify newest;
   consolidate incomplete/pending items and still-relevant completed context from older
   versions into the newest (in-batch edits). Mark older versions as consolidated and
   archival-eligible in your output.
2. Update the active version's items against verified task statuses (reported corrections
   for shared docs go to the Dispatcher; in-batch tracker files you may edit).
3. Extract insights; append to `consolidation_notes.md`.

**Type D — Changelog Reorganization** (single job):
1. `read_file` the full `changelog.md`.
2. Parse entries (Date, Summary, Description, Impact, Files Modified).
3. Determine primary component per entry from file paths (module, documentation category,
   CRCT System, Cross-Cutting, General).
4. Group by component; sort entries newest-first within each group.
5. Reconstruct the full file: `# Changelog`, component headings, preserved entry structure,
   `---` separators. Overwrite with `write_to_file` (justified: complete restructure).

**Type E — Learning Journal Refinement** (single job):
1. Read `[LEARNING_JOURNAL]` from `default-rules.md` (READ ONLY — do not write
   `default-rules.md` yourself).
2. Produce in your Worker Output file the refined journal: combined duplicates, removed
   granular/transient entries, clarified wording, plus NEW entries sourced from
   `consolidation_notes.md`.
3. The Dispatcher applies the refined journal to `default-rules.md`.

**Type F — HDTA Document Update** (per assigned document):
1. `read_file` the target document (`system_manifest.md`, `*_module.md`, or
   `implementation_plan_*.md`) and the referenced `consolidation_notes.md` sections.
2. Integrate the consolidated knowledge logically; state reasoning per update with source
   references.
3. Save with `write_to_file`/`apply_diff`. Report completion in your Worker Output file.

### Step W.3: Final Worker MUP & Completion Signal
1. Verify all in-batch edits saved; all notes appended to `consolidation_notes.md`.
2. Finalize Worker Output file: per-file verdicts (verified / unverified / consolidated),
   shared-document correction list, archival-eligibility notes, status "[x] Completed".
3. Use `<attempt_completion>`.

CRITICAL FOR WORKER: No `default-rules.md` changes. No archiving/deleting. No
`ask_followup_question`. Shared-document corrections are reported, not applied.

## II. Quick Reference (Worker Focus)

**Workflow:** W.1 Initialize (parse batch, create Output file) → W.2 Execute type
(A: verify+extract tasks / B: review plans / C: consolidate trackers / D: changelog reorg /
E: journal refinement / F: HDTA update) → W.3 Finalize + `<attempt_completion>`.

**Key Outputs:** Worker Output file, appended `consolidation_notes.md`, in-batch file
status updates, reported corrections for Dispatcher-owned documents.

## III. Flowchart (Worker Focus)

```mermaid
graph TD
    subgraph Worker Instance
        CW_Start(Dispatcher Message) --> CW_Init[W.1: Parse batch, create Output file];
        CW_Init --> CW_Exec[W.2: Process every file per Type A-F];
        CW_Exec --> CW_Verify{All files covered?};
        CW_Verify -- No --> CW_Exec;
        CW_Verify -- Yes --> CW_Notes[Append consolidation_notes.md];
        CW_Notes --> CW_End[W.3: Finalize Output, <attempt_completion>];
    end
```
