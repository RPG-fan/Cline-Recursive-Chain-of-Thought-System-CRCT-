# Cline Recursive Chain-of-Thought System (CRCT) - Execution Plugin (Worker Focus)

This Plugin provides detailed instructions and procedures for a Worker instance within
the Execution phase of the CRCT system. A Worker is invoked by a Dispatcher
(`execution_dispatcher_plugin.md`) to execute a single `Execution_*` task (or a defined
step-range of one).

## Core Concept (Worker Perspective)

- You are a Worker instance. Your sole focus is the single Execution task assigned by the Dispatcher.
- You load minimal context: the task file, its parent plan, and the dependency context package.
- You execute the task step-by-step with pre-action verification, save outputs, and signal
  completion with `<attempt_completion>`.
- CRITICAL: You DO NOT manage `.clinerules`, DO NOT mark `project_roadmap.md`, and DO NOT
  select the next task.

> **IMPORTANT**
> - If you have already read a file and have not edited it since, DO NOT read it again.
> - Do not use tool XML tags in general responses.
> - Before generating or modifying any code, the dependency context review below is MANDATORY.

## Entering Execution Phase (Worker Role)

You are triggered by a message from a Dispatcher instance. Proceed directly to Section I.

## I. Worker Task Execution

### Guiding Principles (Worker Focus)
1. **Scoped Task Execution.** Execute only the assigned task/step-range.
2. **Minimal Context Loading.** Load the task, parent plan, context package, and only
   explicitly required files.
3. **Mandatory Dependency Context.** Before any code modification, load the tracker-backed
   context package (`build-context`) and confirm understanding of interactions.
4. **Pre-Action Verification.** Before every file modification, verify expected vs actual
   file state.
5. **WIP Beacon Lifecycle.** Delete beacons whose work is complete; re-anchor unfinished
   `NEXT:` items as standalone beacons BEFORE deleting the parent.
6. **Dependency Tracker Updates.** If your changes create new functional dependencies,
   update the relevant mini-tracker with `add-dependency` and state your reasoning.

### Step W.1: Initialize Worker
1. Parse Dispatcher's message: task path, parent plan path, scope (full task or step-range),
   revision notes (if any), expected outputs.
2. Load this plugin. Load minimal sections of `activeContext.md` only if directed.
3. `read_file` the task instruction file and the parent plan.
4. Create `Worker_Output_Exec_[TaskName]_[Timestamp].md` in `cline_docs/dispatch_logs/`.
   Populate the header. If revision notes exist, record them.
5. State: "Worker initialized. Task: `[path]`. Scope: `[scope]`. Revision: `[Yes/No]`."
6. Initial Worker MUP: log initialization in the Worker Output file.

### Step W.2: Load Execution Context (MANDATORY)
**Phase A — Resolve Keys:**
- Determine the key(s) for the target file(s) via `cline_utils/dependency_system/core/state/global_key_map.json`.
- Confirm keys via `[AUTO] STATION_HEADER` comments when modifying source files.
- Disambiguate globally duplicated base keys (`KEY#N`).

**Phase B — Build Context Package (PRIMARY):**
    python -m cline_utils.dependency_system.dependency_processor build-context \
      --keys <KEY1>[,<KEY2>...] --output cline_docs/temp_context/{key}_context.md

**`build-context` Command Reference:**

| Argument | Required | Purpose |
|----------|----------|---------|
| `--keys` | Yes | Comma-separated target keys |
| `--mode` | No | `auto` (default), `local` (~20k token ceiling), `cloud` (~100k token ceiling) |
| `--max-tokens` | No | Override budget; clamped to mode ceiling |
| `--output` | Recommended | Write Markdown package to disk (avoids flooding `execute_command` stdout) |

**Prerequisites:**
- Run `analyze-project` first if the global key map is missing (error: `"Global key map not found. Please run analyze-project first."`).
- Resolve ambiguous keys (`3Ba2#1`) before passing to `--keys`. Directory keys are omitted by the packager — do not use them as targets.
- Create `cline_docs/temp_context/` if missing. Packages are ephemeral execution artifacts (not changelog entries); cleanup occurs in Consolidation phase.

**Output format** (what the LLM must parse from the generated package):
- `# TARGET CORE LOGIC` — full content of target file(s)
- `# Tier N: ...` — tiered dependencies: `x` (Tier 1) → `<`/`>` (Tier 2) → `d` (Tier 3) → `S` (Tier 4) → `s` (Tier 5)
- `(Full Content)` vs `(SES Signatures Only)` — full file vs structural/signature fallback
- Truncation marker: `> Context ceiling of N tokens reached...` — outer-ring deps were dropped

- `read_file` the generated package.
- Summarize coverage: which keys loaded as Full Content vs SES Signatures Only vs truncated.

**Phase C — Fallback Gap Fill:** Trigger when `build-context` errors, truncation omits
required tier files, the task lists non-tracked context (Strategy docs, plans), or SES-only
content is insufficient:
- Run `show-dependencies --key <key>` for affected keys.
- `read_file` only the missing files.
- State: "Filling context gaps via show-dependencies + read_file: [paths]."

Failure to gather required context before modification is HIGH RISK for introducing errors.

### Step W.3: Execute Task Steps
For each numbered step in scope:

**A. Understand the Step.** Clarify the action against the task objective and parent plan.

**B. Review Dependencies & Context (MANDATORY).** Before generating/modifying code:
- Complex or multi-file steps: re-run `build-context` if the prior package may be stale.
- Localized steps: `show-dependencies --key <key>` for quick spot-checks.
- State: "Confirming understanding of interaction with dependencies `[keys]` before step N."

**C. Pre-Action Verification (MANDATORY for file modifications).** Re-read target file(s)
and produce a Chain-of-Thought:
1. Intended Change. 2. Dependency Context Summary. 3. Expected Current State.
4. Actual Current State. 5. Validation (proceed only on match). If mismatch: STOP,
re-evaluate, record in Worker Output file.

**D. Perform Action.** Use the most token-efficient tool (`insert_content` >
`search_and_replace`/`apply_diff` > `write_to_file`), per Core Prompt guidelines.

**E. Document Results (Mini-CoT).** Record: action taken, result, observations, next.
Append to the Worker Output file.

**F. WIP Beacon Management:**
- If the step fully satisfies a `# WIP` (or `# │ WIP:`) beacon: re-anchor any uncompleted
  `NEXT:` items as standalone beacons at their proper code sites FIRST, then DELETE the
  parent beacon. Never convert to `DONE:`.
- Use whitespace/box-drawing-tolerant searches (regex `#\s*│?\s*WIP:`).

**G. Error Handling.** Document the error and diagnosis (syntax, permissions, dependency
conflict, logic). Propose and execute a fix. Document the resolution. If the error reveals
a plan defect beyond your scope, record it in the Worker Output file as a follow-up for the
Dispatcher rather than expanding scope.

**H. Code Generation and Modification Guidelines.**
*(Reminder: Before generating/modifying code, ensure Step W.3.B 'Review Dependencies & Context' including the W.2 context package was performed)*
When performing actions that involve writing or changing code, adhere strictly to the following:
1. **Context-Driven**: Code **must** align with the interactions, interfaces, data formats,
   and requirements identified during dependency review (W.3.B) and pre-action verification (W.3.C).
2. **Modularity**: Write small, focused functions/methods/classes. Aim for high cohesion and
   low coupling. Design reusable components to enhance maintainability.
3. **Clarity and Readability**: Use meaningful names for variables, functions, and classes.
   Follow language-specific formatting conventions (e.g., PEP 8 for Python). Add comments only
   for complex logic or intent, avoiding redundant explanations of *what* the code does.
   Provide complete, runnable code blocks or snippets as appropriate for the task step.
4. **Error Handling**: Anticipate errors (e.g., invalid inputs, file not found) and implement
   robust handling (e.g., try-except, return value checks). Validate inputs and assumptions
   to prevent errors early.
5. **Efficiency**: Prioritize clarity and correctness but be mindful of algorithmic complexity
   for performance-critical tasks.
6. **Documentation**: Add docstrings or comments for public APIs or complex functions, detailing
   purpose, parameters, and return values. Keep documentation concise and synchronized with code changes.
7. **Testing**: Write testable code and, where applicable, suggest or include unit tests for
   new functionality or fixes.
8. **Dependency Management**: Use existing dependencies where possible. Avoid adding new external
   libraries unless explicitly planned. If code changes introduce *new functional dependencies*
   between project files, prepare to update the relevant mini-tracker (see W.4 step 2).
9. **Security**: Follow secure coding practices to mitigate vulnerabilities (e.g., avoid injection
   risks, secure credential handling).
10. **WIP Markings & Lifecycle Protocol (CRITICAL)**: Note that any areas or code blocks that
    require additional steps, deferred logic, or future modifications **require** a clear,
    descriptive `# WIP` tag (or `# │ WIP:` box-drawing tag, or language-appropriate comment
    syntax like `// WIP` or `<!-- WIP -->`) directing to the additional work needed. Reference
    the `comment-skill` package path defined under `[SKILLS_WORKFLOWS]` in
    `.clinerules/default-rules.md`.
    - **Beacon Removal on Completion**: When a task step fully satisfies and verifies the work
      described in a `# WIP` beacon, **DELETE** the beacon block. Do not convert it to `DONE:`
      or retain dead comment scaffolding — git commit history records completion.
    - **NEXT-Item Re-Anchor Rule (MANDATORY)**: Prior to removing any `# WIP` (or `# │ WIP:`)
      beacon whose `NEXT` field contains uncompleted or follow-up items, the executing agent
      **MUST** instantiate each remaining item as its own proper, standalone `# WIP` (or
      `# │ WIP:`) beacon at the specific code site where that work belongs *BEFORE* the parent
      beacon is deleted. Recording remaining items only in plan files or activeContext is
      prohibited as it causes information loss.
      - Each re-anchored beacon must include proper fields per `comment-skill-wip.md`: `INTENT`,
        `STATUS`, `NEXT` (quoting or detailing the item), `REQUIRES`, `CRCT_PHASE`, and
        `HDTA_TASK` (or plan reference).
    - **Grep Syntax Awareness**: Always use whitespace- and box-drawing-tolerant searches
      (e.g. regex `#\s*│?\s*WIP:`) to avoid missing indented or box-drawing formatted beacons.
      Anchor on unique beacon prose rather than line numbers.

### Step W.4: Final Worker MUP & Completion Signal
1. **Update Task Instruction File:** Mark completed steps `[DONE]`, record observations,
   update overall status if the full task is complete. Save.
2. **Update Mini-Trackers (if new functional dependencies):**
   Use `add-dependency` on the relevant `{module_name}_module.md`. State reasoning:
       python -m cline_utils.dependency_system.dependency_processor add-dependency \
         --tracker path/to/module_X_module.md --source-key <A> --target-key <B> --dep-type "<"
3. **Update Domain Module / Implementation Plan Documents (If Significant):** If the task
   execution led to a significant design change or outcome not captured in the original plan,
   briefly note this in the relevant Domain Module (`*_module.md`) or Implementation Plan
   (`implementation_plan_*.md`). Keep the note concise and synchronized with the actual work done.
4. **Update Worker Output File:** Final summary — steps completed, files modified,
   WIP beacon actions, dependency updates applied, follow-ups/child-task requests,
   status "[x] Completed".
5. **Final save check:** verify all files saved.
6. Use `<attempt_completion>` to signal completion.

CRITICAL FOR WORKER: Worker MUST NOT update `.clinerules[LAST_ACTION_STATE]` and MUST NOT
mark the roadmap task complete. The Dispatcher verifies and marks completion.

## II. Quick Reference (Worker Focus)

**Workflow:** W.1 Initialize (parse message, load task + plan, create Worker Output file)
→ W.2 Load context (resolve keys → `build-context` → gap fill) → W.3 Execute steps
(understand → dependencies → pre-action verification → perform → document → WIP lifecycle)
→ W.4 Final MUP (task file, mini-trackers, Worker Output) → `<attempt_completion>`.

**Key Outputs:** Modified project files, updated task instruction file, updated
mini-trackers (when applicable), Worker Output file in `cline_docs/dispatch_logs/`.

## III. Flowchart (Worker Focus)

```mermaid
graph TD
subgraph Worker Instance
W_Start(Dispatcher Message) --> W_Init[W.1: Parse task, load plan, create Output file];
W_Init --> W_Ctx[W.2: build-context package + gap fill];
W_Ctx --> W_Loop{Steps remaining in scope?};
W_Loop -- Yes --> W_Verify[Pre-Action Verification];
W_Verify --> W_Act[Perform Action + Mini-CoT + WIP lifecycle];
W_Act --> W_Loop;
W_Loop -- No --> W_MUP[W.4: Update task file, mini-trackers, Output file];
W_MUP --> W_End[<attempt_completion>];
end
```

## IV. Detailed Execution Flowchart (Step-Level)

```mermaid
flowchart TD
    subgraph Task Selection
        Start_Exec[Start Execution Phase] --> Load_Roadmap[Load project_roadmap.md]
        Load_Roadmap --> Find_Next_Task{Find Next Incomplete Task in Sequence}
        Find_Next_Task -- Task Found --> Load_Task_File[Load Task Instruction File]
        Find_Next_Task -- No Tasks Left --> End_Phase[Exit Execution Phase]
    end

    subgraph Task Execution
        Load_Task_File --> Load_Context["Load Parent Plan & build-context Package<br>(fallback: show-dependencies + read_file)"]
        Load_Context --> A[Start Step] --> B[Understand Step]
        B --> C["Review Dependencies & Context Package<br>MANDATORY"]
        C --> D{File Modification?}
        D -- Yes --> E[Pre-Action Verification<br> with Context]
        D -- No --> G[Perform Action]
        E -- Match & Valid --> G
        E -- No Match or Invalid --> F[Re-evaluate Plan/Context]
        F --> B
        G --> H[Document Results]
        H --> I{Error?}
        I -- Yes --> J[Handle Error]
        I -- No --> K[MUP]
        J --> K
        K --> L{Next Step in Task?}
        L -- Yes --> A
        L -- No --> M[End Task - Update Roadmap]
    end
    
    M --> Find_Next_Task
```
