# **Cline Recursive Chain-of-Thought System (CRCT) - Execution Plugin**

**This Plugin provides detailed instructions and procedures for the Execution phase of the CRCT system. It should be used in conjunction with the Core System Prompt.**

---

## I. Entering and Exiting Execution Phase

**Entering Execution Phase:**
1.  **`.clinerules` Check**: Always read `.clinerules` first. If `[LAST_ACTION_STATE]` shows `current_phase: "Execution"` or `next_phase: "Execution"`, proceed with these instructions, resuming from `next_action` if specified.
2.  **Transition from Strategy**: The `project_roadmap.md` file, updated by the Strategy phase, now contains the definitive, sequenced list of `Execution_*` tasks for the current cycle. This list is the primary input for this phase.
3.  **User Trigger**: Start a new session post-Strategy or to resume execution if paused.

**Exiting Execution Phase:**
1. **Completion Criteria:**
   - **All `Execution_*` tasks listed in the current cycle's "Unified Execution Sequence" within `project_roadmap.md` have been marked as complete (`[x]`).**
   - Expected outputs for all tasks are generated and verified.
   - Results and observations are documented.
   - MUP is followed for all actions.
2. **`.clinerules` Update (MUP):**
   - To proceed to cleanup and consolidation:
     ```
     [LAST_ACTION_STATE]
     last_action: "Completed Execution Phase - Tasks Executed"
     current_phase: "Execution"
     next_action: "Phase Complete - User Action Required"
     next_phase: "Cleanup/Consolidation"
     ```
   - *Alternative: If transitioning back to Set-up/Maintenance for re-verification (less common after standard execution)*:
     ```
     [LAST_ACTION_STATE]
     last_action: "Completed Execution Phase - Tasks Executed, Needs Verification"
     current_phase: "Execution"
     next_action: "Phase Complete - User Action Required"
     next_phase: "Set-up/Maintenance"
     ```
   - For project completion:
     ```
     [LAST_ACTION_STATE]
     last_action: "Completed Execution Phase - Project Objectives Achieved"
     current_phase: "Execution"
     next_action: "Project Completion - User Review"
     next_phase: "Project Complete"
     ```
   *Note: "Project Complete" pauses the system; define further actions if needed.*
3. **User Action**: After updating `.clinerules`, pause for user to trigger the next phase. See Core System Prompt, Section III for a phase transition checklist.

---

## II. Loading Context for Execution

**Action**: Load necessary context for the selected Task Instruction, respecting the planning hierarchy and dependencies.

**Procedure:**
**Load the Master Plan (MANDATORY FIRST STEP)**:
    *   `read_file` the `project_roadmap.md`.
    *   State: "Reading `project_roadmap.md` to identify the execution sequence for the current cycle."

1.  **Identify Next Task from Unified Sequence**:
    *   Locate the "Unified Execution Sequence" checklist for the current cycle within the roadmap.
    *   Scan the list for the **first task that is not marked as complete** (i.e., the first `[ ]`). This is the `[Current_Task_File_Path]`.
    *   **If no incomplete tasks are found**: The execution for this cycle is complete. State: "All tasks in the project roadmap's execution sequence are complete. Proceeding to exit phase." **Go to Section I, Exiting Execution Phase.**
    *   **If a task is found**: State: "Next task identified from roadmap: `[Current_Task_File_Path]`."

2.  **Load Parent Plan (Context)**: Read the parent `implementation_plan_*.md` file (or relevant section of `*_module.md`) that contains the task. This provides higher-level objectives and context. State: "Reading parent plan `{plan_name}.md` for task context."
3.  **Load Task Instruction**: Read the specific `Execution_{task_name}.md` file.

### `build-context` Command Reference

Use `build-context` as the **primary** method for loading tracker-backed dependency context during Execution. It assembles a token-budgeted Markdown package centered on target keys, with tiered dependencies and SES fallbacks.

**Invocation** (always via `execute_command`):

```bash
python -m cline_utils.dependency_system.dependency_processor build-context \
  --keys <KEY1> \
  --output cline_docs/temp_context/{key}_context.md
```

| Argument | Required | Purpose |
|----------|----------|---------|
| `--keys` | Yes | Comma-separated target keys |
| `--mode` | No | `auto` (default), `local` (~20k token ceiling), `cloud` (~100k token ceiling) |
| `--max-tokens` | No | Override budget; clamped to mode ceiling |
| `--output` | Recommended | Write Markdown package to disk (avoids flooding `execute_command` stdout) |

**Prerequisites:**
* Run `analyze-project` first if the global key map is missing (error: `"Global key map not found. Please run analyze-project first."`).
* Resolve ambiguous keys (`3Ba2#1`) before passing to `--keys`. Directory keys are omitted by the packager — do not use them as targets.
* Create `cline_docs/temp_context/` if missing. Packages are ephemeral execution artifacts (not changelog entries); cleanup occurs in Consolidation phase.

**Output format** (what the LLM must parse from the generated package):
* `# TARGET CORE LOGIC` — full content of target file(s)
* `# Tier N: ...` — tiered dependencies: `x` (Tier 1) → `<`/`>` (Tier 2) → `d` (Tier 3) → `S` (Tier 4) → `s` (Tier 5)
* `(Full Content)` vs `(SES Signatures Only)` — full file vs structural/signature fallback
* Truncation marker: `> Context ceiling of N tokens reached...` — outer-ring deps were dropped

4.  **Load Dependencies (MANDATORY PRE-EXECUTION STEP)**:

    **Phase A — Resolve Keys**
    *   Determine what the current key for the target file is. The simplest method is to search for the target file name in `cline_utils\dependency_system\core\state\global_key_map.json`
    *   Confirm keys via `[AUTO] STATION_HEADER` when modifying source files.
    *   Disambiguate with `show-keys` if a base key is globally duplicated (use `KEY#N` suffix).

    **Phase B — Build Context Package (PRIMARY)**
    *   Run `build-context` with `--output` to `cline_docs/temp_context/{key}_context.md`.
    *   `read_file` the generated package.
    *   Summarize coverage: which keys loaded as Full Content vs SES Signatures Only vs truncated/omitted.

    **Phase C — Fallback Gap Fill**
    *   **Trigger fallback** when any of:
        *   `build-context` exits non-zero or output contains `# Error:`
        *   Package shows truncation and the task requires omitted tier files
        *   Task lists explicit non-tracked context (Strategy docs, CRCT task files, plans) not in the package — *note: `show-dependencies` does not work on task files*
        *   SES-only content is insufficient for the planned code change
    *   **Fallback steps:**
        1.  Run `show-dependencies --key <key>` for affected keys.
        2.  `read_file` only the **missing** files (do not re-read content already in the package).
        3.  State: "Filling context gaps via `show-dependencies` + `read_file`: `{file_path_1}`, `{file_path_2}`..."
    *   **Failure to gather required context before coding/modification is a HIGH RISK for introducing errors and logical inconsistencies.**

    **In-Code Verification**: When modifying source files, look for `[AUTO] STATION_HEADER` (to verify the file's primary key) and `[AUTO] CONNECTION_MAP` comments near class/function definitions. These provide immediate, localized context of the file's established dependencies.

    **Load Other Explicit Context**: Use `read_file` to load any other specific Task Instructions, documentation files, or code snippets explicitly listed as required context in the current task file (e.g., Strategy task docs referenced in Execution steps).

---

## III. Executing Tasks from Instruction Files

**Action**: Execute the step-by-step plan detailed in the loaded Task Instruction file, maintaining awareness of its place in the hierarchy and its dependencies.

**Procedure:**
1.  **Iterate Through Steps:** For each numbered step in the Task Instruction file:
    *   **A. Understand the Step**: Read the step's description. Clarify the specific action required, considering the overall task objective and the context from the parent Implementation Plan (loaded in Section II).
    *   **B. Review Dependencies & Context (MANDATORY REINFORCEMENT)**: **Before generating or modifying *any* code or significant file content for this specific step:**
        *   **Complex or multi-file steps**: Re-run `build-context` with updated target keys if the prior package may be stale or new files are in scope.
        *   **Localized steps**: Use `show-dependencies --key <target_file_key>` for quick relationship spot-checks (not full re-load).
        *   **CRITICAL**: Ensure you have **read and understood the relevant content from the Section II.4 context package and/or gap-fill `read_file` results**. How does this step interact with those dependencies (e.g., calling functions, using data structures, implementing interfaces)? State: "Confirming understanding of interaction with dependencies `{key_1}`, `{key_2}` based on context package and/or read files before proceeding with step."
    *   **C. Pre-Action Verification (MANDATORY for File Modifications)**: Before using tools that modify files (`replace_in_file`, `write_to_file` on existing files, `execute_command` that changes files):
        *   Re-read the specific target file(s) for this step using `read_file`.
        *   Generate a "Pre-Action Verification" Chain-of-Thought:
            1.  **Intended Change**: Clearly state the modification planned for this step (e.g., "Insert function X at line Y in file Z").
            2.  **Dependency Context Summary**: Briefly summarize how the intended change relates to the critical dependencies reviewed in III.1.B (e.g., "Function X implements interface defined in dependent file A", "Change adheres to data format expected by dependent function B").
            3.  **Expected Current State**: Describe the specific part of the file you expect to see before the change (e.g., "Expect line Y to be empty", "Expect function signature Z to be present").
            4.  **Actual Current State**: Note the actual state observed from the `read_file` output.
            5.  **Validation**: Compare expected and actual state. Proceed **only if** they match reasonably AND the intended change is consistent with the dependency context summary. If validation fails, **STOP**, state the discrepancy, and re-evaluate the step, plan, or dependencies. Ask for clarification if needed.
        *   Example:
            ```
            Pre-Action Verification:
            1. Intended Change: Replace line 55 in `game_logic.py` (Key: 2Ca1) with `new_score = calculate_score(data, multipliers)`.
            2. Dependency Context Summary: `calculate_score` is imported from `scoring_utils.py` (Key: 2Cb3, confirmed via build-context package Tier 2). It expects `data` (dict) and `multipliers` (list). `game_logic.py` has access to these variables in scope.
            3. Expected Current State: Line 55 contains the old calculation `new_score = data['base'] * 1.1`.
            4. Actual Current State: Line 55 is `new_score = data['base'] * 1.1`.
            5. Validation: Match confirmed. Change is consistent with dependency context. Proceeding with `replace_in_file`.
            ```
    *   **D. Perform Action**: Execute the action described in the step using the appropriate tool (`write_to_file`, `execute_command`, `replace_in_file`, etc.).
    *   **E. Document Results (Mini-CoT)**: Immediately after the action, record the outcome:
        *   **Action Taken**: Briefly restate the action performed.
        *   **Result**: Success, failure, command output, generated content snippet.
        *   **Observations**: Any unexpected behavior, potential issues, or insights gained.
        *   **Next**: Confirm moving to the next step or handling an error.
    *   **F. MUP**: Follow Core MUP (Section VI of Core Prompt) and Section IV additions below. **Perform MUP after each step.**

2.  **Error Handling:** If an action fails or produces unexpected results:
    *   Document the error message and the Mini-CoT leading up to it.
    *   Diagnose the cause: Check command syntax, file paths, permissions, dependency conflicts (referencing context from III.1.B), or logical errors in generated code/instructions. Consult Core Prompt Section VIII for dependency command error details if applicable.
    *   Propose a resolution: Correct the command, revise the code logic based on dependency understanding, adjust the task instructions, or query the user if the plan seems flawed.
    *   Execute the fix.
    *   Document the resolution process.
    *   Apply MUP post-resolution before continuing.

3.  **Code Generation and Modification Guidelines:**
    *(Reminder: Before generating/modifying code, ensure Step III.1.B 'Review Dependencies & Context' including the Section II.4 context package was performed)*
    When performing actions that involve writing or changing code, adhere strictly to the following:
    1.  **Context-Driven**:
     - Code **must** align with the interactions, interfaces, data formats, and requirements identified during dependency review (III.1.B) and pre-action verification (III.1.C).
    2.  **Modularity**:
     - Write small, focused functions/methods/classes. Aim for high cohesion and low coupling.
     - Design reusable components to enhance maintainability.
    3.  **Clarity and Readability**:
     - Use meaningful names for variables, functions, and classes.
     - Follow language-specific formatting conventions (e.g., PEP 8 for Python).
     - Add comments only for complex logic or intent, avoiding redundant explanations of *what* the code does.
     - Provide complete, runnable code blocks or snippets as appropriate for the task step.
    4.  **Error Handling**:
     - Anticipate errors (e.g., invalid inputs, file not found) and implement robust handling (e.g., try-except, return value checks).
     - Validate inputs and assumptions to prevent errors early.
    5.  **Efficiency**:
     - Prioritize clarity and correctness but be mindful of algorithmic complexity for performance-critical tasks.
    6.  **Documentation**:
     - Add docstrings or comments for public APIs or complex functions, detailing purpose, parameters, and return values.
     - Keep documentation concise and synchronized with code changes.
    7.  **Testing**:
     - Write testable code and, where applicable, suggest or include unit tests for new functionality or fixes.
    8.  **Dependency Management**:
     - Use existing dependencies where possible. Avoid adding new external libraries unless explicitly planned.
     - If code changes introduce *new functional dependencies* between project files, prepare to update the relevant mini-tracker (see MUP Additions, Section IV).
    9.  **Security**:
     - Follow secure coding practices to mitigate vulnerabilities (e.g., avoid injection risks, secure credential handling).
    10. **WIP Markings & Lifecycle Protocol (CRITICAL)**:
     - Note that any areas or code blocks that require additional steps, deferred logic, or future modifications **require** a clear, descriptive `# WIP` tag (or `# │ WIP:` box-drawing tag, or language-appropriate comment syntax like `// WIP` or `<!-- WIP -->`) directing to the additional work needed. Reference the `comment-skill` package path defined under `[SKILLS_WORKFLOWS]` in `.clinerules/default-rules.md`.
     - **Beacon Removal on Completion**: When a task step fully satisfies and verifies the work described in a `# WIP` beacon, **DELETE** the beacon block. Do not convert it to `DONE:` or retain dead comment scaffolding — git commit history records completion.
     - **NEXT-Item Re-Anchor Rule (MANDATORY)**: Prior to removing any `# WIP` (or `# │ WIP:`) beacon whose `NEXT` field contains uncompleted or follow-up items, the executing agent **MUST** instantiate each remaining item as its own proper, standalone `# WIP` (or `# │ WIP:`) beacon at the specific code site where that work belongs *BEFORE* the parent beacon is deleted. Recording remaining items only in plan files or activeContext is prohibited as it causes information loss.
       - Each re-anchored beacon must include proper fields per `comment-skill-wip.md`: `INTENT`, `STATUS`, `NEXT` (quoting or detailing the item), `REQUIRES`, `CRCT_PHASE`, and `HDTA_TASK` (or plan reference).
     - **Grep Syntax Awareness**: Always use whitespace- and box-drawing-tolerant searches (e.g. regex `#\s*│?\s*WIP:`) to avoid missing indented or box-drawing formatted beacons. Anchor on unique beacon prose rather than line numbers.

4.  **Execution Flowchart**

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

---

## IV. Execution Plugin - MUP Additions

After Core MUP steps (Section VI of Core Prompt), performed *after each step* of the Task Instruction:
1.  **Update Task Instruction File**:
    *   Mark the just-completed step (e.g., add `[DONE]` or similar marker).
    *   Save any significant observations or results from the Mini-CoT directly into the task file as notes for the relevant step, if useful for context later. Avoid changing the core instructions unless correcting an error found during execution.
    *   If the task is now fully complete, update its overall status section.
    *   Use `write_to_file` to save changes.
2.  **Update Mini-Trackers (If New Functional Dependency Created)**:
    *   **Condition**: If the executed step modified code in file A (key `key_A`) such that it *now* directly imports, calls, or functionally relies on code/data in file B (key `key_B`) *within the same module*, and this dependency didn't exist before or wasn't accurately reflected.
    *   **Action**: Use `add-dependency` on the relevant `{module_name}_module.md` mini-tracker.
    *   **Reasoning (Mandatory)**: Clearly state why the dependency is being added/updated based *specifically* on the code change made in this step.
    *   Example (adding dependency from function/file 2Ca1 to 2Ca3 within module 'C' after adding an import):
        ```bash
        # MUP Trigger: Step X added 'from .file3 import specific_func' to file associated with key 2Ca1.
        # Reasoning: This creates a new functional dependency where 2Ca1 now requires 2Ca3 for specific_func.
        python -m cline_utils.dependency_system.dependency_processor add-dependency --tracker path/to/module_C/module_C_module.md --source-key 2Ca1 --target-key 2Ca3 --dep-type "<"
        ```
        *(Use correct dep-type: '<' if A calls B, '>' if B calls A, 'x' if mutual, 'd' if essential doc link)*
3.  **Update WIP Beacons (Completion Cleanup & NEXT Re-anchoring)**:
    *   **Completed Work**: If this step completed the implementation of a `# WIP` (or `# │ WIP:`) beacon, **delete** the beacon block.
    *   **NEXT Re-Anchoring**: If deleting a beacon whose `NEXT` list contains remaining/uncompleted items, verify that each remaining item has been instantiated as a standalone `# WIP` beacon at its respective code site before deleting the parent beacon.
4.  **Update Domain Module / Implementation Plan Documents (If Significant)**: If the task execution led to a significant design change or outcome not captured in the original plan, briefly note this in the relevant Domain Module (`*_module.md`) or Implementation Plan (`implementation_plan_*.md`).
5.  **Update `.clinerules` [LAST_ACTION_STATE]:** Update `last_action`, `current_phase`, `next_action`, `next_phase`.
    *   After a step:
        ```
        [LAST_ACTION_STATE]
        last_action: "Completed Step {N} in Execution_{task_name}.md"
        current_phase: "Execution"
        next_action: "Execute Step {N+1} in Execution_{task_name}.md"
        next_phase: "Execution"
        ```
    *   After completing the last step in a task:
        ```
        [LAST_ACTION_STATE]
        last_action: "Completed all steps in Execution_{task_name}.md"
        current_phase: "Execution"
        next_action: "Select next Execution task or transition phase"
        next_phase: "Execution" # Default, change only when *all* planned tasks are done.
        ```
    *   Upon exiting the phase (as defined in Section I): Use the appropriate state from Section I.

---

## V. Quick Reference
- **Objective**: Execute planned `Execution_*` tasks step-by-step, modifying files/code according to instructions, dependencies, and quality guidelines.
- **Key Actions**:
    - Load context: Parent Plan -> Task Instruction -> **`build-context` package** (fallback: `show-dependencies` + `read_file` for gaps).
    - Execute steps sequentially.
    - **MANDATORY**: Review dependencies & **context package** before coding/modification.
    - **MANDATORY**: Perform pre-action verification for file modifications.
    - Follow code quality guidelines.
    - Document results (Mini-CoT) after each action.
    - Perform MUP after each action.
    - Update mini-trackers (`add-dependency`) if new functional dependencies are created.
- **Key Inputs**: Prioritized Task list (from Strategy), `implementation_plan_*.md`, `Execution_*.md`, context packages (`cline_docs/temp_context/*_context.md` via `build-context`), gap-fill reads (`show-dependencies` + `read_file`).
- **Key Outputs**: Modified project files (code, docs), updated `activeContext.md`, updated task instruction files, potentially updated mini-trackers, updated `.clinerules`.
- **MUP Additions**: Update instruction files (step completion, notes), mini-trackers (if needed), potentially Plans/Modules, and `.clinerules`.
