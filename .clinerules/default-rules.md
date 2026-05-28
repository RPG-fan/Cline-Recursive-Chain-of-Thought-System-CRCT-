[COUNT]
n + 1 = (x) # Do not alter this row.
*This is a systemwide progressive counter where x = number of current response. This must be displayed at the top of every response.(e.g. 1, 2, ...) Only display the current value for x.*

[LAST_ACTION_STATE]
last_action: "System Initialized"
current_phase: "Set-up/Maintenance"
next_action: "Initialize Core Files"
next_phase: "Set-up/Maintenance"

---

[CODE_ROOT_DIRECTORIES]
- src

[DOC_DIRECTORIES]
- docs

[SKILLS_WORKFLOWS]
- .agent/skills/comment-skill

[LEARNING_JOURNAL]

- **Verification Rigor ("NO TRUST ENVIRONMENT!")**: It is critical to explicitly re-verify task outcomes by examining actual code and documentation, rather than solely relying on task file status. This is essential to identify placeholder logic, unaddressed gaps, or inconsistencies between documented status and actual implementation.
- **`apply_diff` Best Practices**: Exercise extreme caution with `apply_diff` operations. Always re-read the target file immediately before constructing an `apply_diff` to ensure an exact match in the `SEARCH` block (including subtle formatting). Avoid using `:start_line:` in the `REPLACE` section.
- **Proactive Task Management**: Proactively creating missing task files based on higher-level implementation plans and scaffolding execution tasks can significantly improve workflow efficiency and prevent context switching.
- **Feedback Integration**: Actively incorporate user feedback and be prepared for course correction, as this directly impacts workflow efficiency and solution accuracy.
- **Changelog Purpose**: Re-learned that `changelog.md` is exclusively for logging significant *project-related* code or documentation changes. CRCT operational changes are tracked in HDTA documents. `changelog.md` is located in `cline_docs/`.
- **SQL Migration Best Practices**: Implementing data migration functions before dropping columns ensures data integrity during schema changes, as demonstrated by successfully applied user feedback.
- **Dependency Analysis Scope**: The `dependency_processor.py` only tracks *project files*, not CRCT system files (like task files). Therefore, `show-dependencies` will not work on task files directly.
- **Sequential Execution Planning**: When planning a complex feature (like the Task Delegation Service), explicitly break down the execution into sequential, dependency-driven sub-tasks (e.g., Data Models -> DB Layer -> Service Creation -> Integration) to manage complexity and ensure correct build order.
- **Sub-task Verification**: Consistently verify sub-task completion by reviewing detailed worker outputs and dispatcher logs, especially after handoffs, rather than relying on high-level summaries or previous instances' intentions.
- **`Move-Item` Error Handling**: When `Move-Item` reports "Cannot create a file when that file already exists", the user expects the item to be moved with a new name to avoid conflict, rather than assuming it's a duplicate or deleting the source.
- **General Tool Usage**: Avoid mentioning tool names until the exact moment of tool invocation. For `execute_command`, provide full paths and appropriate OS-specific commands. Prioritize `apply_diff` and `insert_content` over `write_to_file` for context window management.

**DO NOT USE THE TOOL NAME UNTIL YOU INTEND TO USE IT**

**Project Management & Workflow Strategy**
This section covers the high-level strategies for managing tasks, context, and feedback to ensure the project stays on track.

**Task Management & Verification**

- **Distinguish Task Types:** Clearly differentiate between **Strategy tasks** (planning, design, analysis) and **Execution tasks** (implementation, coding, testing) to maintain focus and clarity.
- **Verify True Completion:** Do not rely on task file status alone. Verify task completion by cross-referencing actual code and documentation changes to identify placeholder logic or unaddressed gaps.
- **Link Documentation to Tasks:** Module trackers and implementation plans are for tracking project files, not the CRCT documentation itself. For tracking progress on documentation review, the HDTA Review Progress Tracker should be updated to reflect the *actual reading* of content.
- **Systematic Consolidation:** A systematic review of all task files during Cleanup/Consolidation phases is crucial for verifying completion status and preventing information loss.

**State & Context Management**

- **Purpose of MUP:** MUP is for state synchronization. For large context windows, perform a pre-transfer MUP with detailed handoff instructions.
- **Maintain `changelog.md`:** The `changelog.md` is for logging significant, project-related code or documentation changes, not for general state synchronization.
- **Update Context Files:** Regularly update `cline_docs/` and instruction files to maintain task context and prevent information loss between work sessions.
- **Confirm Strategic Documents:** Before consolidation, confirm the status of all strategic tracking documents (e.g., checklists, roadmaps) to ensure a consistent project understanding.

**Responding to Feedback & Ambiguity**

- **Prioritize User Feedback:** User feedback is a critical tool for identifying flaws. If a user raises an architectural question or points out a flaw, pause execution and switch to a Strategy task to re-evaluate.
- **Address Architectural Conflicts:** When a task's objectives conflict with established architectural patterns it necessitates a full re-evaluation and update of all planning documents.
- **Return to Strategy for Clarity:** If data definitions for a task are unclear or documentation is incomplete, return to a Strategy task to fully define the data structures and dependencies before proceeding with implementation.

**Dependency Management & Analysis**
Rigorous dependency management is paramount for preventing errors and ensuring system coherence.

**Core Principles**

- **Single Source of Truth:** Always consult the direct output of `show-keys` and `show-dependencies` to identify actual keys and filenames. Never assume names or keys that are not explicitly listed.
- **Adhere to Definitions:** Strictly follow the ground-truth definitions for dependency characters (`<`, `>`, `x`, `d`, `s`, `S`) when updating trackers.

**Process & Verification**

- **Read Before Acting:** Always perform a full dependency analysis by reading *all* dependent files *before* modifying code. This includes all files marked with positive dependency characters (`<`, `>`, `x`, `d`, `s`, `S`).
- **Strict Content Validation:** Perform *strict* content validation before updating dependencies to prevent errors and ensure tracker accuracy.
- **Inform Workers:** Emphasize to all Worker instances the critical importance of using dependency processor commands and reading *all* positive dependencies before planning or modifying code.

**File Operations & Tool Usage**
This section details best practices for interacting with the file system and using specific tools to ensure reliability and efficiency.

**Best Practices for `apply_diff`**

- **Ensure Exact Matches:** The `SEARCH` block must *exactly* match the current file content. If issues arise, use `read_file` to confirm the content before trying again.
- **Handle Timestamps Carefully:** Timestamps in file content are a common cause of `apply_diff` failures. Always re-read the file immediately before attempting to apply a diff to content containing timestamps. Ensure search patterns for timestamps are precise.
- **Proper `SEARCH` Block Syntax:** Ensure the `start_line` argument is *only* used within `SEARCH` blocks.
- **Use `write_to_file` as an Alternative:** If `apply_diff` continues to fail, `write_to_file` is a reliable alternative for comprehensive updates.

**General Command & Tool Usage**

- **Tool Invocation Protocol:** Avoid mentioning any tool name until the moment of outputting the tool request XML.
- **Use Correct Shell Commands:** Use `execute_command` for file system operations, providing full paths. Use the appropriate commands for the user's active shell (`Rename-Item` or `Move-Item` in PowerShell).
- **Batch Operations:** For efficiency, batch file move operations using appropriate shell commands.
- **Determine Active Shell:** Improve accuracy in determining the user's active shell for `execute_command` proposals, and be prepared to ask for clarification if needed.
- **Manage Context Window:** Prioritize `apply_diff` for targeted changes and `insert_content` for additions to manage the context window size effectively.

**Code Quality & System Maintenance**
These principles focus on writing robust, maintainable code and proactively managing system health.

- **Robust Error Handling:** Implement thorough error handling, such as graceful handling of `None` inputs and precise argument matching, especially after refactoring.
- **Centralize Configuration:** Centralize configuration details, such as character priorities, to improve consistency and maintainability.
- **Correct Data Types:** Ensure all functions return the correct data types (e.g., a `list` versus a `set`) to prevent downstream errors.
- **Performance Profiling:** Leverage profiling tools like `cProfile` to identify performance bottlenecks. Note that excessive `glob` calls were a past issue in the `analyze-project` function.
- **Consult Schemas:** Treat database schema files as a form of documentation and consult them to ensure the implementation aligns with the intended data structure.

**Correcting Core Assumptions**
This section is dedicated to specific, critical corrections of previously held incorrect beliefs.

- **Worker Knowledge is Explicit:** Worker instances have no inherent knowledge of the project. All instructions, context, and file paths must be provided explicitly and accurately.
- I must adhere strictly to the rule of reading all involved files before determining dependency relationships, regardless of my initial assumptions.
- Consistently verify sub-task completion by reviewing detailed worker outputs and dispatcher logs, especially after handoffs, rather than relying on high-level summaries or previous instances' intentions.
- When updating HDTA Review Progress Tracker, reflect actual *reading and review* of content, not just file existence.
- During Cleanup/Consolidation, verify task completion by cross-referencing code/doc changes, not just task file status, to address placeholder logic or unaddressed gaps.
- Emphasize to Workers the critical importance of using the dependency processor commands and reading *all* positive dependencies before planning or modifying code.
- Experienced an `apply_diff` failure due to an outdated search block and incorrect line number markers. Learned to re-read the target file immediately before attempting an `apply_diff` and to ensure strict adherence to the `SEARCH` block format (no `:start_line:` in `REPLACE`).

- **File Existence Verification**: Always verify the existence of files using `list_files` before attempting to add dependencies to non-existent files. This prevents errors with `add-dependency` and ensures accurate tracker updates.
- Always thoroughly verify sub-task completion by reading detailed worker outputs and dispatcher logs, especially after a handoff, before marking an area as planned. Do not rely solely on checklist summaries or previous instances' intentions if they couldn't complete full verification.
- Regularly updating {memory_dir} and any instruction files help me to remember what I have done and what still needs to be done so I don't lose track.
- Verify function call arguments match definitions precisely after refactoring.
- When using `apply_diff`, the SEARCH block must match the current file content exactly, without any +/- markers from previous attempts. Use `read_file` to confirm content if unsure. Pay close attention to the `Best Match Found:` block in the error message, as it shows the *actual* content the tool is searching against, which may differ from your intended SEARCH block due to prior edits or subtle discrepancies.
- Remember: The Best Match Found: content in the error message provides the exact string that the apply_diff tool identified in the file. Using this precise string as the SEARCH block for the next attempt should resolve the matching issue.
- Verify data structures passed between functions (e.g., list vs. dict vs. float) when debugging TypeErrors.
- Carefully respect the ground truth definitions for dependency characters when adding/changing dependencies.
- Ensure correct data types returned by functions (e.g., list vs. set) before applying methods like `.union()`.
- Centralizing configuration like character priorities (`config_manager.py`) improves consistency and maintainability over defining them in multiple places.
- Strict content validation before dependency updates prevents errors and improves tracker accuracy.
- Remember to consult database schema files when documenting systems that rely heavily on database storage to ensure documentation aligns with implementation.
- Leveraging the reciprocal system with `add-dependency` by setting '>' from the source to the targets automatically sets the '<' dependency from the targets back to the source and vice versa.
- Use `execute_command` with appropriate shell commands (like `Rename-Item` for PowerShell) for file system operations such as renaming, instead of trying to simulate them with `write_to_file` or `read_file`. *use the full path*
- It is critical to perform dependency analysis and read dependent files *before* attempting code modifications or dependency assignment to ensure all relevant context is considered. Failing to do so leads to errors and wasted effort.
- Creating a new task file for a specific integration step helps maintain clarity and tracking within the CRCT framework.
- Improve accuracy in determining the user's active shell environment when proposing `execute_command` commands, especially on Windows systems where different shells (CMD, PowerShell) have different syntax. Prioritize environment details but be prepared to ask the user for clarification if necessary.
- MUP itself is not a changelog-worthy event or alteration of core project code; only log significant project-related code or documentation changes in the changelog.
- When updating the HDTA Review Progress Tracker, accurately reflect whether the document's content has been *read and reviewed* in the current session, not just whether the file exists or was created. The status checkboxes should reflect *my* processing of the document's content - the "reviewed" status indicates that I have *already* read the content of the file in the current session, not just noted its existence or dependency listing.
- When just adding to a file it is more efficient and less likely to fail if you use the <insert_content> tool. User confirmed this for changelog updates.
- Manually linking documentation dependencies is crucial when automated analysis may not capture conceptual links essential for context.
- Remember to read ALL files with positive dependency characters (`<`, `>`, `x`, `d`, `s`, `S`) before attempting code modifications.
- Putting a task on hold due to unclear data definitions highlights the importance of fully defining data structures and content during the Strategy phase before attempting implementation in the Execution phase.
- It is essential to verify mutual dependencies from the perspective of *each* key involved to ensure the tracker accurately reflects the relationship from all angles and to clear the `(checks needed: ...)` flags in `show-keys` output.
- It is important to read actual code files (e.g., `.py` files) to understand the current implementation state before attempting to plan or modify functionality, especially when user feedback indicates a gap in understanding.
- User feedback is critical for identifying potential flaws in task definitions or sequencing. Pause execution and switch to Strategy if a fundamental architectural or sequencing question arises.
- When a task objective seems to conflict with established architectural patterns (e.g., `db_middleware` calling `task_management` vs. the reverse for the same functions), it's a strong indicator for strategic review.
- Systematic review of task files during Cleanup/Consolidation is crucial for verifying actual completion status against claims and identifying information for persistent documentation.
- Confirming the status of all strategic tracking documents (checklists, roadmaps) against detailed findings is crucial for ensuring a consistent understanding of project state before major consolidation.
- Batching file move operations using appropriate shell commands (e.g., `Move-Item` with an array in PowerShell) is more efficient than moving files one by one.
- It is important to acknowledge when a previous instance made an incorrect assumption and correct the record in the learning journal.
- **Refactoring Scope Precision (CRITICAL)**: When planning internal decomposition/refactoring, explicitly state that the *host* file/class (e.g., `db_middleware.py`) will remain intact as the central orchestrator and that only a *targeted subset* of monolithic logic will be delegated to new, precise sub-modules. A Worker's assumption to split the *entire* class is invalid and requires explicit correction in the plan.

[Character_Definitions]

```
- `<`: Row **functionally relies on** or requires Column for context/operation.
- `>`: Column **functionally relies on** or requires Row for context/operation.
- `x`: Mutual functional reliance or deep conceptual link.
- `d`: Row is documentation **essential for understanding/using** Column, or vice-versa.
- `o`: Self dependency (diagonal only - managed automatically).
- `n`: **Verified no functional dependency** or essential conceptual link.
- `p`: Placeholder (unverified - requires investigation).
- `s`/`S`: Semantic Similarity suggestion (requires verification for functional/deep conceptual link).
```

---

**IMPORTANT**
1. Understand the Objective: Clearly define the goal of the current step.
2. Analyze the Error: Understand the error message and its context.
3. Formulate a Plan: Develop a plan to address the error, step-by-step.
    *Consider all related aspects* (e.g. files, modules, dependencies, etc.)
4. Execute the Plan (Tool Use): Use the appropriate tool to execute *one* step of the plan.
5. Validate the Result: Check if the tool use was successful and if it addressed the error.
6. Iterate: If the error persists, go back to step 2 and refine the plan based on the new information.
**The Changelog is for tracking changes to the *project's* files, not CRCT operations. CRCT operations are tracked in the HDTA documents.**
**Tracker files serve as their own changelog, dependency operations do not belong in Changelog.md**
