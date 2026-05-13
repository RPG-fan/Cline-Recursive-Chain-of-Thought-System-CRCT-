# **Cline Recursive Chain-of-Thought System (CRCT) - Cleanup/Consolidation Plugin**

**This Plugin provides detailed instructions and procedures for the Cleanup/Consolidation phase of the CRCT system. It should be used in conjunction with the Core System Prompt.**

---

## I. Entering and Exiting Cleanup/Consolidation Phase

**Entering Cleanup/Consolidation Phase:**
1. **`.clinerules` Check**: Always read `.clinerules` first. If `[LAST_ACTION_STATE]` shows `next_phase: "Cleanup/Consolidation"`, proceed with these instructions. This phase typically follows the Execution phase.
2. **User Trigger**: Start a new session if the system is paused after Execution, awaiting this phase.

**Exiting Cleanup/Consolidation Phase:**
1. **Completion Criteria:**
   - Consolidation steps (Section III) are complete: relevant information integrated into persistent docs, changelog reorganized.
   - Cleanup steps (Section IV) are complete: obsolete files identified and archived/removed.
   - `activeContext.md` reflects the clean, consolidated state.
   - MUP is followed for all actions.
2. **`.clinerules` Update (MUP):**
   - Typically transition back to Set-up/Maintenance for verification or Strategy for next planning cycle:
     ```
     [LAST_ACTION_STATE]
     last_action: "Completed Cleanup/Consolidation Phase"
     current_phase: "Cleanup/Consolidation"
     next_action: "Phase Complete - User Action Required"
     next_phase: "Set-up/Maintenance" # Or "Strategy" if planning next cycle immediately
     ```
   - *Alternative: If the project is now considered fully complete:*
     ```
     [LAST_ACTION_STATE]
     last_action: "Completed Cleanup/Consolidation Phase - Project Finalized"
     current_phase: "Cleanup/Consolidation"
     next_action: "Project Completion - User Review"
     next_phase: "Project Complete"
     ```
3. **User Action**: After updating `.clinerules`, pause for user to trigger the next phase.

---

## II. Phase Objective

**Objective**: To systematically review the project state after a cycle of execution, consolidate essential information and learnings into persistent documentation (HDTA, core files), **reorganize the changelog for better readability**, and clean up temporary or obsolete files (like completed Task Instructions, session trackers, and temporary consolidation notes) to maintain a focused and relevant project context.

**Workflow Order**: Consolidation MUST happen *before* Cleanup.

---

## III. Consolidation Workflow

**Goal**: Synthesize key information, decisions, and learnings from the recent execution cycle into the core project documentation, and reorganize the changelog by component/module.

<<<***CRITICAL WARNING***>>>
*You **must** verify the base file's state by manually reading it to determine completion status. For code files read the actual code. For documentation, read the associated documents.*

**Procedure:**

1.  **Review and Verify All Relevant Project Documentation and Task States (CRITICAL)**:
    *   **a. Identify HDTA Document Structure and Other Strategic Trackers**:
        *   **Action**: Review files in `cline_docs/templates/` to understand the standard structure and expected content for all HDTA document tiers (`system_manifest.md`, `*_module.md`, `implementation_plan_*.md`, `task_instruction.md`).
        *   **Action**: Use `list_files` to search within `cline_docs/` (and potentially other relevant project documentation directories if custom locations are used) for any files matching patterns like `*roadmap*.md`, `*checklist*.md`, `*review_progress*.md`, or other strategic tracking documents.
        *   **Purpose**: To ensure a complete understanding of the project's intended documentation structure and to identify all high-level tracking documents that need review and consolidation.
    *   **b. List All Task Instruction Files**:
        *   **Action**: Use `list_files` recursively to identify all `*.md` files within the primary `tasks/` directory (and its subdirectories).
        *   **Action**: Use `list_files` recursively to identify all `*.md` files within the *entire* `cline_docs/archive/` directory (and its subdirectories) to locate any previously archived task files.
        *   **Purpose**: Create a comprehensive list of all available task instruction files, regardless of their current location (active or archived), for subsequent review in batches.
    *   **c. List All Implementation Plan Files**:
        *   **Action**: For each directory listed in `[CODE_ROOT_DIRECTORIES]` (from `.clinerules`), use `list_files` recursively to identify all files matching the pattern `implementation_plan_*.md`. These are typically located within module directories (e.g., `src/module_name/implementation_plan_*.md`).
        *   **Purpose**: Create a comprehensive list of all implementation plan files for subsequent review in batches.
    *   **d. Read Core Project State Files**:
        *   **Action**: Read `activeContext.md`: Identify key decisions, unresolved issues, and summaries of work done.
        *   **Action**: Read `changelog.md`: Review all significant changes. **(Note: Will be reorganized in Step 3)**.
        *   **Action**: Read `progress.md`: Check for all high-level milestones.
        *   **Purpose**: To gather current project state information that will inform the consolidation process.
    *   **e. Review All Identified Task Instruction Files in Batches (CRITICAL)**:
        *   **Purpose**: To identify significant implementation details, design choices, "gotchas," learnings, and to **CRITICALLY verify the actual completion status of tasks.** This review covers all task instruction files identified in step 1b, processed in batches to prevent context overload.
        *   **Procedure**:
            i. **Batch Creation**:
                - **Action**: Divide the comprehensive list of Task Instruction files (from step 1b) into batches, with each batch containing **no more than 10 files**. If the total number of files is not evenly divisible by 10, the final batch may contain fewer than 10 files.
                - **Action**: Maintain a record of which files have been processed to ensure no files are missed or processed twice. This can be done internally or by updating a temporary tracking mechanism (e.g., a note in `activeContext.md` for the duration of this phase).
            ii. **Process Each Batch (Standalone Processing)**:
                - **Action**: For each batch of up to 10 Task Instruction files, **fully process the batch as a standalone task** before proceeding to the next batch. This means completing all verification, extraction, and updating actions for all files in the batch with no inter-batch dependencies for these actions.
                - **Sub-Procedure (for each file in the batch)**:
                    1. **Read the Task File**:
                        - **Action**: Use `read_file` to read the content of the task file.
                    2. **Verify Completion (Manual & CRITICAL)**:
                        - **Verification Approach**:
                            - For **Execution Tasks**: If the task file indicates a tangible action was taken on a project artifact (e.g., "applied x to y," "created file z," "modified function f in file w," "updated documentation q"), you **MUST** manually verify this outcome by examining the target artifact.
                            - For **Strategy Tasks**: Verification involves confirming that the planned output of the strategy task (e.g., a design document, a research summary, a set of defined requirements, a completed exploration) has been produced, is complete, and meets the task's objectives. This may involve reading the output document(s) or assessing the completeness of the strategic analysis presented in the task file itself.
                        - **Action**: Use `read_file` to examine target artifacts or output documents, or use `list_files` to confirm existence/modification as appropriate. Consult `changelog.md` entries if they provide specific file modification details related to the task.
                        - **If the outcome is NOT verified** (e.g., 'x' was not applied to 'y', file 'z' does not contain expected content, function 'f' was not modified as described):
                            - Clearly note this discrepancy for this task file.
                            - **Action (CRITICAL)**: If the task file has a "Status: Completed" marker (or all its internal steps are checked off implying completion), this status is now considered **invalid**. You **MUST** update the task file itself to remove or clearly mark the "Completed" status as incorrect/unverified (e.g., change to "Status: Pending Verification" or "Status: Incomplete - Outcome Not Verified").
                            - **Action (CRITICAL)**: Identify ALL documents that reference this task as completed. This includes (but is not limited to) parent Implementation Plan(s), any `*checklist*.md` files, `*roadmap*.md` files, `progress.md`, and potentially `activeContext.md`. Update these referencing documents to reflect the task's true (unverified/incomplete) status.
                            - **Note**: This task file **MUST NOT** be archived as "complete" in Section IV. It may require a new task to be created in a subsequent phase to address the incompletion.
                    3. **Extract Consolidatable Information (CRITICAL)**:
                        - Regardless of verified completion status, identify any design decisions, new information, important learnings, "gotchas," or deviations from original plans noted *within* the task file.
                        - **Action**: Record this information in a temporary file, `consolidation_notes.md`, located in `cline_docs/`. Append each piece of information with a reference to the source file and batch number (e.g., "Batch 1, task_abc.md: Learned that algorithm X is suboptimal for large datasets"). Use `write_to_file` or `apply_diff` to update `consolidation_notes.md`.
                - **Action**: After fully processing all files in the current batch (i.e., all files have been read, verified, updated if necessary, and their consolidatable information recorded in `consolidation_notes.md`), document the completion of the batch in `activeContext.md` (e.g., "Completed verification and extraction for batch X containing files [file1, file2, ...]. Information recorded in `consolidation_notes.md`."). Only then proceed to the next batch.
                - **Purpose**: Processing each batch as a standalone task ensures that all critical actions are completed without relying on future batches, reducing context overload while maintaining comprehensive verification and extraction.
    *   **f. Review All Identified Implementation Plan Files in Batches (CRITICAL)**:
        *   **Purpose**: To consolidate strategic decisions, outcomes, and ensure alignment with completed (and verified) tasks. This review covers all files identified in step 1c, processed in batches to prevent context overload.
        *   **Procedure**:
            i. **Batch Creation**:
                - **Action**: Divide the comprehensive list of Implementation Plan files (from step 1c) into batches, with each batch containing **no more than 10 files**. If the total number of files is not evenly divisible by 10, the final batch may contain fewer than 10 files.
                - **Action**: Maintain a record of which files have been processed to ensure no files are missed or processed twice.
            ii. **Process Each Batch (Standalone Processing)**:
                - **Action**: For each batch of up to 10 Implementation Plan files, **fully process the batch as a standalone task** before proceeding to the next batch. This means completing all reading, analysis, updating, and extraction actions for all files in the batch with no inter-batch dependencies for these actions.
                - **Sub-Procedure (for each file in the batch)**:
                    1. **Read the Implementation Plan**:
                        - **Action**: Use `read_file` to read the content of the implementation plan.
                    2. Identify any high-level strategic decisions, architectural changes, or overall outcomes described in the plan.
                    3. Cross-reference the tasks listed within the plan against the verification status determined in step 1e. Update the Implementation Plan to accurately reflect the true completion status of its child tasks.
                    4. **Extract Consolidatable Information**:
                        - Earmark any significant strategic information not yet captured in higher-level HDTA documents (like `system_manifest.md` or `*_module.md` files).
                        - **Action**: Record this information in `consolidation_notes.md` in `cline_docs/`, appending each piece with a reference to the source file and batch number (e.g., "Batch 2, implementation_plan_feature_y.md: Decision to use microservices for scalability").
                - **Action**: After fully processing all files in the current batch (i.e., all files have been read, updated, and their consolidatable information recorded in `consolidation_notes.md`), document the completion of the batch in `activeContext.md` (e.g., "Completed verification and extraction for batch Y containing implementation plans [plan1, plan2, ...]. Information recorded in `consolidation_notes.md`."). Only then proceed to the next batch.
                - **Purpose**: Processing each batch as a standalone task ensures manageable processing of Implementation Plans, maintaining alignment with verified task statuses.
    *   **g. Review Other Strategic Tracking Documents (Roadmaps, Checklists, etc.) in Batches (CRITICAL)**:
        *   **Purpose**: To ensure all high-level tracking documents are up-to-date, and that incomplete items from older versions are not lost. This review covers all files identified in step 1a (excluding HDTA templates), processed in batches to prevent context overload.
        *   **Procedure**:
            i. **Batch Creation**:
                - **Action**: Divide the comprehensive list of Strategic Tracking documents (from step 1a, e.g., `*roadmap*.md`, `*checklist*.md`, `*review_progress*.md`, and explicitly `final_review_checklist.md`) into batches, with each batch containing **no more than 10 files**. If the total number of files is not evenly divisible by 10, the final batch may contain fewer than 10 files.
                - **Action**: Maintain a record of which files have been processed to ensure no files are missed or processed twice.
            ii. **Process Each Batch (Standalone Processing)**:
                - **Action**: For each batch of up to 10 Strategic Tracking documents, **fully process the batch as a standalone task** before proceeding to the next batch. This means completing all reading, consolidation, updating, and extraction actions for all files in the batch with no inter-batch dependencies for these actions.
                - **Sub-Procedure (for each file in the batch)**:
                    1. If multiple versions of the same conceptual tracker exist (e.g., `project_checklist_v1.md`, `project_checklist_v2.md`):
                        - **Action**: Read all versions within the batch.
                        - **Action (CRITICAL)**: Identify the *newest* version. Consolidate all incomplete or pending items from *older* versions into this newest version.
                        - **Action (CRITICAL)**: Ensure all significant completed items and learnings noted in older versions are appropriately reflected in persistent project documentation (HDTA, changelog, etc.) or carried over to the newest tracker version if still relevant for context.
                        - Once an older version is fully consolidated (all its unique, still-relevant information is transferred), it can be considered for archival in Section IV. The newest version becomes the active tracker.
                    2. For the active/newest version of each tracker, review its items against the verified task statuses (from step 1e) and Implementation Plan reviews (step 1f). Update the tracker to accurately reflect project progress.
                    3. **Extract Consolidatable Information**:
                        - Earmark any strategic insights or status updates for broader consolidation (e.g., into `activeContext.md` or `progress.md`).
                        - **Action**: Record this information in `consolidation_notes.md` in `cline_docs/`, appending each piece with a reference to the source file and batch number (e.g., "Batch 3, roadmap_v3.md: Updated milestone priorities based on task delays").
                - **Action**: After fully processing all files in the current batch (i.e., all files have been read, consolidated, updated, and their consolidatable information recorded in `consolidation_notes.md`), document the completion of the batch in `activeContext.md` (e.g., "Completed verification and extraction for batch Z containing trackers [tracker1, tracker2, ...]. Information recorded in `consolidation_notes.md`."). Only then proceed to the next batch.
                - **Purpose**: Processing each batch as a standalone task ensures manageable processing of Strategic Trackers, maintaining comprehensive consolidation.

2.  **Identify All Information for Consolidation (CRITICAL)**:
    *   Based on the comprehensive review performed in Step 1 (covering all task instructions, implementation plans, strategic trackers, and core state files), **CRITICALLY** list all specific pieces of information that represent lasting design decisions, architectural changes, significant outcomes, refined requirements, important operational learnings, "gotchas," or any other vital knowledge that **MUST** be integrated into persistent project documentation. This list is not limited to findings from only the most recent operational cycle but encompasses the entire project history as reviewed. (Excluding changelog structural reorganization for this step, which is handled in Step 3b).

3.  **Update Persistent Documentation & Reorganize Changelog**:

    *   **a. Update Standard Documentation (HDTA, Core Files) (CRITICAL)**:
        *   **Purpose**: To ensure all persistent project documentation accurately reflects the consolidated knowledge gathered from `consolidation_notes.md` in Step 2. This is a **CRITICAL** step for maintaining an up-to-date and reliable knowledge base for the project.
        *   **HDTA Documents**:
            *   **Action (CRITICAL)**: Update `system_manifest.md` if the overall architecture, core components, or project goals have evolved or been clarified at any point.
            *   **Action (CRITICAL)**: Update relevant Domain Modules (`*_module.md`) to incorporate refined descriptions, interface changes, key implementation notes, or any other significant learnings discovered.
            *   **Action (CRITICAL)**: Update relevant Implementation Plans (`implementation_plan_*.md`) with notes on final outcomes, deviations from original plans, or significant decisions made during any implementation effort. Ensure they accurately reflect the verified completion status of their child tasks.
            *   **Procedure**: For each HDTA document requiring updates: Use `read_file` to load the target document, integrate the consolidated information logically and clearly, and use `write_to_file` to save changes. **State reasoning for each update, referencing the source of the consolidated information (e.g., specific task file, `activeContext.md` insight).** Example: "Consolidating final algorithm choice for module Y (from archived task `cline_docs/archive/tasks/task_abc.md`) into `src/module_y/module_y_module.md`."
        *   **Core Files**:
            *   **Action (CRITICAL)**: Update `progress.md` to accurately mark all completed high-level checklist items based on verified outcomes.
            *   **Action (CRITICAL)**: Update `userProfile.md` with any newly observed or reinforced user preferences or interaction patterns.
            *   **Action (CRITICAL)**: Review and Consolidate `.clinerules` `[LEARNING_JOURNAL]`:
                i.  **Action**: Read the current `[LEARNING_JOURNAL]` section from `.clinerules`.
                ii. **Purpose**: To refine the journal by grouping similar learnings, combining related entries for conciseness, removing entries that are not strategic or system-level learnings (e.g., very minor tactical notes better suited for `activeContext.md` during a specific task, or temporary observations that are no longer relevant), and ensuring entries are clearly articulated.
                iii. **Procedure**:
                    - Identify entries that are redundant or cover very similar points. Combine them into a single, more comprehensive entry.
                    - Identify entries that are too granular or represent temporary states rather than lasting learnings. Consider removing these if their value is not persistent.
                    - Identify entries that are not appropriate for the Learning Journal's purpose (e.g., simple reminders, task-specific notes that don't represent broader learning). Remove these.
                    - Ensure remaining entries are clear, concise, and genuinely reflect significant learnings about the CRCT process, project management, technical approaches, or user interactions.
                iv. **Action**: Add any *new* significant system-level learnings identified during the comprehensive review (from Step 1e, 1f, 1g) to the refined journal. Example: "Adding to Learning Journal: Comprehensive review during Cleanup/Consolidation revealed a recurring pattern of task underestimation when initial data definitions are incomplete, highlighting the need for more rigorous data strategy upfront."
                v.  **Action**: Use `write_to_file` (or `apply_diff` if more appropriate for `.clinerules` format) to update the `[LEARNING_JOURNAL]` section in `.clinerules` with the consolidated and newly added entries.

    *   **b. Consolidate and Reorganize Changelog (CRITICAL)**:
        *   **Purpose**: To transform the `changelog.md` into a more readable and maintainable format by structuring all historical entries by their primary component/module and then chronologically within each component. This provides a clear, organized history of changes for the entire project lifecycle. This is a **CRITICAL** step for long-term project understanding and maintainability.
        *   **Goal**: Reformat `changelog.md` by grouping entries under component/module headings, sorted chronologically (newest first) within each group.
        *   **Action: Read**: Use `read_file` to load the current content of `changelog.md`.
        *   **Action: Process Internally**:
            1.  **Parse Entries**: Mentally (or by outlining the steps) parse the loaded text into individual changelog entries (likely delimited by `---` or `### Heading - Date`). Extract the Date, Summary, Files Modified list, and the full text block for each entry.
            2.  **Determine Component**: For each entry, determine its primary component/module based on the `Files Modified` paths. Use heuristics:
                *   If most/all files are in `src/module_name/`, component is `Module: module_name`.
                *   If most/all files are in `docs/category/`, component is `Documentation: category`.
                *   If files are in `cline_utils/` or `cline_docs/`, component is `CRCT System`.
                *   If files span multiple major areas, choose the most representative one or create a `Cross-Cutting` category.
                *   Use a default `General` category if no clear component is identifiable.
            3.  **Group Entries**: Create internal lists, grouping the parsed entries by their determined component.
            4.  **Sort Groups**: Within each component group, sort the entries strictly by Date (most recent date first).
            5.  **Format Output**: Construct the *entire new text content* for `changelog.md`.
                *   Start with the main `# Changelog` heading.
                *   For each component group:
                    *   Add a component heading (e.g., `## Component: Game Loop` or `## Documentation: Worldbuilding`).
                    *   List the sorted entries for that component, preserving their original `### Summary - Date`, `Description`, `Impact`, `Files Modified` structure.
                    *   Use `---` between individual entries within the component group.
                *   *(Optional: Add a more distinct separator like `***` between different component groups if helpful for readability)*.
        *   **Action: Write**: Use `write_to_file` to overwrite `changelog.md` with the *complete, reformatted content* generated in the previous step.
        *   **State**: "Reorganized `changelog.md`. Read existing content, parsed entries, grouped by component (e.g., Game Loop, Documentation, CRCT System), sorted entries by date within each group, and overwrote the file with the new structure."

    *   **c. `activeContext.md` (Final Pass & CRITICAL Update)**:
        *   **Action (CRITICAL)**: After all other information has been consolidated into persistent documents (HDTA, core files) and the changelog has been reorganized, update `activeContext.md` one last time.
        *   **Goal**: To ensure `activeContext.md` accurately reflects the *current, fully consolidated baseline state of the entire project*. This involves removing any transient details specific to *any previously completed work cycles or outdated project states* (e.g., step-by-step execution logs from past tasks, outdated considerations, resolved issues that are now documented elsewhere). The file should retain only the current high-level project status, truly outstanding issues that require immediate or near-term attention, and clear pointers to where detailed, persistent information now resides (e.g., "Final design details for feature Y documented in `implementation_plan_feature_y.md`. Changelog comprehensively reorganized. Next focus: Phase X based on `roadmap_v3.md`.").

4.  **MUP**: Perform Core MUP and Section V additions after completing the consolidation steps (including changelog). Update `last_action` in `.clinerules` to indicate consolidation is finished and cleanup is next.

---

## IV. Cleanup Workflow

**Goal**: Remove or archive obsolete files and data to reduce clutter and keep the project context focused on active work. **Proceed only after Consolidation (Section III) is complete.**

**Procedure:**

1.  **Identify Cleanup Targets (CRITICAL)**:
    *   **CRITICAL Pre-condition**: This step relies entirely on the comprehensive review and verification performed in Section III. Only files confirmed as fully completed, verified, and whose essential information has been consolidated into persistent documentation are eligible for cleanup.
    *   **a. Identify Completed and Consolidated Task Instruction Files**:
        *   Refer to the outcomes of Section III, Step 1e. Task Instruction files that were:
            i.  Verified as genuinely completed.
            ii. Had all their critical information (learnings, design choices, "gotchas") successfully consolidated into persistent HDTA documents or the Learning Journal.
        *   These files are primary candidates for archival. **Task files that were found to be unverified or incomplete in Section III, Step 1e, MUST NOT be targeted for cleanup as "complete" items.**
    *   **b. Identify Fulfilled Strategy Task Files**:
        *   Refer to the outcomes of Section III, Step 1f and 1g. Strategy task files whose objectives have been fully met by downstream Execution tasks (which themselves are verified complete and consolidated) and whose own content has been fully consolidated are candidates for archival.
    *   **c. Identify Obsolete Temporary Session Files and Trackers**:
        *   Refer to the outcomes of Section III, Step 1g. Older versions of strategic tracking documents (roadmaps, checklists, review progress files) that have had all their pending items and unique valuable information consolidated into a newer active version (or into persistent HDTA documents) are candidates for archival.
        *   Identify any other temporary session-specific files (e.g., ad-hoc notes from a past phase that are now fully processed and consolidated) that are no longer relevant to the current project state.
    *   **d. Identify Temporary Consolidation Notes File**:
        *   **Action**: Identify `consolidation_notes.md` in `cline_docs/` as a temporary file created during the Consolidation Workflow (Section III). Since its contents have been fully processed and integrated into persistent documentation in Section III, Step 3, it is now obsolete and a candidate for archival.
    *   **e. Identify Other Obsolete Files**:
        *   Consider other temporary files or logs if any were created during any project phase and are confirmed to be no longer relevant and their information (if any) has been consolidated.

2.  **Determine Cleanup Strategy (Archive vs. Delete)**:
    *   **Recommendation**: Archiving is generally safer than permanent deletion.
    *   **Determine Project Root**: Identify the absolute path to the project's root workspace directory from your current environment context. Let's refer to this as `{WORKSPACE_ROOT}`. **Do not hardcode paths.**
    *   **Proposal**: Propose creating an archive structure if it doesn't exist, using **absolute paths**.
        *   Example absolute paths for archive dirs: `{WORKSPACE_ROOT}/cline_docs/archive/tasks/`, `{WORKSPACE_ROOT}/cline_docs/archive/session_trackers/`.
    *   **Action**: First, use `list_files` to check if the proposed archive directories (e.g., `{WORKSPACE_ROOT}/cline_docs/archive/tasks/`, `{WORKSPACE_ROOT}/cline_docs/archive/session_trackers/`) already exist if you have not already done so in a previous step. If they do not, then propose creating them using `execute_command`. Propose the appropriate OS-specific command (e.g., `mkdir -p` for Unix-like, `New-Item -ItemType Directory -Force` for PowerShell, `mkdir` for CMD which might require checking for existence first or handling an error if it already exists) using the absolute path. **Use `ask_followup_question` to confirm this specific command** or allow the user to provide an alternative. Prioritize using the environment details to determine the user's shell for more accurate initial suggestions. If the directories already exist, this creation step can be skipped.
        ```xml
        <!-- Determine Workspace Root as {WORKSPACE_ROOT} -->
        <!-- Proposing command to create archive directories. -->
        <ask_followup_question>
          <question>Create archive directories? Proposed command (uses absolute paths, tailored to detected OS/shell):
          `[Proposed Command Here]`
          Is this command correct for your OS/shell?</question>
          <follow_up>
            <suggest>Yes, execute this command</suggest>
            <suggest>No, I will provide the correct command</suggest>
          </follow_up>
        </ask_followup_question>
        ```
        *   If user selects "Yes", proceed with `execute_command` using the proposed command.
        *   If user selects "No", wait for their input and use that in `execute_command`.
        *(Note: Quoting paths is good practice, especially if the root path might contain spaces. Be mindful of shell-specific syntax for multiple directories or force options.)*

3.  **Execute Cleanup (Using `execute_command` with User Confirmation via `ask_followup_question`) (CRITICAL)**:
    *   **Input**: This step processes the list of files deemed eligible for cleanup (archival or deletion) as determined by the rigorous verification and consolidation checks in Section IV, Step 1, including `consolidation_notes.md`.
    *   **List Files**: Use `list_files` (which uses relative paths based on workspace) to confirm the current existence and *relative paths* of files targeted for cleanup *from the eligible list*.
    *   **Construct Absolute Paths**: For each relative path identified for cleanup (e.g., `tasks/some_task.md`), construct its corresponding **absolute path** by prepending the determined `{WORKSPACE_ROOT}` (e.g., `{WORKSPACE_ROOT}/tasks/some_task.md`). Do the same for target archive locations.
    *   **Propose Actions and Get Command Confirmation (MANDATORY `ask_followup_question` Step)**:
        *   For each file or group of files to be cleaned up:
            1.  Clearly formulate the **question** stating the intended action (archive/delete) and the full absolute path(s) involved.
            2.  Generate **suggested commands** (as `<suggest>` options) for common OS/shell combinations (Linux/macOS/Git Bash, Windows CMD, Windows PowerShell), using the determined `{WORKSPACE_ROOT}` and appropriate path separators (`/` or `\`) for each suggestion. **Prioritize the suggestion matching the detected OS/shell.**
            3.  Include a final `<suggest>` option like: `Provide alternative command`.
            4.  Use the `ask_followup_question` tool with the question and these suggestions.
    *   **Note on Shell Accuracy**: Be aware that the detected shell might not always be the active one. If command execution fails, re-verify the user's active shell or ask for explicit confirmation.
    *   **Example Interaction Flow**: (Examples remain relevant, but suggestions should be tailored and prioritize detected shell)
        *   **LLM**: *(Constructs absolute paths internally based on `{WORKSPACE_ROOT}`)*
          ```xml
          <ask_followup_question>
            <question>Archive completed task file "{WORKSPACE_ROOT}/tasks/Execution_integrate_input_handler_game_loop_task.md" to "{WORKSPACE_ROOT}/cline_docs/archive/tasks/"? Select the correct command for your system:</question>
            <follow_up>
              <suggest>mv "{WORKSPACE_ROOT}/tasks/Execution_integrate_input_handler_game_loop_task.md" "{WORKSPACE_ROOT}/cline_docs/archive/tasks/"</suggest> <!-- Prioritize if Linux/macOS detected -->
              <suggest>move "{WORKSPACE_ROOT}\tasks\Execution_integrate_input_handler_game_loop_task.md" "{WORKSPACE_ROOT}\cline_docs\archive\tasks\"</suggest> <!-- Prioritize if Windows detected -->
              <suggest>Provide alternative command</suggest>
            </follow_up>
          </ask_followup_question>
          ```
        *   **User**: *(Selects a command or provides alternative)*
        *   **LLM**: "Okay, executing the selected command:"
          ```xml
          <execute_command>
          <command>[User Confirmed or Provided Command]</command>
          </execute_command>
          ```
        *   *(Wait for tool result, document action, then proceed to next file)*
    *   **Document Actions**: After successful execution (based on tool output), clearly log which files were archived or deleted using the confirmed commands and absolute paths.

4.  **Final Verification (CRITICAL)**:
    *   **Action (CRITICAL)**: If any archive or delete operations were performed in Step 3, use `list_files` again with the original *relative* locations of the processed files to verify they are no longer present in those locations.
    *   **Action (CRITICAL)**: Ensure `activeContext.md` is clean and does not reference the removed/archived files unless it is explicitly pointing to their new archive location for historical reference. All other pointers should be to active, persistent documentation.

5.  **MUP**: Perform Core MUP and Section V additions after completing cleanup. Update `last_action` and `next_phase` in `.clinerules` to signify the end of this phase.

**Cleanup Flowchart**
```mermaid
flowchart TD
    A[Start Cleanup (Post-Consolidation)] --> B[Identify Cleanup Targets]
    B --> B1[Determine Absolute Workspace Root `{WORKSPACE_ROOT}`]
    B1 --> C{Archive Structure Exists?}
    C -- No --> D[Use `ask_followup_question` to Confirm `mkdir` command w/ Absolute Paths]
    D -- Confirmed --> D1[Execute Confirmed `mkdir` command]
    C -- Yes --> E
    D1 --> E
    E --> F[List Target Files]
    F --> G[For each file/group:]
    G --> G1[Construct Absolute Paths for Source & Target]
    G1 --> H[1. State Intent<br>Archive/Delete]
    H --> I[2. Generate OS-specific command suggestions w/ Absolute Paths]
    I --> J[3. Use `ask_followup_question` w/ suggestions + "Provide Alternative"]
    J -- User Selects Suggested Command --> K[Execute Selected Command via `execute_command`]
    J -- User Selects "Provide Alternative" --> J1[Wait for User Command Input]
    J1 --> K2[Execute User-Provided Command via `execute_command`]
    K --> L[Document Action]
    K2 --> L
    L --> M{More files?}
    M -- Yes --> G
    M -- No --> N[Verify Files Moved/Removed]
    N --> O[MUP & Update .clinerules to Exit Phase]
    O --> P[End Cleanup]

    style J fill:#f9f,stroke:#f6f,stroke-width:2px,color:#000
    style B1 fill:#e6f7ff,stroke:#91d5ff
    style G1 fill:#fffbe6,stroke:#ffe58f
```

---

## V. Cleanup/Consolidation Plugin - MUP Additions (CRITICAL)

**CRITICAL**: These steps **MUST** be performed in addition to Core MUP steps at the appropriate junctures.

1.  **Verify `activeContext.md` State (CRITICAL)**: After any significant consolidation or cleanup action, and especially at the MUP points defined in Section III.4 and IV.5, **CRITICALLY** verify that `activeContext.md` accurately reflects the current, clean, and consolidated state. Ensure it points to persistent documents for details and that all transient information from now-completed cycles or outdated states has been removed.
2.  **Verify `changelog.md` Structure (CRITICAL)**: After the changelog reorganization (Section III.3b), and at the MUP point in Section III.4, **CRITICALLY** verify that the `changelog.md` structure correctly reflects the component grouping and chronological sorting as intended.
3.  **Update `.clinerules` [LAST_ACTION_STATE] (CRITICAL)**:
    *   **After Consolidation step is fully completed (including changelog reorganization - as per Section III.4)**:
      ```
      [LAST_ACTION_STATE]
      last_action: "Completed ALL Consolidation Steps (incl. Changelog Reorg)"
      current_phase: "Cleanup/Consolidation"
      next_action: "Begin Cleanup Workflow"
      next_phase: "Cleanup/Consolidation"
      ```
    *   **After Cleanup step is fully completed (exiting phase - as per Section IV.5)**:
      ```
      [LAST_ACTION_STATE]
      last_action: "Completed Cleanup/Consolidation Phase (All Steps)"
      current_phase: "Cleanup/Consolidation"
      next_action: "Phase Complete - User Action Required to transition to next phase"
      next_phase: "Set-up/Maintenance" # Or "Strategy" or "Project Complete"
      ```

---

## VI. Quick Reference (All Steps are CRITICAL)

- **Objective**: **CRITICALLY** and comprehensively review the **entire project state**. Consolidate all verified learnings, outcomes, and essential information into persistent documentation. **Reorganize `changelog.md` by component/date for the entire project history.** Archive or remove obsolete files based on rigorous verification and consolidation.
- **Order**: Consolidation (Section III) MUST be fully completed BEFORE Cleanup (Section IV).
- **Consolidation (Section III)**:
    - **Inputs (Comprehensive Review)**:
        - HDTA Templates (`cline_docs/templates/`)
        - All Task Instruction files (from `tasks/` and `cline_docs/archive/`)
        - All Implementation Plan files (from Code Root directories)
        - All Strategic Tracking documents (roadmaps, checklists from `cline_docs/`, etc.)
        - Core state files: `activeContext.md`, `changelog.md` (entire history), `progress.md`
    - **Actions (All Mandatory & CRITICAL)**:
        1.  Review HDTA templates; List all Task Instructions, Impl. Plans, Strategic Trackers. Process in batches of ≤10 files; **Fully process each batch as a standalone task**.
        2.  For ALL Task Instructions: Read, **MANUALLY VERIFY OUTCOMES** (if outcome unverified, update task file & all references to show NOT complete; unverified tasks are NOT archived as complete). Extract ALL learnings/design choices.
        3.  For ALL Impl. Plans: Read, cross-reference task verification, update plan status, extract strategic info.
        4.  For ALL Strategic Trackers: Review, consolidate older versions into newest, update status based on verified tasks.
        5.  Identify ALL information for consolidation from the above reviews.
        6.  Update HDTA docs (`system_manifest.md`, `*_module.md`, `implementation_plan_*.md`).
        7.  Update Core Files: `progress.md`, `userProfile.md`.
        8.  Review, Refine, & Update `.clinerules` `[LEARNING_JOURNAL]` (group, combine, remove inappropriate, add new).
        9.  Reorganize ENTIRE `changelog.md` (Parse->Group by Component->Sort by Date->Format->Write).
        10. Update `activeContext.md` to reflect fully consolidated project baseline.
    - **Tools**: `list_files`, `read_file`, `write_to_file`, `apply_diff`.
- **Cleanup (Section IV)**:
    - **Inputs (Derived from Section III)**: Verified list of fully completed & consolidated Task Instructions; Fulfilled Strategy Tasks; Obsolete (fully consolidated) session files/trackers; Other confirmed obsolete files.
    - **Actions (All Mandatory & CRITICAL)**:
        1.  Identify cleanup targets **based on Section III's verified outputs.**
        2.  Determine archive strategy (archive preferred); Check/Create archive dirs (confirm command with `ask_followup_question`).
        3.  For each eligible file: Construct absolute paths, confirm archive/delete command with `ask_followup_question`, execute, document.
        4.  Verify files moved/removed (use `list_files`); Ensure `activeContext.md` is clean.
    - **Tools**: `list_files`, `execute_command`, `ask_followup_question`.
- **MUP Additions (Section V) (CRITICAL)**:
    - After Consolidation: Verify `activeContext.md`, `changelog.md`; Update `.clinerules` (last_action: "Completed ALL Consolidation...", next_action: "Begin Cleanup...").
    - After Cleanup (Exiting Phase): Verify `activeContext.md`; Update `.clinerules` (last_action: "Completed Cleanup/Consolidation Phase (All Steps)...", next_action: "Phase Complete...").
