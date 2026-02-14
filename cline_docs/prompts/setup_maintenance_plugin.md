# **Cline Recursive Chain-of-Thought System (CRCT) - Set-up/Maintenance Plugin**

**This Plugin provides detailed instructions and procedures for the Set-up/Maintenance phase of the CRCT system. It should be used in conjunction with the Core System Prompt.**

## I. Entering and Exiting Set-up/Maintenance Phase

**Entering Set-up/Maintenance Phase:**
1.  **Initial State**: Start here for new projects or if `.clinerules` shows `next_phase: "Set-up/Maintenance"`.
2.  **`.clinerules` Check**: Always read `.clinerules` first. If `[LAST_ACTION_STATE]` indicates `current_phase: "Set-up/Maintenance"` or `next_phase: "Set-up/Maintenance"`, proceed with these instructions, resuming from the `next_action` if specified.
3.  **New Project**: If `.clinerules` is missing/empty, assume this phase, create `.clinerules` (see Section II), and initialize other core files.

**Exiting Set-up/Maintenance Phase:**
1.  **Completion Criteria:**
    *   All core files exist and are initialized (Section II).
    *   `[CODE_ROOT_DIRECTORIES]` and `[DOC_DIRECTORIES]` are populated in `.clinerules` (Core Prompt Sections X & XI).
    *   `doc_tracker.md` exists and has no 'p', 's', or 'S' placeholders remaining (verified via `show-keys` in Section III, Stage 1).
    *   All mini-trackers (`*_module.md`) exist and have no 'p', 's', or 'S' placeholders remaining (verified via `show-keys` in Section III, Stage 2).
    *   `module_relationship_tracker.md` exists and has no 'p', 's', or 'S' placeholders remaining (verified via `show-keys` in Section III, Stage 3).
    *   **Code-Documentation Cross-Reference completed (Section III, Stage 4), ensuring essential 'd' links are added.**
    *   `system_manifest.md` is created and populated (at least minimally from template).
    *   Mini-trackers are created/populated as needed via `analyze-project`.
2.  **`.clinerules` Update (MUP):** Once all criteria are met, update `[LAST_ACTION_STATE]` as follows:
    ```
    last_action: "Completed Set-up/Maintenance Phase"
    current_phase: "Set-up/Maintenance"
    next_action: "Phase Complete - User Action Required"
    next_phase: "Strategy"
    ```
3.  **User Action**: After updating `.clinerules`, pause for user to trigger the next session/phase. Refer to Core System Prompt, Section III for the phase transition checklist.

## II. Initializing Core Required Files & Project Structure

**Action**: Ensure all core files exist, triggering their creation if missing according to the specifications in the Core System Prompt (Section II).

**Procedure:**
1.  **Check for Existence**: Check if each required file listed in Core Prompt Section II (`.clinerules`, `system_manifest.md`, `activeContext.md`, `module_relationship_tracker.md`, `changelog.md`, `doc_tracker.md`, `userProfile.md`, `progress.md`) exists in its specified location.
2.  **Identify Code and Documentation Directories**: If `[CODE_ROOT_DIRECTORIES]` or `[DOC_DIRECTORIES]` in `.clinerules` are empty or missing, **stop** other initialization and follow the procedures in Core Prompt Sections X and XI to identify and populate these sections first. Update `.clinerules` and perform MUP. Resume initialization checks afterwards.
3.  **Trigger Creation of Missing Files:**
    *   **Manual Creation Files** (`.clinerules`, `activeContext.md`, `changelog.md`, `userProfile.md`, `progress.md`): If missing, use `write_to_file` to create them with minimal placeholder content as described in Core Prompt Section II table. State: "File `{file_path}` missing. Creating with placeholder content."
        *   Example Initial `.clinerules` (if creating):
            ```
            [LAST_ACTION_STATE]
            last_action: "System Initialized"
            current_phase: "Set-up/Maintenance"
            next_action: "Initialize Core Files" # Or Identify Code/Doc Roots if needed first
            next_phase: "Set-up/Maintenance"

            [CODE_ROOT_DIRECTORIES]
            # To be identified

            [DOC_DIRECTORIES]
            # To be identified

            [LEARNING_JOURNAL]
            -
            ```
    *   **Template-Based File** (`system_manifest.md`): If missing, first use `write_to_file` to create an empty file named `system_manifest.md` in `{memory_dir}/`. State: "File `system_manifest.md` missing. Creating empty file." Then, read the template content from `cline_docs/templates/system_manifest_template.md` and use `write_to_file` again to *overwrite* the empty `system_manifest.md` with the template content. State: "Populating `system_manifest.md` with template content."
    *   **Tracker Files** (`module_relationship_tracker.md`, `doc_tracker.md`, and mini-trackers `*_module.md`):
        *   **DO NOT CREATE MANUALLY.**
        *   If any of these are missing, or if significant project changes have occurred, or if you are starting verification, run `analyze-project`. This command will create or update all necessary trackers based on the current project structure and identified code/doc roots.
        ```bash
        # Ensure code/doc roots are set in .clinerules first!
        python -m cline_utils.dependency_system.dependency_processor analyze-project
        ```
        *   State: "Tracker file(s) missing or update needed. Running `analyze-project` to create/update trackers."
        *   *(Running `analyze-project` is also the first step in the verification workflow in Section III)*.
        *   *(Optional: Add `--force-analysis` or `--force-embeddings` if needed)*.        
        *   *(Mini-trackers in module directories are also created/updated by `analyze-project`)*.
4.  **MUP**: Follow Core Prompt MUP (Section VI) and Section V additions below after creating files or running `analyze-project`. Update `[LAST_ACTION_STATE]` to reflect progress (e.g., `next_action: "Verify Tracker Dependencies"`).

## III. Analyzing and Verifying Tracker Dependencies (Ordered Workflow)

**DO NOT ASSUME A DEPENDENCY BEFORE THE RELATED FILES HAVE BEEN READ!!**

**Objective**: Ensure trackers accurately reflect project dependencies by systematically resolving placeholders ('p') and verifying suggestions ('s', 'S'), followed by an explicit code-to-documentation cross-referencing step. **This process MUST follow a specific order:**
1.  `doc_tracker.md` (Placeholder/Suggestion Resolution)
2.  All Mini-Trackers (`*_module.md`) (Placeholder/Suggestion Resolution)
3.  `module_relationship_tracker.md` (Placeholder/Suggestion Resolution)
4.  **Code-Documentation Cross-Reference** (Adding explicit dependencies)

This order is crucial because Mini-Trackers capture detailed cross-directory dependencies within modules, which are essential for accurately determining the higher-level module-to-module relationships in `module_relationship_tracker.md`.

**IMPORTANT**:
*   **All tracker modifications MUST use `dependency_processor.py` commands.** See Core Prompt Section VIII for command details.
*   **Do NOT read tracker files directly** for dependency information; use `show-keys` and `show-dependencies`.
*   Run `analyze-project` *before* starting this verification process if significant code/doc changes have occurred since the last run, or upon entering this phase (as done in Section II).

***CRITICAL EMPHASIS***: *It is critical that the documentation is **Exhaustively** cross-referenced against the code. The code cannot be completed properly if the docs that define it are not listed as a dependency. The following verification stages, especially Stage 4, are designed to achieve this.*

**This phase isn't about efficiency, it's about *accuracy*. This is a foundational job. If the accuracy in this phase is low, the entire project will suffer.**

**Procedure:**

1.  **Run Project Analysis (Initial & Updates)**:
    *   Use `analyze-project` to automatically generate/update keys, analyze files, suggest dependencies ('p', 's', 'S'), and update *all* trackers (`module_relationship_tracker.md`, `doc_tracker.md`, and mini-trackers). This command creates trackers if they don't exist and populates/updates the grid based on current code/docs and configuration.
    ```bash
    python -m cline_utils.dependency_system.dependency_processor analyze-project
    ```
    *   *(Optional: Add `--force-analysis` or `--force-embeddings` if needed, e.g., if configuration changed or cache seems stale)*.
    *   **Review logs (`debug.txt`, `suggestions.log`)** for analysis details and suggested changes, but prioritize the verification workflow below. State: "Ran analyze-project. Reviewing logs and proceeding with ordered verification."

2.  **Stage 1: Verify `doc_tracker.md`**:
    *   **A. Identify Keys Needing Verification**:
        *   Run `show-keys` for `doc_tracker.md`:
          ```bash
          python -m cline_utils.dependency_system.dependency_processor show-keys --tracker <path_to_doc_tracker.md>
          ```
          *(Use actual path, likely `{memory_dir}/doc_tracker.md` based on config)*
        *   Examine the output. Keys listed might be base keys (e.g., "1A1") or globally instanced keys (e.g., "2B1#1") if their base key string is used for multiple different paths in the project. Identify all lines ending with `(checks needed: ...)`. This indicates unresolved 'p', 's', or 'S' characters in that key's row *within this tracker*.
        *   Create a list of these keys needing verification for `doc_tracker.md`. If none, state this and proceed to Stage 2.
    *   **B. Verify Placeholders/Suggestions for Identified Keys**:
        *   **Automated Resolution (Optional)**:
            *   You may use the local LLM to resolve 'p' placeholders in batches *before* manual verification.
            *   Command: `python -m cline_utils.dependency_system.dependency_processor resolve-placeholders --tracker <path_to_doc_tracker.md>` (defaults to processing all 'p' placeholders).
            *   *(Optional args: `--limit 50`, `--key <key_string>`, `--dep-char p`)*.
            *   **Review**: Review the changes. The LLM will update 'p' to '<', '>', 'x', 'd', or 'n'.
        *   **Manual/Targeted Verification**:
            *   Iterate through the list of keys from Step 2.A (or remaining keys after automation).
            *   For each `key_string` (row key):
            *   **Get Context**: Run `show-placeholders` targeting the current tracker and key. This command specifically lists the 'p', 's', and 'S' relationships for the given key *within this tracker*, providing a targeted list for verification.
            ```bash
            python -m cline_utils.dependency_system.dependency_processor show-placeholders --tracker <path_to_doc_tracker.md> --key <key_string>
            ```
            *   **Determine Verification Approach**: Assess the number of target files to verify for this key.
                *   **Small number (≤10 files)**: Proceed with direct verification using `read_file` to examine source and target files, then determine relationships.
                *   **Large number (>10 files)**: Use the **new_task tool** to delegate verification work. Group target files into manageable chunks of 5-10 files per task.
            *   **Direct Verification (Small Number)**:
                *   **Examine Files**: Use `read_file` to examine the content of the source file and the relevant target files/folders identified.
                *   **Determine Relationship (CRITICAL STEP)**: Based on file contents, determine the **true relational necessity or essential conceptual link** between the source (`key_string`) and each target key being verified.
                    *   **Go Beyond Semantic Similarity**: Suggestions ('s', 'S') might only indicate related topics. However, if the source file defines the "Why," the rules, or the architecture for the target, it is an essential dependency.
                *   **Focus on Relational and Contextual Necessity**: Ask:
                    *   **Logic & Purpose**: Does the *row file* provide the business logic, requirements, or purpose that the *column file* implements? (Leads to 'd' or '<').
                    *   **Technical Reliance**: Does the code in the *row file* directly **import, call, or inherit from** code in the *column file*? (Leads to '<' or 'x').
                    *   **Knowledge Requirement**: Does a developer/LLM need to read the *column file* to safely or correctly modify the *row file*? (Leads to '<' or 'd').
                    *   **Implementation Link**: Is the *row file* **essential documentation** for understanding or implementing the concepts/code in the *column file*? (Leads to 'd' or potentially '>').
                    *   **Architectural Fit**: Are these files part of the same specific feature or architectural pattern where changing one without the other would cause conceptual drift or technical debt? (Leads to 'x' or 'd').
                *   **Purpose of Dependencies**: Remember, these verified dependencies guide the **Strategy phase** (determining task order) and the **Execution phase** (loading minimal necessary context). A dependency should mean "This file is part of the necessary context required to work effectively on the other."
                *   **Assign 'n' ONLY for Unrelated Content**: If the relationship is purely coincidental, uses similar common terms in a different context, or is an unrelated legacy file, assign 'n' (verified no dependency). **If there is any doubt regarding conceptual relevance, err on the side of 'd' (Documentation/Conceptual link) rather than 'n'.**
                *   **State Reasoning (MANDATORY)**: Before using `add-dependency`, **clearly state your reasoning** for the chosen dependency character (`<`, `>`, `x`, `d`, or `n`) for *each specific relationship* you intend to set, based on your direct file analysis and the relational necessity criteria.
            *   **Delegated Verification (Large Number - Using new_task)**:
                *   **Group Target Files**: Divide the target files into chunks of 5-10 files each. Group by dependency type ('p', 's', 'S') or by logical similarity to improve efficiency.
                *   **Create Verification Task**: For each chunk, use the `new_task` tool with the following structure:
                    ```
                    Dependency Verification Task for Key {key_string}

                    Source File
                    Key: {key_string}
                    Path: {source_file_path}

                    Task Objective
                    Determine the dependency relationship between the source file and each target file listed below using the criteria from cline_docs/prompts/setup_maintenance_plugin.md.

                    Dependency Criteria (from setup_maintenance_plugin.md)
                    < (Row Requires Column): Row relies on Column for context, logic, or operation
                    > (Column Requires Row): Column relies on Row for context, logic, or operation
                    x (Mutual Requirement): Mutual reliance or deep conceptual link requiring co-consideration
                    d (Documentation/Conceptual): Row is documentation or defines the "Why/How" essential for Column (or vice-versa)
                    n (Verified No Dependency): Confirmed NO relational, functional, or conceptual link exists

                    Target Files to Verify (Group {group_number})
                    [List target keys and paths for this chunk]

                    Instructions
                    1. Read the source file: {source_file_path}
                    2. For each target file above:
                       a. Read the target file
                       b. Analyze the relationship (Functional, Logical, or Conceptual) between source and target
                       c. Determine the appropriate dependency character (<, >, x, d, or n)
                       d. State your reasoning for the chosen dependency type
                    3. Provide a summary of your findings in the format:
                       Key {key_string} Dependency Verification Results:

                       [Target Key] [Target Path] -> [Dependency Character]
                       Reasoning: [Your reasoning]

                       [Repeat for each target file]

                    Important Notes
                    - Focus on Relational Necessity: Does one file provide the context or blueprint for the other?
                    - Err on the side of 'd' if the files share a logical flow or implementation goal.
                    - Mark 'n' only if the files are truly unrelated in terms of system operation or design.
                    
                    Expected Output
                    A clear summary of dependency determinations for all target files in this group with reasoning for each.
                    ```
                *   **Wait for Task Completion**: Allow the delegated task to complete and return results.
                *   **Review Results**: Examine the returned dependency determinations and reasoning.
                *   **Apply Dependencies**: Use `add-dependency` to apply the verified relationships from the task results.
            *   **Correct/Confirm Dependencies**: Use `add-dependency`, specifying `--tracker <path_to_doc_tracker.md>`. The `--source-key` is always the `key_string` you are iterating on. The `--target-key` is the column key whose relationship you determined. Set the `--dep-type` based on your reasoned analysis. Batch multiple targets *for the same source key* if they share the *same new dependency type*.
              ```bash
              # Example: Set '>' from 1A2 (source) to 2B1#3 (target) in doc_tracker.md
              # Reasoning: docs/setup.md (1A2) details steps required BEFORE using API described in docs/api/users.md (2B1). Thus, 2B1 depends on 1A2.
              python -m cline_utils.dependency_system.dependency_processor add-dependency --tracker <path_to_doc_tracker.md> --source-key 1A2 --target-key 2B1#3 --dep-type ">"

              # Example: Set 'd' from 1A2 (source) to 3C1 (target) in doc_tracker.md
              # Reasoning: While not a code call, 3C1 contains the user stories that 1A2 implements. 1A2 requires 3C1 for conceptual alignment.
              python -m cline_utils.dependency_system.dependency_processor add-dependency --tracker <path_to_doc_tracker.md> --source-key 1A2 --target-key 3C1 --dep-type "d"
              ```
        *   Repeat Step 2.B for all keys identified in Step 2.A.
    *   **C. Final Check**: Run `show-keys --tracker <path_to_doc_tracker.md>` again to confirm no `(checks needed: ...)` remain.
    *   **MUP**: Perform MUP. Update `last_action`. State: "Completed verification for doc_tracker.md. Proceeding to find and verify mini-trackers."

3.  **Stage 2: Find and Verify Mini-Trackers (`*_module.md`)**:
    *   **A. Find Mini-Tracker Files**:
        *   **Goal**: Locate all `*_module.md` files within the project's code directories.
        *   **Get Code Roots**: Read the `[CODE_ROOT_DIRECTORIES]` list from `.clinerules`. If empty, state this and this stage cannot proceed.
        *   **Scan Directories**: For each code root directory, recursively scan its contents using `list_files` or similar directory traversal logic.
        *   **Identify & Verify**: Identify files matching the pattern `{dirname}_module.md` where `{dirname}` exactly matches the name of the directory containing the file (e.g., `src/user_auth/user_auth_module.md`).
        *   **Create List**: Compile a list of the full, normalized paths to all valid mini-tracker files found.
        *   **Report**: State the list of found mini-tracker paths. If none are found but code roots exist, state this and confirm that `analyze-project` ran successfully (as it should create them if modules exist). If none are found, proceed to Stage 3.
    *   **B. Iterate Through Mini-Trackers**: If mini-trackers were found:
        *   Select the next mini-tracker path from the list. State which one you are processing.
        *   **Repeat Verification Steps**: Follow the same sub-procedure as in Stage 1 (Steps 2.A and 2.B), but substitute the current mini-tracker path for `<path_to_doc_tracker.md>` in all commands (`show-keys`, `add-dependency`).
            *   **Identify Keys**: Use `show-keys --tracker <mini_tracker_path>`. List keys needing checks.
            *   **Verify Keys**: Iterate through keys needing checks. Use `show-placeholders` to get a targeted list of unverified dependencies *within this mini-tracker*.
            ```bash
            python -m cline_utils.dependency_system.dependency_processor show-placeholders --tracker <mini_tracker_path> --key <key_string>
            ```
            *   **Determine Verification Approach**: (As defined in Stage 1, Step 2.B, utilizing the **Relational and Contextual Necessity** criteria).
            *   **Direct Verification (Small Number)**: (As defined in Stage 1, Step 2.B).
            *   **Delegated Verification (Large Number - Using new_task)**: (As defined in Stage 1, Step 2.B).
            *   **Foreign Keys**: Remember, when using `add-dependency` on a mini-tracker, the `--target-key` can be an external (foreign) key if it exists globally (Core Prompt Section VIII). Use this to link internal code to external docs or code in other modules if identified during analysis. State reasoning clearly.
              ```bash
              # Example: Set 'd' from internal code file 1Ba2 to external doc 1Aa6 in agents_module.md
              # Reasoning: combat_agent.py (1Ba2) implements concepts defined in Multi-Agent_Collaboration.md (1Aa6), making doc essential.
              python -m cline_utils.dependency_system.dependency_processor add-dependency --tracker src/agents/agents_module.md --source-key 1Ba2 --target-key 1Aa6 --dep-type "d"
              ```
            *   **Proactive External Links**: While analyzing file content, actively look for explicit references or clear conceptual reliance on *external* files (docs or other modules) missed by automation. Add these using `add-dependency` with the foreign key capability if a true dependency exists. State reasoning.
        *   **C. Final Check (Mini-Tracker)**: Run `show-keys --tracker <mini_tracker_path>` again to confirm no `(checks needed: ...)` remain for *this* mini-tracker.
        *   Repeat Step 3.B and 3.C for all mini-trackers in the list found in Step 3.A.
    *   **MUP**: Perform MUP after verifying all mini-trackers found. Update `last_action`. State: "Completed verification for all identified mini-trackers. Proceeding to module_relationship_tracker.md."

4.  **Stage 3: Verify `module_relationship_tracker.md`**:
    *   Follow the same verification sub-procedure as in Stage 1 (Steps 2.A, 2.B, 2.C), targeting `<path_to_module_relationship_tracker.md>` (likely `{memory_dir}/module_relationship_tracker.md`).
        *   **Identify Keys**: Use `show-keys --tracker <path_to_module_relationship_tracker.md>`. List keys needing checks. If none, state this and verification is complete.
        *   **Verify Keys**: Iterate through keys needing checks.
            *   **Context**: Use `show-placeholders` to get the list of unverified module-level dependencies. When determining relationships here, rely heavily on the verified dependencies established *within* the mini-trackers during Stage 2, as well as the overall system architecture (`system_manifest.md`). A module-level dependency often arises because *some file within* module A depends on *some file within* module B. Read key module files/docs (`read_file`) only if mini-tracker context is insufficient.
            ```bash
            python -m cline_utils.dependency_system.dependency_processor show-placeholders --tracker <path_to_module_relationship_tracker.md> --key <key_string>
            ```
            *   **Determine Verification Approach**: (As defined in Stage 1, Step 2.B, utilizing the **Relational and Contextual Necessity** criteria).
            *   **Direct Verification (Small Number)**:
                *   **Examine Context**: Review the verified dependencies from mini-trackers (Stage 2) and the overall system architecture (`system_manifest.md`). Read key module files/docs (`read_file`) only if mini-tracker context is insufficient.
                *   **Determine Relationship & State Reasoning**: Base decision on aggregated dependencies from mini-trackers and high-level design intent.
                *   **Focus on Module-Level Reliance**: Ask:
                    *   **Logic Flow**: Does Module A provide the data or the trigger that Module B processes? (Leads to '<' or 'x').
                    *   **Architectural Dependence**: Is Module A the "Controller" or "Core" that Module B (as a "Plugin" or "Utility") requires to have purpose? (Leads to '<' or 'd').
                    *   **Direct Code Interaction**: Does *any file within* module A directly **import, call, or inherit from** code in *any file within* module B? (Leads to '<' or 'x').
                    *   **Knowledge/Conceptual Link**: Is there a **deep, direct conceptual link** where understanding or modifying one module *necessitates* understanding the other? (Consider '<', '>', 'x', or 'd' based on the nature of the link).
                *   **State Reasoning (MANDATORY)**: Before using `add-dependency`, **clearly state your reasoning** for the chosen dependency character (`<`, `>`, `x`, `d`, or `n`) for *each specific relationship* you intend to set, based on your analysis of module-level dependencies.
            *   **Delegated Verification (Large Number - Using new_task)**: (As defined in Stage 1, Step 2.B).
            *   **Correct/Confirm**: Use `add-dependency --tracker <path_to_module_relationship_tracker.md>` with appropriate arguments.
        *   **Final Check**: Run `show-keys --tracker <path_to_module_relationship_tracker.md>` again to confirm no checks needed remain.
    *   **MUP**: Perform MUP after verifying `module_relationship_tracker.md`. Update `last_action`. State: "Completed verification for module_relationship_tracker.md. Proceeding to Code-Documentation Cross-Reference."

*Keys must be set from each perspective, as each *row* has its own dependency list.*

5.  **Stage 4: Code-Documentation Cross-Reference (Adding 'd' links)**:
    *   **Objective**: Systematically review code components and ensure they have explicit dependencies pointing to all essential documentation required for their understanding or implementation. This happens *after* initial placeholders/suggestions ('p', 's', 'S') are resolved in Stages 1-3.
    *   **A. Identify Code and Doc Keys**:
        *   Use `show-keys` on relevant trackers (mini-trackers, main tracker) to get lists of code keys.
        *   Use `show-keys --tracker <path_to_doc_tracker.md>` to get a list of documentation keys.
        *The output keys might be `KEY` or `KEY#GI`. You need to resolve these to their specific global `KeyInfo` objects (paths and base keys) to perform the conceptual matching.*
        *   *(Alternatively, use internal logic based on `ConfigManager` and the global key map if more efficient)*.
    *   **B. Iterate Through Code Keys**:
        *   Select a code key (e.g., `code_key_string` representing a specific code file).
        *   **Identify Potential Docs**: Determine which documentation keys (`doc_key_string`) are potentially relevant to `code_key_string`. Consider:
            *   The module the code belongs to.
            *   Functionality described in the code file (`read_file <code_file_path>`).
            *   Existing dependencies shown by `show-dependencies --key <code_key_string>`.
            *   Look for comments in the code referencing specific documentation.
            *   Ask questions like, "Does this documentation provide valuable or useful information for understanding how the code is intended to operate?", and "Does the code need to be aware of this information to perform its intended function?".
            *    Conceptual links and future planned directions should be considered as well. The more information available to inform how the code operates in relation to the systems, the higher quality the end result will be.
        *   **Determine Verification Approach**: Assess the number of potential documentation files to verify for this code key.
            *   **Small number (≤10 docs)**: Proceed with direct verification using `read_file` to examine code and docs, then determine relationships.
            *   **Large number (>10 docs)**: Use the **new_task tool** to delegate verification work. Group documentation files into manageable chunks of 5-10 files per task.
        *   **Direct Verification (Small Number)**:
            *   **Examine Relevant Docs**: Use `read_file` to examine the content of the potentially relevant documentation files.
            *   **Determine Essential Documentation**: For each potential `doc_key_string`, decide if it provides *essential* context, definitions, specifications, or explanations required to understand, implement, or correctly use the code represented by `code_key_string`. This is more than just keyword similarity.
            *   **Add Dependencies (Bi-directionally)**: If `doc_key_string` is essential for `code_key_string`:
                *   **State Reasoning (Mandatory)**: Explain *why* the documentation is essential for the code.
                *   **Add Code -> Doc Link**: Use `add-dependency` targeting the tracker most relevant to the *code file*.
                    ```bash
                    # Reasoning: [Explain why doc_key_string is essential for code_key_string]
                    python -m cline_utils.dependency_system.dependency_processor add-dependency --tracker <code_file_tracker_path> --source-key <code_key_string> --target-key <doc_key_string> --dep-type "<dep_char>"
                    ```
                *   **Add Doc -> Code Link**: Use `add-dependency` targeting `doc_tracker.md`.
                    ```bash
                    # Reasoning: [Same reasoning as above, from doc's perspective]
                    python -m cline_utils.dependency_system.dependency_processor add-dependency --tracker <path_to_doc_tracker.md> --source-key <doc_key_string> --target-key <code_key_string> --dep-type "<dep_char>"
                    ```
        *   **Delegated Verification (Large Number - Using new_task)**: (As defined in Stage 1, Step 2.B, utilizing the **Relational and Contextual Necessity** criteria).
        *   Repeat for all relevant documentation keys for the current `code_key_string`.
    *   **C. Repeat for All Code Keys**: Continue Step 5.B until all relevant code keys have been reviewed against the documentation corpus.
    *   **MUP**: Perform MUP. Update `last_action`. State: "Completed Code-Documentation Cross-Reference."

6.  **Completion**: Once all four stages are complete and `show-keys` reports no `(checks needed: ...)` for `doc_tracker.md`, all mini-trackers, and `module_relationship_tracker.md`, the tracker verification part of Set-up/Maintenance is done. Check if all other phase exit criteria (Section I) are met (e.g., core files exist, code/doc roots identified, system manifest populated). If so, prepare to exit the phase by updating `.clinerules` as per Section I.

*If a dependency is detected in **either** direction 'n' should not be used. Choose the best character to represent the directional dependency or 'd' if it is a more general documentation dependency.*

## IV. Set-up/Maintenance Dependency Workflow Diagram

```mermaid
graph TD
    A[Start Set-up/Maintenance Verification] --> B(Run analyze-project);
    B --> C[Stage 1: Verify doc_tracker.md];

    subgraph Verify_doc_tracker [Stage 1: doc_tracker.md]
        C1[Use show-keys --tracker doc_tracker.md] --> C2{Checks Needed?};
        C2 -- Yes --> C3[Identify Key(s)];
        C3 --> C3a[Optional: Run resolve-placeholders --auto];
        C3a --> C4[For Each (remaining) Key needing check:];
        C4 --> C5(Run show-placeholders --tracker doc_tracker.md --key [key]);
        C5 --> C6{Many Targets >10?};
        C6 -- Yes --> C6a[Group Targets 5-10 per chunk];
        C6a --> C6b[Create new_task for each chunk];
        C6b --> C6c[Wait for task completion];
        C6c --> C6d[Review results & apply dependencies];
        C6 -- No --> C7(Read Source + Target Files);
        C7 --> C8(Determine Relationship & State Reasoning);
        C8 --> C9(Use add-dependency --tracker doc_tracker.md);
        C6d --> C4;
        C9 --> C4;
        C4 -- All Keys Done --> C10[Final Check: show-keys];
        C2 -- No --> C10[doc_tracker Verified];
    end

    C --> Verify_doc_tracker;
    C10 --> D[MUP after Stage 1];

    D --> E[Stage 2: Find & Verify Mini-Trackers];
    subgraph Find_Verify_Minis [Stage 2: Mini-Trackers (`*_module.md`)]
        E1[Identify Code Roots from .clinerules] --> E2[Scan Code Roots Recursively];
        E2 --> E3[Find & Verify *_module.md Files];
        E3 --> E4[Compile List of Mini-Tracker Paths];
        E4 --> E5{Any Mini-Trackers Found?};
        E5 -- Yes --> E6[Select Next Mini-Tracker];
        E6 --> E7[Use show-keys --tracker <mini_tracker>];
        E7 --> E8{Checks Needed?};
        E8 -- Yes --> E9[Identify Key(s)];
        E9 --> E10[For Each Key needing check:];
        E10 --> E11(Run show-placeholders --tracker [mini_tracker] --key [key]);
        E11 --> E12{Many Targets >10?};
        E12 -- Yes --> E12a[Group Targets 5-10 per chunk];
        E12a --> E12b[Create new_task for each chunk];
        E12b --> E12c[Wait for task completion];
        E12c --> E12d[Review results & apply dependencies];
        E12 -- No --> E13(Read Source + Target Files);
        E13 --> E14(Determine Relationship & State Reasoning - Consider Foreign Keys/External);
        E14 --> E15(Use add-dependency --tracker <mini_tracker>);
        E12d --> E10;
        E15 --> E10;
        E10 -- All Keys Done --> E16[Final Check: show-keys];
        E8 -- No --> E16[Mini-Tracker Verified];
        E16 --> E17{All Mini-Trackers Checked?};
        E17 -- No --> E6;
        E17 -- Yes --> E18[All Mini-Trackers Verified];
        E5 -- No --> E18; // Skip if no minis found
    end

    E --> Find_Verify_Minis;
    E18 --> F[MUP after Stage 2];

    F --> G[Stage 3: Verify module_relationship_tracker.md];
    subgraph Verify_main_tracker [Stage 3: module_relationship_tracker.md]
        G1[Use show-keys --tracker module_relationship_tracker.md] --> G2{Checks Needed?};
        G2 -- Yes --> G3[Identify Key(s)];
        G3 --> G4[For Each Key needing check:];
        G4 --> G5(Run show-placeholders --tracker module_relationship_tracker.md --key [key]);
        G5 --> G6{Many Targets >10?};
        G6 -- Yes --> G6a[Group Targets 5-10 per chunk];
        G6a --> G6b[Create new_task for each chunk];
        G6b --> G6c[Wait for task completion];
        G6c --> G6d[Review results & apply dependencies];
        G6 -- No --> G7(Plan Reading / Use Mini-Tracker Context / Read Key Module Files);
        G7 --> G8(Determine Relationship & State Reasoning - Module Level);
        G8 --> G9(Use add-dependency --tracker module_relationship_tracker.md);
        G6d --> G4;
        G9 --> G4;
        G4 -- All Keys Done --> G10[Final Check: show-keys];
        G2 -- No --> G10[Main Tracker Verified];
    end

    G --> Verify_main_tracker;
    G10 --> H[MUP after Stage 3];

    H --> J[Stage 4: Code-Documentation Cross-Ref];
    subgraph CodeDocRef [Stage 4: Code-Doc Cross-Ref]
        J1[Identify Code & Doc Keys] --> J2[For Each Code Key:];
        J2 --> J3(Identify Potential Docs);
        J3 --> J4{Many Docs >10?};
        J4 -- Yes --> J4a[Group Docs 5-10 per chunk];
        J4a --> J4b[Create new_task for each chunk];
        J4b --> J4c[Wait for task completion];
        J4c --> J4d[Review results & apply bi-directional links];
        J4 -- No --> J5(Read Code & Docs);
        J5 --> J6(Determine Essential Docs & Reason);
        J6 -- Yes --> J7(add-dependency -> tracker);
        J7 --> J2;
        J6 -- No --> J2;
        J4d --> J2;
        J2 -- All Code Keys Done --> J8[Stage 4 Complete];
    end

    J --> CodeDocRef;
    J8 --> K[MUP after Stage 4];
    K --> L[End Verification Process - Check All Exit Criteria (Section I)];

    style Verify_doc_tracker fill:#e6f7ff,stroke:#91d5ff
    style Find_Verify_Minis fill:#f6ffed,stroke:#b7eb8f
    style Verify_main_tracker fill:#fffbe6,stroke:#ffe58f
```

## V. Using the new_task Tool for Dependency Verification

**Purpose**: The `new_task` tool enables efficient delegation of dependency verification work when dealing with large numbers of files to verify. This approach improves efficiency by parallelizing verification work and managing context window usage more effectively.

**When to Use new_task**:
- **Large Number of Targets**: When a key has more than 10 target files/modules to verify
- **Context Management**: When reading all source and target files would exceed context limits
- **Efficiency**: When verification work can be logically grouped into independent chunks

**Guidelines for Using new_task**:

1. **Grouping Strategy**:
   - Group target files/modules into chunks of 5-10 items
   - Group by dependency type ('p', 's', 'S') when possible
   - Group by logical similarity (same directory, related functionality)
   - Ensure each chunk is independent and can be verified separately

2. **Task Structure**:
   - **Source File Information**: Always include the source key and path
   - **Task Objective**: Clearly state the verification goal
   - **Dependency Criteria**: Include the full criteria from setup_maintenance_plugin.md
   - **Target Files List**: Provide the complete list of target keys and paths for the chunk
   - **Instructions**: Provide step-by-step instructions for the task
   - **Expected Output**: Specify the exact format for results

3. **Task Template**:
   ```
   Dependency Verification Task for Key {key_string}

   Source File
   Key: {key_string}
   Path: {source_file_path}

   Task Objective
   Determine the dependency relationship between the source file and each target file listed below using the criteria from cline_docs/prompts/setup_maintenance_plugin.md.

   Dependency Criteria (from setup_maintenance_plugin.md)
   < (Row Requires Column): Row functionally relies on or requires Column for context/operation
   > (Column Requires Row): Column functionally relies on or requires Row for context/operation
   x (Mutual Requirement): Mutual functional reliance or deep conceptual link requiring co-consideration
   d (Documentation Link): Row is documentation essential for understanding/using Column, or vice-versa
   n (Verified No Dependency): Confirmed no functional requirement or essential conceptual link exists

   Target Files to Verify (Group {group_number})
   [List target keys and paths for this chunk]

   Instructions
   1. Read the source file: {source_file_path}
   2. For each target file above:
      a. Read the target file
      b. Analyze the functional relationship between source and target
      c. Determine the appropriate dependency character (<, >, x, d, or n)
      d. State your reasoning for the chosen dependency type
   3. Provide a summary of your findings in the format:
      Key {key_string} Dependency Verification Results:

      [Target Key] [Target Path] -> [Dependency Character]
      Reasoning: [Your reasoning]

      [Repeat for each target file]

   Important Notes
   - Focus on functional reliance and necessary knowledge, not just semantic similarity
   - A file mentioning another file's topic does not automatically create a dependency
   - Consider whether the source file would break or be incomplete without the target file
   - The source file is about "{brief_description}" - likely related to {relevant_context}

   Expected Output
   A clear summary of dependency determinations for all target files in this group with reasoning for each.
   ```

4. **Post-Task Processing**:
   - **Wait for Completion**: Allow each delegated task to complete before proceeding
   - **Review Results**: Examine the returned dependency determinations and reasoning
   - **Apply Dependencies**: Use `add-dependency` commands to apply the verified relationships
   - **Batch Applications**: Group multiple `add-dependency` commands for the same source key and dependency type

5. **Best Practices**:
   - **Clear Context**: Provide relevant context about the source file in the task instructions
   - **Consistent Format**: Use the same output format across all tasks for easier processing
   - **Reasoning Quality**: Emphasize the importance of clear, detailed reasoning in task instructions
   - **Independent Chunks**: Ensure each chunk can be verified without dependencies on other chunks

6. **Example Workflow**:
   ```
   1. Run show-placeholders for key 1A3
   2. Identify 196 target files to verify (3 'S', 2 's', 191 'p')
   3. Group into 20 chunks of ~10 files each
   4. Create 20 new_task instances, one for each chunk
   5. Wait for all tasks to complete
   6. Review all returned results
   7. Apply dependencies using add-dependency commands
   8. Run show-keys to confirm verification complete
   ```

**Benefits of Using new_task**:
- **Parallel Processing**: Multiple verification tasks can run concurrently
- **Context Management**: Each task operates within its own context window
- **Improved Efficiency**: Large verification workloads become manageable
- **Better Focus**: Each task focuses on a specific, manageable set of files
- **Scalability**: The approach scales to handle very large projects

**Important Notes**:
- The `new_task` tool should be used judiciously - not every verification requires delegation
- Small numbers of targets (≤10) can be verified directly without creating new tasks
- Always review task results before applying dependencies to ensure accuracy
- The task instructions must be clear and comprehensive to ensure quality results

## VI. Locating and Understanding Mini-Trackers

**Purpose**: Mini-trackers (`{dirname}_module.md`) serve a dual role:
1.  **HDTA Domain Module**: They contain the descriptive text for the module (purpose, components, etc.), managed manually during Strategy.
2.  **Dependency Tracker**: They track file/function-level dependencies *within* that module and potentially *to external* files/docs. The dependency grid is managed via `dependency_processor.py` commands.

**Locating Mini-Trackers:**
1.  **Get Code Roots**: Read the `[CODE_ROOT_DIRECTORIES]` list from `.clinerules`. These are the top-level directories containing project source code.
2.  **Scan Code Roots**: For each directory listed in `[CODE_ROOT_DIRECTORIES]`:
    *   Recursively scan its contents.
    *   Look for files matching the pattern `{dirname}_module.md`, where `{dirname}` is the exact name of the directory containing the file.
    *   Example: In `src/auth/`, look for `auth_module.md`. In `src/game/state/`, look for `state_module.md`.
3.  **Compile List**: Create a list of the full, normalized paths to all valid mini-tracker files found. This list will be used in Section III when it's time to verify mini-trackers.

**Creation and Verification**:
*   **Creation/Update**: The `analyze-project` command (run in Section II.4 and potentially before Section III) automatically creates `{dirname}_module.md` files for detected modules if they don't exist, or updates the dependency grid within them if they do. It populates the grid with keys and initial placeholders/suggestions.
*   **Verification**: The detailed verification process in **Section III** is used to resolve placeholders ('p', 's', 'S') within these mini-trackers *after* `doc_tracker.md` is verified and *before* `module_relationship_tracker.md` is verified. Use the list compiled above to iterate through the mini-trackers during that stage.

## VII. Set-up/Maintenance Plugin - MUP Additions

After performing the Core MUP steps (Core Prompt Section VI):
1.  **Update `system_manifest.md` (If Changed)**: If Set-up actions modified the project structure significantly (e.g., adding a major module requiring a mini-tracker), ensure `system_manifest.md` reflects this, potentially adding the new module.
2.  **Update `.clinerules` [LAST_ACTION_STATE]:** Update `last_action`, `current_phase`, `next_action`, `next_phase` to reflect the specific step completed within this phase. Examples:
    *   After identifying roots:
        ```
        last_action: "Identified Code and Doc Roots"
        current_phase: "Set-up/Maintenance"
        next_action: "Initialize Core Files / Run analyze-project"
        next_phase: "Set-up/Maintenance"
        ```
    *   After initial `analyze-project`:
        ```
        last_action: "Ran analyze-project, Initialized Trackers"
        current_phase: "Set-up/Maintenance"
        next_action: "Verify doc_tracker.md Dependencies"
        next_phase: "Set-up/Maintenance"
        ```
    *   After verifying `doc_tracker.md`:
        ```
        last_action: "Verified doc_tracker.md"
        current_phase: "Set-up/Maintenance"
        next_action: "Verify Mini-Trackers"
        next_phase: "Set-up/Maintenance"
        ```
    *   After verifying the last tracker:
        ```
        last_action: "Completed All Tracker Verification"
        current_phase: "Set-up/Maintenance"
        next_action: "Perform Code-Documentation Cross-Reference"
        next_phase: "Set-up/Maintenance"
        ```
    *   After completing Code-Documentation Cross-Reference:
        ```
        last_action: "Completed Code-Documentation Cross-Reference ('d' links added)"
        current_phase: "Set-up/Maintenance"
        next_action: "Phase Complete - User Action Required"
        next_phase: "Strategy"
        ```