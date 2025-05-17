# **Cline Recursive Chain-of-Thought System (CRCT) - GitOps Integration for Submodules**

**This document provides detailed instructions for ensuring GitOps workflow integration when using the CRCT system as a submodule. It should be used in conjunction with the Core System Prompt and the GitOps Plugin.**

---

## I. Overview

When using the CRCT system as a submodule, the GitOps workflow needs to be explicitly triggered at the beginning and end of task execution. This document provides instructions for ensuring that the GitOps workflow is properly integrated with the parent repository's workflow.

## II. Automatic GitOps Integration

To ensure that the GitOps workflow is automatically triggered when executing tasks, the following steps should be taken at the beginning of **every** task execution:

1. **Check GitOps Configuration**:
   - Read the `[GITOPS_WORKFLOW]` section in `.clinerules`.
   - If `enabled` is `true`, proceed with GitOps integration.

2. **Pre-Task GitOps Operations**:
   - Verify current Git branch.
   - If not on the `default_base_branch` (e.g., `dev`), switch to it.
   - Map task name to branch name based on `task_to_branch_mapping`.
   - Create and checkout a new branch.
   - Update `activeContext.md` with GitOps workflow information.

3. **Post-Task GitOps Operations**:
   - Check Git status to identify changes.
   - Generate a meaningful commit message following the Conventional Commits specification.
   - Stage changes.
   - Commit changes.
   - If `auto_push` is `true`, push changes to the remote repository.
   - Update `activeContext.md` with GitOps post-task operation information.

## III. Integration with Task Execution

### At the Beginning of Task Execution:

```
# GitOps Pre-Task Operations
- Current branch: [Check and display current branch]
- Default branch: [Display default_base_branch from .clinerules]
- Task name: [Display task name]
- Mapped branch name: [Display mapped branch name based on task_to_branch_mapping]

[Execute GitOps pre-task operations as defined in the GitOps Plugin]
```

### At the End of Task Execution:

```
# GitOps Post-Task Operations
- Current branch: [Display current branch]
- Modified files: [List modified files]
- Commit message: [Display generated commit message]

[Execute GitOps post-task operations as defined in the GitOps Plugin]
```

## IV. Implementation in Task Execution Workflow

To ensure that the GitOps workflow is automatically triggered, the following steps should be added to the task execution workflow:

1. **Add to Task Initialization**:
   - After loading the task instruction file, check the GitOps configuration and perform pre-task operations if enabled.
   - This should be done before any actual task execution steps.

2. **Add to Task Completion**:
   - Before marking a task as complete, check the GitOps configuration and perform post-task operations if enabled.
   - This should be done after all task execution steps are complete but before updating the task status.

3. **Update `activeContext.md`**:
   - Include GitOps workflow information in `activeContext.md` to maintain awareness of the current branch and GitOps status.

## V. Example Implementation

```
# Task Initialization
1. Read task instruction file.
2. Check GitOps configuration.
3. If GitOps is enabled, perform pre-task operations.
4. Proceed with task execution.

# Task Execution
[Execute task steps as defined in the task instruction file]

# Task Completion
1. Check GitOps configuration.
2. If GitOps is enabled, perform post-task operations.
3. Update task status.
4. Update `activeContext.md`.
```

## VI. Automatic Reminder System

To ensure that the GitOps workflow is not forgotten, an automatic reminder system should be implemented:

1. **Add to `.clinerules`**:
   ```
   [GITOPS_REMINDER]
   enabled: true
   reminder_message: "Remember to perform GitOps operations at the beginning and end of task execution."
   ```

2. **Check for Reminder**:
   - At the beginning of each task execution, check if `[GITOPS_REMINDER]` is enabled.
   - If enabled, display the reminder message.
   - This serves as a backup in case the automatic GitOps integration fails.

## VII. Conclusion

By following these instructions, the GitOps workflow will be automatically integrated with the task execution workflow when using the CRCT system as a submodule. This ensures that all changes are properly tracked, committed, and pushed to the remote repository without requiring manual reminders.
