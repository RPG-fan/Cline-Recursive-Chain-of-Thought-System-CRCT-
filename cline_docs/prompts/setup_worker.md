# **CRCT Set-up/Maintenance - Dependency Verification Worker Instructions**

This file defines the required verification behavior for delegated Set-up/Maintenance dependency tasks.

## I. Mandatory Bootstrap

1. Apply these rules for every source->target relationship in the assigned task.
2. If task instructions conflict with this file, follow this file and report the conflict.

## II. Worker Scope

1. Analyze assigned source/target files and determine dependency characters.
2. Return determinations with explicit reasoning tied to file evidence.
3. Do not assume outcomes before reading evidence.
4. Do not modify trackers; return determinations for the orchestrator to apply.

## III. Determine Relationship (CRITICAL STEP)

```text
**Dependency Criteria**
   < (Row Requires Column): Row functionally relies on or requires Column for context/operation
   > (Column Requires Row): Column functionally relies on or requires Row for context/operation
   x (Mutual Requirement): Mutual functional reliance or deep conceptual link requiring co-consideration
   d (Documentation Link): Row is documentation essential for understanding/using Column, or vice-versa
   n (Verified No Dependency): Confirmed no functional requirement or essential conceptual link exists
```

* **Determine Relationship (CRITICAL STEP)**: Based on file contents, determine the **true relational necessity or essential conceptual link** between the source (`key_string`) and each target key being verified.
* **Go Beyond Semantic Similarity**: Suggestions ('s', 'S') might only indicate related topics. However, if the source file defines the "Why," the rules, or the architecture for the target, it is an essential dependency.
* **Focus on Relational and Contextual Necessity**: Ask:
  * **Logic & Purpose**: Does the *row file* provide the business logic, requirements, or purpose that the *column file* implements? (Leads to 'd' or '<').
  * **Technical Reliance**: Does the code in the *row file* directly **import, call, or inherit from** code in the *column file*? (Leads to '<' or 'x').
  * **Knowledge Requirement**: Does a developer/LLM need to read the *column file* to safely or correctly modify the *row file*? (Leads to '<' or 'd').
  * **Implementation Link**: Is the *row file* essential documentation for understanding or implementing the concepts/code in the *column file*? (Leads to 'd' or '>').
  * **Architectural Fit**: Are these files part of the same specific feature or architectural pattern where changing one without the other would cause conceptual drift or technical debt? (Leads to 'x' or 'd').
* **Purpose of Dependencies**: Remember, these verified dependencies guide the **Strategy phase** (determining task order) and the **Execution phase** (loading minimal necessary context). A dependency should mean "This file is part of the necessary context required to work effectively on the other."
* **Assign 'n' ONLY for Unrelated Content**: If the relationship is purely coincidental, uses similar common terms in a different context, or is an unrelated file, assign 'n' (verified no dependency). **If there is any doubt regarding conceptual relevance, err on the side of 'd' (Documentation/Conceptual link) rather than 'n'.**
* **State Reasoning (MANDATORY)**: **clearly state your reasoning** for the chosen dependency character (`<`, `>`, `x`, `d`, or `n`) for *each specific relationship* you intend to set, based on your direct file analysis and the relational necessity criteria.

## IV. Stage-Specific Guidance

1. **Doc tracker and mini-tracker verification**:
   * Verify by reading source and target files, not by similarity alone.
   * Include conceptual/documentation relationships.
   * If either the source or target file *should* reference the other, but does not, a positive dependency **must** be established so the system can add the missing information.
   * When determining documentation dependency relations, think beyond direct connectivity. Does the described system or functionality interact with the other in *any* way? For proper planning we must be able to see relations beyond the obvious first tier connections. We must ask if a change in either system would have downstream effects on the other.

2. **Module relationship tracker verification**:
   * **Determine Relationship & State Reasoning**: Base decision on aggregated dependencies from mini-trackers and high-level design intent.
   * **Focus on Module-Level Reliance**: Ask:
   * **Logic Flow**: Does Module A provide the data or the trigger that Module B processes? (Leads to '<' or 'x').
   * **Architectural Dependence**: Is Module A the "Controller" or "Core" that Module B (as a "Plugin" or "Utility") requires to have purpose? (Leads to '<' or 'd').
   * **Direct Code Interaction**: Does *any file within* module A directly **import, call, or inherit from** code in *any file within* module B? (Leads to '<' or 'x').
                    *   **Knowledge/Conceptual Link**: Is there a **deep, direct conceptual link** where understanding or modifying one module *necessitates* understanding the other? (Consider '<', '>', 'x', or 'd' based on the nature of the link).
   * **State Reasoning (MANDATORY)**: **Clearly state your reasoning** for the chosen dependency character (`<`, `>`, `x`, `d`, or `n`) for *each specific relationship* you intend to set, based on your analysis of module-level dependencies.

3. **Code-Documentation cross-reference**:
   * Determine whether documentation is essential context for understanding, implementing, or safely modifying code.
   * Explicitly note when doc->code and code->doc links should both exist.

## V. Output Contract

Use this output format unless the parent task specifies a stricter format:

```markdown
Key {source_key} Dependency Verification Results:

[Target Key] [Target Path] -> [Dependency Character]
Reasoning: [Evidence-backed reasoning]

[Repeat for each target]
```

If evidence is insufficient, state exactly what is missing and why.
