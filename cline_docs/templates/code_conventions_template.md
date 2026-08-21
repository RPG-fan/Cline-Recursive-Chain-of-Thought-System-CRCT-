<!--
Instructions: Fill in the placeholders below to create the project's code_conventions.md.
This file holds PROJECT-SPECIFIC code, database, testing, seed, and tooling conventions —
durable reference material agents apply when writing or verifying code.
*Do NOT include these comments in the created file.*
-->

# {ProjectName} Code Conventions

> **Role in CRCT workflow:** This file holds PROJECT-SPECIFIC conventions — durable reference material kept SEPARATE from `.clinerules/default-rules.md [LEARNING_JOURNAL]`, which stays lean and carries only novel incidents/mechanisms/counter-intuitive findings. When a phase (or cleanup cycle) extracts a new convention from a lesson learned, it lands HERE.
>
> **Maintenance:** Updated during Cleanup/Consolidation (convention extraction from journal entries) and Execution (when a new project-specific pattern is established). Entries carry their origin date where useful. Nothing here may contradict `system_manifest.md`; where they overlap, the manifest wins on architecture, this file wins on mechanical how-to.

---

## {Domain Section 1 — e.g., Database & Schema Conventions}
{Conventions specific to this project's data layer: schema-change policy, connection/singleton handling, transaction semantics, column-type contracts, trigger/constraint behavior. State each as an imperative rule with enough mechanical detail to act on without re-deriving it.}

## {Domain Section 2 — e.g., Testing Conventions}
{What unit tests are mock-based BY DESIGN vs what must run against real infrastructure; fixture requirements; known false-signal artifacts (coverage gates, collected-vs-function counts); policy wording for test records.}

## {Domain Section 3 — e.g., Framework/Library/Stack Gotchas}
{Version- and platform-specific pitfalls that silently break code on THIS stack (driver placeholders, event-loop requirements, escaping rules, OS quirks). One bullet per gotcha with the failure signature.}

## {Domain Section 4 — e.g., Code Assessment Doctrines}
{How agents must treat ambiguous code here before deleting/rewiring: stubs vs dead code, WIP-tag lifecycle and scanning mechanics, what "wired" requires, decomposition patterns preferred by this project.}

## {Domain Section 5 — e.g., Documentation & Tooling Conventions}
{Doc-audience boundaries, plan/task sync duties, changelog model pointer, dependency-tooling notes, shell/tool usage rules.}

<!-- Keep only the domain sections this project needs; add project-specific sections as required.
     Each rule should carry: the rule, the mechanical detail, and (optionally) its origin date/incident. -->
