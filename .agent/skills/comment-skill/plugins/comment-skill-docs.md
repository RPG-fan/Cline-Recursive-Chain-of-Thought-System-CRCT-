## comment-skill-docs.md -- PLUGIN HEADER

EXTENDS: core (SKILL.md)
SCOPE:   `.md`, `.rst`, wiki pages, architecture docs, API reference files
OVERRIDES:
  - Station Header: HTML comment variant (invisible in rendered output)
  - Goto Pointer: heading anchor syntax for rendered cross-links
  - Connection Map: section-level variant for major doc sections
ADDS:    Section-level CONNECTION_MAP; docs-as-code cross-linking rules;
         API reference structure; architecture doc navigation patterns;
         freshness annotations

---

## Station Header (Markdown Variant)

Use an HTML comment block so the Station Header is invisible in rendered output.

```markdown
<!--
ROLE:    [FILL: 1-2 sentence purpose]
CRCT_KEY:   1Ba3                                          [AUTO]
TRACKER_REF: cline_docs/doc_tracker.md                   [AUTO]
-->
```

For RST files:
```rst
.. ROLE:    [FILL: 1-2 sentence purpose]
.. CRCT_KEY:   1Ba3
```

**Rules:**
- Station Header in docs files is always invisible (HTML comment / RST directive).
  Readers see the rendered content; agents read the raw file.

---

## Connection Map (Documentation Variant)

For major doc sections (H2 level), use a single-line CONNECTION_MAP in an HTML
comment immediately after the heading. This is the same core format, adapted
for sections rather than functions.

```markdown
## Transform Layer

<!-- --- CONNECTION_MAP: 1Ba3 d, 2Aa1 d --- Transform Layer [AUTO] -->
```

For non-CRCT projects, use abbreviated section/file path aliases:

```markdown
## Transform Layer

<!-- --- CONNECTION_MAP: docs/arch/data-flow.md#ingestion-layer d, src/pipeline/transform.py d --- Transform Layer -->
```

**Rules:**
- Section CONNECTION_MAP required at H2 in architecture documents.
- Optional at H3; omit for every heading in long-form prose.
- Dep char for doc-to-doc and doc-to-source links is typically `d`
  (documentation dependency).
- `[AUTO]` applies when populated by the `populate_comments.py` utility; agent-written entries
  omit the tag.

---

## Goto Pointer (Documentation Variant)

Two forms depending on target:

**HTML comment (invisible -- for source code refs):**
```markdown
<!-- -> Implementation: src/auth/token_manager.py:TokenManager.verify() -->
<!-- -> DB schema: docs/architecture/db-schema.md#users-table -->
```

**Rendered Markdown link (visible -- for doc-to-doc refs):**
```markdown
For token validation details, see the
[TokenManager implementation](../src/auth/token_manager.py) and the
[auth endpoints spec](./api/auth-endpoints.md#post-authlogin).
```

**Rule:** Cross-references to source code -> invisible HTML comment.
Cross-references to other doc pages -> rendered Markdown link (visible to readers).

---

## API Reference Structure

Per-endpoint layout for API reference docs:

```markdown
### POST /auth/login

<!-- ENDPOINT CONNECT: POST /auth/login
     IMPLEMENTED IN:   src/api/routes/auth.py -> login()
     CALLS:            src/auth/token_manager.py -> issue_token(),
                       src/db/user_store.py -> find_by_credentials()
     RELATED:          POST /auth/refresh, POST /auth/logout
-->

Authenticates a user and returns a JWT access token.
```

---

## Architecture Doc Cross-Linking

Diagrams and architecture sections that describe source code must note their
verification date:

```markdown
<!-- DIAGRAM: Reflects src/pipeline/ as of 2025-Q4.
     If src/pipeline/ changes significantly, update this diagram.
     -> Implementation: src/pipeline/ingest.py, transform.py, store.py
     -> Last verified: 2025-11-15 -->
```

---

## Freshness Annotations

Any doc section describing work in flux must say so explicitly. Agents must not
treat aspirational prose as ground truth.

```markdown
<!--
NOTE: Section 3 (Transform Layer) is a placeholder.
The ETL design in src/pipeline/transform.py is not yet finalized.
Do not treat this section as authoritative.
-->
```

WIP Beacons in docs use HTML comments to stay out of rendered output:

```markdown
<!--
WIP: ## Transform Layer
INTENT:   Document ETL normalization once finalized.
STATUS:   Placeholder prose. Design still in flux.
NEXT:     1. Finalize ETL design in src/pipeline/transform.py
          2. Update this section to reflect final implementation
          3. Add sequence diagram for transform step
REQUIRES: src/pipeline/transform.py implementation complete.
-->
```

---

## File-Type-Specific Rules

1. Station Headers are always invisible (HTML comment / RST directive).
2. Goto Pointers: invisible for source refs; rendered links for doc-to-doc refs.
3. Section CONNECTION_MAP: required at H2 in architecture docs; optional at H3.
4. WIP Beacons in docs use HTML comments -- never rendered prose.
5. Docs must include `DEPENDS ON` pointing to source they describe.
6. Docs must annotate sections in flux with a freshness note. Stale docs are
   worse than no docs for agents.

---

## CRCT Integration Notes

- `CRCT_KEY` and `TRACKER_REF` added inside the Station Header HTML comment block.
- Documentation files appear in `doc_tracker.md`; their keys come from
  `global_key_map.json` like any other file.
- Section CONNECTION_MAP dep chars (`d`) should match the `doc_tracker.md` grid.

-> For CRCT field definitions: plugins/comment-skill-crct.md
-> For core comment categories: SKILL.md