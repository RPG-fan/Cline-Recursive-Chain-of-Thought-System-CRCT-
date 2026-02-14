<!--
STRUCTURED DOCUMENTATION TEMPLATE v1.0
=======================================
Instructions: Use this template for all project documentation. Each section has
a defined separator and purpose. LLMs should author new docs using this format
or convert existing docs to it. The SES parser reads tags and section headers
to generate minimal, high-quality embeddings for dependency analysis.

SECTION FLOW (5-paragraph inspired):
  1. TAGS        - Flat JSONB-style tags for classification
  2. CONTEXT     - What this document is about (the "hook")
  3. OVERVIEW    - High-level architecture or concepts
  4. DETAILS     - Implementation specifics, data models, examples
  5. REFERENCES  - Links, dependencies, related files

SEPARATOR FORMAT: ---SECTION_NAME_START--- / ---SECTION_NAME_END---
These markers are machine-parseable and LLM-respected.
-->

---TAGS_START---
tags: ["doc_type", "domain", "subsystem", "topic_keyword"]
related_tags: ["related_tag_1", "related_tag_2"]
---TAGS_END---

# {Document Title}

---CONTEXT_START---
## Context

{1-2 sentences. What is this document about? What problem does it address?
This section is ALWAYS included in the SES and should be the single most
information-dense paragraph in the document.}
---CONTEXT_END---

---OVERVIEW_START---
## Overview

{High-level explanation. Architecture, design philosophy, or conceptual model.
Use diagrams (mermaid), tables, or bullet points as needed. This section may
be partially included in SES via header extraction.}
---OVERVIEW_END---

---DETAILS_START---
## Details

{Implementation specifics, algorithms, data models, configuration, examples,
code snippets, step-by-step procedures. This is the bulk of the document.
SES does NOT include this section's body — only headers and elements like code blocks within it.}

### {Subsection as needed}

{Content}
---DETAILS_END---

---REFERENCES_START---
## References

- [Related File](file:///path/to/file)
- [External Resource](https://example.com)
---REFERENCES_END---
