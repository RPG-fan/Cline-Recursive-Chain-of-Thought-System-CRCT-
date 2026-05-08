# CRCT Comment-Skill: Automated Dependency Annotation

CRCT v8.4 introduces the **Comment-Skill** system, an automated utility for injecting architectural metadata directly into your source code. This system enhances agent navigability and ensures that documentation remains synchronized with the project's dependency graph.

## Core Concepts

### Station Headers
A **Station Header** is a block of metadata injected at the top of a source file. It provides immediate context about the file's role and its position in the dependency hierarchy.

```python
# --- STATION_HEADER_START --- [AUTO]
# ROLE:    [FILL: describe this file's responsibility]
# LAYER:   [FILL: e.g. Service | Utility | Controller | Model]
# CRCT_KEY:   1A1 [AUTO]
# TRACKER_REF: cline_docs/main_tracker.md [AUTO]
# --- STATION_HEADER_END --- [AUTO]
```

- **[AUTO]**: Fields marked with this tag are managed by CRCT and will be refreshed automatically.
- **[FILL: ...]**: These are prose fields intended for agents to complete. CRCT will preserve any text you write here during updates.

### Connection Maps
A **Connection Map** is a single-line comment injected before each function or class definition. it summarizes the symbol's outbound dependencies using the tracker's "dependency rail" format.

```python
# --- CONNECTION_MAP: 2Ba2>, 3Cf1x --- my_function [AUTO]
def my_function():
    ...
```

- **Dependency Rail**: Shows target keys and their relationship character (e.g., `>` for outbound, `x` for bi-directional).
- **Filtering**: Connection maps are filtered to show only dependencies relevant to the specific symbol, based on AST and runtime analysis.

## Usage

The comment system is integrated into the CRCT `TrackerBatchCollector` and triggers automatically after tracker updates (e.g., during `resolve-placeholders` or `analyze-project`).

### Manual Trigger
You can also manually refresh comments using the `populate_comments.py` utility:

```bash
python -m cline_utils.dependency_system.utils.populate_comments --project-root .
```

## Benefits for Agents
- **Local Context**: Agents can understand a file's dependencies without opening the full tracker.
- **Navigational Pointers**: `TRACKER_REF` and `CRCT_KEY` provide clear "goto" pointers for deeper investigation.
- **Stable Metadata**: Architectural shifts are reflected in the code immediately, reducing the risk of working with stale context.

## Best Practices
1. **Always use [FILL]**: When an agent initializes a file, it should fill in the `ROLE` and `LAYER` fields to provide long-term context.
2. **Commit with Comments**: Include the [AUTO] comment blocks in your version control to share architectural context with other agents or developers.
3. **Audit Regularly**: Use the `code_analysis` reporting tools to find files with missing or incomplete headers.
