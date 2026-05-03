# Modular Reporting System (v8.3)

The CRCT reporting system has been decomposed into a decoupled architecture of **Scanners** (data collection) and **Formatters** (data presentation). This ensures that code quality analysis can be extended with new engines without modifying the output logic.

## 1. System Architecture

### `scanner/` (Intelligence Layer)
- **`static_engine.py`**: Performs deep analysis using Regex and tree-sitter. It detects TODOs, FIXMEs, anti-patterns (`placeholder`, `for now`, `simplified`, etc.), empty/stub functions and classes, and `NotImplementedError` raises. Also parses Pyright output for unused item diagnostics.
- **`runtime_bridge.py`**: Integrates live metadata from the `runtime_inspector` via the `RuntimeIndex` class. Provides `enrich_issue()` to attach runtime context (owning symbol, type annotations, inheritance, callers) to static findings. Also independently emits runtime-only findings (e.g., annotated stubs, exported placeholders, orphan exports) that the static pipeline cannot detect. Contains `score_severity()` for per-issue severity scoring.
- **`heuristics.py`**: Provides low-level symbol classification helpers (`has_trivial_body`, `is_abstract_class`, `is_protocol_class`, `is_data_container_class`, etc.) used by `runtime_bridge.py` to apply suppression logic and avoid false positives.

### `reporting/` (Presentation Layer)
- **`markdown_formatter.py`**: The default generator for human-readable quality reports (`issues_report.md`).
- **`json_exporter.py`**: Generates machine-readable audit trails (`issues_report.json`) for CI/CD pipelines or forensic analysis.

---

## 2. How to Use

### Generating a Report (CLI)
The `report_generator.py` script acts as the main orchestrator. It automatically runs a full suite of analyses including Pyright, runtime inspection, and static walks:

```bash
# Run the full reporting suite
python -m code_analysis.report_generator
```

*Note: The script generates both Markdown and JSON reports to `code_analysis/issues_report.md` and `code_analysis/issues_report.json` respectively.*

### Report Workflow
1.  **Pyright Analysis**: Runs `pyright --outputjson` to identify unused items and type inconsistencies, saving output to `pyright_output.json`.
2.  **Runtime Inspection**: Triggers the `runtime_inspector` to capture live metadata (controlled by the `CRCT_AUTO_RUNTIME=1` environment variable).
3.  **Static Scan**: Walks the project code roots (defined in `.clinerules`) and applies regex + tree-sitter analysis to each file.
4.  **Enrichment**: Merges static findings with runtime context (e.g., owning symbol, type annotations, caller relationships).
5.  **Deduplication**: Collapses redundant issues with the same `(file, line, subtype)` key, keeping the richest context.
6.  **Export**: Writes the final results to `code_analysis/issues_report.md` and `code_analysis/issues_report.json`.

---

## 3. Configuration

The reporting engine respects the core project configuration defined in `.clinerules` and `.clinerules.config.json`:

-   **Code Roots**: Only directories listed in `[CODE_ROOT_DIRECTORIES]` are scanned.
-   **Exclusions**: Respects the `excluded_dirs` and `excluded_paths` settings in `ConfigManager`.
-   **AST Parsing**: Controlled by `analysis.python_ast_enabled` and `analysis.max_ast_file_size_mb`.
-   **Supported Extensions**: `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.md`, `.txt` (defined by `EXTENSIONS` in `static_engine.py`).

---

## 4. Extension Guide

-   **Add a new Scanner**: Create a new module in `scanner/`, then import and call its functions within `report_generator.py:main()`.
-   **Add a new Output Format**: Implement a new formatter in `reporting/` and add a corresponding export call in `report_generator.py`.

---

## 5. Troubleshooting
-   **"No files analyzed"**: Check your `[CODE_ROOT_DIRECTORIES]` in `.clinerules`. Ensure the paths are relative to the project root.
-   **Pyright Failures**: Ensure `pyright` is installed and accessible in your environment's PATH.
-   **Missing Runtime Context**: If the report lacks type information, ensure `CRCT_AUTO_RUNTIME=1` is set in your environment, or run the `runtime_inspector` manually before generating the report.
