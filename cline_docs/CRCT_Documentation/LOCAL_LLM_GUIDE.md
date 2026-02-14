# Local LLM Dependency Resolution Guide

This guide provides detailed instructions on using the local LLM-assisted commands introduced in **v8.2** of the Cline Recursive Chain-of-Thought System (CRCT). These tools are designed to make dependency verification more efficient and less costly by offloading semantic analysis to local models (GGUF).

---

## Why Use Local LLMs?

The dependency verification process traditionally requires manual inspection or expensive API calls to larger models. Local LLM resolution provides:
- **Zero API Cost**: Runs entirely on your local hardware.
- **Batch Efficiency**: Processes hundreds of unverified dependencies ('p') in a single automated pass.
- **Consistency**: Uses specialized prompts to ensure standard dependency characters (`<`, `>`, `x`, `d`, `n`) are strictly applied.

---

## Prerequisites

1.  **Dependencies**: Ensure `llama-cpp-python` is installed.
    ```bash
    pip install llama-cpp-python
    ```
2.  **Model**: You need a GGUF model file. By default, the system looks for:
    `models/Qwen3-4B-Instruct-2507-Q8_0.gguf`
    *(You can override this path using the `--model` flag.)*

---

## Commands

### 1. `resolve-placeholders`
This command parses a tracker file and automatically attempts to resolve all placeholders (defaulting to 'p') using the local LLM.

**Usage:**
```bash
python -m cline_utils.dependency_system.dependency_processor resolve-placeholders --tracker <path_to_tracker.md>
```

**Key Parameters:**
- `--tracker`: (Required) Path to the `doc_tracker.md`, `module_relationship_tracker.md`, or a mini-tracker.
- `--limit`: (Optional, default: 200) Maximum number of dependencies to process in the current batch.
- `--key`: (Optional) Restricts processing to a specific source key (row).
- `--dep-char`: (Optional, default: 'p') The character to resolve.
- `--model`: (Optional) Specify a custom path to a GGUF model.

**Batching Behavior:**
The command processes updates in batches of 10 for performance and reliability. If an error occurs during one pair analysis, the system logs the error and continues to the next.

### 2. `determine-dependency`
This command is for deep-diving into a specific relationship between two files. It provides the full reasoning output from the LLM.

**Usage:**
```bash
python -m cline_utils.dependency_system.dependency_processor determine-dependency --source-key <key1> --target-key <key2>
```

**Key Parameters:**
- `--source-key`: (Required) The starting key (e.g., `1A1#2`).
- `--target-key`: (Required) The key to check against (e.g., `2Ba1`).
- `--model`: (Optional) Custom model path.

---

## Workflow Integration

Effective February 2026, the `resolve-placeholders` command is an **Optional Automated Stage** in the `setup_maintenance_plugin.md` workflow.

1.  **Scan**: Run `show-keys` to see where 'p' placeholders exist.
2.  **Automate**: Run `resolve-placeholders` on the tracker.
3.  **Verify**: Review the updated tracker. The LLM will have converted 'p' into definitive relationships based on the file contents.
4.  **Manual Polish**: Use `show-placeholders` to address any 'n' results or 'p' items that exceeded token limits.

---

## Optimization & Tips

### Dual-Token Sensing
The local LLM processor uses the Dual-Token schema introduced in v8.2. It intelligently checks:
- **`ses_tokens`**: The size of the Symbol Essence String (optimized context).
- **`full_tokens`**: The size of the raw file.

If the combined size of the files exceeds the model's safe context limit (~30k tokens for Qwen3-4B on an 8GB GPU), the command will skip that pair to prevent truncation errors and log a warning.

### Document Templates
For documentation files, the system now uses a structured template (`cline_docs/templates/structured_doc_template.md`). Documentation following this format is parsed much more accurately by the local LLM, leading to higher quality dependency assignments.
