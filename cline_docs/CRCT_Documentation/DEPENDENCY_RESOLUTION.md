# Dependency Resolution & Prefetching (v8.4)

The CRCT dependency system handles complex placeholder patterns and large-scale project graphs through LLM-assisted resolution and parallel prefetching.

## 1. LLM-Assisted Placeholder Resolution

Previous versions relied heavily on exact matches or simple regex. Version 8.2 introduced the `PlaceholderResolver`, which leverages a local LLM to determine relationships between files when placeholders are present.

### Contextual Reasoning
The resolver analyzes the content of both the source and target files to verify if a dependency actually exists:
- **Semantic Verification**: Uses SES (Symbol Essence Strings) or full file content to provide context to the LLM.
- **Dependency Classification**: The LLM determines the specific type of dependency (e.g., `>` for usage, `x` for strict, etc.) and provides a justification.
- **Automated Commit**: Once resolved, the system batches tracker updates and commits them via a dedicated background thread using `TrackerBatchCollector`.

### Pattern Verification
1.  Identifies unverified dependencies (marked with `p`, `s`, or `S`) in the trackers.
2.  Batches tasks for efficient LLM processing.
3.  Utilizes a scoring model (e.g., `Qwen3-4B`) to evaluate the relationship.
4.  Updates the trackers with the verified dependency character and justification.

---

## 2. Multi-threaded Prefetching & Commits

To eliminate I/O wait times and LLM bottlenecks, a robust background processing engine is implemented in `PlaceholderResolver.resolve_batch()`.

- **Parallel Prefetching**: Uses a `ThreadPoolExecutor` (3 workers) to load file content and metadata into memory before the LLM needs them, with a look-ahead depth of 5 items.
- **Background Commits**: Analysis results are batched (every 10 pairs) and committed to the filesystem in a dedicated single-thread executor, ensuring the main analysis loop remains responsive.
- **Hardcoded Performance**: Prefetch workers (3) and prefetch-ahead depth (5) are constants defined directly in code.

---

## 3. Bolt Optimization (v8.4)
 
 Version 8.4 introduces the **Bolt Optimization**, which fundamentally changes how dependency propagation is handled across directories.
 
 - **O(M) Complexity**: Transitioned from an $O(N \cdot M)$ directory resolution loop to an $O(M)$ model using global set sharing.
 - **Global Set Sharing**: Instead of each tracker reconstructing its dependency path from scratch, the system now uses a shared global set of resolved paths, significantly reducing redundant I/O and CPU cycles.
 - **Batched Directory Resolution**: Directory placeholders are now resolved in a single algorithmic pass before LLM-based verification, ensuring that the LLM only handles truly ambiguous relationships.
 
 ---
 
 ## 4. How to Use

### Triggering Resolution (CLI)
You can force the system to resolve placeholders in existing trackers using the `resolve-placeholders` command:

```bash
# Resolve 'p' placeholders across all trackers (default limit: 200)
python -m cline_utils.dependency_system.dependency_processor resolve-placeholders

# Resolve placeholders for a specific tracker file
python -m cline_utils.dependency_system.dependency_processor resolve-placeholders --tracker path/to/tracker_module.md

# Limit the number of items processed
python -m cline_utils.dependency_system.dependency_processor resolve-placeholders --limit 50

# Resolve a different dependency character (e.g., 's')
python -m cline_utils.dependency_system.dependency_processor resolve-placeholders --dep-char s
```

### Manual Determination
Use the `determine-dependency` command to see the LLM's reasoning for a specific pair:

```bash
python -m cline_utils.dependency_system.dependency_processor determine-dependency --source-key 1A1 --target-key 2Ba2
```

---

## 4. Troubleshooting

-   **"Unresolved Placeholder"**: If the system cannot resolve a pattern, it will log a warning. Ensure the GGUF model is correctly placed in the `models/` directory.
-   **Performance Lag**: Large batch resolutions can be CPU/GPU intensive. Use the `--limit` flag to process trackers in smaller chunks.
-   **Model Not Found**: The system defaults to `models/Qwen3-4B-Instruct-2507-Q8_0.gguf`. Ensure this file exists or specify a path using the `--model` flag.
-   **Path Case-Sensitivity**: On Windows, the resolver is case-insensitive, but it will normalize paths to the case found on the filesystem to ensure cross-platform compatibility.
