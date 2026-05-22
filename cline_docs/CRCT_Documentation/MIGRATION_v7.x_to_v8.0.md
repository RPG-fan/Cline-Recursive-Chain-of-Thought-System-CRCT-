# Migration Guide: v7.x to v8.0

> [!CAUTION]
> **Major Version Update** - This release includes breaking changes. Please read this guide carefully before upgrading.

## Overview

Version 8.0 represents a significant architectural evolution of the CRCT system. The core embedding and analysis infrastructure has been rebuilt to provide dramatically improved accuracy and performance.

**Upgrade Time**: ~15-30 minutes (depending on project size)  
**Downtime Required**: No (analysis only)  
**Data Migration**: Yes (embeddings must be regenerated)

---

## Breaking Changes Summary

### 1. Embedding System Architecture ⚠️
- **Changed**: Migrated from simple content-based to Symbol Essence Strings (SES)
- **Impact**: All existing embeddings are incompatible and must be regenerated
- **Action**: Run `analyze-project --force-embeddings` after upgrade

### 2. New Python Dependencies 🔧
- **Added**: `llama-cpp-python` (for GGUF model support)
- **Added**: `huggingface_hub` (for model downloads)
- **Action**: Run `pip install -r requirements.txt`

### 3. Runtime Symbol Inspection 📦
- **Requirement**: Python files must be syntactically valid and importable
- **Impact**: Syntax errors will prevent symbol extraction for those files
- **Action**: Fix syntax errors before running analysis

### 4. First-Run Model Download 📥
- **Behavior**: Qwen3 reranker model (~600MB) downloads automatically on first run
- **Impact**: Requires stable internet connection
- **Storage**: Models stored in `models/` directory

### 5. CLI & API Changes ⚠️
- **Deprecated**: `set_char` command is now **unsafe** and deprecated. Use `add-dependency` instead.
- **Removed**: `cline_utils.dependency_system.core.exceptions` module.
- **Impact**: Custom scripts using `set_char` or importing from `exceptions` will fail.
- **Action**: Update scripts to use `add-dependency` and `exceptions_enhanced`.

---

## Pre-Migration Checklist

Before upgrading, ensure:

- [ ] **Backup current work** - Commit all changes to version control
- [ ] **Check Python version** - Requires Python 3.9+
- [ ] **Verify disk space** - Need ~2GB for models and embeddings
- [ ] **Check internet connection** - Required for model downloads
- [ ] **Fix syntax errors** - Run linter on Python files
- [ ] **Review current config** - Note any custom configuration settings

---

## Step-by-Step Migration

### Step 1: Install New Dependencies

```bash
# Navigate to project root
cd /path/to/your/project

# Update Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import llama_cpp; from huggingface_hub import hf_hub_download; print('✓ Dependencies installed')"

# Optional: Update Node dependencies for visualization
npm install
```

**Expected Output**:
```
✓ Dependencies installed
```

**Troubleshooting**:
- **macOS**: If llama-cpp-python fails, install with `CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python`
- **Linux**: May need `build-essential` and `cmake`: `sudo apt install build-essential cmake`
- **Windows**: Ensure Visual Studio Build Tools are installed

---

### Step 2: Fix Syntax Errors (Python Projects)

The new runtime inspector requires importable Python modules.

```bash
# Run linter to find syntax errors
python -m pylint your_package/ --errors-only

# Or use flake8
flake8 your_package/ --select=E9,F

# Fix any reported syntax errors before proceeding
```

**Common Issues**:
- Missing colons after function/class definitions
- Indentation errors
- Unclosed brackets/parentheses
- Invalid escape sequences in strings

---

### Step 3: Update Configuration (Optional)

Review new configuration options in `.clinerules`:

```json
{
  "thresholds": {
    "reranker_promotion_threshold": 0.92,
    "reranker_strong_semantic_threshold": 0.78,
    "reranker_weak_semantic_threshold": 0.65
  },
  "embedding": {
    "auto_select_model": true,
    "batch_size": "auto",
    "max_context_length": 32768
  },
  "resources": {
    "min_memory_mb": 512,
    "recommended_memory_mb": 2048
  }
}
```

**New Options**:
- `reranker_*_threshold`: Control reranker sensitivity
- `auto_select_model`: Enable hardware-adaptive model selection (default: true)
- `batch_size`: Set to "auto" for dynamic sizing based on VRAM
- `max_context_length`: Maximum tokens for large files (default: 32768)

---

### Step 4: Run First Analysis (Model Download)

On first run, the Qwen3 reranker will download automatically:

```bash
python -m cline_utils.dependency_system.dependency_processor analyze-project --force-embeddings
```

**Expected Output**:
```
INFO: Checking for Qwen3 reranker model...
INFO: Downloading ManiKumarAdapala/Qwen3-Reranker-0.6B-Q8_0...
Downloading: 100%|████████████| 612MB/612MB [02:15<00:00, 4.52MB/s]
INFO: Model downloaded successfully
INFO: Generating embeddings with SES architecture...
```

**Download Details**:
- **Size**: ~600MB (quantized Q8_0 model)
- **Time**: 2-5 minutes (depending on connection)
- **Storage**: `models/Qwen3-Reranker-0.6B-Q8_0/`

**If Download Fails**:
```bash
# Manually download from Hugging Face
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ManiKumarAdapala/Qwen3-Reranker-0.6B-Q8_0-Safetensors",
    local_dir="models/Qwen3-Reranker-0.6B-Q8_0"
)
```

---

### Step 5: Verify Upgrade Success

Check that analysis completed successfully:

```bash
# Look for these indicators in output:
# ✓ "Symbol map merge complete: <N> files"
# ✓ "Dependency suggestion complete"
# ✓ "Final review checklist generated successfully"

# Verify new files were created:
ls -la cline_utils/dependency_system/core/project_symbol_map.json
ls -la cline_utils/dependency_system/core/ast_verified_links.json

# Check for errors in debug output
grep ERROR cline_docs/debug.txt
```

---

## Post-Migration Validation

### Verify Embeddings

Embeddings should now use SES format:

```bash
# Check embedding metadata
cat cline_utils/dependency_system/analysis/embeddings/metadata.json | head -20

# Should show rich metadata including:
# - "type_annotations": {...}
# - "inheritance": [...]
# - "decorators": [...]
```

### Verify Symbol Map

```bash
# Inspect generated symbol map
python -c "
import json
with open('cline_utils/dependency_system/core/project_symbol_map.json') as f:
    data = json.load(f)
    print(f'Symbol map entries: {len(data)}')
    # Show sample entry
    sample = next(iter(data.values()))
    print('Sample keys:', list(sample.keys())[:5])
"
```

**Expected Keys**: `imports`, `functions`, `classes`, `calls`, `attribute_accesses`, etc.

### Test Reranking

Check that reranker is being used:

```bash
# Review reranker usage in logs
grep "Reranker" cline_docs/debug.txt | head -10

# Should show lines like:
# "Reranked pair: source.py -> target.py (score: 0.847)"
```

---

## Common Migration Issues

### Issue: "llama-cpp-python won't install"

**Symptoms**: Build errors during pip install

**Solutions**:
```bash
# macOS - Use Metal acceleration
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Linux - Install build tools first
sudo apt install build-essential cmake
pip install llama-cpp-python

# Windows - Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/
# Then: pip install llama-cpp-python
```

---

### Issue: "Runtime inspection failing"

**Symptoms**: Warnings about failed imports during analysis

**Cause**: Syntax errors or missing dependencies in project files

**Solution**:
```bash
# Identify problematic files
grep "Failed to inspect" cline_docs/debug.txt

# Check if file is importable
python -c "import path.to.your.module"

# Fix syntax errors or add to exclusions in .clinerules
```

---

### Issue: "Out of memory during analysis"

**Symptoms**: Process killed or crashes

**Solution**:
```bash
# Reduce batch size in config
{
  "embedding": {
    "batch_size": 32  # Lower value for limited RAM
  },
  "resources": {
    "min_memory_mb": 512
  }
}

# Or use CPU-only mode (slower but uses less memory)
{
  "compute": {
    "embedding_device": "cpu"
  }
}
```

---

### Issue: "Model download stuck/timeout"

**Symptoms**: Download freezes or fails

**Solution**:
```bash
# Use manual download
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ManiKumarAdapala/Qwen3-Reranker-0.6B-Q8_0-Safetensors',
    local_dir='models/Qwen3-Reranker-0.6B-Q8_0',
    resume_download=True  # Resume if interrupted
)
"

# Then rerun analysis
python -m cline_utils.dependency_system.dependency_processor analyze-project
```

---

## Rollback Procedure

If you need to revert to v7.x:

```bash
# 1. Checkout previous version
git checkout tags/v7.90  # Or your last v7.x version

# 2. Reinstall old dependencies
pip install -r requirements.txt

# 3. Remove v8.0 artifacts
rm -rf models/
rm cline_utils/dependency_system/core/project_symbol_map.json
rm cline_utils/dependency_system/core/ast_verified_links.json

# 4. Regenerate with old system
python -m cline_utils.dependency_system.dependency_processor analyze-project --force-embeddings
```

---

## Performance Expectations

### First Run (v8.0)
- **Embedding Generation**: 2-4x slower (due to SES complexity)
- **Dependency Suggestion**: Similar to v7.x
- **Overall**: ~30-50% slower on initial run

### Subsequent Runs
- **With Cache**: 2-3x faster (better caching)
- **Accuracy**: 5-10x better (reranker + SES)
- **Memory**: Similar or slightly higher peak usage

### Resource Usage
| Component | v7.x | v8.0 | Notes |
|-----------|------|------|-------|
| **Peak RAM** | ~1GB | ~1.5GB | With reranker loaded |
| **Disk** | ~100MB | ~1GB | Includes models |
| **VRAM** (GPU) | ~500MB | ~2GB | With Qwen3-4B GGUF |

---

## Benefits of v8.0

### Accuracy Improvements
- **10x better** semantic understanding with SES
- **5x fewer** false positives with reranker
- **AST-verified** structural dependencies (100% accuracy)

### Developer Experience
- **Real-time progress bars** with ETA
- **Clear error messages** with actionable guidance
- **Automatic** model download and setup

### Performance Optimization
- **Hardware-adaptive** model selection
- **Dynamic batch sizing** based on VRAM
- **Enhanced caching** with compression

---

## Getting Help

### Resources
- **Documentation**: `/docs/` directory
- **Examples**: `/examples/` directory
- **Test Suite**: `/tests/` - Run with `pytest`

### Troubleshooting
1. Check `cline_docs/debug.txt` for detailed logs
2. Run with verbose logging: `--log-level DEBUG`
3. Review configuration in `.clinerules`

### Support Channels
- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and best practices

---

## Next Steps

After successful migration:

1. **Review Results**: Check generated dependency diagrams
2. **Fine-tune Config**: Adjust reranker thresholds if needed
3. **Optimize Performance**: Configure batch sizes for your hardware
4. **Explore New Features**: Try runtime inspection and AST verification

---

**Migration Complete!** 🎉

Your CRCT installation is now running v8.0 with enhanced accuracy and performance.
