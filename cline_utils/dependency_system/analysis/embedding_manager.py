# analysis/embedding_manager.py

"""
Module for managing embeddings generation and similarity calculations using contextual keys.
Handles embedding creation from project files using Symbol Essence Strings (SES) derived from
the project symbol map, and calculates cosine similarity between embeddings.
"""
import json
import logging
import os
import re
import sys
import textwrap
import threading
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import numpy as np
import torch

# from llama_cpp import Llama
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
# )

try:
    import llama_cpp
    from llama_cpp import Llama
except ImportError:
    Llama = None
    llama_cpp = None

import cline_utils.dependency_system.core.key_manager as key_manager_module
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.cache_manager import cache_manager, cached
from cline_utils.dependency_system.utils.cache_manager import (
    get_project_root_cached as get_project_root,
)
from cline_utils.dependency_system.utils.cache_manager import (
    normalize_path_cached as normalize_path,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.phase_tracker import PhaseTracker

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_model_instance: Optional[Any] = None  # Can be Llama or SentenceTransformer
_selected_device_cache: Optional[str] = None
_selected_model_config: Optional[Dict[str, Any]] = None
_tokenizer_instance: Optional[Any] = None

# Locks
MODEL_LOCK = threading.Lock()

# Constants
PROJECT_SYMBOL_MAP_FILENAME = "project_symbol_map.json"

# Model configurations for hardware-based selection
MODEL_CONFIGS = {
    "qwen3-4b": {
        "name": "Qwen3-Embedding-4B-Q6_K",
        "path": None,  # Will be set from config
        "embedding_dim": 2560,
        "min_vram_gb": 3.5,  # Q6_K quantization
        "min_ram_gb": 6.0,  # CPU fallback
        "context_length": 32768,
        "type": "gguf",
    },
    "mpnet": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "embedding_dim": 384,
        "min_vram_gb": 0.5,
        "min_ram_gb": 2.0,
        "context_length": 512,
        "type": "sentence-transformer",
    },
}

# Maximum safe context length to prevent OOM/crashes
MAX_CONTEXT_LENGTH = 32768
SIM_CACHE_MAXSIZE = 100_000
SIM_CACHE_TTL_SEC = 7 * 24 * 60 * 60  # 7 days
SIM_CACHE_NEGATIVE_RESULTS = True
# Bump when SES construction/token accounting changes so unchanged source files
# still get refreshed embeddings + ses/full token metadata.
EMBEDDING_METADATA_VERSION = "2.3_EnhancedSES"


def _get_available_vram() -> float:
    """Get available VRAM in GB for CUDA devices."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        torch.cuda.synchronize()
        # Use mem_get_info() which returns (free_memory, total_memory) directly
        # This is more accurate than manual calculation
        free_memory, _total_memory = torch.cuda.mem_get_info(0)
        return free_memory / (1024**3)  # Convert to GB
    except Exception as e:
        logger.warning(f"Failed to get VRAM info: {e}")
        return 0.0


def _get_available_ram() -> float:
    """Get available system RAM in GB."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        logger.warning("psutil not installed. Cannot check RAM.")
        return 0.0


def _get_best_device() -> str:
    """
    Automatically determines the best available torch device with robust error handling.
    """
    try:
        # 1. Check CUDA
        if torch.cuda.is_available():
            try:
                test_tensor = torch.zeros(1, device="cuda")
                del test_tensor
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                logger.debug("CUDA is available and working. Using CUDA.")
                return "cuda"
            except Exception as e:
                logger.warning(
                    f"CUDA available but failed to initialize: {e}. Falling back."
                )

        # 2. Check MPS (Apple Silicon)
        if (
            sys.platform == "darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            try:
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor
                logger.info("Apple MPS is available and working. Using MPS.")
                return "mps"
            except Exception as e:
                logger.warning(
                    f"MPS available but failed to initialize: {e}. Falling back."
                )

        # 3. Fallback
        logger.info("Using CPU as fallback device.")
        return "cpu"

    except Exception as e:
        logger.warning(f"Device detection failed: {e}. Using CPU as fallback.")
        return "cpu"


def _select_device() -> str:
    """Selects device based on config override or automatic detection."""
    global _selected_device_cache
    if _selected_device_cache is None:
        config_manager = ConfigManager()
        config_device = (
            config_manager.config.get("compute", {})
            .get("embedding_device", "auto")
            .lower()
        )
        if config_device in ["cuda", "mps", "cpu"]:
            if config_device == "cuda" and not torch.cuda.is_available():
                logger.warning(
                    "Config specified 'cuda', but not available. Auto-detecting."
                )
                _selected_device_cache = _get_best_device()
            elif config_device == "mps" and not (
                sys.platform == "darwin"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                logger.warning(
                    "Config specified 'mps', but not available. Auto-detecting."
                )
                _selected_device_cache = _get_best_device()
            else:
                logger.debug(f"Using device specified in config: {config_device}")
                _selected_device_cache = config_device
        elif config_device == "auto":
            logger.debug("Auto-detecting device.")
            _selected_device_cache = _get_best_device()
        else:
            logger.warning(f"Invalid device '{config_device}'. Auto-detecting.")
            _selected_device_cache = _get_best_device()
    return _selected_device_cache or "cpu"


def _verify_qwen3_model(model_path: str) -> bool:
    """Verify that the Qwen3 GGUF model file is valid."""
    if not os.path.exists(model_path):
        return False
    try:
        file_size = os.path.getsize(model_path)
        if file_size < 1000000:  # Less than 1MB is definitely invalid
            logger.warning(f"Qwen3 model file too small: {file_size} bytes")
            return False
        with open(model_path, "rb") as f:
            header = f.read(4)
            if header != b"GGUF":
                logger.warning(f"Invalid GGUF header: {header}")
                return False

        # Try a quick load test with llama-cpp-python
        try:
            from llama_cpp import Llama

            # Quick test load (don't actually initialize fully)
            test_model = Llama(
                model_path=model_path,
                embedding=True,
                n_ctx=16384,
                n_threads=1,
                n_gpu_layers=-1,
                verbose=False,
            )
            # If we get here without exception, model is valid
            del test_model  # Clean up
            logger.debug("Qwen3 model verification successful")
            return True
        except Exception as e:
            logger.warning(f"Qwen3 model verification failed during load test: {e}")
            return False

    except Exception as e:
        logger.warning(f"Qwen3 model verification failed: {e}")
        return False


def _download_qwen3_model(model_path: str) -> bool:
    """Download the Qwen3-Embedding-4B-Q6_K model if it doesn't exist or is invalid."""
    # First check if model exists and is valid
    if os.path.exists(model_path):
        logger.debug(f"Qwen3 model exists at {model_path}, verifying...")
        if _verify_qwen3_model(model_path):
            logger.debug("Qwen3 model verification passed, using existing model")
            return True
        else:
            logger.warning("Qwen3 model verification failed, re-downloading...")
            try:
                os.remove(model_path)
            except:
                pass

    # Create models directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    # Qwen3-Embedding-4B-Q6_K download URL (using resolve endpoint for direct download)
    model_url = "https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/resolve/main/Qwen3-Embedding-4B-Q6_K.gguf"

    logger.info(f"Downloading Qwen3 model from {model_url} to {model_path}")

    try:
        # Download with progress reporting
        with urllib.request.urlopen(model_url) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192

            with open(model_path, "wb") as f:
                with PhaseTracker(
                    total=total_size, phase_name="Downloading Qwen3", unit="bytes"
                ) as tracker:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        tracker.update(
                            len(chunk), description=f"{downloaded}/{total_size} bytes"
                        )

        # Verify download
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            # Final verification after download
            if _verify_qwen3_model(model_path):
                logger.debug(
                    f"Successfully downloaded and verified Qwen3 model to {model_path}"
                )
                return True
            else:
                logger.error("Downloaded Qwen3 model failed verification")
                try:
                    os.remove(model_path)
                except:
                    pass
                return False
        else:
            logger.error(f"Download failed - file not found or empty: {model_path}")
            return False

    except Exception as e:
        logger.error(f"Failed to download Qwen3 model: {e}")
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except OSError:
                pass
        return False


def _select_best_model() -> Dict[str, Any]:
    """Select the best embedding model based on hardware and config."""
    global _selected_model_config
    if _selected_model_config is not None:
        return _selected_model_config

    config_manager = ConfigManager()
    model_selection = config_manager.get_embedding_setting("model_selection", "auto")

    if model_selection == "qwen3-4b":
        model_config = MODEL_CONFIGS["qwen3-4b"].copy()
        model_config["path"] = config_manager.get_embedding_setting("qwen3_model_path")
        if not model_config["path"]:
            # Fallback default path if not in config
            model_config["path"] = os.path.join(
                get_project_root(), "models", "Qwen3-Embedding-4B-Q6_K.gguf"
            )

        if not os.path.exists(model_config["path"]) or not _verify_qwen3_model(
            model_config["path"]
        ):
            _download_qwen3_model(model_config["path"])

        _selected_model_config = model_config
        return model_config
    elif model_selection == "mpnet":
        _selected_model_config = MODEL_CONFIGS["mpnet"].copy()
        return _selected_model_config

    # Auto-detect
    device = _select_device()
    available_mem = _get_available_vram() if device == "cuda" else _get_available_ram()

    # Prefer Qwen3 if VRAM allows, else mpnet
    qwen_config = MODEL_CONFIGS["qwen3-4b"].copy()
    qwen_config["path"] = config_manager.get_embedding_setting("qwen3_model_path")

    mem_req: float = float(
        (qwen_config["min_vram_gb"] if device == "cuda" else qwen_config["min_ram_gb"])
        or 0
    )

    if available_mem >= mem_req:
        # Check if we can actually get the model
        if not qwen_config["path"]:
            qwen_config["path"] = os.path.join(
                get_project_root(), "models", "Qwen3-Embedding-4B-Q6_K.gguf"
            )

        if os.path.exists(qwen_config["path"]) and _verify_qwen3_model(
            qwen_config["path"]
        ):
            _selected_model_config = qwen_config
            return qwen_config
        elif _download_qwen3_model(qwen_config["path"]):
            _selected_model_config = qwen_config
            return qwen_config

    _selected_model_config = MODEL_CONFIGS["mpnet"].copy()
    return _selected_model_config


def _get_tokenizer() -> Optional[Any]:
    """Lazily loads a tokenizer for token counting."""
    global _tokenizer_instance, _reranker_tokenizer

    if _tokenizer_instance is not None:
        return _tokenizer_instance

    # 1. Try reusing Reranker tokenizer if loaded
    if _reranker_tokenizer is not None:
        _tokenizer_instance = _reranker_tokenizer
        return _tokenizer_instance

    # 2. Try loading from local reranker path
    try:
        project_root = get_project_root()
        local_model_path = os.path.join(project_root, "models", "qwen3_reranker")
        if os.path.exists(local_model_path) and os.path.exists(
            os.path.join(local_model_path, "tokenizer.json")
        ):
            from transformers import AutoTokenizer

            _tokenizer_instance = cast(
                Any,
                AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True),  # type: ignore
            )
            return _tokenizer_instance
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from local path: {e}")

    return None


def _count_tokens(text: str, tokenizer: Any = None) -> int:
    """Count tokens in text using tokenizer or fallback estimate."""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass

    # Fallback: Rough estimate (4 chars per token is standard rule of thumb)
    return len(text) // 4


def _load_model(n_ctx: int = 8192):
    """Loads the embedding model based on hardware capabilities."""
    global _model_instance, _selected_model_config

    if _model_instance is not None:
        # Check if we need to reload due to context size
        if _selected_model_config and _selected_model_config["type"] == "gguf":
            try:
                current_n_ctx = _model_instance.n_ctx()
                if current_n_ctx < n_ctx:
                    logger.debug(
                        f"Reloading model to increase context from {current_n_ctx} to {n_ctx}"
                    )
                    _unload_model()
                else:
                    # Existing context is sufficient
                    return _model_instance
            except AttributeError:
                # Fallback if n_ctx() not available
                pass
        elif (
            _selected_model_config
            and _selected_model_config["type"] == "sentence-transformer"
        ):
            # Update max_seq_length for SentenceTransformer without reload
            if hasattr(_model_instance, "max_seq_length"):
                if _model_instance.max_seq_length < n_ctx:
                    _model_instance.max_seq_length = n_ctx
            return _model_instance

    if _model_instance is None:
        _selected_model_config = _select_best_model()
        device = _select_device()

        try:
            if _selected_model_config["type"] == "gguf":
                # Load GGUF model with llama-cpp-python
                if Llama is None or llama_cpp is None:
                    logger.error(
                        "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
                    )
                    raise ImportError("llama-cpp-python not installed")

                n_gpu_layers = (
                    -1 if device == "cuda" else 0
                )  # -1 = Offload ALL layers to GPU
                if device == "mps":
                    n_gpu_layers = 0  # MPS not supported by llama-cpp-python

                # Log callback removed to prevent potential access violations (0xc000001d)
                # llama_cpp.llama_log_set(None, ctypes.c_void_p())

                # Suppress "init: embeddings required..." warning via no-op callback
                # This is the only method that effectively silences the C++ library output.
                # We use a pure no-op to minimize crash risk (access violations).
                import ctypes

                # Define the callback type matching llama.cpp signature
                # typedef void (*llama_log_callback)(enum llama_log_level level, const char * text, void * user_data);
                LogCallback = ctypes.CFUNCTYPE(
                    None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
                )

                def noop_log_callback(level: int, text: bytes, user_data: int) -> None:
                    pass

                # Keep a reference to prevent GC (Critical!)
                global _C_LOG_CALLBACK_REF
                _C_LOG_CALLBACK_REF = LogCallback(noop_log_callback)

                llama_cpp.llama_log_set(_C_LOG_CALLBACK_REF, ctypes.c_void_p())

                _model_instance = Llama(
                    model_path=_selected_model_config["path"],
                    embedding=True,
                    n_ctx=n_ctx,
                    n_batch=512,
                    n_threads=os.cpu_count(),  # Adjust based on CPU
                    n_gpu_layers=n_gpu_layers,
                    use_mmap=True,
                    use_mlock=False,
                    flash_attn=True,
                    verbose=False,
                )
                logger.debug(
                    f"Loaded GGUF model: {_selected_model_config['name']} on device: {device}"
                )

            elif _selected_model_config["type"] == "sentence-transformer":
                # Load sentence-transformers model with proper device handling
                from sentence_transformers import SentenceTransformer

                try:
                    _model_instance = SentenceTransformer(
                        _selected_model_config["name"], device=device
                    )
                except Exception:
                    # Fallback to CPU if device init fails
                    _model_instance = SentenceTransformer(
                        _selected_model_config["name"], device="cpu"
                    )
                logger.debug(
                    f"Loaded sentence transformer: {_selected_model_config['name']}"
                )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    return _model_instance


def _unload_model():
    """Unloads the embedding model."""
    global _model_instance
    if _model_instance is not None:
        _model_instance = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _encode_text(text: str, model_config: Dict[str, Any]) -> np.ndarray:
    """Encode text using the appropriate model type with normalization."""
    model = _load_model()

    if model is None:
        raise RuntimeError("Model not loaded properly")

    if model_config["type"] == "gguf":
        # GGUF model encoding (logging already suppressed via callback)
        embedding = model.embed(text)
        if embedding is None:
            raise RuntimeError("GGUF model returned None")
        arr = np.array(embedding, dtype=np.float32)
    else:
        arr = model.encode(text, show_progress_bar=True, convert_to_numpy=True)
        arr = np.array(arr, dtype=np.float32)

    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


# --- SES (Symbol Essence String) Logic ---
def _load_project_symbol_map() -> Dict[str, Dict[str, Any]]:
    """Loads the project_symbol_map.json."""
    try:
        core_dir = os.path.dirname(os.path.abspath(key_manager_module.__file__))
        map_path = normalize_path(os.path.join(core_dir, PROJECT_SYMBOL_MAP_FILENAME))
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load symbol map: {e}")
    return {}


def _load_token_metadata(project_root: str) -> Dict[str, Dict[str, int]]:
    """Loads token counts from metadata.json."""
    metadata_path = os.path.join(
        project_root,
        "cline_utils",
        "dependency_system",
        "analysis",
        "embeddings",
        "metadata.json",
    )
    token_map: Dict[str, Dict[str, int]] = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                keys = data.get("keys", {})
                for key_data in keys.values():
                    path = key_data.get("path")
                    if not path:
                        continue
                    path = normalize_path(path)

                    ses = key_data.get("ses_tokens")
                    full = key_data.get("full_tokens")

                    if ses is None and "tokens" in key_data:
                        # Fallback for old structure
                        ses = key_data["tokens"]

                    if full is None and ses is not None:
                        # Best guess if full_tokens is missing
                        full = ses

                    if ses is not None and full is not None:
                        token_map[path] = {
                            "ses_tokens": int(ses),
                            "full_tokens": int(full),
                        }
        except Exception as e:
            logger.warning(f"Failed to load token metadata: {e}")
    return token_map


def generate_symbol_essence_string(
    file_path: str,
    symbol_data: Dict[str, Any],
    max_chars: int = 12288,
    symbol_map: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Constructs the Strategic Symbol Essence String (SES) from merged symbol data.

    Now optimized for runtime_symbols structure with:
    - Full type annotations from inspect
    - Inheritance hierarchies
    - Decorator chains
    - Scope references (globals/nonlocals)
    - Attribute access patterns
    - Clean, unescaped source lines

    Still includes AST enhancements:
    - Call graphs with line numbers
    - Import tracking
    - CALLED_BY analysis
    """
    project_root = get_project_root()
    rel_path = os.path.relpath(file_path, project_root)

    parts: List[str] = []

    # 1. Header
    mod_time = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
    parts.append(
        f"[FILE: {rel_path} | TYPE: {symbol_data.get('file_type', 'unknown')} | MOD: {mod_time}]"
    )

    # --- Size-Adaptive Logic ---
    # Check for token count to decide whether to use full content or summarized SES
    full_tokens = symbol_data.get("full_tokens")
    content = ""

    # Try to load content if not already loaded or if we need to count tokens
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            pass

    if full_tokens is None and content:
        # Look up in metadata.json first
        token_meta = _load_token_metadata(get_project_root())
        norm_file_path = normalize_path(file_path)
        if norm_file_path in token_meta:
            full_tokens = token_meta[norm_file_path].get("full_tokens")

    if full_tokens is None and content:
        # Final fallback: load tokenizer
        try:
            tokenizer = _get_tokenizer()
            if tokenizer:
                full_tokens = _count_tokens(content, tokenizer)
        except Exception as e:
            logger.warning(f"Could not count tokens for {file_path}: {e}")

    # Small-file full-content mode is intentionally limited to documentation-like files.
    # Structured/code files (especially SQL/JSON/Svelte) must stay distilled.
    file_type = str(symbol_data.get("file_type", "")).lower()
    is_doc_like = file_type == "md" or file_path.lower().endswith((".md", ".txt", ".rst"))
    if full_tokens and full_tokens < 8192 and content and is_doc_like:
        current_len = len("\n".join(parts))
        max_content_chars = max(0, max_chars - current_len - len("CONTENT:\n"))
        if max_content_chars > 0:
            parts.append(f"CONTENT:\n{content[:max_content_chars]}")
        return "\n".join(parts)

    # --- Large File SES Generation (Summarization) ---

    # 2. Imports (Explicit - Added for Svelte/Large Files)
    imports = cast(List[Dict[str, Any]], symbol_data.get("imports", []))
    if imports:
        parts.append("IMPORTS:")
        # Deduplicate imports
        seen_imports: Set[str] = set()
        for imp in imports:
            if isinstance(imp, str):
                if imp not in seen_imports:
                    seen_imports.add(imp)
                    parts.append(f"  {imp}")
                continue
            path = imp.get("path")
            if path and isinstance(path, str) and path not in seen_imports:
                seen_imports.add(path)
                parts.append(f"  {path}")

    # Compact Python runtime profile near the top so high-signal fields survive truncation.
    if file_type == "py":
        class_count = len(cast(List[Any], symbol_data.get("classes", [])))
        function_count = len(cast(List[Any], symbol_data.get("functions", [])))
        call_count = len(cast(List[Any], symbol_data.get("calls", [])))
        attr_access_count = len(cast(List[Any], symbol_data.get("attribute_accesses", [])))
        type_ref_count = len(cast(List[Any], symbol_data.get("type_references", [])))
        parts.append(
            "PY_RUNTIME_PROFILE: "
            f"classes={class_count}, functions={function_count}, "
            f"calls={call_count}, attribute_accesses={attr_access_count}, "
            f"type_references={type_ref_count}"
        )

        top_decorators: Dict[str, int] = {}
        for d in cast(List[Any], symbol_data.get("decorators_used", [])):
            if not isinstance(d, dict):
                continue
            d_name = cast(str, d.get("name", "")).strip()
            if d_name:
                top_decorators[d_name] = top_decorators.get(d_name, 0) + 1
        if top_decorators:
            decorator_summary = ", ".join(
                f"{k} x{v}" if v > 1 else k
                for k, v in sorted(
                    top_decorators.items(), key=lambda kv: (-kv[1], kv[0])
                )[:20]
            )
            parts.append(f"PY_TOP_DECORATORS: {decorator_summary}")

        top_exceptions: Dict[str, int] = {}
        for ex in cast(List[Any], symbol_data.get("exceptions_handled", [])):
            if not isinstance(ex, dict):
                continue
            ex_name = cast(str, ex.get("type_name_str", "")).strip()
            if ex_name:
                top_exceptions[ex_name] = top_exceptions.get(ex_name, 0) + 1
        if top_exceptions:
            exception_summary = ", ".join(
                f"{k} x{v}" if v > 1 else k
                for k, v in sorted(
                    top_exceptions.items(), key=lambda kv: (-kv[1], kv[0])
                )[:20]
            )
            parts.append(f"PY_TOP_EXCEPTIONS: {exception_summary}")

        top_type_refs: Dict[str, int] = {}
        for t in cast(List[Any], symbol_data.get("type_references", [])):
            if not isinstance(t, dict):
                continue
            t_name = cast(str, t.get("type_name_str", "")).strip()
            if t_name:
                top_type_refs[t_name] = top_type_refs.get(t_name, 0) + 1
        if top_type_refs:
            type_ref_summary = ", ".join(
                f"{k} x{v}" if v > 1 else k
                for k, v in sorted(
                    top_type_refs.items(), key=lambda kv: (-kv[1], kv[0])
                )[:25]
            )
            parts.append(f"PY_TOP_TYPES: {type_ref_summary}")

    # 3. Classes (with Runtime Enhancements)
    classes = symbol_data.get("classes", [])
    if classes:
        for c in cast(List[Any], classes):
            if not isinstance(c, dict):
                continue
            c_name = cast(str, c.get("name", "unknown"))
            c_doc = (c.get("docstring") or "").strip()
            if len(c_doc) > 280:
                c_doc = c_doc[:280] + "..."
            parts.append(f"CLASS: {c_name}")

            # Add inheritance info (runtime)
            inheritance = c.get("inheritance", {})
            bases = inheritance.get("bases", [])
            if bases:
                parts.append(f"  BASES: {', '.join(bases)}")

            # Add decorators (runtime)
            decorators = c.get("decorators", [])
            if decorators:
                parts.append(f"  DECORATORS: {', '.join(decorators)}")

            if c_doc:
                parts.append(f"  DOC: {c_doc}")

            # Methods with enhanced runtime information
            methods = c.get("methods", [])
            if methods:
                method_names: List[str] = []
                for m_item in cast(List[Any], methods):
                    if isinstance(m_item, dict):
                        m_name = cast(str, m_item.get("name", "")).strip()
                        if m_name:
                            method_names.append(m_name)
                if method_names:
                    shown_names = method_names[:60]
                    suffix = ", ..." if len(method_names) > 60 else ""
                    parts.append(f"  METHODS: {', '.join(shown_names)}{suffix}")

                max_method_details = 18 if (full_tokens and full_tokens > 20000) else 60
                for m in cast(List[Any], methods)[:max_method_details]:
                    if not isinstance(m, dict):
                        continue
                    m_name = m["name"]

                    # Prefer runtime signature over params
                    m_sig = m.get("signature")
                    if m_sig:
                        parts.append(f"  METHOD: {m_name}{m_sig}")
                    else:
                        # Fallback to old params-based approach
                        m_params = m.get("params", [])
                        m_param_str = ", ".join(m_params)
                        parts.append(f"  METHOD: {m_name}({m_param_str})")

                    m_doc = (m.get("docstring") or "").strip()
                    if len(m_doc) > 180:
                        m_doc = m_doc[:180] + "..."
                    if m_doc:
                        parts.append(f"    DOC: {m_doc}")

                    # Add type annotations (runtime)
                    type_annot = m.get("type_annotations", {})
                    if type_annot and "parameters" in type_annot:
                        # Only show non-self parameters
                        params_annot = type_annot["parameters"]
                        filtered_annot = {
                            k: v
                            for k, v in params_annot.items()
                            if k not in ["self", "cls", "return"]
                        }
                        if filtered_annot:
                            annot_str = ", ".join(
                                f"{k}={v}" for k, v in list(filtered_annot.items())
                            )
                            parts.append(f"    TYPES: {annot_str}")

                        return_type = type_annot.get("return_type")
                        if return_type and return_type != "<class 'NoneType'>":
                            parts.append(f"    RETURN_TYPE: {return_type}")

                    # Add key scope references (runtime)
                    scope_refs = m.get("scope_references", {})
                    globals_list = scope_refs.get("globals", [])
                    if globals_list:
                        # Filter out builtins and common names, keep significant ones
                        significant_globals = [
                            g
                            for g in globals_list
                            if g
                            not in [
                                "self",
                                "__init__",
                                "__class__",
                                "super",
                                "print",
                                "len",
                                "str",
                                "int",
                                "bool",
                                "list",
                                "dict",
                                "set",
                                "tuple",
                            ]
                        ][
                            :10
                        ]  # Limit to top 10
                        if significant_globals:
                            parts.append(
                                f"    GLOBALS: {', '.join(significant_globals)}"
                            )

                    nonlocals_list = scope_refs.get("nonlocals", [])
                    if nonlocals_list:
                        parts.append(f"    NONLOCALS: {', '.join(nonlocals_list)}")

                    # Closure dependencies (runtime)
                    closure_deps = m.get("closure_dependencies", [])
                    if closure_deps:
                         parts.append(f"    CLOSURE_DEPENDENCIES: {', '.join(closure_deps)}")

                    # Add attribute accesses (runtime) - shows duck-typing contracts
                    attr_accesses = m.get("attribute_accesses", [])
                    if attr_accesses:
                        significant_attrs = [
                            a for a in attr_accesses if a not in ["self", "__class__"]
                        ]
                        if significant_attrs:
                            parts.append(
                                f"    ACCESSES: {', '.join(significant_attrs)}"
                            )
                if len(cast(List[Any], methods)) > max_method_details:
                    parts.append(
                        f"  ... ({len(cast(List[Any], methods)) - max_method_details} methods omitted)"
                    )

    # 3. Top-level Functions (with Runtime Enhancements)
    functions = symbol_data.get("functions", [])
    if functions:
        parts.append("FUNCTIONS:")
        for f in cast(List[Any], functions)[:180]:
            if not isinstance(f, dict):
                continue
            name = f["name"]

            # Prefer runtime signature
            f_sig = f.get("signature")
            if f_sig:
                parts.append(f"  {name}{f_sig}")
            else:
                # Fallback
                params = f.get("params", [])
                param_str = ", ".join(params)
                parts.append(f"  {name}({param_str})")

            doc = (f.get("docstring") or "").strip()
            if len(doc) > 280:
                doc = doc[:280] + "..."
            if doc:
                parts.append(f"    DOC: {doc}")

            # Add type annotations
            type_annot = f.get("type_annotations", {})
            if type_annot and "parameters" in type_annot:
                params_annot = type_annot["parameters"]
                filtered_annot = {
                    k: v for k, v in params_annot.items() if k != "return"
                }
                if filtered_annot:
                    annot_str = ", ".join(
                        f"{k}={v}" for k, v in list(filtered_annot.items())
                    )
                    parts.append(f"    TYPES: {annot_str}")

                # Add return type if available
                return_type = type_annot.get("return_type")
                if return_type and return_type != "<class 'NoneType'>":
                    parts.append(f"    RETURN_TYPE: {return_type}")

            # Add scope references
            scope_refs = f.get("scope_references", {})
            globals_list = scope_refs.get("globals", [])
            if globals_list:
                significant_globals = [
                    g
                    for g in globals_list
                    if g
                    not in [
                        "print",
                        "len",
                        "str",
                        "int",
                        "bool",
                        "list",
                        "dict",
                        "set",
                        "tuple",
                    ]
                ]
                if significant_globals:
                    parts.append(f"    GLOBALS: {', '.join(significant_globals)}")

            nonlocals_list = scope_refs.get("nonlocals", [])
            if nonlocals_list:
                parts.append(f"    NONLOCALS: {', '.join(nonlocals_list)}")

            # Closure dependencies (runtime)
            closure_deps = f.get("closure_dependencies", [])
            if closure_deps:
                parts.append(f"    CLOSURE_DEPENDENCIES: {', '.join(closure_deps)}")
        if len(cast(List[Any], functions)) > 180:
            parts.append("  ... (functions truncated)")

    # 4. Outgoing Calls (from AST/runtime)
    calls = symbol_data.get("calls", [])
    if calls:
        call_counts: Dict[str, int] = {}
        for c in cast(List[Any], calls):
            if not isinstance(c, dict):
                continue
            target = str(c.get("target_name") or "").strip()
            source = str(c.get("potential_source") or "").strip()

            if source and target:
                call_key = f"{source}.{target}"
            else:
                call_key = target or source

            if not call_key:
                continue
            call_counts[call_key] = call_counts.get(call_key, 0) + 1

        if call_counts:
            parts.append("CALLS:")
            for call_key, count in sorted(
                call_counts.items(), key=lambda kv: (-kv[1], kv[0])
            )[:80]:
                if count > 1:
                    parts.append(f"  {call_key} x{count}")
                else:
                    parts.append(f"  {call_key}")

    # 5. Incoming Connections (CALLED_BY) - from imports analysis
    if symbol_map:
        called_by: Set[str] = set()
        fname = os.path.basename(file_path)
        fname_no_ext = os.path.splitext(fname)[0]

        for other_path, other_data in symbol_map.items():
            if other_path == file_path:
                continue

            other_imports = other_data.get("imports", [])
            for imp in other_imports:
                # Handle both string and dict import formats
                if isinstance(imp, str):
                    imp_path = imp
                else:
                    imp_path = imp.get("path")
                if not imp_path or not isinstance(imp_path, str):
                    continue

                # Match on path or filename
                if (
                    rel_path in imp_path
                    or imp_path in rel_path
                    or fname_no_ext in imp_path.replace(".", "/")
                ):
                    called_by.add(os.path.relpath(other_path, project_root))
                    break

        if called_by:
            sorted_called_by = sorted(list(called_by))
            parts.append(f"CALLED_BY: {', '.join(sorted_called_by)}")

    # Exports (all code/data files, not only web)
    exports = symbol_data.get("exports", [])
    if exports:
        parts.append("EXPORTS:")
        seen_exports: Set[str] = set()
        for e in cast(List[Any], exports):
            if not isinstance(e, dict):
                continue
            e_name = cast(str, e.get("name", e.get("default", "unknown"))).strip()
            e_from = cast(str, e.get("from", "")).strip()
            if not e_name:
                continue
            line = f"{e_name} <- {e_from}" if e_from else e_name
            if line not in seen_exports:
                seen_exports.add(line)
                parts.append(f"  {line}")
            if len(seen_exports) >= 80:
                break

    # Significant literal assignments (useful for constants/config wiring)
    lit_assigns = symbol_data.get("literal_assignments", [])
    if lit_assigns:
        parts.append("LITERAL_ASSIGNMENTS:")
        seen_assigns: Set[str] = set()
        for a in cast(List[Any], lit_assigns):
            if not isinstance(a, dict):
                continue
            a_name = cast(str, a.get("name", "unknown")).strip()
            a_val = cast(str, a.get("value", "")).replace("\n", " ").strip()
            if not a_val:
                continue
            if len(a_val) > 240:
                a_val = a_val[:240] + "..."
            line = f"{a_name} = {a_val}"
            if line not in seen_assigns:
                seen_assigns.add(line)
                parts.append(f"  {line}")
            if len(seen_assigns) >= 80:
                break

    # Python runtime-heavy fields (already collected in symbol_map)
    if file_type == "py":
        globals_defined = symbol_data.get("globals_defined", [])
        if globals_defined:
            g_names: List[str] = []
            for g in cast(List[Any], globals_defined):
                if not isinstance(g, dict):
                    continue
                g_name = cast(str, g.get("name", "")).strip()
                if g_name:
                    g_names.append(g_name)
            if g_names:
                parts.append(f"GLOBALS_DEFINED: {', '.join(sorted(set(g_names))[:60])}")

        decorators_used = symbol_data.get("decorators_used", [])
        if decorators_used:
            d_names: List[str] = []
            for d in cast(List[Any], decorators_used):
                if not isinstance(d, dict):
                    continue
                d_name = cast(str, d.get("name", "")).strip()
                if d_name:
                    d_names.append(d_name)
            if d_names:
                parts.append(f"DECORATORS_USED: {', '.join(sorted(set(d_names))[:60])}")

        exceptions_handled = symbol_data.get("exceptions_handled", [])
        if exceptions_handled:
            ex_names: List[str] = []
            for ex in cast(List[Any], exceptions_handled):
                if not isinstance(ex, dict):
                    continue
                ex_name = cast(str, ex.get("type_name_str", "")).strip()
                if ex_name:
                    ex_names.append(ex_name)
            if ex_names:
                parts.append(
                    f"EXCEPTIONS_HANDLED: {', '.join(sorted(set(ex_names))[:60])}"
                )

        with_contexts_used = symbol_data.get("with_contexts_used", [])
        if with_contexts_used:
            with_entries: List[str] = []
            for w in cast(List[Any], with_contexts_used):
                if not isinstance(w, dict):
                    continue
                context_expr = cast(str, w.get("context_expr_str", "")).strip()
                if context_expr:
                    if len(context_expr) > 120:
                        context_expr = context_expr[:120] + "..."
                    with_entries.append(context_expr)
            if with_entries:
                parts.append(f"WITH_CONTEXTS: {', '.join(sorted(set(with_entries))[:40])}")

        inheritance = symbol_data.get("inheritance", [])
        if inheritance:
            pairs: Set[str] = set()
            for h in cast(List[Any], inheritance):
                if not isinstance(h, dict):
                    continue
                cls = cast(str, h.get("class_name", "")).strip()
                base = cast(str, h.get("base_class_name", "")).strip()
                if cls and base:
                    pairs.add(f"{cls} -> {base}")
            if pairs:
                parts.append("INHERITANCE:")
                for pair in sorted(pairs)[:60]:
                    parts.append(f"  {pair}")

        type_references = symbol_data.get("type_references", [])
        if type_references:
            type_counts: Dict[str, int] = {}
            for t in cast(List[Any], type_references):
                if not isinstance(t, dict):
                    continue
                t_name = cast(str, t.get("type_name_str", "")).strip()
                if not t_name:
                    continue
                type_counts[t_name] = type_counts.get(t_name, 0) + 1
            if type_counts:
                parts.append("TYPE_REFERENCES:")
                for t_name, count in sorted(
                    type_counts.items(), key=lambda kv: (-kv[1], kv[0])
                )[:80]:
                    if count > 1:
                        parts.append(f"  {t_name} x{count}")
                    else:
                        parts.append(f"  {t_name}")

        attribute_accesses = symbol_data.get("attribute_accesses", [])
        if attribute_accesses:
            access_counts: Dict[str, int] = {}
            for a in cast(List[Any], attribute_accesses):
                if not isinstance(a, dict):
                    continue
                target = cast(str, a.get("target_name", "")).strip()
                source = cast(str, a.get("potential_source", "")).strip()
                if source and target:
                    access_key = f"{source}.{target}"
                else:
                    access_key = target or source
                if not access_key:
                    continue
                access_counts[access_key] = access_counts.get(access_key, 0) + 1
            if access_counts:
                parts.append("ATTRIBUTE_ACCESS_PATTERNS:")
                for access_key, count in sorted(
                    access_counts.items(), key=lambda kv: (-kv[1], kv[0])
                )[:100]:
                    if count > 1:
                        parts.append(f"  {access_key} x{count}")
                    else:
                        parts.append(f"  {access_key}")

    # 6. Documentation & Web Metadata (NEW)
    # If this is a doc, web, or data file, we may have links, images, scripts, or stylesheets
    is_doc = symbol_data.get("file_type") == "md" or file_path.lower().endswith(
        (".md", ".txt", ".rst")
    )
    is_web = symbol_data.get("file_type") in ("svelte", "html", "htm")
    is_css = symbol_data.get("file_type") == "css"
    is_csv = symbol_data.get("file_type") == "csv"
    is_data = symbol_data.get("file_type") in ("json", "sql")

    # CSS Specific: Collect imports for style-heavy files
    if is_css:
        raw_imports = cast(List[Dict[str, Any]], symbol_data.get("imports", []))
        css_imports = [str(imp.get("url")) for imp in raw_imports if imp.get("url")]
        if css_imports:
            parts.append(f"CSS_IMPORTS: {', '.join(css_imports)}")

    if is_csv and content:
        parts.append("CSV_PREVIEW:")
        preview_lines = content.splitlines()
        preview = "\n".join(preview_lines)
        if len(preview) > 5000:
            preview = preview[:5000] + "..."
        parts.append(textwrap.indent(preview, "  "))

    # Links (Documents, Svelte, JSON, SQL)
    links = symbol_data.get("links", [])
    if links:
        # Capturing the 'url' part of the links
        link_urls: List[str] = []
        seen_link_urls: Set[str] = set()
        for lnk in cast(List[Dict[str, Any]], links):
            u = cast(Optional[str], lnk.get("url") or lnk.get("href") or lnk.get("path"))
            if u:
                # Clean up URL to just filename for essence
                if u.startswith("file:///"):
                    u = os.path.basename(u)
                if u not in seen_link_urls:
                    seen_link_urls.add(u)
                    link_urls.append(u)
        if link_urls:
            parts.append(f"LINKS: {', '.join(link_urls)}")

    # Images (Documents, Svelte)
    images = symbol_data.get("images", [])
    if images:
        img_srcs: List[str] = []
        for img in cast(List[Dict[str, Any]], images):
            src = cast(Optional[str], img.get("src") or img.get("url"))
            if src:
                img_srcs.append(os.path.basename(src))
        if img_srcs:
            parts.append(f"IMAGES: {', '.join(img_srcs)}")

    # Scripts & Stylesheets (Web/Svelte)
    if is_web:
        scripts = symbol_data.get("scripts", [])
        if scripts:
            parts.append(f"SCRIPTS: {len(scripts)} block(s)")
            for s in cast(List[Dict[str, Any]], scripts):
                s_content = cast(str, s["content"]).strip()
                # Show more content (up to 4000 chars)
                lines: List[str] = s_content.split("\n")
                if len(s_content) > 4000:
                    preview = "\n".join(lines)
                    if len(preview) > 4000:
                        preview = preview[:4000] + "..."
                    else:
                        preview += "\n... (truncated)"
                    s_content = preview

                parts.append(f"  [Content]:\n{textwrap.indent(s_content, '    ')}")

        stylesheets = symbol_data.get("stylesheets", [])
        if stylesheets:
            parts.append(f"STYLES: {len(stylesheets)} block(s)")
            for s in cast(List[Dict[str, Any]], stylesheets):
                s_content = cast(str, s["content"]).strip()
                # Same truncation for styles
                lines: List[str] = s_content.split("\n")
                if len(lines) > 40 or len(s_content) > 2000:
                    preview = "\n".join(lines)
                    if len(preview) > 2000:
                        preview = preview[:2000] + "..."
                    else:
                        preview += "\n... (truncated)"
                    s_content = preview

                parts.append(f"  [Content]:\n{textwrap.indent(s_content, '    ')}")

        # Svelte Specifics
        props = symbol_data.get("props", [])
        if props:
            temp_props = cast(List[Dict[str, Any]], props)
            p_names: List[str] = [
                cast(str, p["name"]) for p in temp_props if p.get("name")
            ]
            if p_names:
                parts.append(f"PROPS: {', '.join(p_names)}")

        state = symbol_data.get("state", [])
        if state:
            temp_state = cast(List[Dict[str, Any]], state)
            s_names: List[str] = [
                cast(str, s["name"]) for s in temp_state if s.get("name")
            ]
            if s_names:
                parts.append(f"STATE: {', '.join(s_names)}")

        data_store = symbol_data.get("reactive", [])
        if data_store:
            # Reactive statements ($: ...)
            parts.append("REACTIVE:")
            for r in cast(List[Dict[str, Any]], data_store):
                content_text = cast(str, r.get("content", ""))
                parts.append(f"  $: {content_text.strip()}")

        logic = symbol_data.get("logic", [])
        if logic:
            parts.append("LOGIC:")
            for l in cast(List[Dict[str, Any]], logic):
                l_type = cast(str, l.get("type", "block")).upper()
                l_content = cast(str, l.get("content", "")).replace("\n", " ").strip()
                parts.append(f"  [{l_type}] {l_content}")

        template_outline = symbol_data.get("template_outline", [])
        if template_outline:
            parts.append("TEMPLATE_OUTLINE:")
            for line in cast(List[str], template_outline):
                parts.append(f"  {line}")

        # Render Distilled Template from Full Tree (Runtime distillation)
        template_tree = symbol_data.get("template_tree", [])
        if template_tree and not template_outline:
            parts.append("DISTILLED TEMPLATE:")

            # Helper to render tree to string list
            def _render_tree(nodes: List[Dict[str, Any]], depth: int = 0) -> List[str]:
                lines: List[str] = []
                if depth > 20:
                    return lines
                indent = "  " * depth
                for node in nodes:
                    n_type = node.get("type")
                    if n_type == "element":
                        # Extract tag name from content or children
                        tag = "<unknown>"
                        attrs = ""

                        # Inspect children for start_tag
                        for child in node.get("children", []):
                            if child.get("type") in ("start_tag", "self_closing_tag"):
                                # Extract name/attributes from start_tag children
                                for sub in child.get("children", []):
                                    if sub.get("type") == "tag_name":
                                        tag = sub.get("content", "")
                                    elif sub.get("type") == "attribute":
                                        # name/value
                                        a_name = sub.get("name", "")
                                        a_val = sub.get("value", "")
                                        # clean quotes
                                        if (
                                            a_val.startswith(('"', "'"))
                                            and len(a_val) >= 2
                                        ):
                                            a_val = a_val[1:-1]

                                        if a_name == "id":
                                            attrs += f"#{a_val}"
                                        elif a_name == "class":
                                            classes = a_val.split()
                                            if classes:
                                                attrs += "." + ".".join(classes)
                                        elif a_name == "slot":
                                            attrs += f"[slot={a_val}]"
                                break

                        if tag != "<unknown>":
                            lines.append(f"{indent}<{tag}{attrs}>")
                            lines.extend(
                                _render_tree(node.get("children", []), depth + 1)
                            )

                    elif n_type in (
                        "if_statement",
                        "each_statement",
                        "await_statement",
                        "key_statement",
                    ):
                        # Logic blocks
                        # Try to get the first line of content for the header
                        content = node.get("content", "").split("\n")[0].strip()
                        lines.append(f"{indent}{content}")
                        lines.extend(_render_tree(node.get("children", []), depth + 1))

                    elif n_type in (
                        "else_block",
                        "then_block",
                        "catch_block",
                        "else_if_block",
                    ):
                        # Branch blocks
                        head = (
                            n_type.replace("_block", "").replace("_", " ")
                            if n_type
                            else "branch"
                        )
                        lines.append(f"{indent}{head}")
                        lines.extend(
                            _render_tree(
                                cast(List[Dict[str, Any]], node.get("children", [])),
                                depth + 1,
                            )
                        )
                return lines

            parts.extend(_render_tree(template_tree))

        components = symbol_data.get("components", [])
        if components:
            c_list = cast(List[Dict[str, Any]], components)
            c_names = sorted(
                list(set(cast(str, c.get("name")) for c in c_list if c.get("name")))
            )
            if c_names:
                parts.append(f"COMPONENTS: {', '.join(c_names)}")

    # JS/TS Specifics (outside web-only branch)
    js_comments = symbol_data.get("comments", [])
    if js_comments:
        parts.append("COMMENTS / JSDOC:")
        for c in cast(List[str], js_comments)[:120]:
            parts.append(f"  // {c}")

    js_literals = symbol_data.get("literals", [])
    if js_literals:
        parts.append("LITERALS:")
        significant_literals = [
            l
            for l in cast(List[str], js_literals)
            if any(
                x in l.lower()
                for x in [
                    "/",
                    "http",
                    "select ",
                    "insert ",
                    "update ",
                    "delete ",
                    ".json",
                    ".yaml",
                    ".sql",
                ]
            )
            or len(l) > 10
        ]
        for l in significant_literals[:120]:
            parts.append(f"  \"{l}\"")

    # JS/TS Body Essence (from enhanced analyzer)
    for f in cast(List[Any], symbol_data.get("functions", [])):
        if not isinstance(f, dict):
            continue
        if "body_essence" in f:
            f_name = cast(str, f.get("name", "unknown"))
            body_essence = cast(str, f.get("body_essence", "")).strip()
            if body_essence:
                parts.append(f"  BODY ({f_name}):")
                parts.append(f"    {body_essence}")

    # Definitions (SQL, etc.)
    is_data = file_type in ("sql", "json", "yaml", "xml", "csv")
    if is_data:
        sql_table_ops: Dict[str, Set[str]] = {}

        def _track_sql_operation(table_name: str, op_name: str) -> None:
            if not table_name:
                return
            normalized_table = table_name.strip().strip("\"").lower()
            if not normalized_table:
                return
            if normalized_table not in sql_table_ops:
                sql_table_ops[normalized_table] = set()
            if op_name:
                sql_table_ops[normalized_table].add(op_name.lower())

        definitions = symbol_data.get("definitions", [])
        if definitions:
            parts.append("DEFINITIONS:")
            seen_defs: Set[Tuple[str, str]] = set()
            added_defs = 0
            for defn in cast(List[Any], definitions):
                if not isinstance(defn, dict):
                    continue
                type_name = (
                    cast(str, defn.get("type", "unknown"))
                    .replace("_statement", "")
                    .replace("create.", "CREATE ")
                    .upper()
                )
                summary = cast(str, defn.get("summary", "")).strip()
                if len(summary) > 280:
                    summary = summary[:280] + "..."

                # Deduplicate based on type + summary
                summary_key = (type_name, summary)
                if summary_key not in seen_defs:
                    seen_defs.add(summary_key)
                    parts.append(f"  [{type_name}] {summary}")
                    added_defs += 1
                    if added_defs >= 180:
                        parts.append("  ... (truncated)")
                        break

                table_hint = cast(str, defn.get("table", "")).strip()
                if not table_hint:
                    m_table = re.search(
                        r"(?i)\b(?:from|into|update|join|table|view|copy)\s+([a-zA-Z0-9_.\"]+)",
                        summary,
                    )
                    if m_table:
                        table_hint = cast(str, m_table.group(1))
                if table_hint:
                    _track_sql_operation(table_hint, type_name)

        # SQL Inserts (New)
        inserts = symbol_data.get("inserts", [])
        if inserts:
            parts.append("SQL INSERTS:")
            # De-duplicate inserts by table and column-map
            unique_inserts: List[Dict[str, Any]] = []
            seen_inserts: Set[Tuple[str, str]] = set()
            for i in cast(List[Any], inserts):
                if not isinstance(i, dict):
                    continue
                table = i.get("table", "unknown")
                columns = json.dumps(i.get("columns", {}), sort_keys=True)
                key = (table, columns)
                if key not in seen_inserts:
                    seen_inserts.add(key)
                    unique_inserts.append(i)
            
            for i in unique_inserts[:120]:
                table = i.get("table", "unknown")
                cols = i.get("columns", {})
                cols_str = ", ".join([f"{k}={v}" for k, v in list(cols.items())])
                parts.append(f"  INSERT INTO {table} ({cols_str})")
                _track_sql_operation(cast(str, table), "insert")
            if len(unique_inserts) > 120:
                parts.append("  ... (truncated)")

        columns = symbol_data.get("columns", [])
        if columns:
            parts.append("COLUMNS:")
            for col in cast(List[Any], columns)[:220]:
                if not isinstance(col, dict):
                    continue
                parts.append(f"  {col.get('name')} ({col.get('type')})")
            if len(cast(List[Dict[str, Any]], columns)) > 220:
                parts.append("  ... (truncated)")

        relationships = symbol_data.get("relationships", [])
        if relationships:
            parts.append("RELATIONSHIPS:")
            for rel in cast(List[Any], relationships)[:180]:
                if not isinstance(rel, dict):
                    continue
                parts.append(
                    f"  {rel.get('source_col')} -> {rel.get('target_table')}({rel.get('target_col')})"
                )
                _track_sql_operation(cast(str, rel.get("target_table", "")), "fk_ref")
            if len(cast(List[Dict[str, Any]], relationships)) > 180:
                parts.append("  ... (truncated)")

        # SQL-specific: Tables defined and referenced (from AST analysis)
        tables_defined = symbol_data.get("tables_defined", [])
        if tables_defined:
            unique_tables_defined = sorted(set(cast(List[str], tables_defined)))
            shown_defined = unique_tables_defined
            parts.append(f"TABLES_DEFINED: {', '.join(shown_defined)}")
            for table_name in shown_defined:
                _track_sql_operation(table_name, "define")

        tables_referenced = symbol_data.get("tables_referenced", [])
        if tables_referenced:
            unique_tables_referenced = sorted(set(cast(List[str], tables_referenced)))
            shown_referenced = unique_tables_referenced
            parts.append(f"TABLES_REFERENCED: {', '.join(shown_referenced)}")
            for table_name in shown_referenced:
                _track_sql_operation(table_name, "reference")

        # COPY-aware extraction for SQL dumps where parser captures mostly table refs.
        if file_type == "sql" and content:
            copy_stmt_pattern = re.compile(
                r"(?im)^\s*copy\s+([^\s(]+)\s*\(([^)]*)\)\s+from\s+stdin\s*;"
            )
            copy_matches = list(copy_stmt_pattern.finditer(content))
            if copy_matches:
                parts.append("COPY_BLOCKS:")
                for copy_match in copy_matches[:120]:
                    table_name = copy_match.group(1).strip()
                    raw_columns = copy_match.group(2).strip()
                    column_names = [
                        c.strip().strip("\"")
                        for c in raw_columns.split(",")
                        if c.strip()
                    ]

                    tail = content[copy_match.end() :]
                    end_match = re.search(r"(?m)^\s*\\\.\s*$", tail)
                    data_block = tail[: end_match.start()] if end_match else tail
                    data_rows = [ln for ln in data_block.splitlines() if ln.strip()]
                    row_count = len(data_rows)
                    sample_row = data_rows[0] if data_rows else ""
                    if len(sample_row) > 200:
                        sample_row = sample_row[:200] + "..."

                    column_preview = ", ".join(column_names[:12])
                    if len(column_names) > 12:
                        column_preview = column_preview + ", ..."

                    parts.append(
                        f"  {table_name} cols={len(column_names)} rows~{row_count} [{column_preview}]"
                    )
                    if sample_row:
                        parts.append(f"    sample: {sample_row}")
                    _track_sql_operation(table_name, "copy")

                if len(copy_matches) > 120:
                    parts.append("  ... (truncated)")

        if file_type == "sql" and sql_table_ops:
            parts.append("TABLE_OPERATIONS:")
            sorted_tables = sorted(sql_table_ops.keys())
            for table_name in sorted_tables[:220]:
                ops = ", ".join(sorted(sql_table_ops[table_name]))
                parts.append(f"  {table_name}: {ops}")
            if len(sorted_tables) > 220:
                parts.append("  ... (truncated)")

        # JSON-specific structured context
        json_keys = symbol_data.get("json_keys", [])
        if json_keys:
            parts.append("JSON_KEYS:")
            seen_paths: Set[str] = set()
            for entry in cast(List[Dict[str, Any]], json_keys):
                path = cast(str, entry.get("path") or entry.get("key") or "").strip()
                if path and path not in seen_paths:
                    seen_paths.add(path)
                    parts.append(f"  {path}")

        json_refs = symbol_data.get("json_refs", [])
        if json_refs:
            parts.append("JSON_REFS:")
            seen_refs: Set[str] = set()
            for ref in cast(List[Dict[str, Any]], json_refs):
                key_path = cast(str, ref.get("key_path", "")).strip()
                value = cast(str, ref.get("value", "")).strip()
                if not value:
                    continue
                line = f"{key_path} -> {value}" if key_path else value
                if line not in seen_refs:
                    seen_refs.add(line)
                    parts.append(f"  {line}")

    if is_doc:
        headers = symbol_data.get("headers", [])
        code_blocks = symbol_data.get("code_blocks", [])
        doc_link_count = len(cast(List[Dict[str, Any]], links)) if isinstance(links, list) else 0
        doc_image_count = (
            len(cast(List[Dict[str, Any]], images)) if isinstance(images, list) else 0
        )

        parts.append(
            f"DOC_PROFILE: headers={len(cast(List[Any], headers))}, "
            f"code_blocks={len(cast(List[Any], code_blocks))}, "
            f"links={doc_link_count}, images={doc_image_count}"
        )

        if headers:
            parts.append("HEADERS:")
            seen_headers: Set[str] = set()
            for h in cast(List[Dict[str, Any]], headers)[:220]:
                raw_level = h.get("level", 1)
                level = raw_level if isinstance(raw_level, int) else 1
                indent = "  " * max(0, min(level - 1, 5))
                text = cast(str, h.get("text", "")).strip()
                if not text:
                    continue
                header_line = f"{indent}- {text}"
                if header_line not in seen_headers:
                    seen_headers.add(header_line)
                    parts.append(header_line)
            if len(cast(List[Dict[str, Any]], headers)) > 220:
                parts.append("  ... (truncated)")

        if code_blocks:
            lang_counts: Dict[str, int] = {}
            for cb in cast(List[Any], code_blocks):
                if not isinstance(cb, dict):
                    continue
                lang = cast(str, cb.get("language", "text")).strip().lower() or "text"
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            if lang_counts:
                lang_summary = ", ".join(
                    f"{lang}:{count}"
                    for lang, count in sorted(
                        lang_counts.items(), key=lambda kv: (-kv[1], kv[0])
                    )[:20]
                )
                parts.append(f"CODE_LANGS: {lang_summary}")

            parts.append("CODE_SIGNATURES:")
            seen_signatures: Set[str] = set()
            for cb in cast(List[Any], code_blocks):
                if not isinstance(cb, dict):
                    continue
                lang = cast(str, cb.get("language", "text")).strip().lower() or "text"
                cb_content = cast(str, cb.get("content", "")).strip()
                if not cb_content:
                    continue

                candidate_lines = [ln.strip() for ln in cb_content.splitlines() if ln.strip()]
                signature_line = ""
                for line in candidate_lines[:20]:
                    lowered = line.lower()
                    if (
                        line.startswith(("def ", "class ", "CREATE ", "INSERT ", "UPDATE ", "SELECT ", "DELETE ", "COPY "))
                        or lowered.startswith(
                            (
                                "def ",
                                "class ",
                                "create ",
                                "insert ",
                                "update ",
                                "select ",
                                "delete ",
                                "copy ",
                                "function ",
                                "const ",
                                "let ",
                                "var ",
                            )
                        )
                    ):
                        signature_line = line
                        break
                if not signature_line:
                    signature_line = candidate_lines[0]
                if len(signature_line) > 200:
                    signature_line = signature_line[:200] + "..."

                block_summary = f"  [{lang}] {signature_line}"
                if block_summary not in seen_signatures:
                    seen_signatures.add(block_summary)
                    parts.append(block_summary)
                if len(seen_signatures) >= 120:
                    parts.append("  ... (truncated)")
                    break

        # Essence extraction for docs > 8k (handled at top)
        essence = preprocess_doc_structure(content) if content else ""
        if essence:
            parts.append(f"ESSENCE:\n{essence}")

    # 5. Generic Fallback (Crucial for unanalyzed file types or symbols-sparse files)
    current_ses_len = len("\n".join(parts))
    if content and current_ses_len < 2500 and len(content) > current_ses_len:
        parts.append("CONTENT_PREVIEW:")
        snippet = content[:2500].strip()
        parts.append(textwrap.indent(snippet + ("..." if len(content) > 2500 else ""), "  "))

    # Join and truncate if needed
    result = "\n".join(parts)
    if len(result) > max_chars:
        result = result[:max_chars] + "..."

    return result


def parse_structured_doc(content: str) -> str:
    """
    Parses a structured documentation file (using ---SECTION_START---/---SECTION_END--- markers)
    and extracts the essence for SES generation.

    Extraction rules:
    - TAGS: Full inclusion (tags + related_tags)
    - CONTEXT: Full inclusion (1-2 dense sentences)
    - OVERVIEW: Headers only
    - DETAILS: Sub-headers only (code blocks handled separately via symbol_data)
    - REFERENCES: Full inclusion (links to related files)
    """
    parts: List[str] = []

    # Extract TAGS section (full)
    tags_match = _extract_section(content, "TAGS")
    if tags_match:
        for line in tags_match.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("<!--"):
                parts.append(f"TAG: {line}")

    # Extract CONTEXT section (full)
    context_match = _extract_section(content, "CONTEXT")
    if context_match:
        for line in context_match.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("##"):
                parts.append(line)

    # Extract OVERVIEW section (headers and first bit of content)
    overview_match = _extract_section(content, "OVERVIEW")
    if overview_match:
        o_lines = [l.strip() for l in overview_match.strip().split("\n") if l.strip()]
        for i, line in enumerate(o_lines):
            if line.startswith("#"):
                parts.append(line)
                # Capture next non-header line if available
                if i + 1 < len(o_lines) and not o_lines[i + 1].startswith("#"):
                    snippet = o_lines[i + 1]
                    parts.append(f"  {snippet}")

    # Extract DETAILS section (sub-headers and first bit)
    details_match = _extract_section(content, "DETAILS")
    if details_match:
        d_lines = [l.strip() for l in details_match.strip().split("\n") if l.strip()]
        for i, line in enumerate(d_lines):
            if line.startswith("#"):
                parts.append(line)
                if i + 1 < len(d_lines) and not d_lines[i + 1].startswith("#"):
                    snippet = d_lines[i + 1]
                    parts.append(f"  {snippet}")

    # Extract REFERENCES section (full)
    refs_match = _extract_section(content, "REFERENCES")
    if refs_match:
        for line in refs_match.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("##"):
                parts.append(line)

    return "\n".join(parts)


def _extract_section(content: str, section_name: str) -> Optional[str]:
    """
    Extracts the content between ---SECTION_NAME_START--- and ---SECTION_NAME_END--- markers.
    Returns None if markers are not found.
    """
    start_marker = f"---{section_name}_START---"
    end_marker = f"---{section_name}_END---"

    start_idx = content.find(start_marker)
    if start_idx == -1:
        return None

    start_idx += len(start_marker)
    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        return None

    return content[start_idx:end_idx]


def preprocess_doc_structure(content: str) -> str:
    """
    Preprocesses documentation for embedding/reranking to extract its essence.

    If the document follows the structured template (has ---TAGS_START--- marker),
    uses parse_structured_doc for precise extraction.
    Otherwise, falls back to header + first-paragraph extraction.
    """
    # Check for structured doc markers
    if "---TAGS_START---" in content:
        return parse_structured_doc(content)

    # Fallback: extract headers/snippets and transcript-style turns
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if not lines:
        return ""

    transcript_markers = 0
    for line in lines[:600]:
        lower = line.lower()
        if (
            '"user"' in lower
            or '"model"' in lower
            or '"assistant"' in lower
            or "# \"user\"" in lower
            or "# \"model\"" in lower
        ):
            transcript_markers += 1

    # Transcript-like docs: preserve speaker-turn snippets for semantic retrieval.
    if transcript_markers >= 6:
        turns: List[str] = []
        for i, line in enumerate(lines):
            lower = line.lower()
            role = ""
            if '"user"' in lower or "# \"user\"" in lower:
                role = "USER"
            elif (
                '"model"' in lower
                or '"assistant"' in lower
                or "# \"model\"" in lower
            ):
                role = "MODEL"
            if not role:
                continue

            snippet = ""
            for j in range(i + 1, min(i + 8, len(lines))):
                candidate = lines[j].strip().strip(",")
                candidate = candidate.strip("'\"")
                if (
                    not candidate
                    or candidate.startswith("#")
                    or candidate.startswith("```")
                    or candidate.lower().startswith(("http", "file://"))
                ):
                    continue
                snippet = candidate
                break

            if snippet:
                if len(snippet) > 240:
                    snippet = snippet[:240] + "..."
                turns.append(f"{role}: {snippet}")
            if len(turns) >= 140:
                break

        if turns:
            return "\n".join(turns)

    essence_parts: List[str] = []
    last_header_index = -1
    seen_lines: Set[str] = set()

    for i, line in enumerate(lines):
        if line.startswith("#"):
            header = line[:220]
            if header not in seen_lines:
                seen_lines.add(header)
                essence_parts.append(header)
            last_header_index = i
            continue

        # Keep short snippet lines after each header.
        if last_header_index != -1 and i in (
            last_header_index + 1,
            last_header_index + 2,
            last_header_index + 3,
        ):
            if not any(line.startswith(c) for c in ["http", "file://"]):
                snippet = line
                if len(snippet) > 240:
                    snippet = snippet[:240] + "..."
                candidate = f"  {snippet}"
                if candidate not in seen_lines:
                    seen_lines.add(candidate)
                    essence_parts.append(candidate)
            continue

        # Capture informative bullets in long freeform docs.
        if line.startswith(("- ", "* ")):
            bullet = line[:220]
            if bullet not in seen_lines:
                seen_lines.add(bullet)
                essence_parts.append(bullet)

        # If no headers, keep a few lead paragraphs.
        if last_header_index == -1 and len(essence_parts) < 12:
            if not any(
                line.startswith(c) for c in ["> ", "http", "file://", "```", "|"]
            ):
                para = line[:240]
                if para not in seen_lines:
                    seen_lines.add(para)
                    essence_parts.append(para)

        if len(essence_parts) >= 180:
            break

    return "\n".join(essence_parts)


# --- Reranker Logic ---

_reranker_model: Optional[Any] = None
_reranker_tokenizer: Optional[Any] = None

# Pre-computed tokenizer values to avoid concurrency issues
reranker_false_id: Optional[int] = None
reranker_true_id: Optional[int] = None
_reranker_prefix_tokens: Optional[List[int]] = None
_reranker_suffix_tokens: Optional[List[int]] = None

# Reranking tracking variables
reranked_files: Set[str] = set()
reranking_counter: int = 0
total_files_to_rerank: int = 0

# Reranker scheduling — serializes the VRAM read → batch plan → allocate
# critical section so concurrent threads don't all plan against the same
# stale VRAM snapshot. Forward passes still run concurrently.
_RERANKER_PLAN_LOCK = threading.Lock()
MIN_RERANK_BATCH_SIZE = 6  # Don't thrash with items=1; wait for VRAM instead
RERANK_MAX_PROMPT_TOKENS = 8192
RERANK_WAIT_TIMEOUT_SEC = 10.0
RERANK_ALLOC_TIMEOUT_SEC = 20.0

# Qwen3 Reranker Configuration
RERANKER_REPO_ID = "ManiKumarAdapala/Qwen3-Reranker-0.6B-Q8_0-Safetensors"
RERANKER_FILES = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def _download_file(url: str, path: str, description: str) -> bool:
    """Generic file download with progress reporting."""
    try:
        logger.info(f"Downloading {description} from {url} to {path}")
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192

            with open(path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(
                            f"\rDownload progress ({description}): {progress:.1f}% ({downloaded}/{total_size} bytes)",
                            end="",
                            flush=True,
                        )
            print()  # Newline after progress bar
        return True
    except Exception as e:
        logger.error(f"Failed to download {description}: {e}")
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return False


def _verify_reranker_model(model_dir: str) -> bool:
    """Verify that all required Qwen3 reranker files exist and are valid."""
    if not os.path.exists(model_dir):
        return False

    for filename in RERANKER_FILES:
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            logger.warning(f"Missing reranker file: {filename}")
            return False
        if os.path.getsize(file_path) == 0:
            logger.warning(f"Empty reranker file: {filename}")
            return False

    return True


def _download_reranker_model(model_dir: str) -> bool:
    """Download all required Qwen3 reranker files."""
    os.makedirs(model_dir, exist_ok=True)

    success = True
    for filename in RERANKER_FILES:
        url = f"https://huggingface.co/{RERANKER_REPO_ID}/resolve/main/{filename}"
        path = os.path.join(model_dir, filename)

        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue

        if not _download_file(url, path, filename):
            success = False
            break

    if success and _verify_reranker_model(model_dir):
        logger.debug(f"Successfully downloaded Qwen3 reranker to {model_dir}")
        return True
    else:
        logger.error("Failed to download or verify Qwen3 reranker")
        return False


def _load_reranker_model():
    """Lazy loads the reranker model (Singleton)."""
    global _reranker_model, _reranker_tokenizer
    global reranker_false_id, reranker_true_id, _reranker_prefix_tokens, _reranker_suffix_tokens

    with MODEL_LOCK:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if _reranker_model is not None:
            return _reranker_tokenizer, _reranker_model

        try:
            # Use local model path if available, otherwise download
            project_root = get_project_root()
            local_model_path = os.path.join(project_root, "models", "qwen3_reranker")

            if _verify_reranker_model(local_model_path):
                logger.debug(f"Loading reranker from local path: {local_model_path}")
            else:
                logger.info(
                    f"Local reranker not found or incomplete at {local_model_path}. Downloading..."
                )
                if _download_reranker_model(local_model_path):
                    logger.info(f"Download complete. Loading from: {local_model_path}")
                else:
                    raise RuntimeError("Failed to download reranker model")

            model_name_or_path = local_model_path

            device = _select_device()

            _reranker_tokenizer = cast(
                Any,
                AutoTokenizer.from_pretrained(  # type: ignore
                    model_name_or_path,
                    trust_remote_code=True,
                    padding_side="left",  # Left padding for generation/classification to align last token
                ),
            )
            assert _reranker_tokenizer is not None

            # Pre-compute special tokens and IDs under lock to avoid concurrency issues
            reranker_false_id = _reranker_tokenizer.convert_tokens_to_ids("no")
            reranker_true_id = _reranker_tokenizer.convert_tokens_to_ids("yes")

            if reranker_false_id is None or reranker_true_id is None:
                logger.error(
                    f"Could not find 'yes' (id={reranker_true_id}) or 'no' (id={reranker_false_id}) tokens in tokenizer. Model may be corrupted."
                )
                raise RuntimeError("Invalid tokenizer state for reranker")

            prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

            _reranker_prefix_tokens = _reranker_tokenizer.encode(
                prefix, add_special_tokens=False
            )
            _reranker_suffix_tokens = _reranker_tokenizer.encode(
                suffix, add_special_tokens=False
            )

            # Optimizations: Flash Attention 2 and float16 for CUDA
            try:
                if device == "cuda":
                    _reranker_model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                        model_name_or_path,
                        dtype=torch.float16,  # Use 'dtype' instead of 'torch_dtype'
                        attn_implementation="flash_attention_2",
                        trust_remote_code=True,
                    )
                else:
                    _reranker_model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                        model_name_or_path, trust_remote_code=True
                    )
            except Exception as e:
                logger.warning(
                    f"Optimization failed, falling back to standard load: {e}"
                )
                _reranker_model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                    model_name_or_path, trust_remote_code=True
                )

                # Only move non-quantized models manually
                if not getattr(_reranker_model, "is_quantized", False):
                    _reranker_model.to(device)

            _reranker_model.eval()

            # Create a dummy tensor to prime the CUDA context with the expected dtype and device
            if device == "cuda":
                dummy_tensor = torch.zeros(1, dtype=torch.float16, device=device)
                del dummy_tensor
                torch.cuda.empty_cache()
                logger.debug("CUDA context primed with dummy tensor.")

            # Verify Flash Attention
            if hasattr(_reranker_model.config, "_attn_implementation"):
                attn_impl = getattr(
                    _reranker_model.config, "_attn_implementation", "unknown"
                )
                if attn_impl == "flash_attention_2":
                    logger.info("Flash Attention 2 is active!")
                else:
                    logger.warning(
                        f"Using {attn_impl} (not Flash Attention). Check flash-attn install."
                    )

            # Measure model memory footprint for adaptive worker calculation
            if device == "cuda":
                torch.cuda.synchronize()
                model_memory_gb = torch.cuda.memory_allocated() / (1024**3)
                logger.info(
                    f"Loaded Qwen3-Reranker (Q8 quantized) on {_reranker_model.device}. "
                    f"Model footprint: {model_memory_gb:.2f}GB"
                )
            else:
                logger.info(f"Loaded Qwen3-Reranker-0.6B on {_reranker_model.device}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}", exc_info=True)
            # Clean up partial load
            _reranker_model = None
            _reranker_tokenizer = None
            reranker_false_id = None
            reranker_true_id = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None

    return _reranker_tokenizer, _reranker_model


def unload_reranker_model():
    """Unloads reranker model to free memory."""
    global _reranker_model, _reranker_tokenizer
    _reranker_model = None
    _reranker_tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()


def get_instruction_for_relation_type(source_path: str, target_path: str) -> str:
    """Returns the appropriate instruction for the SES reranking task based on file types."""
    from cline_utils.dependency_system.utils.path_utils import get_file_type

    source_ext_type = get_file_type(source_path)
    target_ext_type = get_file_type(target_path)

    def map_to_category(ext_type: str) -> str:
        if ext_type in ["md", "txt", "rst"]:
            return "doc"
        return "code"  # Treat everything else (py, js, html, etc.) as code for instruction purposes

    source_cat = map_to_category(source_ext_type)
    target_cat = map_to_category(target_ext_type)

    if source_cat == "code" and target_cat == "code":
        return "Retrieve the code file that is structurally or semantically related to the query code file, sharing dependencies or functionality. Will file A be affected by changes in file B, or vice versa?"
    elif (source_cat == "doc" and target_cat == "code") or (
        source_cat == "code" and target_cat == "doc"
    ):
        return "Retrieve the file that explains, provides relevant context, or implements the concepts described in the query."
    elif source_cat == "doc" and target_cat == "doc":
        return "Retrieve the documentation file that explains, informs, or supports the query documentation file."
    else:
        return "Retrieve the structural dependency match based on symbol definitions and references"


def _calculate_dynamic_batch_size(
    available_mem_gb: float, context_length: int, device: str
) -> int:
    """
    Calculates a safe batch size based on available memory and context length.
    Uses empirical measurements.

    Empirical observations on RTX 4060 8GB with Qwen3-Reranker-0.6B (1.1GB):
    - Model footprint: ~1.1GB
    - Context=1000: ~0.3GB per sample -> Batch size ~15
    - Context=4000: ~0.5GB per sample -> Batch size ~8
    - Context=8000: ~0.8GB per sample -> Batch size ~5
    - Context=16000: ~1.5GB per sample -> Batch size ~3
    - Context=32000: ~2.5GB per sample -> Batch size ~2
    """
    # Empirical formula: MB per sample = base_overhead + (context_length * kb_per_token)
    # Set base_overhead_mb to 30MB to cover per-item fixed metadata/overhead + safety.
    # Set mb_per_1k_tokens to 120MB to cover KV cache + activations + padding overhead + safety.
    base_overhead_mb = 30  # Base overhead per sample in MB
    mb_per_1k_tokens = 120  # MB per 1000 tokens

    estimated_mb_per_sample = (
        base_overhead_mb + (context_length / 1000.0) * mb_per_1k_tokens
    )
    estimated_gb_per_sample = estimated_mb_per_sample / 1024.0

    # Safety buffer: leave 10% or 0.5GB, whichever is larger
    # Reduced constant buffer from 1.0GB to 0.5GB as reliability improves
    reserved_buffer = max(0.5, available_mem_gb * 0.1)
    usable_mem_gb = max(0.0, available_mem_gb - reserved_buffer)

    max_batch = int(usable_mem_gb / estimated_gb_per_sample)

    # Clamp batch size
    max_batch = max(1, min(max_batch, 50))  # Cap at 50

    # logger.debug(
    #     f"Dynamic Batch Sizing: Available={available_mem_gb:.2f}GB, "
    #     f"Context={context_length}, Est.PerSample={estimated_gb_per_sample:.2f}GB "
    #     f"-> Batch Size={max_batch}"
    # )
    return max_batch


def _get_rerank_cache_key(
    query_text: str,
    candidate_texts: List[str],
    top_k: int = 10,
    source_file_path: Optional[str] = None,
    instruction: Optional[str] = None,
) -> str:
    """Generates a deterministic cache key for reranking."""
    # Hash the candidate texts to create a compact key part
    import hashlib

    candidates_hash = hashlib.md5(
        "".join(sorted(candidate_texts)).encode("utf-8")
    ).hexdigest()
    return f"rerank:{hashlib.md5(query_text.encode('utf-8')).hexdigest()}:{candidates_hash}:{top_k}"


@cached("reranking", key_func=_get_rerank_cache_key)
def rerank_candidates_with_qwen3(
    query_text: str,
    candidate_texts: List[str],
    top_k: int = 10,
    source_file_path: Optional[str] = None,
    instruction: Optional[str] = None,
) -> List[Tuple[int, float]]:
    """
    Rerank candidate texts using Qwen3 reranker model.
    Implements official Qwen3-Reranker-0.6B format from HuggingFace with special token handling.
    Optimizes throughput by sorting candidates by length and using dynamic batch sizing.
    """
    tokenizer, model = _load_reranker_model()

    # Fallback: If reranker failed to load (e.g. low memory), return original candidates
    if tokenizer is None or model is None:
        logger.warning(
            "Reranker unavailable (likely due to memory constraints). Skipping reranking."
        )
        # Return candidates with default score 1.0, preserving original order (which is usually vector sim order)
        return [(i, 1.0) for i in range(len(candidate_texts))][:top_k]

    # Use pre-computed values
    token_false_id = reranker_false_id
    token_true_id = reranker_true_id

    if token_false_id is None or token_true_id is None:
        logger.error("Pre-computed token IDs are missing.")
        raise RuntimeError("Invalid tokenizer state for reranker")

    # Setup special tokens and template structure
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    # Default instruction
    if instruction is None:
        instruction = get_instruction_for_relation_type(source_file_path or "", "")

    device = next(model.parameters()).device.type  # 'cuda', 'cpu', 'mps'

    # 1. Prepare and Tokenize All Candidates
    # We tokenize everything upfront to get accurate lengths for sorting and batching.
    query_text = query_text[:12000]
    full_prompts_data: List[Dict[str, Any]] = []
    for i, doc in enumerate(candidate_texts):
        doc_limited = doc[:12000]
        prompt = f"{prefix}<Instruct>: {instruction}\n<Query>: {query_text}\n<Document>: {doc_limited}{suffix}"
        full_prompts_data.append({"index": i, "text": prompt})

    try:
        # Tokenize without padding first to get raw lengths
        # Cap prompt length to keep VRAM usage predictable under concurrent reranking.
        all_inputs = tokenizer(
            [p["text"] for p in full_prompts_data],
            padding=False,
            truncation=True,
            max_length=RERANK_MAX_PROMPT_TOKENS,
            add_special_tokens=False,  # We added them manually in the prompt string
        )

        for i, input_ids in enumerate(all_inputs["input_ids"]):
            full_prompts_data[i]["input_ids"] = input_ids
            full_prompts_data[i]["length"] = len(input_ids)

    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        return []

    # 2. Sort by Length (Ascending)
    # This groups short items together (large batches) and long items together (small batches).
    sorted_items = sorted(full_prompts_data, key=lambda x: x["length"])

    all_scores: List[Tuple[int, float]] = []
    start_idx = 0
    total_candidates = len(sorted_items)

    # 3. Process in Dynamic Batches with VRAM coordination
    # Get the VRAM manager directly for coordinated VRAM management
    vram_manager = None
    if device == "cuda":
        from cline_utils.dependency_system.utils.resource_validator import (
            get_vram_manager,
        )

        vram_manager = get_vram_manager()

    while start_idx < total_candidates:
        # Check for backpressure from the VRAM manager
        if vram_manager is not None and vram_manager.should_pause_for_backpressure():
            logger.debug("Reranking paused due to VRAM backpressure")
            vram_manager.wait_for_available_vram(0.75, timeout=RERANK_WAIT_TIMEOUT_SEC)

        # ── Serialized batch planning ────────────────────────────────
        # Acquire the planning lock so that only one thread at a time
        # reads VRAM, calculates batch size, and submits an allocation
        # request.  This prevents N threads from all planning against
        # the same stale GB reading simultaneously.  The lock is
        # held only during planning (~0.1 ms), NOT during the forward
        # pass, so compute still runs concurrently.
        _wait_for_vram_gb: float = 0.0
        with _RERANKER_PLAN_LOCK:
            # Re-poll available memory with fresh hardware reading
            if device == "cuda" and vram_manager is not None:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                available_mem = vram_manager.get_available_for_allocation()
            else:
                available_mem = _get_available_ram()

            # Peek at the next item to establish a baseline length
            current_item = sorted_items[start_idx]
            current_len = current_item["length"]

            # Calculate initial batch size based on shortest remaining item
            batch_size = _calculate_dynamic_batch_size(
                available_mem, current_len, device
            )

            # Look ahead to find the actual max length in this tentative batch
            end_idx = min(start_idx + batch_size, total_candidates)
            last_item_in_batch = sorted_items[end_idx - 1]
            max_len_in_batch = last_item_in_batch["length"]

            # Recalculate batch size based on the ACTUAL longest item
            real_batch_size = _calculate_dynamic_batch_size(
                available_mem, max_len_in_batch, device
            )

            # If the real batch size is smaller than our lookahead, shrink
            if real_batch_size < (end_idx - start_idx):
                end_idx = min(start_idx + real_batch_size, total_candidates)
                max_len_in_batch = sorted_items[end_idx - 1]["length"]

            actual_batch_count = end_idx - start_idx
            remaining_items = total_candidates - start_idx

            # ── Minimum batch enforcement ────────────────────────────
            # If batch is too small AND there are enough items remaining,
            # wait for VRAM instead of thrashing with items=1 batches.
            if (
                actual_batch_count < MIN_RERANK_BATCH_SIZE
                and remaining_items >= MIN_RERANK_BATCH_SIZE
                and vram_manager is not None
            ):
                min_batch_end = min(start_idx + MIN_RERANK_BATCH_SIZE, total_candidates)
                min_batch_max_len: int = int(sorted_items[min_batch_end - 1]["length"])
                _wait_for_vram_gb = (
                    float(
                        MIN_RERANK_BATCH_SIZE
                        * (175 + (min_batch_max_len / 1000.0) * 80)
                    )
                    / 1024.0
                )
                # logger.debug(
                #     f"Batch too small ({actual_batch_count} < {MIN_RERANK_BATCH_SIZE}), "
                #     f"waiting for {_wait_for_vram_gb:.2f}GB VRAM"
                # )
            # Lock released here by 'with' block exit

        # After releasing the lock, wait for VRAM if needed and retry
        if _wait_for_vram_gb > 0.0 and vram_manager is not None:
            became_available = vram_manager.wait_for_available_vram(
                _wait_for_vram_gb, timeout=RERANK_WAIT_TIMEOUT_SEC
            )
            if became_available:
                continue  # Re-enter loop: re-acquire lock, re-poll VRAM
            logger.debug(
                f"Timed out waiting for {_wait_for_vram_gb:.2f}GB VRAM; "
                f"continuing with smaller rerank batch."
            )

        batch_items = sorted_items[start_idx:end_idx]

        # Estimate VRAM needed for this batch and request allocation
        # Use the same verified model as _calculate_dynamic_batch_size:
        # base_overhead=30MB/sample + 120MB per 1k tokens per sample
        estimated_mb = len(batch_items) * (30 + (max_len_in_batch / 1000.0) * 120)
        estimated_vram_gb = estimated_mb / 1024.0

        logger.debug(
            f"Batch plan: items={len(batch_items)}, max_tokens={max_len_in_batch}, "
            f"est_vram={estimated_vram_gb:.2f}GB, available={available_mem:.2f}GB"
        )
        allocation_id = None

        if vram_manager is not None and device == "cuda":
            # Request VRAM allocation through the VRAM manager (blocking)
            granted, allocation_id = vram_manager.request_allocation(
                size_gb=max(estimated_vram_gb, 0.75),  # Minimum 0.75GB per batch
                blocking=True,
                timeout=RERANK_ALLOC_TIMEOUT_SEC,
            )
            if not granted:
                logger.warning(
                    f"VRAM allocation denied/timed out for batch after "
                    f"{RERANK_ALLOC_TIMEOUT_SEC:.0f}s, using fallback scores"
                )
                for item in batch_items:
                    all_scores.append((item["index"], 0.0))
                start_idx = end_idx
                continue

        try:
            # Clear cache before allocation to reduce fragmentation
            if device == "cuda":
                torch.cuda.empty_cache()

            with torch.no_grad():
                # Prepare batch tensors
                # We manually pad using tokenizer.pad which handles the list of dicts
                batch_inputs_list = [
                    {
                        "input_ids": item["input_ids"],
                        "attention_mask": [1] * len(item["input_ids"]),
                    }
                    for item in batch_items
                ]

                # Pad to the longest in THIS batch
                padded_batch = tokenizer.pad(
                    batch_inputs_list, padding="longest", return_tensors="pt"
                )

                # Move to device
                for k in padded_batch:
                    padded_batch[k] = padded_batch[k].to(device)

                # Get logits
                logits = model(**padded_batch).logits[:, -1, :]

                # Extract yes/no token scores
                true_vector = logits[:, token_true_id]
                false_vector = logits[:, token_false_id]
                batch_scores_tensor = torch.stack([false_vector, true_vector], dim=1)

                # Compute probabilities
                batch_scores_tensor = torch.nn.functional.log_softmax(
                    batch_scores_tensor, dim=1
                )
                scores = batch_scores_tensor[:, 1].exp().tolist()

                # Collect scores with original indices
                for item, score in zip(batch_items, scores):
                    all_scores.append((item["index"], score))

                # Explicitly delete tensors to free memory immediately
                del padded_batch
                del logits
                del true_vector
                del false_vector
                del batch_scores_tensor
                if device == "cuda":
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Reranking batch failed: {e}")
            # Fallback: assign zero scores
            for item in batch_items:
                all_scores.append((item["index"], 0.0))
        finally:
            # Always release the VRAM allocation
            if allocation_id is not None and vram_manager is not None:
                vram_manager.release_allocation(allocation_id)

        # Move to next batch
        start_idx = end_idx

    # 4. Sort and Return Top-K
    # all_scores contains (original_index, score)
    all_scores.sort(key=lambda x: x[1], reverse=True)
    return all_scores[:top_k]


def _get_text_content_for_embedding(
    file_path: str, symbol_map: Dict[str, Any], project_root: str
) -> str:
    """Helper to retrieve text content for embedding from symbol map or file."""
    ext = os.path.splitext(file_path)[1].lower()
    is_doc = ext in [".md", ".txt", ".rst"]

    if file_path in symbol_map:
        # For docs in symbol map, we still want the essence string if it's been updated to handle them
        return generate_symbol_essence_string(
            file_path, symbol_map[file_path], symbol_map=symbol_map
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if is_doc:
            return preprocess_doc_structure(content)

        rel_path = os.path.relpath(file_path, project_root)
        return f"[FILE: {rel_path}]\n{content[:32000]}"
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return ""


# --- Main Embedding Generation ---


def generate_embeddings(
    project_paths: List[str],
    path_to_key_info: Dict[str, KeyInfo],
    force: bool = False,
    batch_size: Optional[int] = None,
    symbol_map: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Generates embeddings for project files.
    Uses SES (Symbol Essence Strings) derived from symbol_map where available.
    """
    if not project_paths or not path_to_key_info:
        logger.error("No project paths or key info provided.")
        return False

    config_manager = ConfigManager()
    project_root = get_project_root()
    embeddings_dir = config_manager.get_path(
        "embeddings_dir", "cline_utils/dependency_system/analysis/embeddings"
    )
    if not os.path.isabs(embeddings_dir):
        embeddings_dir = os.path.join(project_root, embeddings_dir)
    os.makedirs(embeddings_dir, exist_ok=True)

    # 1. Load Symbol Map (if not provided)
    if symbol_map is None:
        symbol_map = _load_project_symbol_map()

    # Load existing metadata to preserve tokens for skipped files
    metadata_path = os.path.join(embeddings_dir, "metadata.json")
    existing_metadata = {}
    existing_metadata_by_path: Dict[str, Dict[str, Any]] = {}
    metadata_version = ""
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                metadata_version = str(data.get("version", ""))
                existing_metadata = data.get("keys", {})
                for m_value in cast(Dict[str, Any], existing_metadata).values():
                    if not isinstance(m_value, dict):
                        continue
                    path_value = m_value.get("path")
                    if not isinstance(path_value, str) or not path_value:
                        continue
                    try:
                        existing_metadata_by_path[normalize_path(path_value)] = m_value
                    except Exception:
                        continue
                if metadata_version != EMBEDDING_METADATA_VERSION:
                    logger.info(
                        f"Embedding metadata version mismatch "
                        f"(found='{metadata_version}', expected='{EMBEDDING_METADATA_VERSION}'). "
                        "Forcing embedding regeneration to refresh SES token counts."
                    )
                    existing_metadata = {}
                    existing_metadata_by_path = {}
                    force = True
        except Exception as e:
            logger.warning(f"Failed to load existing metadata: {e}")

    # Track token counts for all files
    final_token_counts: Dict[str, Dict[str, int]] = {}

    # 2. Identification Phase
    files_to_process: List[KeyInfo] = []

    for key_info in path_to_key_info.values():
        if key_info.is_directory:
            continue

        if not _is_valid_file(key_info.norm_path):
            continue

        # Calculate where the embedding should be
        rel_path = os.path.relpath(key_info.norm_path, project_root)
        embedding_path = os.path.join(embeddings_dir, rel_path) + ".npy"

        should_process = False
        if force:
            should_process = True
        elif not os.path.exists(embedding_path):
            should_process = True
        else:
            try:
                src_mtime = os.path.getmtime(key_info.norm_path)
                emb_mtime = os.path.getmtime(embedding_path)
                if src_mtime > emb_mtime:
                    should_process = True
            except OSError:
                should_process = True

        if should_process:
            files_to_process.append(key_info)

    if not files_to_process and all(
        (
            (
                key_info.key_string in existing_metadata
                and (
                    "tokens" in existing_metadata[key_info.key_string]
                    or (
                        "ses_tokens" in existing_metadata[key_info.key_string]
                        and "full_tokens" in existing_metadata[key_info.key_string]
                    )
                )
            )
            or (
                key_info.norm_path in existing_metadata_by_path
                and (
                    "tokens" in existing_metadata_by_path[key_info.norm_path]
                    or (
                        "ses_tokens" in existing_metadata_by_path[key_info.norm_path]
                        and "full_tokens"
                        in existing_metadata_by_path[key_info.norm_path]
                    )
                )
            )
        )
        for key_info in path_to_key_info.values()
        if not key_info.is_directory
    ):
        logger.info("All embeddings and metadata are up to date.")
        return True

    logger.info(
        f"Generating embeddings for {len(files_to_process)} files using Symbol Essence..."
    )

    # 3. Processing Phase
    # Pre-calculate token counts and sort to optimize model loading
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        logger.warning("Tokenizer not found. Using character-based token estimation.")

    processing_queue: List[Dict[str, Any]] = []

    with PhaseTracker(
        total=len(files_to_process), phase_name="Preparing Embeddings"
    ) as prep_tracker:
        for key_info in files_to_process:
            file_path = key_info.norm_path
            rel_path = os.path.relpath(file_path, project_root)
            prep_tracker.set_description(f"Reading {os.path.basename(rel_path)}")

            text_to_embed = _get_text_content_for_embedding(
                file_path, symbol_map, project_root
            )

            if not text_to_embed.strip():
                prep_tracker.update()
                continue

            # Count tokens
            ses_token_count = _count_tokens(text_to_embed, tokenizer)

            # Count Full Context tokens (for raw file content)
            full_token_count = 0

            with open(file_path, "r", encoding="utf-8") as f:
                full_content = f.read()
                full_token_count = _count_tokens(full_content, tokenizer)

            final_token_counts[key_info.key_string] = {
                "ses_tokens": ses_token_count,
                "full_tokens": full_token_count,
            }

            processing_queue.append(
                {
                    "key_info": key_info,
                    "text": text_to_embed,
                    "tokens": ses_token_count,
                    "rel_path": rel_path,
                }
            )
            prep_tracker.update()

    # Sort by token count (ascending) to grow context window monotonically
    # Sort by tokens to optimize model window
    processing_queue.sort(key=lambda x: cast(int, x["tokens"]))
    current_batch_texts: List[str] = []
    current_batch_paths: List[str] = []

    # Determine batch size
    effective_batch_size = batch_size or (
        64 if _selected_device_cache == "cuda" else 16
    )

    with PhaseTracker(
        total=len(processing_queue), phase_name="Generating Embeddings"
    ) as tracker:
        for item in processing_queue:
            tracker.set_description(f"Embedding {os.path.basename(item['rel_path'])}")

            # Calculate required n_ctx for this item
            # Formula: max(actual_tokens + 512, 8192)
            # Cap at MAX_CONTEXT_LENGTH
            required_n_ctx = min(max(item["tokens"] + 512, 8192), MAX_CONTEXT_LENGTH)

            # Ensure model is loaded with sufficient context
            try:
                _load_model(n_ctx=required_n_ctx)
            except Exception as e:
                logger.error(f"Could not load model for embedding generation: {e}")
                return False

            current_batch_texts.append(item["text"])

            # Calculate save path
            save_path = os.path.join(embeddings_dir, item["rel_path"]) + ".npy"
            current_batch_paths.append(save_path)

            if len(current_batch_texts) >= effective_batch_size:
                _flush_batch(current_batch_texts, current_batch_paths)
                tracker.update(len(current_batch_texts))
                current_batch_texts = []
                current_batch_paths = []

        # Flush remaining
        if current_batch_texts:
            _flush_batch(current_batch_texts, current_batch_paths)
            tracker.update(len(current_batch_texts))

    # 4. Create/Update Metadata
    metadata_path = os.path.join(embeddings_dir, "metadata.json")
    new_metadata: Dict[str, Any] = {
        "version": EMBEDDING_METADATA_VERSION,
        "model": (
            _selected_model_config["name"] if _selected_model_config else "unknown"
        ),
        "keys": {},
    }

    # If we have existing metadata, carry it over by stable file path.
    # This prevents token drift when key instance suffixes (#n) are reassigned.
    if existing_metadata:
        for v in cast(Dict[str, Any], existing_metadata).values():
            if not isinstance(v, dict):
                continue
            m_path = v.get("path")
            if not isinstance(m_path, str) or not m_path:
                continue

            norm_m_path = normalize_path(m_path)
            current_ki = path_to_key_info.get(norm_m_path)
            if not current_ki or current_ki.is_directory:
                continue

            migrated_item: Dict[str, Any] = dict(v)
            # Migrate old tokens if needed
            if "tokens" in migrated_item and (
                "full_tokens" not in migrated_item or "ses_tokens" not in migrated_item
            ):
                migrated_item["ses_tokens"] = migrated_item["tokens"]
                # Safe fallback for legacy metadata without full token counts
                migrated_item["full_tokens"] = migrated_item["tokens"]
                del migrated_item["tokens"]

            migrated_item["path"] = current_ki.norm_path
            new_metadata["keys"][current_ki.key_string] = migrated_item

    # Overwrite with new items or add new ones
    for key_info in path_to_key_info.values():
        if key_info.is_directory:
            continue

        rel_path = os.path.relpath(key_info.norm_path, project_root)
        npy_path = os.path.join(embeddings_dir, rel_path) + ".npy"

        if os.path.exists(npy_path):
            try:
                # Get token count: new > existing > calculate
                token_data = final_token_counts.get(key_info.key_string)

                ses_tokens = 0
                full_tokens = 0

                if token_data is not None:
                    ses_tokens = token_data["ses_tokens"]
                    full_tokens = token_data["full_tokens"]
                elif (
                    existing_metadata_by_path
                    and key_info.norm_path in existing_metadata_by_path
                ):
                    m_item = existing_metadata_by_path[key_info.norm_path]
                    if "ses_tokens" in m_item:
                        ses_tokens = cast(int, m_item["ses_tokens"])
                        full_tokens = cast(int, m_item.get("full_tokens", ses_tokens))
                    elif "tokens" in m_item:
                        ses_tokens = cast(int, m_item["tokens"])
                        full_tokens = ses_tokens
                else:
                    # Fallback: calculate now
                    try:
                        text = _get_text_content_for_embedding(
                            key_info.norm_path, symbol_map, project_root
                        )
                        ses_tokens = _count_tokens(text, tokenizer)
                        # Try to get full too
                        with open(key_info.norm_path, "r", encoding="utf-8") as f:
                            full_tokens = _count_tokens(f.read(), tokenizer)
                    except Exception:
                        full_tokens = ses_tokens

                new_metadata["keys"][key_info.key_string] = {
                    "path": key_info.norm_path,
                    "mtime": os.path.getmtime(key_info.norm_path),
                    "ses_tokens": ses_tokens,
                    "full_tokens": full_tokens,
                }
            except OSError:
                pass

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(new_metadata, f, indent=2)
        logger.info(f"Updated metadata at {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")

    try:
        # Invalidate similarity cache as embeddings have changed
        cache_manager.get_cache("similarity_calculation").invalidate(".*")
        logger.debug(
            "Invalidated similarity_calculation cache after embedding generation."
        )
    except Exception as e:
        logger.warning(f"Failed to invalidate similarity cache: {e}")

    _unload_model()
    return True


def _flush_batch(texts: List[str], save_paths: List[str]):
    """Helper to encode a batch of texts and save them to their respective paths."""
    if not texts:
        return

    try:
        if _model_instance is None:
            logger.error("Model instance lost during batch flush")
            return

        if _selected_model_config and _selected_model_config["type"] == "gguf":
            # GGUF (llama-cpp) handles one by one in loop usually unless batched explicitly
            embeddings: List[np.ndarray] = []
            for t in texts:
                res = _model_instance.embed(t)
                embeddings.append(np.array(res, dtype=np.float32))
        else:
            # SentenceTransformer handles batches natively
            embeddings = _model_instance.encode(
                texts, show_progress_bar=False, convert_to_numpy=True
            )

        for i, emb in enumerate(embeddings):
            # Normalize
            emb = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            path = save_paths[i]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, emb)

    except Exception as e:
        logger.error(f"Failed to flush batch: {e}")


# --- Similarity Calculation ---


def _get_similarity_cache_key(key1: str, key2: str, *args: Any, **kwargs: Any) -> str:
    """Generates a deterministic cache key for similarity."""
    # deterministic order
    k1, k2 = sorted((key1, key2))
    return f"sim_ses:{k1}:{k2}"


def _get_similarity_file_deps(
    key1_str: str,
    key2_str: str,
    embeddings_dir: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    *args: Any,
    **kwargs: Any,
) -> List[str]:
    """
    Returns the .npy file paths for the two keys.
    Used by @cached decorator to track file dependencies and mtimes.
    """
    file_paths: List[str] = []

    # Ensure embeddings_dir is an absolute path
    if embeddings_dir and not os.path.isabs(embeddings_dir):
        embeddings_dir = os.path.join(project_root, embeddings_dir)

    for key_str in [key1_str, key2_str]:
        npy_path: Optional[str] = None

        # Try to resolve via path_to_key_info first
        ki = next(
            (k for k in path_to_key_info.values() if k.key_string == key_str), None
        )
        if ki:
            rel = os.path.relpath(ki.norm_path, project_root)
            npy_path = os.path.join(embeddings_dir, rel) + ".npy"
        else:
            # Fallback: Try direct key-to-filename mapping (for testing/simple cases)
            # e.g., key "1A1" -> "embeddings_dir/1A1.npy"
            direct_npy = os.path.join(embeddings_dir, f"{key_str}.npy")
            if os.path.exists(direct_npy):
                npy_path = direct_npy

        if npy_path and os.path.exists(npy_path):
            file_paths.append(npy_path)

    return file_paths


@cached(
    "similarity_calculation",
    key_func=_get_similarity_cache_key,
    ttl=SIM_CACHE_TTL_SEC,
    file_deps=_get_similarity_file_deps,
    check_mtime=True,
)
def calculate_similarity(
    key1_str: str,
    key2_str: str,
    embeddings_dir: str,
    path_to_key_info: Dict[str, KeyInfo],
    project_root: str,
    code_roots: List[str],
    doc_roots: List[str],
) -> float:
    """
    Calculates cosine similarity between two keys.
    Requires the embeddings to be generated and saved on disk.
    """
    # 1. Validate Keys
    ki1 = next((k for k in path_to_key_info.values() if k.key_string == key1_str), None)
    ki2 = next((k for k in path_to_key_info.values() if k.key_string == key2_str), None)

    if not ki1 or not ki2:
        return 0.0

    # 2. Locate NPY files
    def get_npy_path(ki: KeyInfo) -> Optional[str]:
        rel = os.path.relpath(ki.norm_path, project_root)
        path = os.path.join(embeddings_dir, rel) + ".npy"
        return path if os.path.exists(path) else None

    p1 = get_npy_path(ki1)
    p2 = get_npy_path(ki2)

    if not p1 or not p2:
        return 0.0

    # 3. Load and Compute
    try:
        v1 = np.load(p1)
        v2 = np.load(p2)

        # Ensure flat (1D) arrays
        v1 = v1.flatten()
        v2 = v2.flatten()

        # Dot product (vectors are already normalized in generation)
        score = np.dot(v1, v2)
        return float(max(0.0, min(1.0, score)))
    except Exception as e:
        logger.warning(f"Similarity calc error ({key1_str}, {key2_str}): {e}")
        return 0.0


# --- File Validation Helper ---


@cached(
    "file_validation",
    key_func=lambda file_path: f"is_valid:{normalize_path(str(file_path))}:{os.path.getmtime(ConfigManager().config_path) if os.path.exists(ConfigManager().config_path) else 0}",
    track_path_args=[0],
)
def _is_valid_file(file_path: str) -> bool:
    """Check if a file is valid for processing (not excluded, size limit)."""
    try:
        config = ConfigManager()
        project_root = get_project_root()
        norm_path = normalize_path(file_path)

        # Excluded paths/dirs
        excluded_paths = set(config.get_excluded_paths())
        if norm_path in excluded_paths:
            return False

        excluded_dirs = [
            normalize_path(os.path.join(project_root, d))
            for d in config.get_excluded_dirs()
        ]
        if any(norm_path.startswith(d + os.sep) for d in excluded_dirs):
            return False

        # Extensions
        ext = os.path.splitext(norm_path)[1].lower()
        if ext in config.get_excluded_extensions():
            return False

        # Size check (10MB limit)
        if os.path.getsize(norm_path) > 10 * 1024 * 1024:
            return False

        return True
    except Exception:
        return False


# --- CLI Placeholders ---
# def register_parser(subparsers):
#    parser = subparsers.add_parser("generate-embeddings", help="Generate embeddings")
#    parser.add_argument("project_paths", nargs="+")
#    parser.add_argument("--force", action="store_true")
#    parser.set_defaults(func=command_handler)

# def command_handler(args):
#     logger.error("Direct CLI usage deprecated. Use project_analyzer.")
#     return 1

# EoF
