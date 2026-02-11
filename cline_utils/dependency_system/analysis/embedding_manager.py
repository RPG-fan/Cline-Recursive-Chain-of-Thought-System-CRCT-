# analysis/embedding_manager.py

"""
Module for managing embeddings generation and similarity calculations using contextual keys.
Handles embedding creation from project files using Symbol Essence Strings (SES) derived from
the project symbol map, and calculates cosine similarity between embeddings.
"""
import json
import logging
import os
import sys
import threading
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

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
MODEL_INSTANCE: Optional[Any] = None  # Can be Llama or SentenceTransformer
SELECTED_DEVICE: Optional[str] = None
SELECTED_MODEL_CONFIG: Optional[Dict[str, Any]] = None
TOKENIZER_INSTANCE: Optional[Any] = None

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


def _get_available_vram() -> float:
    """Get available VRAM in GB for CUDA devices."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        torch.cuda.synchronize()
        # Use mem_get_info() which returns (free_memory, total_memory) directly
        # This is more accurate than manual calculation
        free_memory, total_memory = torch.cuda.mem_get_info(0)
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
    global SELECTED_DEVICE
    if SELECTED_DEVICE is None:
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
                SELECTED_DEVICE = _get_best_device()
            elif config_device == "mps" and not (
                sys.platform == "darwin"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                logger.warning(
                    "Config specified 'mps', but not available. Auto-detecting."
                )
                SELECTED_DEVICE = _get_best_device()
            else:
                logger.debug(f"Using device specified in config: {config_device}")
                SELECTED_DEVICE = config_device
        elif config_device == "auto":
            logger.debug("Auto-detecting device.")
            SELECTED_DEVICE = _get_best_device()
        else:
            logger.warning(f"Invalid device '{config_device}'. Auto-detecting.")
            SELECTED_DEVICE = _get_best_device()
    return SELECTED_DEVICE or "cpu"


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
    global SELECTED_MODEL_CONFIG
    if SELECTED_MODEL_CONFIG is not None:
        return SELECTED_MODEL_CONFIG

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

        SELECTED_MODEL_CONFIG = model_config
        return model_config
    elif model_selection == "mpnet":
        SELECTED_MODEL_CONFIG = MODEL_CONFIGS["mpnet"].copy()
        return SELECTED_MODEL_CONFIG

    # Auto-detect
    device = _select_device()
    available_mem = _get_available_vram() if device == "cuda" else _get_available_ram()

    # Prefer Qwen3 if VRAM allows, else mpnet
    qwen_config = MODEL_CONFIGS["qwen3-4b"].copy()
    qwen_config["path"] = config_manager.get_embedding_setting("qwen3_model_path")

    mem_req = (
        qwen_config["min_vram_gb"] if device == "cuda" else qwen_config["min_ram_gb"]
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
            SELECTED_MODEL_CONFIG = qwen_config
            return qwen_config
        elif _download_qwen3_model(qwen_config["path"]):
            SELECTED_MODEL_CONFIG = qwen_config
            return qwen_config

    SELECTED_MODEL_CONFIG = MODEL_CONFIGS["mpnet"].copy()
    return SELECTED_MODEL_CONFIG


def _get_tokenizer():
    """Lazily loads a tokenizer for token counting."""
    global TOKENIZER_INSTANCE, RERANKER_TOKENIZER

    if TOKENIZER_INSTANCE is not None:
        return TOKENIZER_INSTANCE

    # 1. Try reusing Reranker tokenizer if loaded
    if RERANKER_TOKENIZER is not None:
        TOKENIZER_INSTANCE = RERANKER_TOKENIZER
        return TOKENIZER_INSTANCE

    # 2. Try loading from local reranker path
    try:
        project_root = get_project_root()
        local_model_path = os.path.join(project_root, "models", "qwen3_reranker")
        if os.path.exists(local_model_path) and os.path.exists(
            os.path.join(local_model_path, "tokenizer.json")
        ):
            from transformers import AutoTokenizer

            TOKENIZER_INSTANCE = AutoTokenizer.from_pretrained(
                local_model_path, trust_remote_code=True
            )
            return TOKENIZER_INSTANCE
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
    global MODEL_INSTANCE, SELECTED_MODEL_CONFIG

    if MODEL_INSTANCE is not None:
        # Check if we need to reload due to context size
        if SELECTED_MODEL_CONFIG and SELECTED_MODEL_CONFIG["type"] == "gguf":
            try:
                current_n_ctx = MODEL_INSTANCE.n_ctx()
                if current_n_ctx < n_ctx:
                    logger.debug(
                        f"Reloading model to increase context from {current_n_ctx} to {n_ctx}"
                    )
                    _unload_model()
                else:
                    # Existing context is sufficient
                    return MODEL_INSTANCE
            except AttributeError:
                # Fallback if n_ctx() not available
                pass
        elif (
            SELECTED_MODEL_CONFIG
            and SELECTED_MODEL_CONFIG["type"] == "sentence-transformer"
        ):
            # Update max_seq_length for SentenceTransformer without reload
            if hasattr(MODEL_INSTANCE, "max_seq_length"):
                if MODEL_INSTANCE.max_seq_length < n_ctx:
                    MODEL_INSTANCE.max_seq_length = n_ctx
            return MODEL_INSTANCE

    if MODEL_INSTANCE is None:
        SELECTED_MODEL_CONFIG = _select_best_model()
        device = _select_device()

        try:
            if SELECTED_MODEL_CONFIG["type"] == "gguf":
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

                def noop_log_callback(level, text, user_data):
                    pass

                # Keep a reference to prevent GC (Critical!)
                global _C_LOG_CALLBACK_REF
                _C_LOG_CALLBACK_REF = LogCallback(noop_log_callback)

                llama_cpp.llama_log_set(_C_LOG_CALLBACK_REF, ctypes.c_void_p())

                MODEL_INSTANCE = Llama(
                    model_path=SELECTED_MODEL_CONFIG["path"],
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
                    f"Loaded GGUF model: {SELECTED_MODEL_CONFIG['name']} on device: {device}"
                )

            elif SELECTED_MODEL_CONFIG["type"] == "sentence-transformer":
                # Load sentence-transformers model with proper device handling
                from sentence_transformers import SentenceTransformer

                try:
                    MODEL_INSTANCE = SentenceTransformer(
                        SELECTED_MODEL_CONFIG["name"], device=device
                    )
                except Exception:
                    # Fallback to CPU if device init fails
                    MODEL_INSTANCE = SentenceTransformer(
                        SELECTED_MODEL_CONFIG["name"], device="cpu"
                    )
                logger.debug(
                    f"Loaded sentence transformer: {SELECTED_MODEL_CONFIG['name']}"
                )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    return MODEL_INSTANCE


def _unload_model():
    """Unloads the embedding model."""
    global MODEL_INSTANCE
    if MODEL_INSTANCE is not None:
        MODEL_INSTANCE = None
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


def generate_symbol_essence_string(
    file_path: str,
    symbol_data: Dict[str, Any],
    max_chars: int = 4000,
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

    parts = []

    # 1. Header
    mod_time = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
    parts.append(
        f"[FILE: {rel_path} | TYPE: {symbol_data.get('file_type', 'unknown')} | MOD: {mod_time}]"
    )

    # 2. Classes (with Runtime Enhancements)
    classes = symbol_data.get("classes", [])
    if classes:
        for c in classes:
            c_name = c["name"]
            c_doc = (c.get("docstring") or "").strip()
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
                for m in methods:
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

    # 3. Top-level Functions (with Runtime Enhancements)
    functions = symbol_data.get("functions", [])
    if functions:
        parts.append("FUNCTIONS:")
        for f in functions:
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

    # 4. Outgoing Calls (from AST - still useful for call graphs)
    calls = symbol_data.get("calls", [])
    if calls:
        unique_calls = set()
        for c in calls:
            src = c.get("potential_source")
            if src:
                unique_calls.add(src)

        if unique_calls:
            sorted_calls = sorted(list(unique_calls))
            parts.append(f"CALLS: {', '.join(sorted_calls)}")

    # 5. Incoming Connections (CALLED_BY) - from imports analysis
    if symbol_map:
        called_by = set()
        fname = os.path.basename(file_path)
        fname_no_ext = os.path.splitext(fname)[0]

        for other_path, other_data in symbol_map.items():
            if other_path == file_path:
                continue

            other_imports = other_data.get("imports", [])
            for imp in other_imports:
                imp_path = imp.get("path") if isinstance(imp, dict) else imp
                if not imp_path:
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

    # Join and truncate if needed
    result = "\n".join(parts)
    if len(result) > max_chars:
        result = result[:max_chars] + "..."

    return result


def preprocess_doc_structure(content: str) -> str:
    """
    Preprocesses documentation for embedding/reranking.
    Returns the full content (truncated to 32k chars) to preserve context.
    """
    # Return full content with a generous safety cap
    return content[:32000]


# --- Reranker Logic ---

RERANKER_MODEL: Optional[Any] = None
RERANKER_TOKENIZER: Optional[Any] = None

# Pre-computed tokenizer values to avoid concurrency issues
RERANKER_FALSE_ID: Optional[int] = None
RERANKER_TRUE_ID: Optional[int] = None
RERANKER_PREFIX_TOKENS: Optional[List[int]] = None
RERANKER_SUFFIX_TOKENS: Optional[List[int]] = None

# Reranking tracking variables
RERANKED_FILES: set = set()
RERANKING_COUNTER: int = 0
TOTAL_FILES_TO_RERANK: int = 0

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
    global RERANKER_MODEL, RERANKER_TOKENIZER
    global RERANKER_FALSE_ID, RERANKER_TRUE_ID, RERANKER_PREFIX_TOKENS, RERANKER_SUFFIX_TOKENS

    with MODEL_LOCK:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if RERANKER_MODEL is not None:
            return RERANKER_TOKENIZER, RERANKER_MODEL

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

            RERANKER_TOKENIZER = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                padding_side="left",  # Left padding for generation/classification to align last token
            )

            # Pre-compute special tokens and IDs under lock to avoid concurrency issues
            RERANKER_FALSE_ID = RERANKER_TOKENIZER.convert_tokens_to_ids("no")
            RERANKER_TRUE_ID = RERANKER_TOKENIZER.convert_tokens_to_ids("yes")

            if RERANKER_FALSE_ID is None or RERANKER_TRUE_ID is None:
                logger.error(
                    f"Could not find 'yes' (id={RERANKER_TRUE_ID}) or 'no' (id={RERANKER_FALSE_ID}) tokens in tokenizer. Model may be corrupted."
                )
                raise RuntimeError("Invalid tokenizer state for reranker")

            prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

            RERANKER_PREFIX_TOKENS = RERANKER_TOKENIZER.encode(
                prefix, add_special_tokens=False
            )
            RERANKER_SUFFIX_TOKENS = RERANKER_TOKENIZER.encode(
                suffix, add_special_tokens=False
            )

            # Optimizations: Flash Attention 2 and float16 for CUDA
            try:
                if device == "cuda":
                    RERANKER_MODEL = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,  # FIXED: was model_name
                        dtype=torch.float16,  # Use 'dtype' instead of 'torch_dtype'
                        attn_implementation="flash_attention_2",
                        trust_remote_code=True,
                    )
                else:
                    RERANKER_MODEL = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path, trust_remote_code=True
                    )
            except Exception as e:
                logger.warning(
                    f"Optimization failed, falling back to standard load: {e}"
                )
                RERANKER_MODEL = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, trust_remote_code=True
                )

                # Only move non-quantized models manually
                if not getattr(RERANKER_MODEL, "is_quantized", False):
                    RERANKER_MODEL.to(device)

            RERANKER_MODEL.eval()

            # Create a dummy tensor to prime the CUDA context with the expected dtype and device
            if device == "cuda":
                dummy_tensor = torch.zeros(1, dtype=torch.float16, device=device)
                del dummy_tensor
                torch.cuda.empty_cache()
                logger.debug("CUDA context primed with dummy tensor.")

            # Verify Flash Attention
            if hasattr(RERANKER_MODEL.config, "_attn_implementation"):
                attn_impl = RERANKER_MODEL.config._attn_implementation
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
                    f"Loaded Qwen3-Reranker (Q8 quantized) on {RERANKER_MODEL.device}. "
                    f"Model footprint: {model_memory_gb:.2f}GB"
                )
            else:
                logger.info(f"Loaded Qwen3-Reranker-0.6B on {RERANKER_MODEL.device}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}", exc_info=True)
            # Clean up partial load
            RERANKER_MODEL = None
            RERANKER_TOKENIZER = None
            RERANKER_FALSE_ID = None
            RERANKER_TRUE_ID = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None

    return RERANKER_TOKENIZER, RERANKER_MODEL


def unload_reranker_model():
    """Unloads reranker model to free memory."""
    global RERANKER_MODEL, RERANKER_TOKENIZER
    RERANKER_MODEL = None
    RERANKER_TOKENIZER = None
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
    # These values are tuned from actual VRAM observations
    base_overhead_mb = 175  # Base overhead per sample in MB
    mb_per_1k_tokens = 80  # Additional MB per 1000 tokens

    estimated_mb_per_sample = (
        base_overhead_mb + (context_length / 1000.0) * mb_per_1k_tokens
    )
    estimated_gb_per_sample = estimated_mb_per_sample / 1024.0

    # Safety buffer: leave 10% or 1GB, whichever is larger
    reserved_buffer = max(1.0, available_mem_gb * 0.1)
    usable_mem_gb = max(0.0, available_mem_gb - reserved_buffer)

    max_batch = int(usable_mem_gb / estimated_gb_per_sample)

    # Clamp batch size
    max_batch = max(1, min(max_batch, 15))  # Cap at 15

    logger.debug(
        f"Dynamic Batch Sizing: Available={available_mem_gb:.2f}GB, "
        f"Context={context_length}, Est.PerSample={estimated_gb_per_sample:.2f}GB "
        f"-> Batch Size={max_batch}"
    )
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
    token_false_id = RERANKER_FALSE_ID
    token_true_id = RERANKER_TRUE_ID

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
    full_prompts_data = []
    for i, doc in enumerate(candidate_texts):
        prompt = f"{prefix}<Instruct>: {instruction}\n<Query>: {query_text}\n<Document>: {doc}{suffix}"
        full_prompts_data.append({"index": i, "text": prompt})

    try:
        # Tokenize without padding first to get raw lengths
        # We use a large max_length here to avoid premature truncation, but clamp to model max if needed.
        # Qwen3 typically handles 32k, but let's use a reasonable upper bound.
        all_inputs = tokenizer(
            [p["text"] for p in full_prompts_data],
            padding=False,
            truncation=True,
            max_length=32000,  # Hard cap to prevent insane memory usage
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

    all_scores = []
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
            # Wait for VRAM to become available (blocking, no timeout)
            vram_manager.wait_for_available_vram(0.75)

        # CRITICAL: Re-poll available memory before each batch calculation
        # This ensures we get accurate, real-time values after previous batches free memory
        if device == "cuda":
            available_mem = vram_manager.get_available_for_allocation()
        else:
            available_mem = _get_available_ram()

        # Peek at the next item to establish a baseline length
        # (In a sorted list, this is the shortest in the remaining set)
        current_item = sorted_items[start_idx]
        current_len = current_item["length"]

        # Calculate initial batch size based on this length
        # This gives us a "maximum possible batch size" if all subsequent items were this short.
        batch_size = _calculate_dynamic_batch_size(available_mem, current_len, device)

        # Look ahead to find the actual max length in this tentative batch
        end_idx = min(start_idx + batch_size, total_candidates)
        last_item_in_batch = sorted_items[end_idx - 1]
        max_len_in_batch = last_item_in_batch["length"]

        # Recalculate batch size based on the ACTUAL longest item in the batch
        real_batch_size = _calculate_dynamic_batch_size(
            available_mem, max_len_in_batch, device
        )

        # If the real batch size is smaller than our lookahead, we need to shrink
        if real_batch_size < (end_idx - start_idx):
            end_idx = min(start_idx + real_batch_size, total_candidates)
            # Update max length for the new, smaller batch
            max_len_in_batch = sorted_items[end_idx - 1]["length"]

        batch_items = sorted_items[start_idx:end_idx]

        # Estimate VRAM needed for this batch and request allocation
        estimated_vram_gb = (max_len_in_batch / 1000.0) * 0.08 + 0.175 * len(
            batch_items
        ) / 1024.0
        allocation_id = None

        if vram_manager is not None and device == "cuda":
            # Request VRAM allocation through the VRAM manager (blocking, no timeout)
            granted, allocation_id = vram_manager.request_allocation(
                size_gb=max(estimated_vram_gb, 0.75),  # Minimum 0.75GB per batch
                blocking=True,
                timeout=None,  # Wait indefinitely for resources
            )
            if not granted:
                logger.warning(
                    f"VRAM allocation denied for batch, using fallback scores"
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
                for key in padded_batch:
                    padded_batch[key] = padded_batch[key].to(device)

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
    if file_path in symbol_map:
        return generate_symbol_essence_string(
            file_path, symbol_map[file_path], symbol_map=symbol_map
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".md", ".txt", ".rst"]:
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
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_metadata = data.get("keys", {})
        except Exception as e:
            logger.warning(f"Failed to load existing metadata: {e}")

    # Track token counts for all files
    final_token_counts: Dict[str, int] = {}

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
        k in existing_metadata and "tokens" in existing_metadata[k]
        for k in path_to_key_info.keys()
        if not path_to_key_info[k].is_directory
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

    processing_queue = []

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
            token_count = _count_tokens(text_to_embed, tokenizer)
            final_token_counts[key_info.key_string] = token_count

            processing_queue.append(
                {
                    "key_info": key_info,
                    "text": text_to_embed,
                    "tokens": token_count,
                    "rel_path": rel_path,
                }
            )
            prep_tracker.update()

    # Sort by token count (ascending) to grow context window monotonically
    processing_queue.sort(key=lambda x: x["tokens"])

    current_batch_texts = []
    current_batch_paths = []

    # Determine batch size
    effective_batch_size = batch_size or (64 if SELECTED_DEVICE == "cuda" else 16)

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

    # Create/Update Metadata
    metadata_path = os.path.join(embeddings_dir, "metadata.json")
    new_metadata = {
        "version": "2.0_SES",
        "model": SELECTED_MODEL_CONFIG["name"] if SELECTED_MODEL_CONFIG else "unknown",
        "keys": {},
    }

    # Populate metadata with current state of all valid files
    for key_info in path_to_key_info.values():
        if key_info.is_directory:
            continue
        rel_path = os.path.relpath(key_info.norm_path, project_root)
        npy_path = os.path.join(embeddings_dir, rel_path) + ".npy"

        if os.path.exists(npy_path):
            try:
                # Get token count: new > existing > calculate
                token_count = final_token_counts.get(key_info.key_string)
                if token_count is None:
                    if (
                        key_info.key_string in existing_metadata
                        and "tokens" in existing_metadata[key_info.key_string]
                    ):
                        token_count = existing_metadata[key_info.key_string]["tokens"]
                    else:
                        # Fallback: calculate now (avoiding full re-embed)
                        # This happens if file was skipped but metadata is missing tokens
                        text = _get_text_content_for_embedding(
                            key_info.norm_path, symbol_map, project_root
                        )
                        token_count = _count_tokens(text, tokenizer)
                        final_token_counts[key_info.key_string] = token_count

                new_metadata["keys"][key_info.key_string] = {
                    "path": key_info.norm_path,
                    "mtime": os.path.getmtime(key_info.norm_path),
                    "tokens": token_count,
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
        if MODEL_INSTANCE is None:
            logger.error("Model instance lost during batch flush")
            return

        if SELECTED_MODEL_CONFIG and SELECTED_MODEL_CONFIG["type"] == "gguf":
            # GGUF (llama-cpp) handles one by one in loop usually unless batched explicitly
            embeddings = []
            for t in texts:
                res = MODEL_INSTANCE.embed(t)
                embeddings.append(np.array(res, dtype=np.float32))
        else:
            # SentenceTransformer handles batches natively
            embeddings = MODEL_INSTANCE.encode(
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


def _get_similarity_cache_key(key1: str, key2: str, *args, **kwargs) -> str:
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
    *args,
    **kwargs,
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
def register_parser(subparsers):
    parser = subparsers.add_parser("generate-embeddings", help="Generate embeddings")
    parser.add_argument("project_paths", nargs="+")
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(func=command_handler)


def command_handler(args):
    logger.error("Direct CLI usage deprecated. Use project_analyzer.")
    return 1


# EoF
