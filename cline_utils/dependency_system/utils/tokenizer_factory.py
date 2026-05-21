# cline_utils/dependency_system/utils/tokenizer_factory.py

import os
import logging
from typing import Any, Optional
from cline_utils.dependency_system.utils.path_utils import get_project_root

logger = logging.getLogger(__name__)

# Thread-local or global cache to avoid loading the same tokenizer multiple times
_tokenizer_cache = {}


def get_tokenizer(model_path_or_name: Optional[str] = None) -> Optional[Any]:
    """
    Resolves, loads, and caches a tokenizer offline-first.
    
    If transformers is not installed, returns None.
    If the tokenizer folder exists under models/<sanitized_name>_tokenizer, loads locally.
    Otherwise, attempts to download from HuggingFace Hub and saves to the models folder.
    
    Args:
        model_path_or_name: Path to a GGUF model or a HF repository identifier.
        
    Returns:
        The loaded AutoTokenizer instance, or None if unavailable/unsupported.
    """
    global _tokenizer_cache

    if not model_path_or_name:
        model_path_or_name = "Qwen/Qwen2.5-7B-Instruct"

    if model_path_or_name in _tokenizer_cache:
        return _tokenizer_cache[model_path_or_name]

    # Determine the most suitable HF repository ID
    repo_id = "Qwen/Qwen2.5-7B-Instruct"
    name_lower = model_path_or_name.lower()
    if "mpnet" in name_lower:
        repo_id = "sentence-transformers/all-mpnet-base-v2"
    elif "qwen" in name_lower:
        repo_id = "Qwen/Qwen2.5-7B-Instruct"

    # Safe import of transformers
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning("transformers package is not installed. Tokenizer cannot be loaded via factory.")
        return None

    project_root = get_project_root()
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Sanitize repository name for local folder use
    repo_sanitized = repo_id.replace("/", "_")
    
    # Sanitize custom model/path name for local folder use
    custom_name = os.path.basename(model_path_or_name)
    if custom_name.endswith(".gguf"):
        custom_name = custom_name[:-5]
    custom_sanitized = custom_name.replace("/", "_").replace("\\", "_")

    # List of candidate paths to check offline in priority order
    candidate_paths = [
        os.path.join(models_dir, f"{custom_sanitized}_tokenizer"),
        os.path.join(models_dir, f"{repo_sanitized}_tokenizer"),
        os.path.join(models_dir, "qwen3_reranker"),
    ]

    # Filter out duplicate paths while preserving order
    candidate_paths = list(dict.fromkeys(candidate_paths))

    # 1. Offline Mode: Try loading from local candidates first
    for local_path in candidate_paths:
        if os.path.exists(local_path) and (
            os.path.exists(os.path.join(local_path, "tokenizer.json")) or 
            os.path.exists(os.path.join(local_path, "vocab.json")) or
            os.path.exists(os.path.join(local_path, "tokenizer_config.json"))
        ):
            try:
                logger.info(f"Loading tokenizer locally from: {local_path}")
                tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
                _tokenizer_cache[model_path_or_name] = tokenizer
                return tokenizer
            except Exception as e:
                logger.warning(f"Failed to load local tokenizer from {local_path}: {e}")

    # 2. Try download and save locally for future offline runs
    target_local_path = os.path.join(models_dir, f"{repo_sanitized}_tokenizer")
    try:
        logger.info(f"Downloading tokenizer '{repo_id}' from HuggingFace to '{target_local_path}'")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        tokenizer.save_pretrained(target_local_path)
        _tokenizer_cache[model_path_or_name] = tokenizer
        return tokenizer
    except Exception as e:
        logger.warning(
            f"Failed to download tokenizer '{repo_id}' from HuggingFace: {e}. "
            "Offline fallback will be used."
        )

    # 3. Fallback: Search for any other directory containing 'tokenizer' in models folder
    try:
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path) and "tokenizer" in item.lower():
                if os.path.exists(os.path.join(item_path, "tokenizer.json")):
                    try:
                        logger.info(f"Attempting fallback load from available directory: {item_path}")
                        tokenizer = AutoTokenizer.from_pretrained(item_path, local_files_only=True)
                        _tokenizer_cache[model_path_or_name] = tokenizer
                        return tokenizer
                    except Exception:
                        pass
    except Exception:
        pass

    return None


def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """
    Count the number of tokens in the given text using the tokenizer.
    If tokenizer is None, uses character-based fallback of len(text) // 4.
    
    Args:
        text: Input string to be tokenized.
        tokenizer: Optional tokenizer instance supporting encode() or tokenize().
        
    Returns:
        Integer representing the token count.
    """
    if not text:
        return 0

    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "encode"):
                return len(tokenizer.encode(text, add_special_tokens=False))
            elif hasattr(tokenizer, "tokenize"):
                return len(tokenizer.tokenize(text.encode("utf-8")))
        except Exception as e:
            logger.warning(f"Tokenizer encoding failed: {e}. Falling back to estimate.")

    return len(text) // 4
