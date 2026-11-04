"""
Locates the local LLM (GGUF) used by determine-dependency, resolve-placeholders and
normalize-docs, downloading the default model on first use.

Only the default model is downloaded: an explicit --model path that does not exist is
an error, never a download. A file that is structurally intact (GGUF header, plausible
size) is never replaced; only a corrupt one is re-downloaded. The download goes to a
'.part' file that is renamed into place only once complete, so an interrupted download
never leaves a truncated model at the real path.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_LLM_FILENAME = "Qwen3-4B-Instruct-2507-Q8_0.gguf"
DEFAULT_LOCAL_LLM_URL = (
    "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/"
    + DEFAULT_LOCAL_LLM_FILENAME
)


def default_local_llm_path(project_root: str) -> str:
    return os.path.join(project_root, "models", DEFAULT_LOCAL_LLM_FILENAME)


def ensure_local_llm_model(model_arg: Optional[str], project_root: str) -> Optional[str]:
    """
    Path of the local LLM to load, or None when it is unavailable (the reason is
    printed). `model_arg` is the --model value; without it the default model under
    <project_root>/models is used and downloaded if missing or corrupt.
    """
    from .embedding_manager import (  # pyright: ignore[reportPrivateUsage]
        _download_with_retry,
        _qwen3_file_is_intact,
    )

    if model_arg:
        if os.path.isfile(model_arg):
            return model_arg
        print(f"Error: Local LLM model not found at {model_arg} (--model).")
        return None

    path = default_local_llm_path(project_root)
    if _qwen3_file_is_intact(path):
        return path
    if os.path.exists(path):
        logger.warning(f"Local LLM model at {path} is corrupt; downloading it again.")

    part = path + ".part"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Local LLM model not found; downloading {DEFAULT_LOCAL_LLM_FILENAME} (~4.3 GB) to {path}...")
    if not _download_with_retry(
        url=DEFAULT_LOCAL_LLM_URL,
        path=part,
        description=f"Downloading {DEFAULT_LOCAL_LLM_FILENAME}",
        use_phase_tracker=True,
    ):
        print(
            f"Error: could not download the local LLM model from {DEFAULT_LOCAL_LLM_URL}. "
            f"Download it to {path}, or pass --model."
        )
        return None
    if not _qwen3_file_is_intact(part):
        print(f"Error: the downloaded local LLM model is not a valid GGUF file ({part}).")
        try:
            os.remove(part)
        except OSError:
            pass
        return None
    os.replace(part, path)
    logger.info(f"Downloaded local LLM model to {path}.")
    return path
