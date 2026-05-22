"""
Core package initialization.
"""

import os
import shutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def _migrate_state_files(core_dir: Optional[str] = None):
    """Migrate any .json and .json.lock files from core to core/state."""
    if core_dir is None:
        core_dir = os.path.dirname(os.path.abspath(__file__))
    
    state_dir = os.path.join(core_dir, "state")
    
    try:
        if not os.path.exists(state_dir):
            os.makedirs(state_dir, exist_ok=True)
            
        for file in os.listdir(core_dir):
            if file.endswith(".json") or file.endswith(".json.lock"):
                source = os.path.join(core_dir, file)
                dest = os.path.join(state_dir, file)
                if os.path.isfile(source):
                    try:
                        shutil.move(source, dest)
                        logger.info(f"Migrated state file from {source} to {dest}")
                    except FileNotFoundError as e:
                        logger.debug(f"Failed to migrate state file {file} (likely already migrated): {e}")
    except Exception as e:
        logger.error(f"Error during state file migration: {e}")

def resolve_state_path(filename: str, core_dir: Optional[str] = None) -> str:
    """
    Resolve the path for a state file, preferring the 'state/' subdirectory.
    Falls back to the core directory if the file exists there but not in 'state/'.
    """
    if core_dir is None:
        core_dir = os.path.dirname(os.path.abspath(__file__))
    
    state_path = os.path.join(core_dir, "state", filename)
    if os.path.exists(state_path):
        return state_path
        
    legacy_path = os.path.join(core_dir, filename)
    if os.path.exists(legacy_path):
        logger.debug(f"Using legacy state file path for {filename}: {legacy_path}")
        return legacy_path
        
    return state_path

_migrate_state_files()

