"""
Configuration module for dependency tracking system.
Handles reading and writing configuration settings.
"""

import glob
import os
import json
from typing import Dict, List, Any, Optional, Union
import logging

from .path_utils import normalize_path, get_project_root

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "excluded_dirs": [
        "__pycache__",
        ".git",
        ".svn",
        ".hg",
        ".vscode",
        ".idea",
        "__MACOSX",
        "venv",
        "env",
        ".venv",
        "node_modules",
        "bower_components",
        "build",
        "dist",
        "target",
        "out",
        "tmp",
        "temp",
        "tests",
        "examples",
        "embeddings",
        # Explicitly exclude the default embeddings directory path relative to project root
        # to prevent processing its own contents (metadata.json, .npy files)
        "cline_utils/dependency_system/analysis/embeddings",
        "cline_docs/dependency_diagrams", 
    ],
    "excluded_extensions": [
        ".pyc",
        ".pyo",
        ".pyd",
        ".dll",
        ".exe",
        ".so",
        ".o",
        ".a",
        ".lib",
        ".dll",
        ".pdb",
        ".sdf",
        ".suo",
        ".user",
        ".swp",
        ".log",
        ".tmp",
        ".bak",
        ".d",
        ".DS_Store",
        ".jar",
        ".war",
        ".ear",
        ".zip",
        ".tar.gz",
        ".tar",
        ".tgz",
        ".rar",
        ".7z",
        ".dmg",
        ".iso",
        ".img",
        ".bin",
        ".dat",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".dbf",
        ".mdb",
        ".sav",
        ".eot",
        ".ttf",
        ".woff",
        ".woff2",
        ".otf",
        ".swf",
        ".bak",
        ".old",
        ".orig",
        ".embedding",
        ".npy",
        ".mermaid"
    ],
    "thresholds": {"doc_similarity": 0.65, "code_similarity": 0.7},
    "models": {
        "doc_model_name": "all-mpnet-base-v2",
        "code_model_name": "all-mpnet-base-v2",
    },
    "compute": {
        "embedding_device": "auto"  # Options: "auto", "cuda", "mps", "cpu"
    },
    "paths": {
        "doc_dir": "docs",
        "memory_dir": "cline_docs",
        "embeddings_dir": "cline_utils/dependency_system/analysis/embeddings",
        "backups_dir": "cline_docs/backups",
    },
    "excluded_paths": [
        "src/node_modules",
        "src/client/node_modules"
    ],
    "allowed_dependency_chars": ['<', '>', 'x', 'd', 's', 'S'],
    "excluded_file_patterns": [
        "*_module.md",
        "implementation_plan_*.md",
        "*_task.md" # Future task file pattern (to be confirmed)
    ],
    "file_exclusion_patterns": [
        "**/bin/**",
        "**/obj/**",
        "**/*.dll",
        "**/*.pdb",
        "**/*.exe",
        "**/node_modules/**",
        "**/.git/**"
    ],
    "directory_exclusion_patterns": [
        "bin",
        "obj",
        "packages",
        "node_modules",
        ".vs",
        ".vscode"
    ],
    "visualization": {
        "auto_generate_on_analyze": True, # Enable auto-generation by default
        "auto_diagram_output_dir": None   # Default to None, meaning derive from memory_dir
                                          # If user sets this (e.g., "my_diagrams"), it overrides the default derivation
    }
}

# Define character priorities (Higher number = higher priority) - Centralized definition
# Conforms to the existing convention in dependency_suggester.py
CHARACTER_PRIORITIES = {
    'x': 5,
    '<': 4, '>': 4, 'd': 4, 'n': 4,
    'S': 3,
    's': 2,
    'p': 1, 'o': 1,
    '-': 0, # Placeholder_char (Assign lowest numeric priority > 0)
    ' ': 0  # Empty_char
}
DEFAULT_PRIORITY = 0 # Default for unknown characters (lowest priority)

class ConfigManager:
    """
    Configuration manager for dependency tracking system.
    Handles reading and writing configuration settings, and provides
    convenience methods for accessing specific settings.
    """

    _instance = None

    def __new__(cls):
        """
        Singleton pattern implementation to ensure only one config instance.
        
        Returns:
            ConfigManager instance
        """
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False  # Moved inside the singleton check
        return cls._instance

    def __init__(self):
        """Initialize the configuration manager and load config."""
        if self._initialized:  # Correct check for double initialization
            return

        self._config = None  # Initialize config
        self._config_path = None # Initialize config path
        self._load_config()  # Load or create default config
        
        # Set default exclusions if not present
        if "excluded_dirs" not in self._config:
            self._config["excluded_dirs"] = DEFAULT_CONFIG["excluded_dirs"]
        if "excluded_extensions" not in self._config:
            self._config["excluded_extensions"] = DEFAULT_CONFIG["excluded_extensions"]

        self._initialized = True

    def get_compute_setting(self, setting_name: str, default: Any = None) -> Any:
        """Gets a setting from the 'compute' section of the config."""
        compute_settings = self.config.get("compute", {})
        return compute_settings.get(setting_name, default)

    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary.
        
        Returns:
            Configuration dictionary
        """
        from .cache_manager import cached

        @cached("config_data",
                key_func=lambda self: f"config:{os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else 'missing'}")
        def _get_config(self) -> Dict[str, Any]:
            if self._config is None:
                self._load_config()
            return self._config

        return _get_config(self)

    @property
    def config_path(self) -> str:
        """
        Get the path to the configuration file.
        
        Returns:
            Path to the configuration file
        """
        from .cache_manager import cached

        def _get_config_path(self) -> str:
            if self._config_path is None:
                project_root = get_project_root()
                self._config_path = normalize_path(os.path.join(project_root, ".clinerules.config.json"))
            return self._config_path

        return _get_config_path(self)

    def _load_config(self) -> None:
        """Load configuration from file or create default."""
        try:
            if os.path.exists(normalize_path(self.config_path)):
                with open(normalize_path(self.config_path), 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            else:
                self._config = DEFAULT_CONFIG.copy()
                self._save_config()
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_path}: {e}")
            self._config = DEFAULT_CONFIG.copy()

    def _save_config(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(normalize_path(self.config_path)), exist_ok=True)
            with open(normalize_path(self.config_path), 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2)
            return True
        except OSError as e:
            logger.error(f"Error writing configuration file {self.config_path}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error saving configuration to {self.config_path}: {e}")
            return False

    def update_config_setting(self, key: str, value: Union[str, int, float, List, Dict]) -> bool:
        """Update a specific configuration setting."""
        keys = key.split('.')
        current = self.config
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                logger.error(f"Invalid configuration key: {key}")
                return False
            current = current[k]
        last_key = keys[-1]
        if last_key not in current:
            logger.error(f"Invalid configuration key: {key}")
            return False
        current[last_key] = value
        return self._save_config()

    def get_excluded_dirs(self) -> List[str]:
        """
        Get list of excluded directories.
        
        Returns:
            List of excluded directory names
        """
        from .cache_manager import cached

        @cached("excluded_dirs",
                key_func=lambda self: f"excluded_dirs:{os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else 'missing'}")
        def _get_excluded_dirs(self) -> List[str]:
            return self.config.get("excluded_dirs", DEFAULT_CONFIG["excluded_dirs"])

        return _get_excluded_dirs(self)

    def get_excluded_extensions(self) -> List[str]:
        """
        Get list of excluded file extensions.
        
        Returns:
            List of excluded file extensions
        """
        from .cache_manager import cached

        @cached("excluded_extensions",
                key_func=lambda self: f"excluded_extensions:{os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else 'missing'}")
        def _get_excluded_extensions(self) -> List[str]:
            return self.config.get("excluded_extensions", DEFAULT_CONFIG["excluded_extensions"])

        return _get_excluded_extensions(self)

    def get_excluded_paths(self) -> List[str]:
        """
        Get list of excluded paths from configuration.
        
        Returns:
            List of excluded path patterns or absolute paths
        """
        from .cache_manager import cached

        @cached("excluded_paths",
                key_func=lambda self: f"excluded_paths:{os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else 'missing'}")
        def _get_excluded_paths(self) -> List[str]:
            # Retrieve excluded_paths from config, defaulting to DEFAULT_CONFIG value
            excluded_paths_config = self.config.get("excluded_paths", DEFAULT_CONFIG["excluded_paths"])
            excluded_file_patterns = self.config.get("excluded_file_patterns", DEFAULT_CONFIG.get("excluded_file_patterns", [])) # Get file patterns, default to empty list if not set
            
            excluded_paths = []
            project_root = get_project_root()

            # 1. Explicitly excluded paths
            excluded_paths.extend([normalize_path(os.path.join(project_root, p)) if not os.path.isabs(p) else normalize_path(p)
                                     for p in excluded_paths_config])

            # 2. Paths from excluded file patterns
            for pattern in excluded_file_patterns:
                # Construct the full pattern relative to the project root
                full_pattern = normalize_path(os.path.join(project_root, '**', pattern)) # Use '**' for recursion
                # Use glob with recursive=True to find matching paths
                matching_paths = glob.glob(full_pattern, recursive=True)
                excluded_paths.extend([normalize_path(p) for p in matching_paths])

            return excluded_paths

        return _get_excluded_paths(self)


    def get_threshold(self, threshold_type: str) -> float:
        """
        Get threshold value.
        
        Args:
            threshold_type: Type of threshold ('doc_similarity' or 'code_similarity')
            
        Returns:
            Threshold value
        """
        thresholds = self.config.get("thresholds", DEFAULT_CONFIG["thresholds"])
        return thresholds.get(threshold_type, 0.7)
    
    def get_model_name(self, model_type: str) -> str:
        """
        Get model name.
        
        Args:
            model_type: Type of model ('doc_model_name' or 'code_model_name')
            
        Returns:
            Model name
        """
        models = self.config.get("models", DEFAULT_CONFIG["models"])
        return models.get(model_type, "all-mpnet-base-v2")

    def get_path(self, path_type: str, default_path: Optional[str] = None) -> str:
        """
        Get path from configuration.
        
        Args:
            path_type: Type of path ('doc_dir', 'memory_dir', or 'embeddings_dir')
            default_path: Default path to use if not found in configuration
            
        Returns:
            Path from configuration or default
        """
        paths = self.config.get("paths", DEFAULT_CONFIG["paths"])
        path = paths.get(path_type, default_path if default_path else DEFAULT_CONFIG["paths"].get(path_type, ""))
        if path_type == "embeddings_dir":
            return normalize_path(os.path.join(get_project_root(), path))
        return normalize_path(path)

    def get_code_root_directories(self) -> List[str]:
        """
        Get list of code root directories from .clinerules, sorted alphabetically.

        Returns:
            Sorted list of code root directories
        """
        from .cache_manager import cached

        @cached("code_roots",
                key_func=lambda self: f"code_roots:{os.path.getmtime(os.path.join(get_project_root(), '.clinerules')) if os.path.exists(os.path.join(get_project_root(), '.clinerules')) else 'missing'}")
        def _get_code_root_directories(self) -> List[str]:
            clinerules_path = os.path.join(get_project_root(), ".clinerules")
            code_root_dirs = []
            try:
                with open(clinerules_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                in_code_root_section = False
                for line in lines:
                    line = line.strip()
                    if line == "[CODE_ROOT_DIRECTORIES]":
                        in_code_root_section = True
                        continue
                    if in_code_root_section:
                        if line.startswith("-"):
                            # Normalize path *before* adding to list
                            path_part = line[1:].strip() # Get content after '-'
                            if path_part: # Ensure it's not just '-'
                                code_root_dirs.append(normalize_path(path_part))
                        elif line.startswith("["): break # Reached next section
            except FileNotFoundError:
                logger.warning("'.clinerules' file not found. Cannot read code root directories.")
            except Exception as e:
                logger.error(f"Error reading .clinerules for code roots: {e}")
            # *** SORT the result alphabetically ***
            code_root_dirs.sort()
            logger.debug(f"Found and sorted code roots: {code_root_dirs}")
            return code_root_dirs

        return _get_code_root_directories(self)

    def get_doc_directories(self) -> List[str]:
        """
        Get list of doc directories from .clinerules, sorted alphabetically.

        Returns:
            Sorted list of doc directories
        """
        from .cache_manager import cached

        @cached("doc_dirs",
                key_func=lambda self: f"doc_dirs:{os.path.getmtime(os.path.join(get_project_root(), '.clinerules')) if os.path.exists(os.path.join(get_project_root(), '.clinerules')) else 'missing'}")
        def _get_doc_directories(self) -> List[str]:
            clinerules_path = os.path.join(get_project_root(), ".clinerules")
            doc_dirs = []
            try:
                with open(clinerules_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                in_doc_section = False
                for line in lines:
                    line = line.strip()
                    if line == "[DOC_DIRECTORIES]":
                        in_doc_section = True
                        continue
                    if in_doc_section:
                        if line.startswith("-"):
                             # Normalize path *before* adding to list
                            path_part = line[1:].strip() # Get content after '-'
                            if path_part: # Ensure it's not just '-'
                                doc_dirs.append(normalize_path(path_part))
                        elif line.startswith("["): break # Reached next section
            except FileNotFoundError:
                 logger.warning("'.clinerules' file not found. Cannot read doc directories.")
            except Exception as e:
                logger.error(f"Error reading .clinerules for doc dirs: {e}")
            # *** SORT the result alphabetically ***
            doc_dirs.sort()
            logger.debug(f"Found and sorted doc dirs: {doc_dirs}")
            return doc_dirs

        return _get_doc_directories(self)

    def get_allowed_dependency_chars(self) -> List[str]:
        """Get the allowed dependency characters from configuration."""
        # Correctly fetch from the config dictionary, falling back to default
        return self.config.get("allowed_dependency_chars", DEFAULT_CONFIG["allowed_dependency_chars"])

    def get_file_exclusion_patterns(self) -> List[str]:
        """
        Get list of file exclusion patterns from configuration.
        
        Returns:
            List of file exclusion patterns (glob patterns)
        """
        from .cache_manager import cached

        @cached("file_exclusion_patterns",
                key_func=lambda self: f"file_exclusion_patterns:{os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else 'missing'}")
        def _get_file_exclusion_patterns(self) -> List[str]:
            return self.config.get("file_exclusion_patterns", DEFAULT_CONFIG.get("file_exclusion_patterns", []))

        return _get_file_exclusion_patterns(self)

    def get_directory_exclusion_patterns(self) -> List[str]:
        """
        Get list of directory exclusion patterns from configuration.
        
        Returns:
            List of directory exclusion patterns (glob patterns)
        """
        from .cache_manager import cached

        @cached("directory_exclusion_patterns",
                key_func=lambda self: f"directory_exclusion_patterns:{os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else 'missing'}")
        def _get_directory_exclusion_patterns(self) -> List[str]:
            return self.config.get("directory_exclusion_patterns", DEFAULT_CONFIG.get("directory_exclusion_patterns", []))

        return _get_directory_exclusion_patterns(self)

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Deep update of nested dictionaries
            self._deep_update(self.config, updates)
            return self._save_config()
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            return False

    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

    def reset_to_defaults(self) -> bool:
        """
        Reset configuration to default values.
        
        Returns:
            True if successful, False otherwise
        """
        self._config = DEFAULT_CONFIG.copy()
        return self._save_config()

    def get_char_priority(self, char: str) -> int:
        """
        Get the priority tier for a given dependency character.
        Higher numbers indicate higher priority.

        Args:
            char: The dependency character.

        Returns:
            The priority tier (integer).
        """
        # Uses the centrally defined dictionary
        return CHARACTER_PRIORITIES.get(char, DEFAULT_PRIORITY)