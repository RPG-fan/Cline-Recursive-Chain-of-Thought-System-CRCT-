# config_manager.py

"""
Configuration module for dependency tracking system.
Handles reading and writing configuration settings.
"""

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from .path_utils import get_project_root, normalize_path
from .resource_validator import ResourceValidator

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
        "__tests__",
        "examples",
        "embeddings",
        "cline_utils/dependency_system/analysis/embeddings",
        "cline_docs/dependency_diagrams",
        "cline_utils",
        "cline_docs",
        ".roo",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        ".tox",
        ".coverage",
        ".cache",
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
        ".mermaid",
        ".roomodes",
        ".env",
        ".clinerules",
        ".clinerules.config.json",
    ],
    "thresholds": {
        "doc_similarity": 0.65,
        "code_similarity": 0.7,
        "doc_code_similarity": 0.68,  # Threshold for doc<->code relations
        "reranker_promotion_threshold": 0.9,  # Threshold for < promotion
        "reranker_strong_semantic_threshold": 0.78,  # Threshold for S
        "reranker_weak_semantic_threshold": 0.65,  # Threshold for s
    },
    "models": {
        "doc_model_name": "all-mpnet-base-v2",
        "code_model_name": "all-mpnet-base-v2",
    },
    "compute": {"embedding_device": "auto"},  # Options: "auto", "cuda", "mps", "cpu"
    "embedding": {
        "model_selection": "auto",  # "auto", "qwen3-4b", "mpnet"
        "qwen3_model_path": "models/Qwen3-Embedding-4B-Q6_K.gguf",
        "qwen3_embedding_dim": 2560,
        "qwen3_context_length": 32768,
        "qwen3_gpu_layers": 35,  # Number of layers to offload to GPU (0 for CPU only)
        "mpnet_embedding_dim": 384,
        "mpnet_context_length": 512,
        "reranker_model_path": "models/Qwen3-Reranker-0.6B",  # New reranker model path
    },
    "paths": {
        "doc_dir": "docs",
        "memory_dir": "cline_docs",
        "embeddings_dir": "cline_utils/dependency_system/analysis/embeddings",
        "backups_dir": "cline_docs/backups",
    },
    "excluded_paths": [
        "src/node_modules",
        "src/client/node_modules",
        "**/.mypy_cache",
        "**/.pytest_cache",
        "**/__pycache__",
    ],
    "allowed_dependency_chars": ["<", ">", "x", "d", "s", "S", "n"],
    "excluded_file_patterns": [
        "*_module.md",
        "implementation_plan_*.md",
        "*_task.md",
        "*-checkpoint.md",
        "*debug.txt",
        "*suggestions.log",
    ],
    "visualization": {
        "auto_generate_on_analyze": True,  # Enable auto-generation by default
        "auto_diagram_output_dir": None,  # Default to None, meaning derive from memory_dir
        # If user sets this (e.g., "my_diagrams"), it overrides the default derivation
    },
    "recovery": {
        "auto_restore_corrupt_tracker_from_backup": False,  # Default: False (safer, prompts or rebuilds)
        "backup_on_restore_attempt": True,  # Default: True (backup the corrupt file before overwriting with backup)
    },
    # Enhanced performance configuration
    "performance": {
        "default_batch_size": 32,  # Batch size for parallel processing
        "embedding_batch_size": 16,  # Smaller batch for embedding generation
        "enable_parallel_processing": True,  # Enable parallel file analysis
        "max_workers": None,  # None = auto-detect based on CPU cores
        "cache_size_limit": 5000,  # Maximum cache entries
        "cache_ttl_seconds": 300,  # Cache time-to-live (5 minutes)
        "memory_limit_mb": 2048,  # Memory limit for analysis
        "strict_mode": False,  # Fail on warnings if True
    },
    # Enhanced analysis configuration
    "analysis": {
        "strict_binary_detection": True,  # Enhanced binary file detection
        "check_file_signatures": True,  # Check file signatures for binary detection
        "null_byte_threshold": 0.1,  # Threshold for binary detection
        "python_ast_enabled": True,  # Enable AST parsing for Python
        "max_ast_file_size_mb": 1,  # Maximum file size for AST parsing
        "js_tree_sitter_enabled": True,  # Enable tree-sitter for JavaScript
        "typescript_tree_sitter_enabled": True,  # Enable tree-sitter for TypeScript
        "extract_comments": False,  # Extract comments from code
        "extract_docstrings": True,  # Extract docstrings from Python
        "min_function_length": 1,  # Minimum function length to extract
        "min_class_length": 1,  # Minimum class length to extract
    },
    # Enhanced resource management
    "resources": {
        "min_memory_mb": 512,  # Minimum memory required (MB)
        "recommended_memory_mb": 2048,  # Recommended memory (MB)
        "min_disk_space_mb": 100,  # Minimum disk space required (MB)
        "recommended_disk_space_mb": 500,  # Recommended disk space (MB)
        "temp_space_required_mb": 100,  # Temporary space required (MB)
        "temp_dir_path": None,  # Custom temp directory (None = system default)
        "strict_resource_validation": False,  # Fail on resource warnings
        "allow_partial_analysis": True,  # Allow analysis with limited resources
        "resource_check_enabled": True,  # Enable pre-analysis resource checks
    },
    # Enhanced output configuration
    "output": {
        "auto_generate_diagrams": True,  # Auto-generate dependency diagrams
        "diagram_output_dir": "dependency_diagrams",  # Directory for diagrams
        "max_diagram_nodes": 100,  # Maximum nodes in diagrams
        "log_level": "INFO",  # Logging level
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_file_enabled": True,  # Enable log file output
        "debug_logging": False,  # Enable debug logging
        "create_backups": True,  # Create backups before changes
        "backup_suffix": "_old",  # Backup file suffix
        "max_backups": 3,  # Maximum number of backup files
        "structured_logging": False,  # Use structured logging format
        "progress_reporting": True,  # Show progress during analysis
    },
    # Environment variable overrides
    "environment": {
        "allow_overrides": True,  # Allow environment variable overrides
        "override_prefix": "ANALYZER_",  # Prefix for environment variables
        "supported_overrides": [
            "LOG_LEVEL",
            "BATCH_SIZE",
            "CHUNK_SIZE",
            "MAX_WORKERS",
            "EMBEDDING_MODEL",
            "USE_STREAMING",
            "DEBUG",
            "AUTO_SELECT_MODEL",
            "ENABLE_GGUF",
            "GGUF_PATH",
            "SIMILARITY_THRESHOLD",
        ],
    },
}

# Define character priorities (Higher number = higher priority) - Centralized definition
# Conforms to the existing convention in dependency_suggester.py
CHARACTER_PRIORITIES = {
    "x": 5,
    "<": 4,
    ">": 4,
    "n": 4,
    "d": 3,
    "S": 3,
    "s": 2,
    "p": 1,
    "o": 1,
    "-": 0,  # Placeholder_char (Assign lowest numeric priority > 0)
    " ": 0,  # Empty_char
}
DEFAULT_PRIORITY = 0  # Default for unknown characters (lowest priority)


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
        super().__init__()
        if self._initialized:
            return
        self._config: Optional[Dict[str, Any]] = None
        self._config_path: Optional[str] = None
        self._environment_overrides: Dict[str, Any] = {}
        self._config_files_loaded: List[str] = []
        self._resource_validation_results: Optional[Dict[str, Any]] = None

        self._load_and_merge_config()

        # Ensure defaults for all top-level keys if missing after load
        for key, default_value in [
            ("excluded_dirs", DEFAULT_CONFIG["excluded_dirs"]),
            ("excluded_extensions", DEFAULT_CONFIG["excluded_extensions"]),
            ("paths", DEFAULT_CONFIG["paths"]),
            ("recovery", DEFAULT_CONFIG["recovery"]),
            ("performance", DEFAULT_CONFIG["performance"]),
            ("analysis", DEFAULT_CONFIG["analysis"]),
            ("resources", DEFAULT_CONFIG["resources"]),
            ("output", DEFAULT_CONFIG["output"]),
            ("environment", DEFAULT_CONFIG["environment"]),
        ]:
            if key not in self._config:  # type: ignore
                self._config[key] = default_value  # type: ignore

        # Apply environment variable overrides
        self._apply_environment_overrides()

        # NOTE: Resource adjustments are no longer applied automatically here.
        # They are triggered by perform_resource_validation_and_adjustments().

        self._initialized = True

    def perform_resource_validation_and_adjustments(self) -> None:
        """
        Performs system resource validation and applies adjustments to the
        current configuration if enabled. This is intended to be called
        explicitly before heavy operations like project analysis.
        """
        if self.config.get("resources", {}).get("resource_check_enabled", True):
            self._apply_resource_adjustments()

    def get_recovery_setting(
        self, setting_name: str, default_override: Any = None
    ) -> Any:
        """Gets a setting from the 'recovery' section of the config."""
        recovery_settings = self.config.get(
            "recovery", DEFAULT_CONFIG.get("recovery", {})
        )

        # Determine the ultimate default value
        # 1. Use default_override if provided
        # 2. Else, use default from DEFAULT_CONFIG for this specific setting_name
        # 3. Else, None (though our DEFAULT_CONFIG for recovery is complete)
        if default_override is not None:
            ultimate_default = default_override
        else:
            ultimate_default = DEFAULT_CONFIG.get("recovery", {}).get(setting_name)

        return recovery_settings.get(setting_name, ultimate_default)

    def get_compute_setting(self, setting_name: str, default: Any = None) -> Any:
        """Gets a setting from the 'compute' section of the config."""
        compute_settings = self.config.get("compute", {})
        return compute_settings.get(setting_name, default)

    def get_embedding_setting(self, setting_name: str, default: Any = None) -> Any:
        """Gets a setting from the 'embedding' section of the config."""
        embedding_settings = self.config.get("embedding", {})
        return embedding_settings.get(setting_name, default)

    def get_reranker_model_path(self) -> str:
        """Gets the path to the reranker model."""
        reranker_path = self.config.get("embedding", {}).get(
            "reranker_model_path", DEFAULT_CONFIG["embedding"]["reranker_model_path"]
        )
        return normalize_path(os.path.join(get_project_root(), reranker_path))

    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary.

        Returns:
            Configuration dictionary
        """
        from .cache_manager import cached

        @cached(
            "config_data",
            key_func=lambda self: f"config:{os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else 'missing'}",
        )
        def _get_config(self) -> Dict[str, Any]:
            # Always reload if this function is called (cache miss/invalidation)
            self._load_and_merge_config()
            return self._config

        return _get_config(self)

    @property
    def config_path(self) -> str:
        """
        Get the path to the configuration file.

        Returns:
            Path to the configuration file
        """

        def _get_config_path(self) -> str:
            if self._config_path is None:
                project_root = get_project_root()
                self._config_path = normalize_path(
                    os.path.join(project_root, ".clinerules.config.json")
                )
            return self._config_path

        return _get_config_path(self)

    def _load_and_merge_config(self) -> None:
        """Load configuration from file and deep merge with defaults."""
        # Start with a deep copy of the defaults
        final_config = json.loads(json.dumps(DEFAULT_CONFIG))  # Simple deep copy

        user_config = self._load_user_config_file()
        if user_config:
            # Deep update the defaults with the user's settings
            self._deep_update(final_config, user_config)

        self._config = final_config

        # If no config file existed, save the generated default config
        if not user_config:
            self._save_config()

        # Re-apply environment overrides (they are lost when self._config is reset)
        self._apply_environment_overrides()

        # Re-apply resource adjustments if they were previously calculated
        if self._resource_validation_results:
            self._apply_adjustments_from_results(self._resource_validation_results)

    def _load_user_config_file(self) -> Optional[Dict[str, Any]]:
        """Loads user config from file, returns None if not found or error."""
        try:
            if os.path.exists(normalize_path(self.config_path)):
                with open(normalize_path(self.config_path), "r", encoding="utf-8") as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_path}: {e}")
            return None

    def _save_config(self) -> bool:
        """
        Save configuration to file.

        Returns:
            True if successful, False otherwise
        """
        config_path = self.config_path  # Ensure path is initialized
        try:
            os.makedirs(
                os.path.dirname(normalize_path(self.config_path)), exist_ok=True
            )
            with open(normalize_path(self.config_path), "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {config_path}")

            # Invalidate config cache if using cache_manager
            try:
                from .cache_manager import invalidate_dependent_entries

                invalidate_dependent_entries(
                    "config_data", f"config:{os.path.getmtime(config_path)}"
                )
            except ImportError:
                pass  # Cache manager not available
            except Exception as e_cache:
                logger.warning(f"Could not invalidate config cache: {e_cache}")

            return True
        except OSError as e:
            logger.error(f"Error writing configuration file {self.config_path}: {e}")
            return False
        except Exception as e:
            logger.exception(
                f"Unexpected error saving configuration to {self.config_path}: {e}"
            )
            return False

    def update_config_setting(
        self, key: str, value: Union[str, int, float, List[Any], Dict[str, Any]]
    ) -> bool:
        """Update a specific configuration setting."""
        keys = key.split(".")
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

        @cached(
            "excluded_dirs",
            key_func=lambda self: f"excluded_dirs:{os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else 'missing'}",
        )
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

        @cached(
            "excluded_extensions",
            key_func=lambda self: f"excluded_extensions:{os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else 'missing'}",
        )
        def _get_excluded_extensions(self) -> List[str]:
            return self.config.get(
                "excluded_extensions", DEFAULT_CONFIG["excluded_extensions"]
            )

        return _get_excluded_extensions(self)

    def get_excluded_paths(self) -> List[str]:
        """
        Get list of excluded paths from configuration.

        Returns:
            List of excluded path patterns or absolute paths
        """
        from .cache_manager import cached

        @cached(
            "excluded_paths",
            key_func=lambda self: f"excluded_paths:{os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else 'missing'}",
        )
        def _get_excluded_paths(self) -> List[str]:
            # Retrieve excluded_paths from config, defaulting to DEFAULT_CONFIG value
            excluded_paths_config = self.config.get(
                "excluded_paths", DEFAULT_CONFIG["excluded_paths"]
            )
            excluded_file_patterns = self.config.get(
                "excluded_file_patterns",
                DEFAULT_CONFIG.get("excluded_file_patterns", []),
            )  # Get file patterns, default to empty list if not set

            excluded_paths = []
            project_root = get_project_root()

            # 1. Explicitly excluded paths
            excluded_paths.extend(
                [
                    (
                        normalize_path(os.path.join(project_root, p))
                        if not os.path.isabs(p)
                        else normalize_path(p)
                    )
                    for p in excluded_paths_config
                ]
            )

            # 2. Paths from excluded file patterns
            for pattern in excluded_file_patterns:
                # Construct the full pattern relative to the project root
                full_pattern = normalize_path(
                    os.path.join(project_root, "**", pattern)
                )  # Use '**' for recursion
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
        return thresholds.get(threshold_type, 0.65)

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
        path = paths.get(
            path_type,
            (
                default_path
                if default_path
                else DEFAULT_CONFIG["paths"].get(path_type, "")
            ),
        )
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

        @cached(
            "code_roots",
            key_func=lambda self: (
                lambda pr: f"code_roots:{os.path.getmtime(os.path.join(pr, '.clinerules', 'default-rules.md')) if os.path.exists(os.path.join(pr, '.clinerules', 'default-rules.md')) else (os.path.getmtime(os.path.join(pr, '.clinerules')) if os.path.exists(os.path.join(pr, '.clinerules')) else 'missing')}"
            )(get_project_root()),
        )
        def _get_code_root_directories(self) -> List[str]:
            project_root = get_project_root()
            new_rules_path = os.path.join(
                project_root, ".clinerules", "default-rules.md"
            )
            legacy_rules_path = os.path.join(project_root, ".clinerules")
            clinerules_path = (
                new_rules_path if os.path.exists(new_rules_path) else legacy_rules_path
            )
            code_root_dirs = []
            try:
                with open(clinerules_path, "r", encoding="utf-8") as f:
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
                            path_part = line[1:].strip()  # Get content after '-'
                            if path_part:  # Ensure it's not just '-'
                                code_root_dirs.append(normalize_path(path_part))
                        elif line.startswith("["):
                            break  # Reached next section
            except FileNotFoundError:
                logger.warning(
                    "'.clinerules/default-rules.md' not found and legacy '.clinerules' missing. Cannot read code root directories."
                )
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

        @cached(
            "doc_dirs",
            key_func=lambda self: (
                lambda pr: f"doc_dirs:{os.path.getmtime(os.path.join(pr, '.clinerules', 'default-rules.md')) if os.path.exists(os.path.join(pr, '.clinerules', 'default-rules.md')) else (os.path.getmtime(os.path.join(pr, '.clinerules')) if os.path.exists(os.path.join(pr, '.clinerules')) else 'missing')}"
            )(get_project_root()),
        )
        def _get_doc_directories(self) -> List[str]:
            project_root = get_project_root()
            new_rules_path = os.path.join(
                project_root, ".clinerules", "default-rules.md"
            )
            legacy_rules_path = os.path.join(project_root, ".clinerules")
            clinerules_path = (
                new_rules_path if os.path.exists(new_rules_path) else legacy_rules_path
            )
            doc_dirs = []
            try:
                with open(clinerules_path, "r", encoding="utf-8") as f:
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
                            path_part = line[1:].strip()  # Get content after '-'
                            if path_part:  # Ensure it's not just '-'
                                doc_dirs.append(normalize_path(path_part))
                        elif line.startswith("["):
                            break  # Reached next section
            except FileNotFoundError:
                logger.warning(
                    "'.clinerules/default-rules.md' not found and legacy '.clinerules' missing. Cannot read doc directories."
                )
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
        return self.config.get(
            "allowed_dependency_chars", DEFAULT_CONFIG["allowed_dependency_chars"]
        )

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

    # Enhanced configuration methods

    def get_performance_setting(self, setting_name: str, default: Any = None) -> Any:
        """Gets a setting from the 'performance' section of the config."""
        performance_settings = self.config.get("performance", {})
        return performance_settings.get(setting_name, default)

    def get_analysis_setting(self, setting_name: str, default: Any = None) -> Any:
        """Gets a setting from the 'analysis' section of the config."""
        analysis_settings = self.config.get("analysis", {})
        return analysis_settings.get(setting_name, default)

    def get_resource_setting(self, setting_name: str, default: Any = None) -> Any:
        """Gets a setting from the 'resources' section of the config."""
        resource_settings = self.config.get("resources", {})
        return resource_settings.get(setting_name, default)

    def get_output_setting(self, setting_name: str, default: Any = None) -> Any:
        """Gets a setting from the 'output' section of the config."""
        output_settings = self.config.get("output", {})
        return output_settings.get(setting_name, default)

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        environment_config = self._config.get("environment", {})

        if not environment_config.get("allow_overrides", True):
            return

        prefix = environment_config.get("override_prefix", "ANALYZER_")
        supported_overrides = environment_config.get("supported_overrides", [])

        # Environment variable to configuration key mappings
        env_mappings = {
            "LOG_LEVEL": ("output", "log_level"),
            "BATCH_SIZE": ("performance", "default_batch_size"),
            "CHUNK_SIZE": ("performance", "chunk_size"),
            "MAX_WORKERS": ("performance", "max_workers"),
            "EMBEDDING_MODEL": ("embedding", "preferred_model"),
            "USE_STREAMING": ("performance", "use_streaming_analysis"),
            "DEBUG": ("output", "debug_logging"),
            "AUTO_SELECT_MODEL": ("embedding", "auto_select_model"),
            "ENABLE_GGUF": ("embedding", "enable_gguf_models"),
            "GGUF_PATH": ("embedding", "gguf_model_path"),
            "SIMILARITY_THRESHOLD": ("embedding", "similarity_threshold"),
        }

        for env_var, (section, key) in env_mappings.items():
            if env_var in supported_overrides:
                env_value = os.environ.get(f"{prefix}{env_var}")
                if env_value is not None:
                    converted_value = self._convert_env_value(env_value)
                    self._set_config_value(section, key, converted_value)
                    self._environment_overrides[f"{section}.{key}"] = env_value

    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ("true", "1", "yes", "on"):
            return True
        elif value.lower() in ("false", "0", "no", "off"):
            return False

        # Numeric conversion
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _set_config_value(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value in the config dictionary."""
        if self._config is None:
            self._config = {}
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def _apply_resource_adjustments(self) -> None:
        """Adjust configuration based on available system resources."""
        try:
            validator = ResourceValidator(
                strict_mode=self.get_resource_setting(
                    "strict_resource_validation", False
                )
            )
            project_root = get_project_root()
            results = validator.validate_system_resources(str(project_root))

            # Store validation results for later access
            self._resource_validation_results = results

            self._apply_adjustments_from_results(results)

        except Exception as e:
            logger.warning(f"Resource-based configuration adjustment failed: {e}")

    def _apply_adjustments_from_results(self, results: Dict[str, Any]) -> None:
        """Apply configuration adjustments based on validation results."""
        try:
            # Memory-based adjustments
            memory_check = results.get("resource_check", {}).get("memory", {})
            available_mb = memory_check.get("available_mb", 0)

            if available_mb < 1024:
                # Low memory configuration
                logger.info("Applying low-memory configuration adjustments")
                self._set_config_value("performance", "use_streaming_analysis", True)
                self._set_config_value(
                    "performance",
                    "default_batch_size",
                    min(
                        16,
                        self._config.get("performance", {}).get(
                            "default_batch_size", 32
                        ),
                    ),
                )
                self._set_config_value(
                    "performance",
                    "embedding_batch_size",
                    min(
                        8,
                        self._config.get("performance", {}).get(
                            "embedding_batch_size", 16
                        ),
                    ),
                )
                self._set_config_value("performance", "max_workers", 1)

            elif available_mb < 2048:
                # Medium memory configuration
                logger.info("Applying medium-memory configuration adjustments")
                self._set_config_value(
                    "performance",
                    "default_batch_size",
                    min(
                        24,
                        self._config.get("performance", {}).get(
                            "default_batch_size", 32
                        ),
                    ),
                )

            # CPU-based adjustments
            cpu_check = results.get("resource_check", {}).get("cpu", {})
            cores = cpu_check.get("cores", 1)

            if cores == 1:
                self._set_config_value(
                    "performance", "enable_parallel_processing", False
                )

            # Disk space adjustments
            disk_check = results.get("resource_check", {}).get("disk_space", {})
            free_space_mb = disk_check.get("free_space_mb", 0)

            if free_space_mb < 500:
                logger.warning("Low disk space detected, reducing cache sizes")
                self._set_config_value(
                    "performance",
                    "cache_size_limit",
                    min(
                        1000,
                        self._config.get("performance", {}).get(
                            "cache_size_limit", 5000
                        ),
                    ),
                )
                self._set_config_value("output", "auto_generate_diagrams", False)
        except Exception as e:
            logger.warning(f"Failed to apply adjustments from results: {e}")

    def validate_system_resources(
        self, project_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate system resources and return validation results.

        Args:
            project_path: Path to project (uses current project if None)

        Returns:
            Dictionary with validation results
        """
        if project_path is None:
            project_path = get_project_root()

        try:
            validator = ResourceValidator(
                strict_mode=self.get_resource_setting(
                    "strict_resource_validation", False
                )
            )
            results = validator.validate_system_resources(project_path)
            self._resource_validation_results = results
            return results
        except Exception as e:
            logger.error(f"Resource validation failed: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {e}"],
                "warnings": [],
                "recommendations": [],
            }

    def get_analysis_settings(self) -> Dict[str, Any]:
        """Get optimized analysis settings for current project."""
        return {
            "use_streaming": self.get_performance_setting(
                "use_streaming_analysis", True
            ),
            "chunk_size": self.get_performance_setting("chunk_size", 8192),
            "batch_size": self.get_performance_setting("default_batch_size", 32),
            "embedding_batch_size": self.get_performance_setting(
                "embedding_batch_size", 16
            ),
            "enable_parallel": self.get_performance_setting(
                "enable_parallel_processing", True
            ),
            "max_workers": self.get_performance_setting("max_workers", None),
            "strict_binary_detection": self.get_analysis_setting(
                "strict_binary_detection", True
            ),
            "check_file_signatures": self.get_analysis_setting(
                "check_file_signatures", True
            ),
            "python_ast_enabled": self.get_analysis_setting("python_ast_enabled", True),
            "max_ast_file_size_mb": self.get_analysis_setting(
                "max_ast_file_size_mb", 1
            ),
            "similarity_threshold": self.get_embedding_setting(
                "similarity_threshold", 0.7
            ),
            "weak_similarity_threshold": self.get_embedding_setting(
                "weak_similarity_threshold", 0.06
            ),
            "memory_limit_mb": self.get_performance_setting("memory_limit_mb", 4096),
            "cache_size_limit": self.get_performance_setting("cache_size_limit", 5000),
            "resource_validation_enabled": self.get_resource_setting(
                "resource_check_enabled", True
            ),
            "progress_reporting": self.get_output_setting("progress_reporting", True),
        }

    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current configuration and resources."""
        recommendations = []

        # Memory-based recommendations
        if self._resource_validation_results:
            memory_check = self._resource_validation_results.get(
                "resource_check", {}
            ).get("memory", {})
            available_mb = memory_check.get("available_mb", 0)

            if available_mb < 1024:
                recommendations.extend(
                    [
                        "Consider enabling streaming analysis for large files",
                        "Reduce batch sizes for memory efficiency",
                        "Close other applications to free up memory",
                    ]
                )
            elif available_mb < 2048:
                recommendations.append(
                    "Monitor memory usage during large project analysis"
                )

            # CPU-based recommendations
            cpu_check = self._resource_validation_results.get("resource_check", {}).get(
                "cpu", {}
            )
            cores = cpu_check.get("cores", 1)
            if cores < 4:
                recommendations.append(
                    "Analysis may be slower due to limited CPU cores"
                )

            # Disk space recommendations
            disk_check = self._resource_validation_results.get(
                "resource_check", {}
            ).get("disk_space", {})
            free_space_mb = disk_check.get("free_space_mb", 0)
            if free_space_mb < 1000:
                recommendations.append(
                    "Consider freeing up disk space for optimal performance"
                )

        # Configuration-based recommendations
        if not self.get_performance_setting("enable_parallel_processing", True):
            recommendations.append(
                "Enable parallel processing for faster analysis on multi-core systems"
            )

        if not self.get_output_setting("progress_reporting", True):
            recommendations.append(
                "Enable progress reporting for better user experience"
            )

        return recommendations

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            "config_files_loaded": self._config_files_loaded,
            "environment_overrides": self._environment_overrides,
            "performance": self.config.get("performance", {}),
            "analysis": self.config.get("analysis", {}),
            "embedding": self.config.get("embedding", {}),
            "resources": self.config.get("resources", {}),
            "output": self.config.get("output", {}),
            "resource_validation_results": self._resource_validation_results,
            "project_root": get_project_root(),
        }

    def export_config_template(self, file_path: Optional[str] = None) -> str:
        """
        Export configuration template to file.

        Args:
            file_path: Path to export to (default: project_root/analyzer_config_template.json)

        Returns:
            Path to exported file
        """
        if file_path is None:
            file_path = os.path.join(
                get_project_root(), "analyzer_config_template.json"
            )

        template_config = {
            "_template": "Configuration template for project analyzer",
            "_description": "Copy this template and modify as needed, then rename to analyzer_config.json",
            "performance": DEFAULT_CONFIG["performance"],
            "analysis": DEFAULT_CONFIG["analysis"],
            "embedding": DEFAULT_CONFIG["embedding"],
            "resources": DEFAULT_CONFIG["resources"],
            "output": DEFAULT_CONFIG["output"],
            "environment": DEFAULT_CONFIG["environment"],
            "_usage_notes": [
                "This template shows all available configuration options",
                "Remove sections you don't need to customize",
                "File will be merged with existing configuration on load",
                "Environment variables override file settings",
            ],
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(template_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration template exported to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to export configuration template: {e}")
            raise
