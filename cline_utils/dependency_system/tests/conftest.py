import pytest
import os
from pathlib import Path
from typing import Generator


@pytest.fixture(scope="function", autouse=True)
def sandbox_dependency_system(tmp_path: Path) -> Generator[None, None, None]:
    """
    Automatically isolates the dependency system internal state files
    for every test run to prevent modifying the live core project files.
    """
    # Create isolated directories inside the pytest tmp_path
    sandbox_core_dir = tmp_path / "sandbox_core"
    sandbox_core_dir.mkdir(parents=True, exist_ok=True)

    sandbox_cache_dir = tmp_path / "sandbox_cache"
    sandbox_cache_dir.mkdir(parents=True, exist_ok=True)

    # Import the modules we want to sandbox
    import cline_utils.dependency_system.core.key_manager as km
    import cline_utils.dependency_system.core as core
    import cline_utils.dependency_system.utils.cache_manager as cm
    import cline_utils.dependency_system.io.transparency_manager as tm
    import cline_utils.dependency_system.utils.config_manager as config_manager

    # Reset ConfigManager singleton to ensure it picks up the correct project root for each test
    config_manager.ConfigManager._instance = None  # pyright: ignore[reportPrivateUsage]

    # Keep track of original values so we can restore them cleanly after the test
    orig_km_file = km.__file__
    orig_core_file = core.__file__
    orig_cm_cache_dir = getattr(cm, "CACHE_DIR", None)
    orig_tm_registry_path = getattr(tm, "REGISTRY_PATH", None)
    orig_tm_defaults = tm.TransparencyManager.__init__.__defaults__

    # 1. Sandbox key_manager.py paths by redirecting its __file__ attribute.
    # We point __file__ to a dummy file within our sandbox core directory.
    sandbox_km_file = os.path.join(str(sandbox_core_dir), "key_manager.py")
    km.__file__ = sandbox_km_file

    # 2. Sandbox core package path for resource_validator's validation cache.
    sandbox_core_init = os.path.join(str(sandbox_core_dir), "__init__.py")
    core.__file__ = sandbox_core_init

    # 3. Sandbox cache_manager's CACHE_DIR
    cm.CACHE_DIR = str(sandbox_cache_dir)
    # Clear the existing caches dict to force reloading from the new sandbox path
    if hasattr(cm, "cache_manager"):
        cm.cache_manager.caches.clear()

    # 4. Sandbox transparency_manager registry path
    sandbox_registry_path = os.path.join(
        str(sandbox_core_dir), "transparency_registry.json"
    )
    tm.REGISTRY_PATH = sandbox_registry_path
    # Override TransparencyManager.__init__ default registry_path value
    tm.TransparencyManager.__init__.__defaults__ = (sandbox_registry_path,)

    # If a global _manager_instance already exists, clear or recreate it
    if hasattr(tm, "_manager_instance"):
        tm._manager_instance = None  # pyright: ignore[reportPrivateUsage]

    yield

    # Restore everything to original values to prevent leakage across sessions
    km.__file__ = orig_km_file
    core.__file__ = orig_core_file
    if orig_cm_cache_dir is not None:
        cm.CACHE_DIR = orig_cm_cache_dir
    if orig_tm_registry_path is not None:
        tm.REGISTRY_PATH = orig_tm_registry_path
    tm.TransparencyManager.__init__.__defaults__ = orig_tm_defaults
    if hasattr(tm, "_manager_instance"):
        tm._manager_instance = None  # pyright: ignore[reportPrivateUsage]
    if hasattr(cm, "cache_manager"):
        cm.cache_manager.caches.clear()
    config_manager.ConfigManager._instance = None  # pyright: ignore[reportPrivateUsage]

    try:
        from cline_utils.dependency_system.analysis import embedding_manager as em

        em._model_instance = None  # pyright: ignore[reportPrivateUsage]
        em._selected_model_config = None  # pyright: ignore[reportPrivateUsage]
    except Exception:
        pass
