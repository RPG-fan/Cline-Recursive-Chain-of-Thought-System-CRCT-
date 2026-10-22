import pytest
import os
from pathlib import Path
from typing import Generator


_REAL_MODELS_DIR = os.path.normcase(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
)


def _in_real_models_dir(path: object) -> bool:
    try:
        p = os.path.normcase(os.path.abspath(os.fspath(path)))  # type: ignore[arg-type]
    except TypeError:
        return False
    return p == _REAL_MODELS_DIR or p.startswith(_REAL_MODELS_DIR + os.sep)


@pytest.fixture(scope="function", autouse=True)
def protect_real_models_dir(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """
    Tests must never delete or (re)download the project's real models (multi-GB GGUF
    files; many CRCT users have metered bandwidth). Any delete or download that
    targets <repo>/models raises instead of touching the file.
    """
    import shutil

    real_remove, real_unlink, real_rmtree = os.remove, os.unlink, shutil.rmtree
    real_path_unlink = Path.unlink

    def guarded(fn):  # type: ignore[no-untyped-def]
        def _wrapper(path, *args, **kwargs):  # type: ignore[no-untyped-def]
            if _in_real_models_dir(path):
                raise PermissionError(f"Test tried to delete a real model file: {path}")
            return fn(path, *args, **kwargs)
        return _wrapper

    monkeypatch.setattr(os, "remove", guarded(real_remove))
    monkeypatch.setattr(os, "unlink", guarded(real_unlink))
    monkeypatch.setattr(shutil, "rmtree", guarded(real_rmtree))
    monkeypatch.setattr(Path, "unlink", guarded(real_path_unlink))

    from cline_utils.dependency_system.analysis import embedding_manager as em

    real_download = em._download_with_retry  # pyright: ignore[reportPrivateUsage]

    def guarded_download(*args, **kwargs):  # type: ignore[no-untyped-def]
        target = kwargs.get("path", args[1] if len(args) > 1 else None)
        if target is not None and _in_real_models_dir(target):
            raise PermissionError(f"Test tried to download into the real models dir: {target}")
        return real_download(*args, **kwargs)

    monkeypatch.setattr(em, "_download_with_retry", guarded_download)
    yield


def _real_vram_files() -> tuple:
    import tempfile

    temp_dir = tempfile.gettempdir()
    return tuple(
        os.path.normcase(os.path.abspath(os.path.join(temp_dir, name)))
        for name in ("vram_lock.lock", "vram_registry.json")
    )


@pytest.fixture(scope="function", autouse=True)
def isolate_vram_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    """
    The VRAM manager coordinates live processes through %TEMP%\\vram_lock.lock and
    %TEMP%\\vram_registry.json. Tests must never lock, rewrite or delete those: a
    running analyze-project would lose its allocations or stall on the lock. Each test
    gets its own lock and registry in tmp_path, and touching the real ones raises.
    """
    from cline_utils.dependency_system.utils import resource_validator as rv

    real_files = _real_vram_files()

    def is_real(path: object) -> bool:
        try:
            return os.path.normcase(os.path.abspath(os.fspath(path))) in real_files  # type: ignore[arg-type]
        except TypeError:
            return False

    vram_dir = tmp_path / "vram"
    vram_dir.mkdir(exist_ok=True)
    mgr = rv.VRAMResourceManager()
    monkeypatch.setattr(mgr, "_lockfile_path", str(vram_dir / "vram_lock.lock"))
    monkeypatch.setattr(mgr, "_registry_path", str(vram_dir / "vram_registry.json"))
    monkeypatch.setattr(mgr, "_active_allocations", {})
    monkeypatch.setattr(mgr, "_total_allocated_gb", 0.0)
    monkeypatch.setattr(mgr, "_physical_vram_gb", None, raising=False)

    real_lock_init = rv.CrossProcessLock.__init__

    def guarded_lock_init(self, lockfile_path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if is_real(lockfile_path):
            raise PermissionError(f"Test tried to use the real VRAM lock: {lockfile_path}")
        real_lock_init(self, lockfile_path, *args, **kwargs)

    monkeypatch.setattr(rv.CrossProcessLock, "__init__", guarded_lock_init)

    def guarded(fn):  # type: ignore[no-untyped-def]
        def _wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            if any(is_real(a) for a in args[:2]):
                raise PermissionError(f"Test tried to modify a real VRAM file: {args[:2]}")
            return fn(*args, **kwargs)
        return _wrapper

    # Chains onto protect_real_models_dir's wrappers (same os functions).
    for name in ("remove", "unlink", "replace"):
        monkeypatch.setattr(os, name, guarded(getattr(os, name)))
    yield


@pytest.fixture(scope="function", autouse=True)
def sandbox_dependency_system(tmp_path: Path) -> Generator[None, None, None]:
    """
    Automatically isolates the dependency system internal state files
    for every test run to prevent modifying the live core project files.
    """
    # Create isolated directories inside the pytest tmp_path
    sandbox_core_dir = tmp_path / "sandbox_core"
    sandbox_core_dir.mkdir(parents=True, exist_ok=True)
    
    state_dir = sandbox_core_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    sandbox_cache_dir = tmp_path / "sandbox_cache"
    sandbox_cache_dir.mkdir(parents=True, exist_ok=True)

    # Import the modules we want to sandbox
    import cline_utils.dependency_system.core.key_manager as km
    import cline_utils.dependency_system.core as core
    import cline_utils.dependency_system.core.analysis_state_manager as asm
    import cline_utils.dependency_system.utils.cache_manager as cm
    import cline_utils.dependency_system.io.transparency_manager as tm
    import cline_utils.dependency_system.utils.config_manager as config_manager

    # Reset ConfigManager singleton to ensure it picks up the correct project root for each test
    config_manager.ConfigManager._instance = None  # pyright: ignore[reportPrivateUsage]

    # Keep track of original values so we can restore them cleanly after the test
    orig_km_file = km.__file__
    orig_core_file = core.__file__
    orig_asm_file = asm.__file__
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
    sandbox_asm_file = os.path.join(str(sandbox_core_dir), "analysis_state_manager.py")
    asm.__file__ = sandbox_asm_file

    # 3. Sandbox cache_manager's CACHE_DIR
    cm.CACHE_DIR = str(sandbox_cache_dir)
    # Clear the existing caches dict to force reloading from the new sandbox path
    if hasattr(cm, "cache_manager"):
        cm.cache_manager.caches.clear()

    # 4. Sandbox transparency_manager registry path
    # Use resolve_state_path internally via mocking tm.REGISTRY_PATH
    sandbox_registry_path = os.path.join(
        str(state_dir), "transparency_registry.json"
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
    asm.__file__ = orig_asm_file
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
