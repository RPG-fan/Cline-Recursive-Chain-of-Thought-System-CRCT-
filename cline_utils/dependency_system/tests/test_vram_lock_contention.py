"""
The VRAM manager's locks must be held only for quick bookkeeping:
- No CUDA call at all (synchronize, mem_get_info, memory_allocated) while the
  cross-process lock or _allocation_lock is held. A sync waits for every in-flight
  forward pass, and even mem_get_info can stall behind GPU work (the first reranker
  batch warming up kernels for ~20s). Either one made other threads' lock attempts time
  out ("Could not acquire cross-process lock ...", 5s each). Total VRAM is read once.
- Read-only queries do not rewrite the shared registry file unless pruning removed a
  dead process's allocation.
No GPU or model is used: torch is replaced by a fake, and the lock and registry
live in tmp_path, never in the real %TEMP% files.
"""

import json
import os
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import pytest

from cline_utils.dependency_system.utils import resource_validator as rv

GB = 1024**3


@pytest.fixture
def manager(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Tuple[Any, Dict[str, Any]]:
    state: Dict[str, Any] = {"held": False, "cuda_calls": [], "saves": 0}
    mgr = rv.VRAMResourceManager()

    def cuda_call(name: str, result: Any):
        def call(*_a: Any) -> Any:
            owned = mgr._allocation_lock._is_owned()  # pyright: ignore[reportPrivateUsage]
            state["cuda_calls"].append((name, state["held"], owned))
            return result

        return call

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=cuda_call("mem_get_info", (8 * GB, 8 * GB)),
        synchronize=cuda_call("synchronize", None),
        memory_allocated=cuda_call("memory_allocated", 0),
    )
    monkeypatch.setattr(rv, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(rv, "TORCH_AVAILABLE", True)

    real_acquire, real_release = rv.CrossProcessLock.acquire, rv.CrossProcessLock.release

    def acquire(self, *a, **k):
        ok = real_acquire(self, *a, **k)
        state["held"] = state["held"] or ok
        return ok

    def release(self):
        state["held"] = False
        real_release(self)

    monkeypatch.setattr(rv.CrossProcessLock, "acquire", acquire)
    monkeypatch.setattr(rv.CrossProcessLock, "release", release)

    monkeypatch.setattr(mgr, "_physical_vram_gb", None, raising=False)
    monkeypatch.setattr(mgr, "_lockfile_path", str(tmp_path / "vram_lock.lock"))
    monkeypatch.setattr(mgr, "_registry_path", str(tmp_path / "vram_registry.json"))
    monkeypatch.setattr(mgr, "_active_allocations", {})
    monkeypatch.setattr(mgr, "_total_allocated_gb", 0.0)

    real_save = mgr._save_registry  # pyright: ignore[reportPrivateUsage]

    def counting_save(registry):
        state["saves"] += 1
        real_save(registry)

    monkeypatch.setattr(mgr, "_save_registry", counting_save)
    return mgr, state


def _write_registry(mgr: Any, entries: Dict[str, Dict[str, Any]]) -> None:
    with open(mgr._registry_path, "w", encoding="utf-8") as f:
        json.dump(entries, f)


def test_no_cuda_call_while_any_lock_is_held(manager) -> None:
    mgr, state = manager

    mgr.get_available_for_allocation()
    granted, alloc_id = mgr.request_allocation(size_gb=1.0)
    assert granted
    mgr.should_pause_for_backpressure()
    assert mgr.wait_for_available_vram(0.5, timeout=1.0)
    mgr.get_stats()
    mgr.get_recommended_max_workers()
    mgr.release_allocation(alloc_id)

    assert state["cuda_calls"], "the fake GPU was never queried"
    under_lock = [c for c in state["cuda_calls"] if c[1] or c[2]]
    assert under_lock == []


def test_total_vram_is_read_once(manager, monkeypatch) -> None:
    mgr, _ = manager
    assert mgr._get_physical_vram_gb() == pytest.approx(8.0)  # pyright: ignore[reportPrivateUsage]

    def gone(*_a):
        raise AssertionError("total VRAM re-queried")

    monkeypatch.setattr(rv.torch.cuda, "mem_get_info", gone)
    assert mgr._get_physical_vram_gb() == pytest.approx(8.0)  # pyright: ignore[reportPrivateUsage]
    assert mgr.get_reserved_vram_gb() == pytest.approx(1.2)


def test_read_only_queries_do_not_rewrite_the_registry(manager) -> None:
    mgr, state = manager
    live = {"alloc_live": {"allocation_id": "alloc_live", "size_gb": 1.0, "pid": os.getpid()}}
    _write_registry(mgr, live)
    before = os.path.getmtime(mgr._registry_path)

    available = mgr.get_available_for_allocation()
    mgr.should_pause_for_backpressure()
    mgr.wait_for_available_vram(0.5, timeout=1.0)
    mgr.get_stats()

    assert state["saves"] == 0
    assert os.path.getmtime(mgr._registry_path) == before
    # The live allocation is still counted: 8GB free - 1.2GB reserved - 1GB booked.
    assert available == pytest.approx(5.8)


def test_dead_allocation_is_pruned_and_saved_once(manager, monkeypatch) -> None:
    mgr, state = manager
    _write_registry(
        mgr,
        {
            "alloc_live": {"allocation_id": "alloc_live", "size_gb": 1.0, "pid": os.getpid()},
            "alloc_dead": {"allocation_id": "alloc_dead", "size_gb": 2.0, "pid": 999_999},
        },
    )
    monkeypatch.setattr(rv, "is_pid_alive", lambda pid: pid == os.getpid())

    assert mgr.get_available_for_allocation() == pytest.approx(5.8)
    assert state["saves"] == 1
    with open(mgr._registry_path, encoding="utf-8") as f:
        assert set(json.load(f)) == {"alloc_live"}

    mgr.get_available_for_allocation()  # nothing left to prune
    assert state["saves"] == 1


def test_allocation_bookkeeping_still_saves(manager) -> None:
    mgr, state = manager
    granted, alloc_id = mgr.request_allocation(size_gb=1.0)
    assert granted and state["saves"] == 1
    with open(mgr._registry_path, encoding="utf-8") as f:
        assert alloc_id in json.load(f)

    mgr.release_allocation(alloc_id)
    assert state["saves"] == 2
    with open(mgr._registry_path, encoding="utf-8") as f:
        assert json.load(f) == {}


def test_tests_never_touch_the_real_vram_files() -> None:
    """conftest.isolate_vram_registry: a running analyze-project shares these files."""
    import tempfile

    real_lock = os.path.join(tempfile.gettempdir(), "vram_lock.lock")
    real_registry = os.path.join(tempfile.gettempdir(), "vram_registry.json")
    mgr = rv.get_vram_manager()
    assert os.path.normcase(mgr._lockfile_path) != os.path.normcase(real_lock)  # pyright: ignore[reportPrivateUsage]
    assert os.path.normcase(mgr._registry_path) != os.path.normcase(real_registry)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(PermissionError):
        rv.CrossProcessLock(real_lock)
    with pytest.raises(PermissionError):
        os.remove(real_registry)
