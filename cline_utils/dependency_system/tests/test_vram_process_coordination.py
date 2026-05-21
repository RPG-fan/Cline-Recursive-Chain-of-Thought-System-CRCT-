import os
import tempfile
import time
import pytest
from unittest.mock import patch

from cline_utils.dependency_system.utils.resource_validator import (
    VRAMResourceManager,
    VRAMAllocation,
    AllocationStatus
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Fixture to ensure a fresh, clean VRAM registry and lock file for each test."""
    temp_dir = tempfile.gettempdir()
    lockfile_path = os.path.join(temp_dir, "vram_lock.lock")
    registry_path = os.path.join(temp_dir, "vram_registry.json")
    
    for path in (lockfile_path, registry_path):
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
                
    yield
    
    for path in (lockfile_path, registry_path):
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def test_cross_process_allocation_coordination():
    """Verify that multiple VRAMResourceManager instances share state and respect limits."""
    # We will instantiate two different VRAMResourceManager instances.
    # Note that in production they share a singleton per-process, but since they
    # read/write to the same shared files, they simulate two separate processes!
    mgr1 = VRAMResourceManager()
    mgr2 = VRAMResourceManager()

    # Let's mock _get_physical_vram_gb and _get_available_vram_gb so they return predictable values
    with patch.object(mgr1, "_get_physical_vram_gb", return_value=16.0), \
         patch.object(mgr1, "_get_available_vram_gb", return_value=16.0), \
         patch.object(mgr2, "_get_physical_vram_gb", return_value=16.0), \
         patch.object(mgr2, "_get_available_vram_gb", return_value=16.0):

        # Calculate initial available space
        # System reservation = physical * 0.10 = 1.6 GB
        # Buffer = 0.5 GB
        # Initial available VRAM = 16.0 - 1.6 = 14.4 GB
        available_before = mgr1.get_available_for_allocation()
        assert abs(available_before - 14.4) < 0.1

        # Manager 1 requests 4.0 GB
        granted1, alloc_id1 = mgr1.request_allocation(size_gb=4.0)
        assert granted1 is True
        assert alloc_id1 is not None

        # Manager 2 should immediately see the decreased available VRAM!
        available_after_1 = mgr2.get_available_for_allocation()
        # available = 14.4 - 4.0 = 10.4 GB
        assert abs(available_after_1 - 10.4) < 0.1

        # Manager 2 requests 8.0 GB (leaves 2.4 GB)
        granted2, alloc_id2 = mgr2.request_allocation(size_gb=8.0)
        assert granted2 is True
        assert alloc_id2 is not None

        # Manager 1 should see decreased available VRAM: 10.4 - 8.0 = 2.4 GB
        available_after_2 = mgr1.get_available_for_allocation()
        assert abs(available_after_2 - 2.4) < 0.1

        # Manager 1 tries to allocate another 3.0 GB (should be denied since only 2.4 GB is available and buffer is needed)
        granted3, alloc_id3 = mgr1.request_allocation(size_gb=3.0, blocking=False)
        assert granted3 is False
        assert alloc_id3 is None

        # Manager 1 releases its 4.0 GB allocation
        assert mgr1.release_allocation(alloc_id1) is True

        # Manager 2 should see available VRAM increased to: 2.4 + 4.0 = 6.4 GB
        available_after_release = mgr2.get_available_for_allocation()
        assert abs(available_after_release - 6.4) < 0.1

        # Clean up the remaining allocation
        assert mgr2.release_allocation(alloc_id2) is True


def test_dead_process_pruning():
    """Verify that allocations registered to dead PIDs are automatically pruned."""
    mgr = VRAMResourceManager()

    with patch.object(mgr, "_get_physical_vram_gb", return_value=16.0), \
         patch.object(mgr, "_get_available_vram_gb", return_value=16.0):

        # Request an allocation (it will register with our current PID which is alive)
        granted1, alloc_id1 = mgr.request_allocation(size_gb=2.0)
        assert granted1 is True

        # Manually corrupt or append a mock dead process allocation to the registry
        registry = mgr._load_registry()
        registry["alloc_dead_12345"] = {
            "allocation_id": "alloc_dead_12345",
            "size_gb": 5.0,
            "pid": 999999,  # Non-existent mock PID
            "requested_at": time.time(),
            "granted_at": time.time(),
            "worker_id": "dead_worker",
            "batch_id": None,
            "status": "granted",
        }
        mgr._save_registry(registry)

        # Under is_pid_alive patch:
        # PIDs: current pid is alive, 999999 is dead.
        def mock_is_pid_alive(pid):
            return pid == os.getpid()

        with patch("cline_utils.dependency_system.utils.resource_validator.is_pid_alive", mock_is_pid_alive):
            # Calling get_available_for_allocation should trigger pruning!
            # If pruned: dead allocation of 5.0 GB is removed, leaving only our 2.0 GB allocation.
            # Available VRAM = 14.4 - 2.0 = 12.4 GB.
            # If not pruned, available VRAM would be 7.4 GB.
            available = mgr.get_available_for_allocation()
            assert abs(available - 12.4) < 0.1

            # Let's check stats to confirm that only 1 allocation is active globally now
            stats = mgr.get_stats()
            assert stats["active_allocations"] == 1
            assert abs(stats["current_allocated_gb"] - 2.0) < 0.1

        # Clean up
        mgr.release_allocation(alloc_id1)
