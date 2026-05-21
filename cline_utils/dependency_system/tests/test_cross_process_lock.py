import os
import tempfile
import pytest

from cline_utils.dependency_system.utils.resource_validator import CrossProcessLock


def test_lock_acquire_release():
    """Verify basic lock acquisition and release."""
    temp_dir = tempfile.gettempdir()
    lock_path = os.path.join(temp_dir, "test_basic_lock.lock")
    if os.path.exists(lock_path):
        try:
            os.unlink(lock_path)
        except OSError:
            pass

    lock = CrossProcessLock(lock_path)
    assert lock.acquire(timeout=2.0) is True
    # Verify file is created and has at least 1 byte (for Windows)
    assert os.path.exists(lock_path)
    assert os.path.getsize(lock_path) > 0

    lock.release()
    # Can acquire again after release
    assert lock.acquire(timeout=2.0) is True
    lock.release()

    # Clean up
    try:
        os.unlink(lock_path)
    except OSError:
        pass


def test_lock_exclusivity():
    """Verify that lock is exclusive (other instances cannot acquire it concurrently)."""
    temp_dir = tempfile.gettempdir()
    lock_path = os.path.join(temp_dir, "test_exclusive_lock.lock")
    if os.path.exists(lock_path):
        try:
            os.unlink(lock_path)
        except OSError:
            pass

    lock1 = CrossProcessLock(lock_path)
    lock2 = CrossProcessLock(lock_path)

    # Acquire lock1
    assert lock1.acquire(timeout=2.0) is True

    # Try to acquire lock2 (should fail because lock1 holds it)
    assert lock2.acquire(timeout=0.5) is False

    # Release lock1
    lock1.release()

    # Now lock2 should be able to acquire it
    assert lock2.acquire(timeout=2.0) is True
    lock2.release()

    # Clean up
    try:
        os.unlink(lock_path)
    except OSError:
        pass


def test_lock_context_manager():
    """Verify that context manager works as expected and auto-releases."""
    temp_dir = tempfile.gettempdir()
    lock_path = os.path.join(temp_dir, "test_ctx_lock.lock")
    if os.path.exists(lock_path):
        try:
            os.unlink(lock_path)
        except OSError:
            pass

    with CrossProcessLock(lock_path) as lock1:
        assert lock1._fd is not None
        # Try to acquire another lock concurrently
        lock2 = CrossProcessLock(lock_path)
        assert lock2.acquire(timeout=0.5) is False

    # After exiting block, lock should be released, and lock2 can acquire it
    lock2 = CrossProcessLock(lock_path)
    assert lock2.acquire(timeout=2.0) is True
    lock2.release()

    # Clean up
    try:
        os.unlink(lock_path)
    except OSError:
        pass
