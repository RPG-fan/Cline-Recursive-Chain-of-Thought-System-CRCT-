"""
CrossProcessLock queues threads of one process on an in-process lock before the file
lock. When every thread polled the file lock itself (try, sleep 50ms, retry), a
waiter could starve for its whole 5s timeout although each holder kept the lock for
under 10ms ("Could not acquire cross-process lock ..."). Everything here runs in
tmp_path; no GPU or model is used.
"""

import os
import subprocess
import sys
import textwrap
import threading
import time
from typing import List

from cline_utils.dependency_system.utils import resource_validator as rv


def test_waiters_do_not_starve_under_heavy_thread_contention(tmp_path, monkeypatch) -> None:
    """Also guards the lock order: the availability query used to hold
    _allocation_lock while waiting for the file lock that request_allocation held
    while waiting for _allocation_lock."""
    mgr = rv.get_vram_manager()  # lock/registry redirected to tmp_path by conftest
    monkeypatch.setattr(mgr, "_get_available_vram_gb", lambda: 6.0)
    monkeypatch.setattr(mgr, "_physical_vram_gb", 8.0, raising=False)
    stop = threading.Event()

    def poller() -> None:
        while not stop.is_set():
            mgr.get_available_for_allocation()
            time.sleep(0.02)

    def allocator() -> None:
        while not stop.is_set():
            ok, alloc = mgr.request_allocation(0.3, blocking=True, timeout=10)
            time.sleep(0.05)
            if ok:
                mgr.release_allocation(alloc)

    threads = [threading.Thread(target=poller) for _ in range(8)]
    threads += [threading.Thread(target=allocator) for _ in range(6)]
    for t in threads:
        t.start()
    waits: List[float] = []
    try:
        for _ in range(25):
            lock = rv.CrossProcessLock(mgr._lockfile_path)  # pyright: ignore[reportPrivateUsage]
            start = time.perf_counter()
            got = lock.acquire(timeout=5.0)
            waits.append(time.perf_counter() - start)
            assert got, "starved for the whole timeout"
            lock.release()
            time.sleep(0.05)
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert max(waits) < 2.0, waits


def test_timeout_is_honoured_and_leaves_nothing_held(tmp_path) -> None:
    path = str(tmp_path / "q.lock")
    holder = rv.CrossProcessLock(path)
    assert holder.acquire(timeout=1.0)

    result: List[bool] = []
    start = time.perf_counter()
    t = threading.Thread(target=lambda: result.append(rv.CrossProcessLock(path).acquire(timeout=0.3)))
    t.start()
    t.join()
    assert result == [False]
    assert 0.25 <= time.perf_counter() - start < 2.0

    holder.release()
    again = rv.CrossProcessLock(path)
    assert again.acquire(timeout=1.0)  # the timed-out waiter left no thread lock behind
    again.release()


def test_same_path_shares_one_queue_and_other_paths_are_independent(tmp_path) -> None:
    a1 = rv.CrossProcessLock(str(tmp_path / "a.lock"))
    a2 = rv.CrossProcessLock(str(tmp_path / "sub" / ".." / "a.lock"))
    b = rv.CrossProcessLock(str(tmp_path / "b.lock"))
    assert a1._thread_lock is a2._thread_lock  # pyright: ignore[reportPrivateUsage]
    assert a1.acquire(timeout=1.0)
    assert b.acquire(timeout=0.2)  # unrelated lock file: not blocked
    assert not a2.acquire(timeout=0.2)
    b.release()
    a1.release()
    assert a2.acquire(timeout=1.0)
    a2.release()


def test_still_exclusive_across_processes(tmp_path) -> None:
    path = str(tmp_path / "x.lock")
    ready = tmp_path / "ready"
    script = textwrap.dedent(
        f"""
        import os, sys, time
        fd = os.open({path!r}, os.O_RDWR | os.O_CREAT)
        if sys.platform == "win32":
            import msvcrt
            if os.lseek(fd, 0, os.SEEK_END) == 0:
                os.write(fd, b"\\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        open({str(ready)!r}, "w").close()
        time.sleep(3)
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", script])
    try:
        deadline = time.time() + 20
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert ready.exists()
        assert not rv.CrossProcessLock(path).acquire(timeout=0.5)
    finally:
        proc.wait(timeout=20)
    lock = rv.CrossProcessLock(path)
    assert lock.acquire(timeout=2.0)
    lock.release()
    os.remove(str(ready))
