"""
Tracker files are written to a temp file and swapped in with a short retry. Writing in
place failed on Windows with "[Errno 22] Invalid argument" while another process
briefly held module_relationship_tracker.md (e.g. a scan after the backup copy).
"""

import errno
import os
import sys
import threading
import time

import pytest

from cline_utils.dependency_system.io import tracker_io

_REAL_SLEEP = time.sleep  # captured before the autouse fixture patches time.sleep


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(tracker_io.time, "sleep", lambda s: None)


def _leftover_temps(directory) -> list:
    return [n for n in os.listdir(directory) if n.endswith(".tmp")]


def test_transient_errors_are_retried(tmp_path, monkeypatch) -> None:
    target = tmp_path / "module_relationship_tracker.md"
    target.write_text("old", encoding="utf-8")
    real_replace = os.replace
    calls = []

    def flaky_replace(src, dst):
        calls.append(dst)
        if len(calls) <= 2:
            raise OSError(errno.EINVAL, "Invalid argument", str(dst))
        real_replace(src, dst)

    monkeypatch.setattr(tracker_io.os, "replace", flaky_replace)
    tracker_io._atomic_write_text(str(target), "new content\n")  # pyright: ignore[reportPrivateUsage]
    assert target.read_text(encoding="utf-8") == "new content\n"
    assert len(calls) == 3
    assert _leftover_temps(tmp_path) == []


def test_persistent_failure_leaves_the_old_file_intact(tmp_path, monkeypatch) -> None:
    target = tmp_path / "doc_tracker.md"
    target.write_text("previous content", encoding="utf-8")

    def always_locked(src, dst):
        raise PermissionError(errno.EACCES, "Access is denied", str(dst))

    monkeypatch.setattr(tracker_io.os, "replace", always_locked)
    with pytest.raises(PermissionError):
        tracker_io._atomic_write_text(str(target), "new")  # pyright: ignore[reportPrivateUsage]
    assert target.read_text(encoding="utf-8") == "previous content"  # never truncated
    assert _leftover_temps(tmp_path) == []


def test_non_transient_errors_are_not_retried(tmp_path, monkeypatch) -> None:
    calls = []

    def missing_dir(src, dst):
        calls.append(dst)
        raise FileNotFoundError(errno.ENOENT, "No such file", str(dst))

    monkeypatch.setattr(tracker_io.os, "replace", missing_dir)
    with pytest.raises(FileNotFoundError):
        tracker_io._atomic_write_text(str(tmp_path / "x.md"), "new")  # pyright: ignore[reportPrivateUsage]
    assert len(calls) == 1


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing semantics")
def test_real_open_handle_on_windows_is_waited_out(tmp_path, monkeypatch) -> None:
    """Another handle on the target blocks the swap until it is released."""
    monkeypatch.setattr(tracker_io.time, "sleep", _REAL_SLEEP)
    target = tmp_path / "module_relationship_tracker.md"
    target.write_text("old", encoding="utf-8")
    holder = open(target, "r", encoding="utf-8")  # no FILE_SHARE_DELETE: blocks replace
    threading.Timer(0.12, holder.close).start()
    tracker_io._atomic_write_text(str(target), "new\n")  # pyright: ignore[reportPrivateUsage]
    assert target.read_text(encoding="utf-8") == "new\n"
    assert _leftover_temps(tmp_path) == []
