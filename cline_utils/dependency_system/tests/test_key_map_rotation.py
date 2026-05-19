"""Atomic global key map rotation (Windows-safe os.replace with retries)."""

import json
import os
import threading
from pathlib import Path
from unittest.mock import patch

from cline_utils.dependency_system.core.key_manager import (
    _rotate_global_key_map_atomically,
)


def test_rotate_global_key_map_atomically_moves_file(tmp_path: Path) -> None:
    current = tmp_path / "global_key_map.json"
    old = tmp_path / "global_key_map_old.json"
    current.write_text('{"1A": {}}', encoding="utf-8")

    _rotate_global_key_map_atomically(str(current), str(old))

    assert not current.exists()
    assert old.read_text(encoding="utf-8") == '{"1A": {}}'


def test_rotate_global_key_map_atomically_overwrites_existing_dest(
    tmp_path: Path,
) -> None:
    current = tmp_path / "global_key_map.json"
    old = tmp_path / "global_key_map_old.json"
    current.write_text('{"new": true}', encoding="utf-8")
    old.write_text('{"stale": true}', encoding="utf-8")

    _rotate_global_key_map_atomically(str(current), str(old))

    assert not current.exists()
    assert json.loads(old.read_text(encoding="utf-8")) == {"new": True}


def test_rotate_global_key_map_atomically_retries_on_lock(tmp_path: Path) -> None:
    current = tmp_path / "global_key_map.json"
    old = tmp_path / "global_key_map_old.json"
    current.write_text("{}", encoding="utf-8")
    calls = {"count": 0}
    real_replace = os.replace

    def flaky_replace(src: str, dest: str) -> None:
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError("file is locked")
        real_replace(src, dest)

    with patch(
        "cline_utils.dependency_system.core.key_manager.os.replace",
        side_effect=flaky_replace,
    ):
        with patch("cline_utils.dependency_system.core.key_manager.time.sleep"):
            _rotate_global_key_map_atomically(
                str(current), str(old), max_retries=5, base_delay=0.01
            )

    assert calls["count"] == 3
    assert not current.exists()
    assert old.exists()


def test_concurrent_rotation_completes_without_unhandled_errors(
    tmp_path: Path,
) -> None:
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5)
            for iteration in range(20):
                current = tmp_path / f"w{worker_id}_iter{iteration}_current.json"
                old = tmp_path / f"w{worker_id}_iter{iteration}_old.json"
                current.write_text(json.dumps({"worker": worker_id, "i": iteration}))
                if old.exists():
                    old.unlink()
                _rotate_global_key_map_atomically(
                    str(current), str(old), max_retries=8, base_delay=0.001
                )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    assert errors == []
