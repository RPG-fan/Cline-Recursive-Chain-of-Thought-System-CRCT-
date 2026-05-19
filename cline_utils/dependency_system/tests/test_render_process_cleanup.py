"""Scoped render subprocess cleanup (no global taskkill)."""

import builtins
import inspect
import os
from unittest.mock import patch

from cline_utils.dependency_system.utils.viz.renderer import (
    cleanup_orphaned_render_processes,
)


def test_cleanup_source_has_no_global_taskkill_by_image_name() -> None:
    source = inspect.getsource(cleanup_orphaned_render_processes)
    assert '["taskkill", "/F", "/IM"' not in source
    assert "node.exe" not in source


def test_cleanup_terminates_tracked_root_pids_only() -> None:
    with patch(
        "cline_utils.dependency_system.utils.viz.renderer._kill_process_tree"
    ) as mock_kill:
        mock_kill.return_value = True
        cleanup_orphaned_render_processes(
            os.getpid(), tracked_root_pids=[101, 202, 101]
        )

    mock_kill.assert_any_call(101)
    mock_kill.assert_any_call(202)
    assert mock_kill.call_count == 2


def test_kill_process_tree_uses_scoped_taskkill_on_windows_without_psutil() -> None:
    original_import = builtins.__import__

    def import_mock(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name == "psutil":
            raise ImportError("mocked for test")
        return original_import(name, globals, locals, fromlist, level)

    with (
        patch("builtins.__import__", side_effect=import_mock),
        patch(
            "cline_utils.dependency_system.utils.viz.renderer.platform.system",
            return_value="Windows",
        ),
        patch(
            "cline_utils.dependency_system.utils.viz.renderer.subprocess.run"
        ) as mock_run,
    ):
        from cline_utils.dependency_system.utils.viz import renderer as renderer_mod

        result = renderer_mod._kill_process_tree(4242)

    assert result is True
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[:3] == ["taskkill", "/F", "/PID"]
    assert args[3] == "4242"
    assert args[4] == "/T"


def test_popen_render_subprocess_uses_process_group_flags() -> None:
    with patch(
        "cline_utils.dependency_system.utils.viz.renderer.subprocess.Popen"
    ) as mock_popen:
        from cline_utils.dependency_system.utils.viz import renderer as renderer_mod

        renderer_mod._popen_render_subprocess(["mmdc"], "graph TD; A-->B")

    kwargs = mock_popen.call_args.kwargs
    assert kwargs.get("start_new_session") is True or kwargs.get("creationflags")
