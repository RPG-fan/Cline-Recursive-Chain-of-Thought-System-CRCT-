# cline_utils/dependency_system/utils/viz/renderer.py

"""
Logic for invoking 'mmdc' and managing Puppeteer/VRAM for high-resolution rendering.
"""

import json
import logging
import os
import platform
import subprocess
import threading
from typing import Any, Dict, List, Optional, Sequence

from .layout_config import MERMAID_CONFIG, PUPPETEER_CONFIG

logger = logging.getLogger(__name__)

# mmdc / Puppeteer child process basenames (used for parent-tree sweeps only).
_RENDER_PROCESS_NAMES = frozenset({"mmdc", "node", "chrome", "chromium"})

_local_render_root_pids: List[int] = []
_local_render_pid_lock = threading.Lock()


def _find_mmdc_executable() -> str:
    """Dynamically finds the path to the 'mmdc' executable."""
    executable_name = "mmdc"
    try:
        process = subprocess.run(
            ["npm", "config", "get", "prefix"],
            capture_output=True,
            text=True,
            check=True,
            shell=(platform.system() == "Windows"),
        )
        npm_prefix = process.stdout.strip()
        if platform.system() == "Windows":
            potential_path = os.path.join(npm_prefix, "mmdc.cmd")
        else:
            potential_path = os.path.join(npm_prefix, "bin", "mmdc")

        if os.path.isfile(potential_path):
            logger.debug(f"Found 'mmdc' executable at: {potential_path}")
            return potential_path
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.debug(
            "Could not query 'npm' for its prefix. Assuming 'mmdc' is in PATH."
        )

    return executable_name


def _ensure_config_file(file_path: str, config_data: Dict[str, Any]) -> bool:
    """Writes a config file if it doesn't exist."""
    if not os.path.exists(file_path):
        try:
            with open(file_path, "w") as f:
                json.dump(config_data, f, indent=2)
            logger.debug(f"Created config file at: {file_path}")
            return True
        except IOError as e:
            logger.error(f"Could not write config file {file_path}: {e}")
            return False
    return True


def register_render_process_root(
    pid: int, process_registry: Optional[Any] = None
) -> None:
    """Record the root PID of a render subprocess tree for later scoped cleanup."""
    if process_registry is not None:
        process_registry.append(pid)
    with _local_render_pid_lock:
        if pid not in _local_render_root_pids:
            _local_render_root_pids.append(pid)


def drain_local_render_process_roots() -> List[int]:
    """Return and clear render PIDs registered in this process."""
    with _local_render_pid_lock:
        pids = list(_local_render_root_pids)
        _local_render_root_pids.clear()
    return pids


def _popen_render_subprocess(
    command: List[str], mermaid_syntax: str
) -> subprocess.Popen[bytes]:
    """Start mmdc in an isolated process group for targeted teardown."""
    popen_kwargs: Dict[str, Any] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    return subprocess.Popen(command, **popen_kwargs)


def _kill_process_tree(root_pid: int) -> bool:
    """Terminate *root_pid* and its descendants without touching unrelated processes."""
    try:
        import psutil

        try:
            root = psutil.Process(root_pid)
        except psutil.NoSuchProcess:
            return False

        targets = root.children(recursive=True) + [root]
        for proc in targets:
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        _, alive = psutil.wait_procs(targets, timeout=5)
        for proc in alive:
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return True
    except ImportError:
        if platform.system() != "Windows":
            return False
        try:
            subprocess.run(
                ["taskkill", "/F", "/PID", str(root_pid), "/T"],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return True
        except Exception:
            return False


def cleanup_orphaned_render_processes(
    parent_pid: int,
    tracked_root_pids: Optional[Sequence[int]] = None,
) -> None:
    """Kill render subprocess trees owned by this analysis run only.

    Never uses global ``taskkill /IM`` patterns that would terminate unrelated
    developer Node.js or browser processes.
    """
    killed = 0
    root_pids = list(tracked_root_pids or ())
    root_pids.extend(drain_local_render_process_roots())

    seen_roots: set[int] = set()
    for root_pid in root_pids:
        if root_pid in seen_roots:
            continue
        seen_roots.add(root_pid)
        if _kill_process_tree(root_pid):
            killed += 1
            logger.debug("Terminated render process tree rooted at pid=%d", root_pid)

    try:
        import psutil

        try:
            parent = psutil.Process(parent_pid)
        except psutil.NoSuchProcess:
            logger.debug(
                "Cleanup: parent PID %d no longer exists, skipping tree walk.",
                parent_pid,
            )
            if killed:
                logger.info("Cleaned up %d render process tree(s).", killed)
            return

        render_children = []
        for child in parent.children(recursive=True):
            try:
                basename = child.name().lower().split(".")[0]
                if basename in _RENDER_PROCESS_NAMES:
                    render_children.append(child)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        render_child_pids = {child.pid for child in render_children}
        for child in render_children:
            try:
                if child.pid in seen_roots:
                    continue
                parent_render_pid = child.ppid()
                if parent_render_pid in render_child_pids:
                    continue
                logger.warning(
                    "Killing orphaned render process: pid=%d name=%s",
                    child.pid,
                    child.name(),
                )
                if _kill_process_tree(child.pid):
                    killed += 1
                    seen_roots.add(child.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if killed:
            logger.info("Cleaned up %d orphaned render process(es).", killed)
        else:
            logger.debug("Cleanup: no orphaned render processes found.")
    except ImportError:
        if killed:
            logger.info("Cleaned up %d render process tree(s).", killed)
        else:
            logger.warning(
                "Cannot sweep render child processes: psutil is not installed. "
                "Install psutil or ensure render subprocesses exit cleanly."
            )


def render_mermaid_to_image(
    mermaid_syntax: str,
    output_file_path: str,
    process_registry: Optional[Any] = None,
):
    """Renders Mermaid syntax to an image file."""
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mermaid_config_path = os.path.join(output_dir, "mermaid_config.json")
    puppeteer_config_path = os.path.join(output_dir, "puppeteer_config.json")

    _ensure_config_file(mermaid_config_path, MERMAID_CONFIG)
    _ensure_config_file(puppeteer_config_path, PUPPETEER_CONFIG)

    mmdc_executable = _find_mmdc_executable()
    command = [
        mmdc_executable,
        "--input",
        "-",
        "--output",
        os.path.normpath(output_file_path),
        "--backgroundColor",
        "transparent",
        "--configFile",
        os.path.normpath(mermaid_config_path),
        "--puppeteerConfigFile",
        os.path.normpath(puppeteer_config_path),
    ]

    subprocess_timeout_seconds = 900
    proc: Optional[subprocess.Popen[bytes]] = None
    try:
        proc = _popen_render_subprocess(command, mermaid_syntax)
        register_render_process_root(proc.pid, process_registry)
        stdout, stderr = proc.communicate(
            input=mermaid_syntax.encode("utf-8"),
            timeout=subprocess_timeout_seconds,
        )
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, command, output=stdout, stderr=stderr
            )
        logger.debug(f"Successfully rendered diagram to {output_file_path}")
        if stderr:
            logger.warning(f"Mermaid CLI Warnings:\n{stderr.decode('utf-8')}")
    except FileNotFoundError:
        logger.error(
            f"Error: '{mmdc_executable}' command not found. Install with: npm install -g @mermaid-js/mermaid-cli"
        )
    except subprocess.TimeoutExpired:
        if proc is not None:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
        logger.error(
            f"Mermaid rendering timed out after {subprocess_timeout_seconds}s."
        )
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error during Mermaid rendering: {e}")
        error_output = e.stderr.decode("utf-8")
        logger.warning(f"Mermaid CLI Error Output:\n{error_output}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during rendering: {e}")
