# cline_utils/dependency_system/utils/viz/renderer.py

"""
Logic for invoking 'mmdc' and managing Puppeteer/VRAM for high-resolution rendering.
"""

import json
import logging
import os
import platform
import subprocess
from typing import Any, Dict
from .layout_config import MERMAID_CONFIG, PUPPETEER_CONFIG

logger = logging.getLogger(__name__)


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


def render_mermaid_to_image(mermaid_syntax: str, output_file_path: str):
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
    try:
        process = subprocess.run(
            command,
            input=mermaid_syntax.encode("utf-8"),
            capture_output=True,
            text=False,
            check=True,
            timeout=subprocess_timeout_seconds,
        )
        logger.debug(f"Successfully rendered diagram to {output_file_path}")
        if process.stderr:
            logger.warning(f"Mermaid CLI Warnings:\n{process.stderr.decode('utf-8')}")
    except FileNotFoundError:
        logger.error(
            f"Error: '{mmdc_executable}' command not found. Install with: npm install -g @mermaid-js/mermaid-cli"
        )
    except subprocess.TimeoutExpired:
        logger.error(
            f"Mermaid rendering timed out after {subprocess_timeout_seconds}s."
        )
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error during Mermaid rendering: {e}")
        error_output = e.stderr.decode("utf-8")
        logger.warning(f"Mermaid CLI Error Output:\n{error_output}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during rendering: {e}")
