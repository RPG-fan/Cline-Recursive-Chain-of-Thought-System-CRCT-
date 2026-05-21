import os
import pytest
from cline_utils.dependency_system.analysis.dependency_suggester import _find_and_parse_tsconfig


def test_find_and_parse_tsconfig_casing_mismatch(tmp_path):
    # Create start_dir and project_root
    project_root = tmp_path / "my_project"
    start_dir = project_root / "src" / "sub"
    start_dir.mkdir(parents=True, exist_ok=True)

    # Write a tsconfig.json in a parent directory of the project root to ensure that if it walks past
    # the project root, it might find this unrelated tsconfig.
    unrelated_tsconfig = tmp_path / "tsconfig.json"
    unrelated_tsconfig.write_text(
        '{"compilerOptions": {"target": "es5"}}', encoding="utf-8"
    )

    # Make sure we don't have tsconfig inside the project root or start_dir
    root_str = str(project_root)
    start_str = str(start_dir)

    # Mismatch the drive letter casing
    if len(root_str) > 1 and root_str[1] == ":":
        drive = root_str[0]
        mismatched_root_str = (
            (drive.lower() if drive.isupper() else drive.upper()) + root_str[1:]
        )
    else:
        mismatched_root_str = root_str

    if len(start_str) > 1 and start_str[1] == ":":
        drive = start_str[0]
        mismatched_start_str = (
            (drive.lower() if drive.isupper() else drive.upper()) + start_str[1:]
        )
    else:
        mismatched_start_str = start_str

    # Call _find_and_parse_tsconfig with start_dir and mismatched_root_str
    # Since there is no tsconfig within the project root, it should stop at project_root_norm and return None.
    # If the casing bug is present, it will walk up past the project root, find the unrelated_tsconfig, and return it.
    result = _find_and_parse_tsconfig(mismatched_start_str, mismatched_root_str)

    assert (
        result is None
    ), "Should break cleanly at project root and not find unrelated_tsconfig outside the root"


def test_find_and_parse_tsconfig_finds_correct_config(tmp_path):
    project_root = tmp_path / "my_project"
    start_dir = project_root / "src" / "sub"
    start_dir.mkdir(parents=True, exist_ok=True)

    # Write tsconfig.json in project root
    project_tsconfig = project_root / "tsconfig.json"
    project_tsconfig.write_text(
        '{"compilerOptions": {"target": "es6"}}', encoding="utf-8"
    )

    root_str = str(project_root)
    start_str = str(start_dir)

    if len(root_str) > 1 and root_str[1] == ":":
        drive = root_str[0]
        mismatched_root_str = (
            (drive.lower() if drive.isupper() else drive.upper()) + root_str[1:]
        )
    else:
        mismatched_root_str = root_str

    if len(start_str) > 1 and start_str[1] == ":":
        drive = start_str[0]
        mismatched_start_str = (
            (drive.lower() if drive.isupper() else drive.upper()) + start_str[1:]
        )
    else:
        mismatched_start_str = start_str

    result = _find_and_parse_tsconfig(mismatched_start_str, mismatched_root_str)

    assert (
        result is not None
    ), "Should find the correct tsconfig inside the project root"
    config_path, parsed_data = result
    assert parsed_data["compilerOptions"]["target"] == "es6"
