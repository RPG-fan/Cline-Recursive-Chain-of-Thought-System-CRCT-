"""Top-level root path limit (A-Z) in generate_keys."""

from pathlib import Path

import pytest

from cline_utils.dependency_system.core.key_manager import (
    MAX_TOP_LEVEL_ROOTS,
    KeyGenerationError,
    generate_keys,
)


def _make_root_dirs(base: Path, count: int) -> list[str]:
    paths: list[str] = []
    for i in range(count):
        root = base / f"root_{i:02d}"
        root.mkdir()
        paths.append(str(root))
    return paths


def test_generate_keys_rejects_more_than_26_top_level_roots(tmp_path: Path) -> None:
    root_paths = _make_root_dirs(tmp_path, MAX_TOP_LEVEL_ROOTS + 1)

    with pytest.raises(KeyGenerationError, match="at most 26 top-level root paths"):
        generate_keys(
            root_paths,
            excluded_dirs=set(),
            excluded_extensions={".pyc"},
            precomputed_excluded_paths=set(),
        )


def test_generate_keys_assigns_through_z_for_26_top_level_roots(tmp_path: Path) -> None:
    root_paths = _make_root_dirs(tmp_path, MAX_TOP_LEVEL_ROOTS)

    path_to_key_info, _ = generate_keys(
        root_paths,
        excluded_dirs=set(),
        excluded_extensions={".pyc"},
        precomputed_excluded_paths=set(),
    )

    top_level_keys = {
        info.key_string.split("#", 1)[0]
        for info in path_to_key_info.values()
        if info.is_directory and info.tier == 1
    }
    assert top_level_keys == {
        f"1{chr(ord('A') + i)}" for i in range(MAX_TOP_LEVEL_ROOTS)
    }
    assert "1[" not in top_level_keys


def test_generate_keys_filters_nested_roots(tmp_path: Path) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    nested_dir = parent_dir / "nested"
    nested_dir.mkdir()

    (parent_dir / "parent_file.txt").write_text("parent content", encoding="utf-8")
    (nested_dir / "nested_file.txt").write_text("nested content", encoding="utf-8")

    root_paths = [str(parent_dir), str(nested_dir)]

    path_to_key_info, _ = generate_keys(
        root_paths,
        excluded_dirs=set(),
        excluded_extensions=set(),
        precomputed_excluded_paths=set(),
    )

    from cline_utils.dependency_system.utils.path_utils import normalize_path

    norm_parent_file = normalize_path(str(parent_dir / "parent_file.txt"))
    norm_nested_file = normalize_path(str(nested_dir / "nested_file.txt"))

    assert norm_parent_file in path_to_key_info
    assert norm_nested_file in path_to_key_info

    nested_file_key = path_to_key_info[norm_nested_file].key_string
    # Under the nested roots exclusion logic, nested_dir is processed as a separate top-level root (1B)
    # and skipped during the parent_dir (1A) processing.
    assert nested_file_key.startswith("1B")
    assert not nested_file_key.startswith("1A")
