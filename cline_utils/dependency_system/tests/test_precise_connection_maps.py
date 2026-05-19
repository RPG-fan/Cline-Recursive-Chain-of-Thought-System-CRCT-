from pathlib import Path

from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.io.transparency_manager import (
    TransparencyManager,
    extract_connection_map_metadata,
    overlay_connection_maps,
)
from cline_utils.dependency_system.utils.populate_comments import (
    SymbolReferences,
    TargetSymbol,
    _build_target_symbol_index,
    _collect_symbol_references,
    _resolve_relevant_target_symbols,
    _resolve_target_symbols,
    build_connection_map,
    process_file,
    backup_and_write,
)


def _ki(key: str, path: Path) -> KeyInfo:
    return KeyInfo(
        key_string=key,
        norm_path=str(path).replace("\\", "/"),
        parent_path=str(path.parent).replace("\\", "/"),
        tier=1,
        is_directory=False,
    )


def _dir_ki(key: str, path: Path) -> KeyInfo:
    return KeyInfo(
        key_string=key,
        norm_path=str(path).replace("\\", "/"),
        parent_path=str(path.parent).replace("\\", "/"),
        tier=1,
        is_directory=True,
    )


def test_populate_comments_target_symbol_index_covers_symbols():
    symbol_map_entry = {
        "functions": [{"name": "target_func", "line": 7}],
        "classes": [
            {
                "name": "TargetClass",
                "source_context": {"line_range": [20, 40]},
                "methods": [{"name": "target_method", "line": 24}],
            }
        ],
        "globals_defined": [{"name": "TARGET_CONST", "line": 3}],
        "exports": [{"name": "exported_alias", "line": 50}],
    }

    index = _build_target_symbol_index(symbol_map_entry)

    assert index is not None
    matches = _resolve_target_symbols(
        {
            "target_func",
            "TargetClass",
            "target_method",
            "TARGET_CONST",
            "exported_alias",
        },
        index,
    )

    assert matches == [
        TargetSymbol("TARGET_CONST", 3),
        TargetSymbol("TargetClass", 20),
        TargetSymbol("exported_alias", 50),
        TargetSymbol("target_func", 7),
        TargetSymbol("target_method", 24),
    ]


def test_populate_comments_symbol_references_include_attributes_and_inheritance():
    refs = _collect_symbol_references(
        {
            "name": "SourceClass",
            "attribute_accesses": ["target_method"],
            "closure_dependencies": ["target_func"],
            "inheritance": {"bases": ["TargetClass"], "mro": []},
        },
        "class",
    )

    assert refs == SymbolReferences(
        actionable={"target_method", "target_func", "TargetClass"},
        ambient=set(),
    )


def test_populate_comments_low_value_logger_does_not_create_target_map(tmp_path: Path):
    index = _build_target_symbol_index(
        {"globals_defined": [{"name": "logger", "line": 18}]}
    )

    refs = _collect_symbol_references(
        {
            "name": "close",
            "scope_references": {
                "globals": ["logger"],
                "nonlocals": [],
            },
        },
        "function",
    )

    assert refs is not None
    assert refs == SymbolReferences(actionable=set(), ambient=set())
    assert _resolve_relevant_target_symbols(refs, index) == []


def test_populate_comments_build_connection_map_formats_precise_targets(tmp_path: Path):
    source = _ki("1A", tmp_path / "source.py")
    target = _ki("1B", tmp_path / "target.py")
    unknown = _ki("1C", tmp_path / "unknown.py")
    directory = _dir_ki("1D", tmp_path / "pkg")

    line = build_connection_map(
        "source_func",
        "1A",
        [source, target, unknown, directory],
        "odxx",
        "#",
        target_symbols_by_key={
            "1B": [TargetSymbol("target_func", 7), TargetSymbol("TargetClass", 20)],
            "1C": None,
            "1D": None,
        },
    )

    assert (
        line == "# --- CONNECTION_MAP: 1B(target_func:7|TargetClass:20) {d}, "
        "1C(file:?) {x} --- source_func [AUTO]"
    )


def test_populate_comments_build_connection_map_returns_empty_for_no_entries(
    tmp_path: Path,
):
    source = _ki("1A", tmp_path / "source.py")
    directory = _dir_ki("1B", tmp_path / "pkg")

    line = build_connection_map(
        "source_func",
        "1A",
        [source, directory],
        "ox",
        "#",
    )

    assert line == ""


def test_populate_comments_process_file_dry_run_refreshes_old_flat_map(tmp_path: Path):
    source_path = tmp_path / "source.py"
    target_path = tmp_path / "target.py"
    unknown_path = tmp_path / "unknown.py"
    source_path.write_text(
        "# --- CONNECTION_MAP: 1Bd --- source_func [AUTO]\n"
        "def source_func():\n"
        "    return target_func() + missing_func()\n",
        encoding="utf-8",
    )
    target_path.write_text("def target_func():\n    return 1\n", encoding="utf-8")

    source = _ki("1A", source_path)
    target = _ki("1B", target_path)
    unknown = _ki("1C", unknown_path)
    symbol_data = {
        "functions": [
            {
                "name": "source_func",
                "line": 2,
                "scope_references": {
                    "globals": ["target_func", "missing_func"],
                    "nonlocals": [],
                },
            }
        ],
        "classes": [],
    }
    full_symbol_map = {
        source.norm_path: symbol_data,
        target.norm_path: {"functions": [{"name": "target_func", "line": 1}]},
    }

    result = process_file(
        file_path=source_path,
        source_key="1A",
        key_info_list=[source, target, unknown],
        grid_row="odx",
        tracker_ref="mini_tracker.md",
        entry_from=[],
        exits_to=[],
        symbol_data=symbol_data,
        project_root=tmp_path,
        dry_run=True,
        verbose=False,
        full_symbol_map=full_symbol_map,
    )

    assert result["maps_updated"] == 1
    assert result["generated_connection_maps"] == [
        "# --- CONNECTION_MAP: 1B(target_func:1) {d}, 1C(file:?) {x} "
        "--- source_func [AUTO]"
    ]


def test_populate_comments_process_file_refreshes_class_methods(tmp_path: Path):
    source_path = tmp_path / "source.py"
    target_path = tmp_path / "target.py"
    source_path.write_text(
        "class Source:\n"
        "    # --- CONNECTION_MAP: 1Bd --- is_ready [AUTO]\n"
        "    @property\n"
        "    def is_ready(self):\n"
        "        return target_func()\n",
        encoding="utf-8",
    )
    target_path.write_text("def target_func():\n    return True\n", encoding="utf-8")

    source = _ki("1A", source_path)
    target = _ki("1B", target_path)
    directory = _dir_ki("1C", tmp_path / "pkg")
    symbol_data = {
        "functions": [],
        "classes": [
            {
                "name": "Source",
                "line": 1,
                "methods": [
                    {
                        "name": "is_ready",
                        "source_context": {"line_range": [4, 5]},
                        "scope_references": {
                            "globals": ["target_func"],
                            "nonlocals": [],
                        },
                    }
                ],
            }
        ],
    }
    full_symbol_map = {
        source.norm_path: symbol_data,
        target.norm_path: {"functions": [{"name": "target_func", "line": 1}]},
    }

    result = process_file(
        file_path=source_path,
        source_key="1A",
        key_info_list=[source, target, directory],
        grid_row="odx",
        tracker_ref="mini_tracker.md",
        entry_from=[],
        exits_to=[],
        symbol_data=symbol_data,
        project_root=tmp_path,
        dry_run=True,
        verbose=False,
        full_symbol_map=full_symbol_map,
    )

    assert (
        "    # --- CONNECTION_MAP: 1B(target_func:1) {d} --- is_ready [AUTO]"
        in result["generated_connection_maps"]
    )
    assert all("1C(" not in line for line in result["generated_connection_maps"])


def test_transparency_extract_connection_map_metadata_parses_precise_entries():
    content = (
        "# --- CONNECTION_MAP: 1B(target_func:7|TargetClass:20) {d}, "
        "1C(file:?) {x} --- source_func [AUTO]\n"
        "def source_func():\n"
        "    pass\n"
    )

    records = extract_connection_map_metadata(content)

    assert records == [
        {
            "source_symbol": "source_func",
            "source_line": 2,
            "target_key": "1B",
            "target_symbol": "target_func",
            "target_line": 7,
            "dep_char": "d",
        },
        {
            "source_symbol": "source_func",
            "source_line": 2,
            "target_key": "1B",
            "target_symbol": "TargetClass",
            "target_line": 20,
            "dep_char": "d",
        },
        {
            "source_symbol": "source_func",
            "source_line": 2,
            "target_key": "1C",
            "target_symbol": "file",
            "target_line": None,
            "dep_char": "x",
        },
    ]


def test_transparency_virtualizes_connection_maps_and_overlays_hidden_layer(
    tmp_path: Path,
):
    source_path = tmp_path / "source.py"
    source_path.write_text(
        "# --- STATION_HEADER: 1A ---\n"
        "# --- CONNECTION_MAP: 1B(target_func:7|TargetClass:20) {d} "
        "--- source_func [AUTO]\n"
        "def source_func():\n"
        "    return target_func()\n",
        encoding="utf-8",
    )

    manager = TransparencyManager(str(tmp_path / "transparency_registry.json"))

    assert manager.virtualize_connection_maps(str(source_path))

    clean_content = source_path.read_text(encoding="utf-8")
    assert "CONNECTION_MAP:" not in clean_content
    assert "# --- STATION_HEADER: 1A ---" in clean_content

    metadata = manager.get_file_metadata(str(source_path))
    assert metadata is not None
    assert metadata["connection_maps"] == [
        {
            "source_symbol": "source_func",
            "source_line": 2,
            "target_key": "1B",
            "target_symbol": "target_func",
            "target_line": 7,
            "dep_char": "d",
        },
        {
            "source_symbol": "source_func",
            "source_line": 2,
            "target_key": "1B",
            "target_symbol": "TargetClass",
            "target_line": 20,
            "dep_char": "d",
        },
    ]

    assert (
        extract_connection_map_metadata(clean_content, metadata)
        == metadata["connection_maps"]
    )
    overlaid = overlay_connection_maps(clean_content, metadata)
    assert "CONNECTION_MAP: 1B(target_func:7|TargetClass:20) {d}" in overlaid
    assert overlaid.splitlines()[1].startswith("# --- CONNECTION_MAP:")


def test_transparency_virtualize_clears_stale_maps_when_populate_emits_none(
    tmp_path: Path,
):
    source_path = tmp_path / "source.py"
    source_path.write_text("def source_func():\n    return 1\n", encoding="utf-8")
    manager = TransparencyManager(str(tmp_path / "transparency_registry.json"))

    manager._write_connection_map_metadata(
        str(source_path),
        source_path.read_text(encoding="utf-8"),
        [
            {
                "source_symbol": "source_func",
                "source_line": 1,
                "target_key": "1B",
                "target_symbol": "old_target",
                "target_line": 9,
                "dep_char": "d",
            }
        ],
        [
            {
                "line": 1,
                "content": "# --- CONNECTION_MAP: 1B(old_target:9) {d} --- source_func [AUTO]",
            }
        ],
    )

    assert manager.virtualize_connection_maps(str(source_path), clear_if_absent=True)
    metadata = manager.get_file_metadata(str(source_path))
    assert metadata is not None
    assert "connection_maps" not in metadata
    assert "connection_map_lines" not in metadata


def test_backup_and_write_cleanup(tmp_path: Path):
    # Setup source file and project root structure
    project_root = tmp_path
    file_path = project_root / "test_module.py"
    file_path.write_text("original content", encoding="utf-8")

    # Create backup directory and populate it with old timestamped backups
    backup_dir = project_root / ".comment_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    old_backup_1 = backup_dir / "test_module.py.20251112_010203.bak"
    old_backup_2 = backup_dir / "test_module.py.20251113_040506.bak"
    old_backup_other = backup_dir / "other_module.py.20251114_070809.bak"
    unrelated_file = backup_dir / "unrelated.txt"

    old_backup_1.write_text("old content 1", encoding="utf-8")
    old_backup_2.write_text("old content 2", encoding="utf-8")
    old_backup_other.write_text("old other content", encoding="utf-8")
    unrelated_file.write_text("some text", encoding="utf-8")

    # Execute backup_and_write
    new_content = "new updated content"
    backup_and_write(file_path, new_content, project_root)

    # Assertions
    # 1. Source file has the new content
    assert file_path.read_text(encoding="utf-8") == new_content

    # 2. Single backup file is created
    single_backup_path = backup_dir / "test_module.py.bak"
    assert single_backup_path.exists()
    assert single_backup_path.read_text(encoding="utf-8") == "original content"

    # 3. All old timestamped backups have been cleaned up
    assert not old_backup_1.exists()
    assert not old_backup_2.exists()
    assert not old_backup_other.exists()

    # 4. Unrelated files/directories are NOT touched or deleted
    assert unrelated_file.exists()
    assert unrelated_file.read_text(encoding="utf-8") == "some text"


def test_bulk_prune_stale_virtual_maps(tmp_path: Path):
    manager = TransparencyManager(str(tmp_path / "transparency_registry.json"))

    file_a = tmp_path / "file_a.py"
    file_b = tmp_path / "file_b.py"
    file_c = tmp_path / "file_c.py"

    file_a.write_text("def a(): pass", encoding="utf-8")
    file_b.write_text("def b(): pass", encoding="utf-8")
    file_c.write_text("def c(): pass", encoding="utf-8")

    # Write connection maps for A, B, and C
    manager._write_connection_map_metadata(
        str(file_a),
        "def a(): pass",
        [
            {
                "source_symbol": "a",
                "source_line": 1,
                "target_key": "1B",
                "target_symbol": "b",
                "target_line": 1,
                "dep_char": "d",
            }
        ],
        [{"line": 1, "content": "# --- CONNECTION_MAP: 1B(b:1) {d} --- a [AUTO]"}],
    )

    manager._write_connection_map_metadata(
        str(file_b),
        "def b(): pass",
        [
            {
                "source_symbol": "b",
                "source_line": 1,
                "target_key": "1A",
                "target_symbol": "a",
                "target_line": 1,
                "dep_char": "d",
            }
        ],
        [{"line": 1, "content": "# --- CONNECTION_MAP: 1A(a:1) {d} --- b [AUTO]"}],
    )

    manager._write_connection_map_metadata(
        str(file_c),
        "def c(): pass",
        [
            {
                "source_symbol": "c",
                "source_line": 1,
                "target_key": "1A",
                "target_symbol": "a",
                "target_line": 1,
                "dep_char": "d",
            }
        ],
        [{"line": 1, "content": "# --- CONNECTION_MAP: 1A(a:1) {d} --- c [AUTO]"}],
    )

    # Execute bulk prune where only file_a was processed with maps, and all files were processed
    files_processed = {str(file_a), str(file_b), str(file_c)}
    files_with_maps = {str(file_a)}

    manager.bulk_prune_stale_virtual_maps(files_processed, files_with_maps)

    # Assertions
    meta_a = manager.get_file_metadata(str(file_a))
    meta_b = manager.get_file_metadata(str(file_b))
    meta_c = manager.get_file_metadata(str(file_c))

    assert meta_a is not None
    assert "connection_maps" in meta_a
    assert meta_b is not None
    assert "connection_maps" not in meta_b
    assert meta_c is not None
    assert "connection_maps" not in meta_c
