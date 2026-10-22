# tests/test_path_centric_migration.py
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cline_utils.dependency_system.core.analysis_state_manager import (
    AnalysisPhase,
    AnalysisStateManager,
)
from cline_utils.dependency_system.core.key_manager import KeyInfo, save_tracker_map
from cline_utils.dependency_system.core.dependency_grid import compress, decompress
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import normalize_path
from cline_utils.dependency_system.utils.tracker_batch_collector import (
    TrackerBatchCollector,
    TrackerUpdate,
)


def test_import_external_relationships_uses_paths_not_stale_keys(tmp_path: Path) -> None:
    """
    Verify that _import_external_relationships imports dependencies keyed by file paths.
    Even if keys in the home tracker on disk shift in the new generation, the dependency
    must stay attached to the true file path and not mis-transfer to an unrelated file.
    """
    collector = TrackerBatchCollector()
    config = ConfigManager()
    project_root = normalize_path(str(tmp_path))

    # 1. Create a home tracker on disk with Generation 0 keys:
    # 1A1: /proj/src/engine.py
    # 1A2: /proj/src/parser.py
    # 1A1 depends on 1A2 with '<'
    module_dir = tmp_path / "src"
    module_dir.mkdir(parents=True, exist_ok=True)
    home_tracker = module_dir / "src_module.md"

    engine_path = normalize_path(str(module_dir / "engine.py"))
    parser_path = normalize_path(str(module_dir / "parser.py"))
    alpha_path = normalize_path(str(module_dir / "alpha.py"))

    row1 = compress("\\<")
    row2 = compress(".\\")
    grid_content = (
        "# src Module Tracker\n\n"
        "---KEY_DEFINITIONS_START---\n"
        f"1A1: {engine_path}\n"
        f"1A2: {parser_path}\n"
        "---KEY_DEFINITIONS_END---\n\n"
        "---GRID_START---\n"
        "X 1A1 1A2\n"
        f"1A1 = {row1}\n"
        f"1A2 = {row2}\n"
        "---GRID_END---\n"
    )
    home_tracker.write_text(grid_content, encoding="utf-8")

    # 2. In Generation 2: a new file alpha.py was added.
    # New keys:
    # 1A1: alpha.py (shifted!)
    # 1A2: engine.py (was 1A1)
    # 1A3: parser.py (was 1A2)
    ki_alpha = KeyInfo(
        key_string="1A1",
        norm_path=alpha_path,
        parent_path=normalize_path(str(module_dir)),
        tier=1,
        is_directory=False,
    )
    ki_engine = KeyInfo(
        key_string="1A2",
        norm_path=engine_path,
        parent_path=normalize_path(str(module_dir)),
        tier=1,
        is_directory=False,
    )
    ki_parser = KeyInfo(
        key_string="1A3",
        norm_path=parser_path,
        parent_path=normalize_path(str(module_dir)),
        tier=1,
        is_directory=False,
    )
    new_kis = [ki_alpha, ki_engine, ki_parser]

    # Empty 3x3 grid for pending update
    initial_rows = [compress("..."), compress("..."), compress("...")]
    other_tracker_file = str(tmp_path / "consumer_module.md")

    # Create a pending update with foreign references needing home tracker import
    update = TrackerUpdate(
        output_file=other_tracker_file,
        tracker_type="mini",
        key_info_list=new_kis,
        grid_rows=initial_rows,
        last_key_edit="none",
        last_grid_edit="none",
        manual_foreign_pins=[engine_path, parser_path, alpha_path],
        module_path=normalize_path(str(tmp_path / "other")),
    )
    collector.add(update)

    # tracker_map.json is authoritative: only home trackers it lists are imported from.
    save_tracker_map([normalize_path(str(home_tracker))])

    # 3. Trigger _import_external_relationships
    collector._import_external_relationships(config, project_root, set())

    # 4. Assert highest_dependency_cache has dependency under (engine_path, parser_path),
    # and NOT under (alpha_path, engine_path)!
    assert (engine_path, parser_path) in collector.highest_dependency_cache
    assert collector.highest_dependency_cache[(engine_path, parser_path)][0] == "<"
    assert (alpha_path, engine_path) not in collector.highest_dependency_cache

    # 5. Run grid consolidation
    collector._consolidate_grids()

    # Verify that the update's row for engine (index 1) has '<' at column parser (index 2)
    row_engine = list(decompress(update.grid_rows[1]))
    assert row_engine[2] == "<", f"Expected '<' at engine->parser, got row: {row_engine}"

    # Verify that row for alpha (index 0) has NO dependency on engine (index 1)
    row_alpha = list(decompress(update.grid_rows[0]))
    assert row_alpha[1] != "<", f"Alpha was incorrectly given engine's dependency: {row_alpha}"


def test_resume_preserves_old_key_map_across_timeout(tmp_path: Path) -> None:
    """
    Verify that when an analysis run times out after key generation, the subsequent
    run resumes without re-rotating global_key_map_old.json, preventing two-cycle erasure.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    current_map = state_dir / "global_key_map.json"
    old_map = state_dir / "global_key_map_old.json"

    # Generation 0 on disk
    old_map.write_text(json.dumps({"G_minus_1": {"path": "root.py"}}), encoding="utf-8")
    current_map.write_text(json.dumps({"G_0": {"path": "engine.py"}}), encoding="utf-8")

    state_mgr = AnalysisStateManager(core_dir=str(tmp_path))
    is_resuming, run_state = state_mgr.start_or_resume_run(str(tmp_path))
    assert not is_resuming

    # Simulate Run 1 rotating keys: current becomes old (G0), new becomes current (G1)
    old_map.write_text(json.dumps({"G_0": {"path": "engine.py"}}), encoding="utf-8")
    current_map.write_text(json.dumps({"G_1": {"path": "engine.py"}}), encoding="utf-8")
    state_mgr.record_phase(AnalysisPhase.EMBEDDINGS_GENERATED)

    # Simulate TIMEOUT occurs here: process ends without committing trackers.

    # Run 2 begins:
    new_mgr = AnalysisStateManager(core_dir=str(tmp_path))
    resuming, state2 = new_mgr.start_or_resume_run(str(tmp_path))

    assert resuming
    assert state2["phase"] == AnalysisPhase.EMBEDDINGS_GENERATED.value

    # Because it is resuming, global_key_map_old.json is NOT overwritten again.
    # It still contains G_0!
    assert "G_0" in old_map.read_text(encoding="utf-8")
    assert "G_1" in current_map.read_text(encoding="utf-8")


def test_should_skip_suggestion_path_first() -> None:
    """Verify that should_skip_suggestion prioritizes path-based lookups directly."""
    from cline_utils.dependency_system.analysis.dependency_suggester import (
        should_skip_suggestion,
    )

    src = normalize_path("/project/src/a.py")
    tgt = normalize_path("/project/src/b.py")
    other = normalize_path("/project/src/other.py")

    path_to_ki = {
        src: KeyInfo("1A1", src, "/project/src", 1, False),
        tgt: KeyInfo("1A2", tgt, "/project/src", 1, False),
        other: KeyInfo("1A3", other, "/project/src", 1, False),
    }

    # existing_state indexed by path pair
    existing_state = {
        (src, tgt): ("x", {"tracker.md"}),
    }

    # Direct match should skip
    assert should_skip_suggestion(src, tgt, existing_state, path_to_ki) is True
    # Reverse match should skip
    assert should_skip_suggestion(tgt, src, existing_state, path_to_ki) is True
    # Unrelated file should NOT skip
    assert should_skip_suggestion(other, tgt, existing_state, path_to_ki) is False

