"""
Tests for handle_add_dependency in dependency_processor.py.

Coverage map:
    AD-01  Invalid dep_type -> returns 1, prints error
    AD-02  Bad instance format in source key -> returns 1
    AD-03  Unknown source base key -> returns 1
    AD-04  Ambiguous (multi-instance) source key without specifier -> returns 1
    AD-05  Bad instance format in target key -> rejected, update_tracker not called
    AD-06  Unknown target base key -> rejected, update_tracker not called
    AD-07  Ambiguous target key -> rejected, update_tracker not called
    AD-08  Self-dependency skipped silently -> update_tracker not called
    AD-09  PATH A -- explicit --tracker, non-existent non-mini tracker -> returns 1
    AD-10  PATH A -- explicit --tracker, successful write -> returns 0
    AD-11  PATH B -- broadcast, no eligible tracker -> returns 1 with helpful message
    AD-12  PATH B -- broadcast, updates only trackers containing both keys
    AD-13  PATH B -- broadcast, partial tracker failure -> returns 1
    AD-14  PATH B -- broadcast, all succeed -> returns 0 with summary
"""

import argparse
import os
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

ALLOWED_CHARS = [">", "<", "x", "s", "S", "d", "D"]

SRC_PATH = "/proj/src/alpha.py"
TGT_PATH = "/proj/src/beta.py"
OTHER_PATH = "/proj/src/gamma.py"


def _make_ki(key_string: str, norm_path: str) -> MagicMock:
    ki = MagicMock()
    ki.key_string = key_string
    ki.norm_path = norm_path
    ki.is_directory = False
    ki.parent_path = os.path.dirname(norm_path)
    return ki


SRC_KI = _make_ki("1A", SRC_PATH)
TGT_KI = _make_ki("1B", TGT_PATH)
OTHER_KI = _make_ki("1C", OTHER_PATH)
AMBIG_KI_1 = _make_ki("1D#1", "/proj/src/d1.py")
AMBIG_KI_2 = _make_ki("1D#2", "/proj/src/d2.py")

GLOBAL_MAP: dict[str, Any] = {
    SRC_PATH: SRC_KI,
    TGT_PATH: TGT_KI,
    OTHER_PATH: OTHER_KI,
    "/proj/src/d1.py": AMBIG_KI_1,
    "/proj/src/d2.py": AMBIG_KI_2,
}

_GI_MAP: dict[int, str] = {
    id(SRC_KI): "1A",
    id(TGT_KI): "1B",
    id(OTHER_KI): "1C",
    id(AMBIG_KI_1): "1D#1",
    id(AMBIG_KI_2): "1D#2",
}


def _gi_func(ki: MagicMock, _map: Any) -> str:
    return _GI_MAP.get(id(ki), ki.key_string)


MAIN_DEFS = [("1A", SRC_PATH), ("1B", TGT_PATH)]
MINI_DEFS = [("1A", SRC_PATH), ("1B", TGT_PATH), ("1C", OTHER_PATH)]
UNRELATED_DEFS = [("1C", OTHER_PATH)]


def _make_args(
    *,
    tracker: str | None = None,
    source_key: str = "1A",
    target_key: list[str] | None = None,
    dep_type: str = ">",
) -> argparse.Namespace:
    return argparse.Namespace(
        tracker=tracker,
        source_key=source_key,
        target_key=target_key or ["1B"],
        dep_type=dep_type,
    )


MODULE = "cline_utils.dependency_system.dependency_processor"


@pytest.fixture()
def mocks(tmp_path: Any) -> Generator[dict[str, MagicMock], None, None]:
    """Patches all external I/O so tests remain pure unit tests."""
    config_inst = MagicMock()
    config_inst.get_allowed_dependency_chars.return_value = ALLOWED_CHARS
    config_inst.get_char_priority.return_value = 1

    active_patches: list[Any] = []
    out: dict[str, MagicMock] = {}

    def _start(attr: str, target: Any) -> MagicMock:
        p = patch(f"{MODULE}.{attr}", target)
        m = p.start()
        active_patches.append(p)
        out[attr] = m
        return m

    _start("ConfigManager", MagicMock(return_value=config_inst))
    _start("_load_global_map_or_exit", MagicMock(return_value=GLOBAL_MAP))
    _start("get_project_root", MagicMock(return_value="/proj"))
    _start("get_key_global_instance_string", MagicMock(side_effect=_gi_func))
    _start("get_item_type_for_checklist", MagicMock(return_value="code"))
    _start(
        "build_dependency_suggestions_with_reciprocals",
        MagicMock(side_effect=lambda d: d),
    )
    _start("update_tracker", MagicMock())
    _start("find_all_tracker_paths", MagicMock(return_value=set()))
    _start("read_key_definitions_from_lines", MagicMock(return_value=[]))
    _start("normalize_path", MagicMock(side_effect=lambda p: p))
    _start("add_code_doc_dependency_to_checklist", MagicMock(return_value=None))

    yield out

    for p in active_patches:
        p.stop()


def _run(args: argparse.Namespace) -> int:
    from cline_utils.dependency_system.dependency_processor import handle_add_dependency

    return handle_add_dependency(args)


def _setup_broadcast(
    mocks: dict[str, MagicMock],
    tracker_defs_by_path: dict[str, list[tuple[str, str]]],
) -> None:
    """Wire broadcast-mode mocks: each tracker identified by its first line."""
    mocks["find_all_tracker_paths"].return_value = set(tracker_defs_by_path.keys())

    def _read_defs(lines: list[str]) -> list[tuple[str, str]]:
        path_key = lines[0].strip() if lines else ""
        return tracker_defs_by_path.get(path_key, [])

    mocks["read_key_definitions_from_lines"].side_effect = _read_defs


def _tracker_file(tmp_path: Any, name: str) -> str:
    """Create a dummy tracker whose first line is its own path (sentinel for _read_defs)."""
    p = tmp_path / name
    p.write_text(str(p) + "\n", encoding="utf-8")
    return str(p)


# ===========================================================================
# AD-01  Invalid dep_type
# ===========================================================================


def test_ad01_invalid_dep_type(
    mocks: dict[str, MagicMock], capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid dep_type is caught before any I/O; exits with 1."""
    result = _run(_make_args(dep_type="INVALID"))
    assert result == 1
    assert "Invalid dependency type" in capsys.readouterr().out
    mocks["update_tracker"].assert_not_called()


# ===========================================================================
# AD-02  Bad instance number in source key
# ===========================================================================


def test_ad02_bad_source_instance_format(
    mocks: dict[str, MagicMock], capsys: pytest.CaptureFixture[str]
) -> None:
    """Source key with a non-integer '#suffix' returns 1."""
    result = _run(_make_args(source_key="1A#abc"))
    assert result == 1
    assert "Invalid instance number format" in capsys.readouterr().out


# ===========================================================================
# AD-03  Unknown source base key
# ===========================================================================


def test_ad03_unknown_source_key(
    mocks: dict[str, MagicMock], capsys: pytest.CaptureFixture[str]
) -> None:
    """A source key absent from the global map returns 1."""
    result = _run(_make_args(source_key="ZZZZ"))
    assert result == 1
    assert "not found in global key map" in capsys.readouterr().out


# ===========================================================================
# AD-04  Ambiguous source key (two instances, no specifier)
# ===========================================================================


def test_ad04_ambiguous_source_key(
    mocks: dict[str, MagicMock], capsys: pytest.CaptureFixture[str]
) -> None:
    """Source key matching two KeyInfo objects without '#N' specifier returns 1."""
    result = _run(_make_args(source_key="1D"))
    assert result == 1
    assert "ambiguous" in capsys.readouterr().out.lower()


# ===========================================================================
# AD-05  Bad instance format in target key
# ===========================================================================


def test_ad05_bad_target_instance_rejected(
    mocks: dict[str, MagicMock], capsys: pytest.CaptureFixture[str]
) -> None:
    """An invalid target instance format is rejected; update_tracker is never called."""
    _run(_make_args(target_key=["1B#xyz"]))
    mocks["update_tracker"].assert_not_called()
    assert "Invalid instance number format" in capsys.readouterr().out


# ===========================================================================
# AD-06  Unknown target base key
# ===========================================================================


def test_ad06_unknown_target_key_rejected(
    mocks: dict[str, MagicMock], capsys: pytest.CaptureFixture[str]
) -> None:
    """Unknown target key is rejected gracefully."""
    _run(_make_args(target_key=["ZZZZ"]))
    mocks["update_tracker"].assert_not_called()
    assert "not found in global key map" in capsys.readouterr().out


# ===========================================================================
# AD-07  Ambiguous target key
# ===========================================================================


def test_ad07_ambiguous_target_key_rejected(
    mocks: dict[str, MagicMock], capsys: pytest.CaptureFixture[str]
) -> None:
    """Ambiguous target key is rejected without crashing."""
    _run(_make_args(target_key=["1D"]))
    mocks["update_tracker"].assert_not_called()
    assert "ambiguous" in capsys.readouterr().out.lower()


# ===========================================================================
# AD-08  Self-dependency (source == target path)
# ===========================================================================


def test_ad08_self_dependency_skipped(mocks: dict[str, MagicMock]) -> None:
    """When source and target resolve to the same path no tracker write occurs."""
    _run(_make_args(source_key="1A", target_key=["1A"]))
    mocks["update_tracker"].assert_not_called()


# ===========================================================================
# PATH A -- Explicit --tracker
# ===========================================================================


def test_ad09_explicit_tracker_nonexistent_non_mini(
    mocks: dict[str, MagicMock],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Any,
) -> None:
    """PATH A: A non-existent, non-mini-tracker path returns 1."""
    missing = str(tmp_path / "ghost.md")
    result = _run(_make_args(tracker=missing))
    assert result == 1
    assert "not found" in capsys.readouterr().out.lower()
    mocks["update_tracker"].assert_not_called()


def test_ad10_explicit_tracker_writes_once(
    mocks: dict[str, MagicMock], tmp_path: Any
) -> None:
    """PATH A: A valid existing tracker file triggers exactly one update_tracker call."""
    tracker = tmp_path / "module_relationship_tracker.md"
    tracker.write_text("# tracker\n", encoding="utf-8")

    result = _run(_make_args(tracker=str(tracker)))

    assert result == 0
    mocks["update_tracker"].assert_called_once()
    kw = mocks["update_tracker"].call_args.kwargs
    assert kw["force_apply_suggestions"] is True
    assert kw["apply_ast_overrides"] is False
    assert kw["output_file_suggestion"] == str(tracker)


# ===========================================================================
# PATH B -- Global broadcast
# ===========================================================================


def test_ad11_broadcast_no_eligible_tracker(
    mocks: dict[str, MagicMock],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Any,
) -> None:
    """PATH B: No tracker contains both keys -> returns 1 and advises --tracker."""
    unrelated = _tracker_file(tmp_path, "module_relationship_tracker.md")
    _setup_broadcast(mocks, {unrelated: UNRELATED_DEFS})

    result = _run(_make_args())

    assert result == 1
    assert "--tracker" in capsys.readouterr().out
    mocks["update_tracker"].assert_not_called()


def test_ad12_broadcast_only_eligible_updated(
    mocks: dict[str, MagicMock], tmp_path: Any
) -> None:
    """PATH B: Only trackers whose grids contain both keys are written."""
    main_t = _tracker_file(tmp_path, "module_relationship_tracker.md")
    mini_t = _tracker_file(tmp_path, "src_module.md")
    unrel_t = _tracker_file(tmp_path, "other_module.md")

    _setup_broadcast(
        mocks,
        {
            main_t: MAIN_DEFS,        # eligible: has 1A + 1B
            mini_t: MINI_DEFS,        # eligible: has 1A + 1B (+ extra)
            unrel_t: UNRELATED_DEFS,  # NOT eligible: only 1C
        },
    )

    result = _run(_make_args())

    assert result == 0
    assert mocks["update_tracker"].call_count == 2
    written = {
        c.kwargs["output_file_suggestion"]
        for c in mocks["update_tracker"].call_args_list
    }
    assert main_t in written
    assert mini_t in written
    assert unrel_t not in written


def test_ad13_broadcast_partial_failure(
    mocks: dict[str, MagicMock],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Any,
) -> None:
    """PATH B: One tracker write fails -> exit code 1, warning printed, both attempted."""
    main_t = _tracker_file(tmp_path, "module_relationship_tracker.md")
    mini_t = _tracker_file(tmp_path, "src_module.md")

    _setup_broadcast(mocks, {main_t: MAIN_DEFS, mini_t: MAIN_DEFS})
    mocks["update_tracker"].side_effect = [None, RuntimeError("disk full")]

    result = _run(_make_args())

    assert result == 1
    assert mocks["update_tracker"].call_count == 2
    assert "failed" in capsys.readouterr().out.lower()


def test_ad14_broadcast_all_succeed(
    mocks: dict[str, MagicMock],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Any,
) -> None:
    """PATH B: All eligible trackers updated successfully -> returns 0 with summary."""
    main_t = _tracker_file(tmp_path, "module_relationship_tracker.md")
    mini_t = _tracker_file(tmp_path, "src_module.md")

    _setup_broadcast(mocks, {main_t: MAIN_DEFS, mini_t: MAIN_DEFS})

    result = _run(_make_args())

    assert result == 0
    assert mocks["update_tracker"].call_count == 2
    out = capsys.readouterr().out
    assert "broadcast" in out.lower() or "successfully" in out.lower()
