"""
update_tracker logs one structured summary line per call, with no per-cell or per-step
lines (it used to write ~15 DEBUG lines per tracker plus one per ignored cell).
Model-free; trackers in tmp_path via the mirror-pair test helper.
"""

import logging

from cline_utils.dependency_system.io import tracker_io
from cline_utils.dependency_system.tests.test_tracker_mirror_pair import _grid

NOISE = (
    "--- update_tracker CALLED", "Suggestion Ignored", "Fast path triggered",
    "Using cached state", "Final sorted list", "Populated suggestions_to_process",
    "Finalizing write", "Returning prepared data", "Keeping existing last_GRID_edit",
    "Cleaning up", "Successfully wrote mini",
)


def test_one_summary_line_per_tracker_update(tmp_path, monkeypatch, caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger=tracker_io.logger.name):
        _grid(
            tmp_path, monkeypatch, {"1A": [("1B", "<")]}, False,
            then=[({"1A": [("1B", "s")]}, False)],  # lower priority: kept existing
        )
    messages = [r.getMessage() for r in caplog.records if r.name == tracker_io.logger.name]
    assert not [m for m in messages if any(n in m for n in NOISE)]
    summaries = [m for m in messages if m.startswith("Tracker summary ")]
    assert len(summaries) == 2  # one per update_tracker call
    assert "kept_existing=1" in summaries[-1]
    assert all("type=" in m and "applied=" in m for m in summaries)
