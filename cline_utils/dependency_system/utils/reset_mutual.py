# utils/reset_mutual.py

"""
One-time migration: reset every mutual ('x') dependency cell in every tracker to
the placeholder character ('p').

Why this exists
---------------
A since-fixed defect in ``io/tracker_io.py`` turned directional import edges into
'x' in both cells of a pair, and the old JS/TS resolver linked to compiled ``.js``
twins. CRCT never overwrites 'x' (the suggester skips pairs holding 'x'/'n' and the
tracker write path keeps higher-priority characters; 'x' has priority 6), and it
keeps no per-cell provenance, so valid and bogus 'x' cells cannot be told apart.
This module performs a blanket reset instead; ``analyze-project`` then restores
valid code relations and doc pairs are re-verified (live or ``resolve-placeholders``).

Rules
-----
* Only off-diagonal cells holding exactly 'x' change (-> 'p'). The diagonal 'o' and
  every other character ('n', '<', '>', 'd', 's', 'S', 'p') are never touched.
* Trackers are read with ``read_tracker_file_structured`` and written back in ONE batch
  through ``TrackerBatchCollector.commit_all`` (its temp backups + rollback, permanent
  backups, standard writers and cache invalidation). The collector's grid-consolidation
  pass is disabled for this use (see ``_PureRewriteCollector``): a reset must not re-derive cells.
* Every tracker that is modified is backed up first (``backup_tracker_file`` plus a
  run-scoped snapshot, because ``backup_tracker_file`` keeps only the 2 newest backups
  per file *name* and many mini trackers share a name).
* After each write the tracker is re-read and verified: identical definitions/headers
  and a grid that differs from the original only by x -> p. On any mismatch the
  tracker is restored from its backup.

The functions take all external inputs (tracker paths, global key map, doc roots)
as arguments so they can be exercised without live CRCT state.
"""

import datetime
import logging
import os
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

from cline_utils.dependency_system.core.dependency_grid import (
    PLACEHOLDER_CHAR,
    compress,
    decompress,
)
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.io.tracker_io import (
    _read_manual_foreign_pins_from_lines,  # pyright: ignore[reportPrivateUsage]
)
from cline_utils.dependency_system.io.update_mini_tracker import get_mini_tracker_data
from cline_utils.dependency_system.utils.tracker_batch_collector import (
    TrackerBatchCollector,
    TrackerUpdate,
)
from cline_utils.dependency_system.utils.cache_manager import (
    invalidate_dependent_entries,
)
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.path_utils import is_subpath, normalize_path
from cline_utils.dependency_system.utils.tracker_utils import (
    clear_runtime_aggregation_cache,
    read_tracker_file_structured,
)

logger = logging.getLogger(__name__)

MUTUAL_CHAR = "x"
_DOC_EXTENSIONS = (".md", ".rst", ".txt", ".adoc", ".mdx")

# Pair categories (unordered pair of paths)
CAT_DOC_DOC = "doc-doc"
CAT_DOC_CODE = "doc-code"
CAT_CODE_CODE = "code-code"
CATEGORIES = (CAT_DOC_DOC, CAT_DOC_CODE, CAT_CODE_CODE)

PathPair = Tuple[str, str]


@dataclass
class TrackerResetResult:
    """Outcome of scanning (and optionally resetting) one tracker."""

    path: str
    tracker_type: str  # "main" | "doc" | "mini"
    status: str  # "clean" | "reset" | "would_reset" | "skipped" | "error"
    x_cells: int = 0
    asymmetric_cells: int = 0  # 'x' cells whose mirror cell is not 'x'
    pair_keys: Set[PathPair] = field(default_factory=lambda: set())
    category_pairs: DefaultDict[str, Set[PathPair]] = field(
        default_factory=lambda: defaultdict(set)
    )
    structural_pairs: Set[PathPair] = field(default_factory=lambda: set())
    dir_pairs: Set[PathPair] = field(default_factory=lambda: set())  # either side is a directory key
    unmapped_pairs: Set[PathPair] = field(default_factory=lambda: set())  # a path is not in the global key map
    examples: List[str] = field(default_factory=lambda: [])
    reason: str = ""
    backup_path: str = ""
    snapshot_path: str = ""

    @property
    def unique_pairs(self) -> int:
        return len(self.pair_keys)


@dataclass
class ResetReport:
    """Aggregated report across all scanned trackers."""

    applied: bool
    results: List[TrackerResetResult] = field(default_factory=lambda: [])
    snapshot_dir: str = ""

    @property
    def total_cells(self) -> int:
        return sum(r.x_cells for r in self.results if r.status not in ("skipped", "ignored"))

    @property
    def unique_pairs(self) -> Set[PathPair]:
        out: Set[PathPair] = set()
        for r in self.results:
            if r.status not in ("skipped", "ignored"):
                out |= r.pair_keys
        return out

    def pairs_by_category(self) -> Dict[str, Set[PathPair]]:
        out: Dict[str, Set[PathPair]] = {c: set() for c in CATEGORIES}
        for r in self.results:
            if r.status in ("skipped", "ignored"):
                continue
            for cat, pairs in r.category_pairs.items():
                out.setdefault(cat, set()).update(pairs)
        return out

    def _union(self, attr: str) -> Set[PathPair]:
        out: Set[PathPair] = set()
        for r in self.results:
            if r.status not in ("skipped", "ignored"):
                out |= getattr(r, attr)
        return out

    @property
    def structural_pairs(self) -> Set[PathPair]:
        return self._union("structural_pairs")

    @property
    def dir_pairs(self) -> Set[PathPair]:
        return self._union("dir_pairs")

    @property
    def unmapped_pairs(self) -> Set[PathPair]:
        return self._union("unmapped_pairs")

    @property
    def file_pairs(self) -> Set[PathPair]:
        """Pairs between two files that are both in the current global key map."""
        return self.unique_pairs - self.dir_pairs - self.unmapped_pairs

    @property
    def skipped(self) -> List[TrackerResetResult]:
        return [r for r in self.results if r.status in ("skipped", "error", "ignored")]

    @property
    def has_failures(self) -> bool:
        return any(r.status in ("skipped", "error") for r in self.results)

    def basename_collisions(self) -> Dict[str, List[str]]:
        """Trackers that would be/were modified and share a file name (backup rotation hazard)."""
        groups: DefaultDict[str, List[str]] = defaultdict(list)
        for r in self.results:
            if r.status in ("would_reset", "reset"):
                groups[os.path.basename(r.path)].append(r.path)
        return {k: v for k, v in groups.items() if len(v) > 1}


# --- Classification helpers ---


def classify_path(path: str, doc_roots: Sequence[str]) -> str:
    """Returns 'doc' or 'code' for a tracker-key path."""
    norm = normalize_path(path)
    for root in doc_roots:
        if norm == root or is_subpath(norm, root):
            return "doc"
    if not doc_roots and norm.lower().endswith(_DOC_EXTENSIONS):
        return "doc"
    return "code"


def classify_pair(path_a: str, path_b: str, doc_roots: Sequence[str]) -> str:
    kinds = sorted(
        [classify_path(path_a, doc_roots), classify_path(path_b, doc_roots)]
    )  # "code" < "doc"
    if kinds == ["doc", "doc"]:
        return CAT_DOC_DOC
    if kinds == ["code", "code"]:
        return CAT_CODE_CODE
    return CAT_DOC_CODE


def _tracker_type_for(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith("_module.md"):
        return "mini"
    if "doc_tracker" in base:
        return "doc"
    return "main"


# --- Scanning ---


@dataclass
class _ParsedTracker:
    definitions: List[Tuple[str, str]]
    headers: List[str]
    row_labels: List[str]
    rows: List[List[str]]  # decompressed
    last_key_edit: str
    last_grid_edit: str
    raw_lines: List[str]


def _parse_tracker(path: str) -> Tuple[Optional[_ParsedTracker], str]:
    """Reads a tracker into a validated structure. Returns (parsed, error_reason)."""
    if not os.path.exists(path):
        return None, "tracker file not found"
    data = read_tracker_file_structured(path, include_raw_lines=True)
    defs: List[Tuple[str, str]] = data.get("definitions_ordered") or []
    headers: List[str] = data.get("grid_headers_ordered") or []
    grid_rows: List[Tuple[str, str]] = data.get("grid_rows_ordered") or []
    if not defs or not grid_rows:
        return None, "no key definitions or grid rows could be read"
    n = len(defs)
    if not (len(headers) == len(grid_rows) == n):
        return None, (
            f"inconsistent counts (definitions={n}, headers={len(headers)}, "
            f"rows={len(grid_rows)})"
        )
    def_keys = [k for k, _ in defs]
    labels = [lab for lab, _ in grid_rows]
    if headers != def_keys or labels != def_keys:
        return None, "grid headers / row labels do not match definition order"
    rows: List[List[str]] = []
    for label, comp in grid_rows:
        row = list(decompress(comp))
        if len(row) != n:
            return None, f"row '{label}' has {len(row)} cells, expected {n}"
        rows.append(row)
    return (
        _ParsedTracker(
            definitions=defs,
            headers=headers,
            row_labels=labels,
            rows=rows,
            last_key_edit=data.get("last_key_edit") or "",
            last_grid_edit=data.get("last_grid_edit") or "",
            raw_lines=list(data.get("raw_lines") or []),
        ),
        "",
    )


def _own_key_infos(
    parsed: _ParsedTracker,
) -> Tuple[List[KeyInfo], Dict[str, KeyInfo]]:
    """
    Builds KeyInfo objects from the tracker's *own* definitions, plus a tracker-local path map.

    The CRCT writers derive the on-disk labels from ``KeyInfo.key_string``. Using the
    labels already in the file (instead of the current global key map) keeps every label
    exactly as it is, which also works for stale trackers whose keys no longer match the
    global map (e.g. trackers of since-excluded directories). Only 'x' cells change.
    """
    infos: List[KeyInfo] = []
    local_map: Dict[str, KeyInfo] = {}
    for label, p in parsed.definitions:
        norm = normalize_path(p)
        ki = KeyInfo(
            key_string=label,
            norm_path=norm,
            parent_path=normalize_path(os.path.dirname(norm)) or None,
            tier=0,
            is_directory=False,
        )
        infos.append(ki)
        local_map[norm] = ki
    return infos, local_map


def _is_structural(
    pa: str, pb: str, path_to_key_info: Dict[str, KeyInfo]
) -> bool:
    """True for a directory <-> contained-path pair (what update_tracker's structural rule writes as 'x')."""
    for parent, child in ((pa, pb), (pb, pa)):
        ki = path_to_key_info.get(parent)
        is_dir = ki.is_directory if ki is not None else os.path.isdir(parent)
        if is_dir and is_subpath(child, parent):
            return True
    return False


def scan_tracker(
    path: str,
    path_to_key_info: Dict[str, KeyInfo],
    doc_roots: Sequence[str] = (),
    sample_limit: int = 5,
) -> Tuple[TrackerResetResult, Optional[_ParsedTracker], Optional[List[KeyInfo]]]:
    """Counts 'x' cells in one tracker and validates that it can be rewritten safely."""
    path = normalize_path(path)
    result = TrackerResetResult(
        path=path, tracker_type=_tracker_type_for(path), status="clean"
    )
    parsed, err = _parse_tracker(path)
    if parsed is None:
        # A *_module.md without a readable key/grid section holds no tracker cells.
        unreadable = err.startswith("no key definitions")
        result.status = "ignored" if unreadable else "skipped"
        result.reason = err
        return result, None, None

    defs = parsed.definitions
    n = len(defs)
    for i in range(n):
        for j in range(n):
            if i == j or parsed.rows[i][j] != MUTUAL_CHAR:
                continue
            result.x_cells += 1
            if parsed.rows[j][i] != MUTUAL_CHAR:
                result.asymmetric_cells += 1
            pa, pb = normalize_path(defs[i][1]), normalize_path(defs[j][1])
            pair: PathPair = (pa, pb) if pa <= pb else (pb, pa)
            pair_paths = (pa, pb)
            if pair not in result.pair_keys:
                result.pair_keys.add(pair)
                result.category_pairs[classify_pair(pa, pb, doc_roots)].add(pair)
                if _is_structural(pa, pb, path_to_key_info):
                    result.structural_pairs.add(pair)
                kis = (path_to_key_info.get(pa), path_to_key_info.get(pb))
                if path_to_key_info and (kis[0] is None or kis[1] is None):
                    result.unmapped_pairs.add(pair)
                elif any(
                    (k.is_directory if k is not None else os.path.isdir(q))
                    for k, q in zip(kis, pair_paths)
                ):
                    result.dir_pairs.add(pair)
                if len(result.examples) < sample_limit:
                    result.examples.append(f"{defs[i][0]} <-> {defs[j][0]}: {pair[0]} <-> {pair[1]}")

    if result.x_cells == 0:
        return result, parsed, None

    infos, _ = _own_key_infos(parsed)
    if result.tracker_type == "mini":
        start, end = get_mini_tracker_data()["markers"]
        stripped = [ln.strip() for ln in parsed.raw_lines]
        if start not in stripped or end not in stripped:
            result.status = "skipped"
            result.reason = (
                "mini tracker template markers not found; rewriting would replace its header/footer"
            )
            return result, parsed, None
    result.status = "would_reset"
    return result, parsed, infos


# --- Applying ---


def _backups_dir(project_root: str) -> str:
    rel = ConfigManager().get_path("backups_dir", "cline_docs/backups")
    return normalize_path(rel if os.path.isabs(rel) else os.path.join(project_root, rel))


def _snapshot_tracker(path: str, snapshot_dir: str, project_root: str) -> str:
    """
    Copies a tracker into the run-scoped snapshot dir, mirroring its project-relative path.

    Every snapshot file gets a ``.bak`` suffix: CRCT globs ``*_module.md`` under the project
    root, so a snapshot named like a mini tracker could be re-registered as a tracker and
    bring the pre-reset 'x' cells back into the suggester's skip state.
    """
    norm_root = normalize_path(project_root)
    if path.lower().startswith(norm_root.lower().rstrip("/") + "/"):
        rel = path[len(norm_root.rstrip("/")) + 1 :]
    else:
        rel = re.sub(r"[:/\\]+", "_", path)
    dest = os.path.join(snapshot_dir, *rel.split("/")) + ".bak"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(path, dest)
    return dest


def _expected_rows(parsed: _ParsedTracker) -> List[List[str]]:
    out: List[List[str]] = []
    for i, row in enumerate(parsed.rows):
        out.append(
            [
                PLACEHOLDER_CHAR if (j != i and c == MUTUAL_CHAR) else c
                for j, c in enumerate(row)
            ]
        )
    return out


class _PureRewriteCollector(TrackerBatchCollector):
    """
    TrackerBatchCollector for updates that already hold the final grid.

    ``commit_all`` keeps everything we want from the collector: pre-validation, temp
    backups with rollback on error, ``backup_tracker_file`` permanent backups, the
    standard writers, and one cache invalidation for the whole batch. Its
    ``_consolidate_grids`` pass is switched off: it re-derives cells (reciprocals, AST
    overrides, external relationships, doc-root pruning), but a reset must change
    nothing except the 'x' cells. A failed write aborts and rolls back the whole batch.
    """

    def _consolidate_grids(self) -> None:
        return None

    def _write_single_tracker(self, update: TrackerUpdate) -> bool:
        # The base class swallows write errors and returns False, and commit_all only rolls
        # back when a write *raises*. Raise so a failed write rolls back the whole batch.
        if not super()._write_single_tracker(update):
            raise RuntimeError(f"write failed for {update.output_file}")
        return True


def _build_update(
    result: TrackerResetResult, parsed: _ParsedTracker, infos: List[KeyInfo]
) -> TrackerUpdate:
    """Builds the TrackerUpdate (final grid, original labels/header/footer/pins) for one tracker."""
    stamp = datetime.datetime.now().isoformat()
    is_mini = result.tracker_type == "mini"
    return TrackerUpdate(
        tracker_type=result.tracker_type,
        output_file=result.path,
        key_info_list=infos,
        grid_rows=[compress("".join(r)) for r in _expected_rows(parsed)],
        last_key_edit=parsed.last_key_edit,
        last_grid_edit=(
            f"Reset {result.x_cells} mutual 'x' cell(s) to 'p' (one-time migration) ({stamp})"
        ),
        module_path=normalize_path(os.path.dirname(result.path)) if is_mini else None,
        manual_foreign_pins=(
            sorted(_read_manual_foreign_pins_from_lines(parsed.raw_lines))
            if is_mini
            else None
        ),
        template_data=get_mini_tracker_data() if is_mini else None,
        existing_lines=parsed.raw_lines if is_mini else None,
        tracker_exists=True,
        path_to_key_info={ki.norm_path: ki for ki in infos},
    )


def _latest_regular_backup(backups_dir: str, tracker_path: str, since: float) -> str:
    """Newest ``<name>.<timestamp>.bak`` that backup_tracker_file wrote since `since` ('' if none)."""
    base = os.path.basename(tracker_path) + "."
    best, best_m = "", since - 1
    try:
        for f in os.listdir(backups_dir):
            full = os.path.join(backups_dir, f)
            if f.startswith(base) and f.endswith(".bak") and os.path.isfile(full):
                m = os.path.getmtime(full)
                if m >= since - 1 and m >= best_m:
                    best, best_m = full, m
    except OSError:
        pass
    return best


def _invalidate_tracker_caches(path: str) -> None:
    for name, pattern in (
        ("tracker_data", f"tracker_data:{path}:.*"),
        ("tracker_data_structured", f"tracker_data_structured:{path}:.*"),
        ("aggregation_v2_gi", ".*"),
    ):
        try:
            invalidate_dependent_entries(name, pattern)
        except Exception as e:  # cache hygiene must never fail the migration
            logger.debug(f"Cache invalidation for {name} failed: {e}")


def _verify_reset(path: str, before: _ParsedTracker) -> str:
    """Re-reads a rewritten tracker; returns '' if it differs from `before` only by x -> p."""
    _invalidate_tracker_caches(path)
    after, err = _parse_tracker(path)
    if after is None:
        return f"re-read failed: {err}"
    if after.definitions != before.definitions:
        return "key definitions changed"
    if after.headers != before.headers or after.row_labels != before.row_labels:
        return "grid headers/labels changed"
    if after.rows != _expected_rows(before):
        return "grid differs from the expected x -> p result"
    return ""


def reset_mutual_dependencies(
    tracker_paths: Sequence[str],
    path_to_key_info: Dict[str, KeyInfo],
    doc_roots: Sequence[str] = (),
    apply: bool = False,
    project_root: Optional[str] = None,
    sample_limit: int = 5,
) -> ResetReport:
    """
    Scans `tracker_paths` and (when `apply` is True) resets every 'x' cell to 'p'.

    Dry run (apply=False) never writes anything. Trackers that cannot be rewritten
    faithfully are skipped (and reported) rather than rewritten approximately.
    `path_to_key_info` (the global key map) is only used to tell directories from files
    when counting structural pairs; labels are always taken from each tracker itself.
    """
    from cline_utils.dependency_system.utils.cache_manager import get_project_root_cached

    root = normalize_path(project_root or get_project_root_cached())
    norm_doc_roots = [normalize_path(d) for d in doc_roots]
    report = ResetReport(applied=apply)

    planned: List[Tuple[TrackerResetResult, _ParsedTracker, List[KeyInfo]]] = []
    for p in sorted({normalize_path(t) for t in tracker_paths}):
        try:
            res, parsed, infos = scan_tracker(
                p, path_to_key_info, norm_doc_roots, sample_limit
            )
        except Exception as e:
            logger.exception(f"Failed to scan tracker {p}")
            res = TrackerResetResult(
                path=p, tracker_type=_tracker_type_for(p), status="error", reason=str(e)
            )
            parsed, infos = None, None
        report.results.append(res)
        if res.status == "would_reset" and parsed is not None and infos is not None:
            planned.append((res, parsed, infos))

    if not apply:
        return report

    backups_dir = _backups_dir(root)
    snapshot_dir = os.path.join(
        backups_dir, "x_reset_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    report.snapshot_dir = snapshot_dir
    started = time.time()

    # 1. Snapshot every tracker we are about to change (also the restore source below).
    collector = _PureRewriteCollector()
    batch: List[Tuple[TrackerResetResult, _ParsedTracker]] = []
    for res, parsed, infos in planned:
        try:
            res.snapshot_path = _snapshot_tracker(res.path, snapshot_dir, root)
            collector.add(_build_update(res, parsed, infos))
            batch.append((res, parsed))
        except Exception as e:
            logger.exception(f"Cannot prepare reset for {res.path}")
            res.status, res.reason = "error", f"snapshot/update preparation failed: {e}; left untouched"

    def _restore(res: TrackerResetResult) -> None:
        if res.snapshot_path and os.path.exists(res.snapshot_path):
            shutil.copy2(res.snapshot_path, res.path)
        _invalidate_tracker_caches(res.path)

    # 2. One batch commit (atomic: the collector rolls every tracker back if a write raises).
    results: Dict[str, bool] = {}
    try:
        if not collector.is_empty():
            results = collector.commit_all(skip_populate_hook=True)
    except Exception as e:
        logger.exception("Batch commit of the reset failed")
        for res, _ in batch:
            try:
                _restore(res)
            except OSError:
                logger.exception(f"Could not restore {res.path}")
            res.status, res.reason = "error", f"batch write failed ({e}); restored from backup"
        batch = []
    finally:
        # commit_all seeds the in-process aggregation cache with the collector's (empty) map.
        clear_runtime_aggregation_cache()

    # 3. Verify each tracker individually; restore the ones that are not exactly x -> p.
    for res, parsed in batch:
        try:
            if not results.get(res.path, False):
                problem = "collector reported a failed write"
            else:
                problem = _verify_reset(res.path, parsed)
        except Exception as e:
            problem = f"verification raised: {e}"
        if problem:
            try:
                _restore(res)
            except OSError:
                logger.exception(f"Could not restore {res.path}")
            res.status = "error"
            res.reason = f"verification failed ({problem}); restored from backup"
            continue
        res.backup_path = _latest_regular_backup(backups_dir, res.path, started)
        res.status = "reset"
    return report


# --- Reporting ---


def format_report(report: ResetReport, example_limit: int = 8) -> str:
    """Renders a human-readable (markdown-friendly) report."""
    mode = "APPLIED" if report.applied else "DRY RUN (no files changed)"
    cats = report.pairs_by_category()
    lines: List[str] = [f"# Mutual 'x' -> 'p' reset: {mode}", ""]
    lines += [
        f"- Trackers scanned: {len(report.results)}",
        f"- Trackers with 'x' cells: {sum(1 for r in report.results if r.x_cells and r.status not in ('skipped', 'ignored'))}",
        f"- 'x' cells to reset (all trackers): {report.total_cells}",
        f"- Unique pairs (deduplicated across trackers): {len(report.unique_pairs)}",
        f"  - doc-doc: {len(cats[CAT_DOC_DOC])}",
        f"  - doc-code: {len(cats[CAT_DOC_CODE])}",
        f"  - code-code: {len(cats[CAT_CODE_CODE])}",
        f"- Breakdown of the unique pairs by kind:",
        f"  - file <-> file (both paths in the current global key map): {len(report.file_pairs)}",
        f"  - involving a directory key (main-tracker module cells, parent/child, doc dirs): {len(report.dir_pairs)}",
        f"  - with a path missing from the current global key map (stale trackers): {len(report.unmapped_pairs)}",
        f"  - (of the directory pairs, directory <-> contained path: {len(report.structural_pairs)}; "
        "doc and mini trackers re-create these as 'x' on the next update_tracker run)",
        f"- Asymmetric 'x' cells (mirror cell is not 'x'; reset anyway): "
        f"{sum(r.asymmetric_cells for r in report.results if r.status not in ('skipped', 'ignored'))}",
    ]
    if report.applied:
        done = sum(1 for r in report.results if r.status == "reset")
        lines.append(f"- Trackers rewritten: {done}")
        lines.append(f"- Run snapshot dir: {report.snapshot_dir}")
    lines += ["", "## Per tracker", "", "| Tracker | Type | Status | x cells | Unique pairs | doc-doc | doc-code | code-code |", "|---|---|---|---|---|---|---|---|"]
    for r in report.results:
        if r.status == "ignored" or (r.x_cells == 0 and r.status == "clean"):
            continue
        c = r.category_pairs
        lines.append(
            f"| {r.path} | {r.tracker_type} | {r.status} | {r.x_cells} | {r.unique_pairs} | "
            f"{len(c.get(CAT_DOC_DOC, ()))} | {len(c.get(CAT_DOC_CODE, ()))} | {len(c.get(CAT_CODE_CODE, ()))} |"
        )
    clean = sum(1 for r in report.results if r.x_cells == 0 and r.status == "clean")
    lines.append(f"\n({clean} tracker(s) with no 'x' cells omitted.)")
    if report.skipped:
        lines += ["", "## Skipped / errors (left untouched)", ""]
        for r in report.skipped:
            lines.append(f"- {r.path} [{r.status}]: {r.reason}")
    coll = report.basename_collisions()
    if coll:
        lines += [
            "",
            "## Backup note",
            "",
            f"{len(coll)} tracker file name(s) are shared by several modified trackers. "
            "`backup_tracker_file` keeps only the 2 newest backups per file name, so "
            "`--apply` also writes a complete run-scoped snapshot under the backups dir.",
        ]
    shown = 0
    lines += ["", "## Examples", ""]
    for r in report.results:
        for ex in r.examples:
            if shown >= example_limit:
                break
            lines.append(f"- [{os.path.basename(r.path)}] {ex}")
            shown += 1
    if shown == 0:
        lines.append("- (none)")
    return "\n".join(lines) + "\n"
