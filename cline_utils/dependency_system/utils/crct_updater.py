#!/usr/bin/env python3
"""
crct_updater.py - CRCT System File Updater
============================================
Checks upstream GitHub for newer versions of CRCT system files using the
Git Trees API (2 API calls total regardless of file count) and updates local
copies whose blob SHA differs from the upstream version and whose modification
time (mtime) indicates they are older/outdated.

Comparison strategy:
  - Fetch the branch tip commit date and recursive tree in two API calls.
  - Compute local Git blob SHAs for comparison.
  - If a file is identical (local SHA == remote SHA), skip.
  - If a file differs:
    - If local mtime is older than remote commit date, update it.
    - If local mtime is newer or equal, preserve it as a local customization.

Managed paths (never touches anything outside these):
  cline_docs/CRCT_Documentation/
  cline_docs/templates/
  code_analysis/
  cline_utils/
  .agent/
  .clinerules/   (excluding default-rules.md)

Usage:
  python -m cline_utils.dependency_system.utils.crct_updater              # Interactive opt-in check
  python -m cline_utils.dependency_system.utils.crct_updater --check      # Dry-run: show what would change
  python -m cline_utils.dependency_system.utils.crct_updater --force      # Skip cooldown and update immediately
  python -m cline_utils.dependency_system.utils.crct_updater --status     # Show config and last-run info
  python -m cline_utils.dependency_system.utils.crct_updater --enable-auto   # Enable dependency_processor hook
  python -m cline_utils.dependency_system.utils.crct_updater --disable-auto  # Disable dependency_processor hook

Auto-update hook (add to dependency_processor.py):
  from cline_utils.dependency_system.utils.crct_updater import auto_update_check
  auto_update_check()          # silent no-op until --enable-auto is run
"""

import os
import sys
import json
import time
import stat
import shutil
import hashlib
import argparse
import platform
import tempfile
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union, Any

# ─── Configuration ─────────────────────────────────────────────────────────────

REPO_OWNER = "RPG-fan"
REPO_NAME = "Cline-Recursive-Chain-of-Thought-System-CRCT-"
BRANCH = "main"

GITHUB_API_BASE = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}"

# Paths managed by this updater (relative to project root, forward-slash)
MANAGED_PATHS: tuple[str, ...] = (
    "cline_docs/CRCT_Documentation",
    "cline_docs/templates",
    "code_analysis",
    "cline_utils",
    ".agent",
    ".clinerules",
)

# Files that are explicitly excluded from updates (user-customisable content)
EXCLUDED_FILES: frozenset[str] = frozenset(
    {
        ".clinerules/default-rules.md",
        "ast_verified_links.json",
        "ast_verified_links_old.json",
        "validation_cache.json",
        "transparency_registry.json",
        "tracker_map.json",
        "runtime_symbols.json",
        "project_symbol_map.json",
        "project_symbol_map_old.json",
        "global_key_map.json",
        "global_key_map_old.json",
        "activeContext.md",
        "changelog.md",
        "userProfile.md",
    }
)

# State / config file relocated to cline_utils/dependency_system/core/state/
STATE_FILE_PATH_REL = (
    Path("cline_utils")
    / "dependency_system"
    / "core"
    / "state"
    / "crct_updater_state.json"
)
COOLDOWN_SECONDS = 3600  # 1 hour

# State-file keys
CFG_AUTO_UPDATE = "auto_update_enabled"
CFG_LAST_RUN = "last_run_timestamp"
CFG_LAST_UPDATED = "last_updated_files"
CFG_KNOWN_SHAS = "known_shas"  # {rel_path: git_blob_sha}

# ─── Utility helpers ───────────────────────────────────────────────────────────


def _configure_stdio_for_unicode() -> None:
    """Avoid UnicodeEncodeError on Windows terminals using legacy code pages."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                _ = reconfigure(errors="backslashreplace")
            except (OSError, ValueError):
                pass


def _get_project_root() -> Path:
    """Walk up from this script until project_root.cfg or .clinerules is found."""
    start = Path(__file__).resolve().parent
    for candidate in [start, *start.parents]:
        if (candidate / "project_root.cfg").exists() or (
            candidate / ".clinerules"
        ).exists():
            return candidate
    return start  # fallback: script directory


def _get_state_path(root: Path) -> Path:
    return root / STATE_FILE_PATH_REL


def _load_state(root: Path) -> dict[str, Any]:
    p = _get_state_path(root)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    state_dict: dict[str, Any] = data
                    return state_dict
        except (json.JSONDecodeError, OSError):
            pass
    return {
        CFG_AUTO_UPDATE: False,
        CFG_LAST_RUN: 0.0,
        CFG_LAST_UPDATED: [],
        CFG_KNOWN_SHAS: {},
    }


def _save_state(root: Path, state: dict[str, Any]) -> None:
    p = _get_state_path(root)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except OSError as e:
        _warn(f"Could not save updater state: {e}")


def _log(msg: str) -> None:
    print(f"[CRCT Updater] {msg}")


def _warn(msg: str) -> None:
    print(f"[CRCT Updater] WARNING: {msg}", file=sys.stderr)


def _git_blob_sha(path: Path) -> str:
    """Calculate the Git blob SHA-1 of a local file."""
    try:
        data = path.read_bytes()
        sha1 = hashlib.sha1()
        sha1.update(f"blob {len(data)}\x00".encode("utf-8"))
        sha1.update(data)
        return sha1.hexdigest()
    except OSError:
        return ""


def _make_request(url: str, *, timeout: int = 20) -> Optional[bytes]:
    """Single low-level HTTP GET; injects GITHUB_TOKEN when present."""
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "CRCT-Updater/2.0 (" + platform.system() + ")",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 403:
            _warn("GitHub API rate-limited. Set GITHUB_TOKEN env var for 5 000 req/hr.")
        elif e.code == 404:
            _warn(f"404 Not Found: {url}")
        else:
            _warn(f"HTTP {e.code}: {url}")
        return None
    except urllib.error.URLError as e:
        _warn(f"Network error ({url}): {e}")
        return None


def _api_json(url: str) -> Optional[Union[dict[str, Any], list[Any]]]:
    raw = _make_request(url)
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict):
            dict_data: dict[str, Any] = data
            return dict_data
        elif isinstance(data, list):
            list_data: list[Any] = data
            return list_data
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        _warn(f"JSON parse error for {url}: {e}")
        return None


def _atomic_write(dest: Path, data: bytes) -> bool:
    """
    Write data to dest via a sibling temp file, preserving existing permissions.
    Safe on Windows (same-volume move) and POSIX (atomic rename).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Avoid permissions/sharing errors on Windows:
    # use mkstemp with sibling dir, close file descriptor immediately.
    fd, tmp_path_str = tempfile.mkstemp(dir=dest.parent, prefix=".tmp_crct_")
    tmp = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as f:
            _ = f.write(data)
        if dest.exists():
            try:
                os.chmod(tmp, stat.S_IMODE(dest.stat().st_mode))
            except OSError:
                pass
            # On Windows, shutil.move/os.rename raises OSError if target exists.
            # We must remove target first on Windows, or use shutil.move which handles it.
            if platform.system() == "Windows":
                try:
                    dest.unlink()
                except OSError:
                    pass
        _ = shutil.move(str(tmp), str(dest))
        return True
    except OSError as e:
        _warn(f"Write failed for {dest}: {e}")
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _set_mtime(path: Path, ts: float) -> None:
    try:
        os.utime(path, (ts, ts))
    except OSError:
        pass  # non-fatal


def _is_managed(path: str) -> bool:
    """Return True if the forward-slash path falls under a managed directory."""
    p = path.replace("\\", "/")
    return any(p == m or p.startswith(m + "/") for m in MANAGED_PATHS)


def _is_excluded(path: str) -> bool:
    return path.replace("\\", "/") in EXCLUDED_FILES


# ─── GitHub Git Trees API ──────────────────────────────────────────────────────


def _resolve_branch_commit(branch: str = BRANCH) -> Optional[tuple[str, float]]:
    """
    Return (commit_sha, committed_date_ts) for the tip of `branch`.
    One API call.
    """
    url = f"{GITHUB_API_BASE}/branches/{branch}"
    data = _api_json(url)
    if not data or not isinstance(data, dict):
        return None
    try:
        commit_info = data.get("commit")
        if not isinstance(commit_info, dict):
            return None
        commit_info_dict: dict[str, Any] = commit_info
        sha = str(commit_info_dict.get("sha", ""))
        commit_details = commit_info_dict.get("commit")
        if not isinstance(commit_details, dict):
            return None
        commit_details_dict: dict[str, Any] = commit_details
        committer = commit_details_dict.get("committer")
        if not isinstance(committer, dict):
            return None
        committer_dict: dict[str, Any] = committer
        date_str = str(committer_dict.get("date", ""))
        # Convert date to timestamp, handling standard 'Z' format
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return sha, dt.timestamp()
    except (KeyError, ValueError) as e:
        _warn(f"Could not parse branch response: {e}")
        return None


def _fetch_tree(tree_sha: str) -> Optional[list[dict[str, Any]]]:
    """
    Fetch the full recursive Git tree.  Returns the 'tree' array.
    One API call.
    """
    url = f"{GITHUB_API_BASE}/git/trees/{tree_sha}?recursive=1"
    data = _api_json(url)
    if not data or not isinstance(data, dict):
        return None
    if data.get("truncated"):
        _warn(
            "Git tree response was truncated by GitHub (very large repo). "
            + "Some files may be missed; consider using a GITHUB_TOKEN."
        )
    tree_list = data.get("tree")
    if isinstance(tree_list, list):
        return tree_list
    return None


def _build_managed_index(tree: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """
    Filter the full tree down to managed blobs, excluding explicitly excluded files.
    Returns { rel_path: { "sha": str, "url": str } }
    """
    index: dict[str, dict[str, str]] = {}
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = str(item.get("path", "")).replace("\\", "/")
        if not path:
            continue
        if _is_managed(path) and not _is_excluded(path):
            encoded_path = urllib.parse.quote(path)
            index[path] = {
                "sha": str(item.get("sha", "")),
                "url": f"{GITHUB_RAW_BASE}/{encoded_path}",
            }
    return index


# ─── Core update logic ──────────────────────────────────────────────────────────


def check_for_updates(
    project_root: Path,
    state: dict[str, Any],
    *,
    dry_run: bool = False,
    force: bool = False,
    commit_ts: Optional[float] = None,
) -> list[str]:
    """
    Compare the upstream managed-file index against local files.
    Calculates actual local Git blob SHAs and checks mtimes to safely
    detect outdated files while protecting local modifications.
    """
    _log("Fetching branch HEAD...")
    branch_info = _resolve_branch_commit()
    if branch_info is None:
        _warn("Could not reach GitHub. Aborting update check.")
        return []

    head_sha, head_ts = branch_info
    effective_ts = commit_ts if commit_ts is not None else head_ts

    _log(f"Fetching repository tree ({BRANCH} @ {head_sha[:8]})...")
    tree = _fetch_tree(head_sha)
    if tree is None:
        _warn("Could not fetch repository tree. Aborting.")
        return []

    managed = _build_managed_index(tree)
    _log(f"Found {len(managed)} managed file(s) in upstream tree.")

    known_shas: dict[str, str] = state.get(CFG_KNOWN_SHAS, {})
    updated: list[str] = []

    for rel_path, remote in managed.items():
        local_path = project_root / Path(rel_path)
        local_exists = local_path.exists()
        remote_sha = remote["sha"]

        needs_update = False
        status_label = "UP-TO-DATE"

        if not local_exists:
            needs_update = True
            status_label = "MISSING"
        else:
            local_sha = _git_blob_sha(local_path)
            if local_sha != remote_sha:
                # File is different. Check modification time.
                local_mtime = local_path.stat().st_mtime
                if force:
                    needs_update = True
                    status_label = "FORCE-UPDATE"
                elif local_mtime < effective_ts:
                    # Outdated and older than remote branch commit
                    needs_update = True
                    status_label = "OUTDATED"
                else:
                    # Modified locally (mtime is newer or equal to remote branch HEAD commit)
                    status_label = "PRESERVED"
                    # Keep track of local SHA to avoid unnecessary updates
                    known_shas[rel_path] = local_sha

        if not needs_update:
            if dry_run and status_label == "PRESERVED":
                _log(f"  [PRESERVED] {rel_path} (local modifications detected)")
            continue

        if dry_run:
            _log(f"  [{status_label}] {rel_path}")
            updated.append(rel_path)
            continue

        # Download
        raw = _make_request(remote["url"], timeout=30)
        if raw is None:
            _warn(f"  Skipping {rel_path} — download failed.")
            continue

        if _atomic_write(local_path, raw):
            _set_mtime(local_path, effective_ts)
            known_shas[rel_path] = remote_sha
            _log(f"  [UPDATED] {rel_path}  (sha: {remote_sha[:8]})")
            updated.append(rel_path)

    if not updated:
        _log("All managed files are up to date.")
    elif not dry_run:
        _log(f"Updated {len(updated)} file(s).")
        state[CFG_KNOWN_SHAS] = known_shas

    return updated


# ─── Cooldown-gated auto-update ────────────────────────────────────────────────


def auto_update_check(
    project_root: Optional[Path] = None,
    force: bool = False,
) -> bool:
    """
    Intended to be called by dependency_processor on every invocation.
    Silently no-ops unless the user has run --enable-auto.
    Enforces a 1-hour cooldown between actual network checks.
    """
    if project_root is None:
        project_root = _get_project_root()

    state = _load_state(project_root)

    if not state.get(CFG_AUTO_UPDATE, False):
        return False  # opted out — silent

    if not force:
        elapsed = time.time() - float(state.get(CFG_LAST_RUN, 0.0))
        if elapsed < COOLDOWN_SECONDS:
            return False  # cooldown active — silent

    # Stamp BEFORE network calls to prevent re-entry on slow connections
    state[CFG_LAST_RUN] = time.time()
    _save_state(project_root, state)

    try:
        updated = check_for_updates(project_root, state, dry_run=False, force=force)
    except Exception as e:
        _warn(f"Auto-update encountered an unexpected error: {e}")
        return False

    if updated:
        state[CFG_LAST_UPDATED] = updated
        _save_state(project_root, state)
        return True

    return False


# ─── Config helpers ────────────────────────────────────────────────────────────


def configure_auto_update(root: Path, enable: bool) -> None:
    state = _load_state(root)
    state[CFG_AUTO_UPDATE] = enable
    _save_state(root, state)
    verb = "enabled" if enable else "disabled"
    _log(f"Auto-update {verb}.")
    if enable:
        _log(
            f"  Update checks will fire when dependency_processor runs "
            + f"(cooldown: {COOLDOWN_SECONDS // 60} min)."
        )


def show_status(root: Path) -> None:
    state = _load_state(root)
    auto = state.get(CFG_AUTO_UPDATE, False)
    last_run: float = float(state.get(CFG_LAST_RUN, 0.0))
    updated: list[str] = list(state.get(CFG_LAST_UPDATED, []))
    shas: dict[str, str] = dict(state.get(CFG_KNOWN_SHAS, {}))

    _log("─── CRCT Updater Status ──────────────────────────────────")
    _log(f"  Auto-update : {'ENABLED ✓' if auto else 'DISABLED'}")
    if last_run:
        last_dt = datetime.fromtimestamp(last_run, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        elapsed = int(time.time() - last_run)
        _log(f"  Last check  : {last_dt}  ({elapsed}s ago)")
        remaining = COOLDOWN_SECONDS - elapsed
        _log(
            f"  Cooldown    : {'ready' if remaining <= 0 else f'{remaining}s remaining'}"
        )
    else:
        _log("  Last check  : never")
    _log(f"  Tracked SHAs: {len(shas)} file(s)")
    if updated:
        _log(f"  Last updated: {len(updated)} file(s)")
        for f in updated[:15]:
            _log(f"    - {f}")
        if len(updated) > 15:
            _log(f"    ... and {len(updated) - 15} more")
    _log("  Managed paths:")
    for p in MANAGED_PATHS:
        _log(f"    {p}/")
    _log("  Excluded files:")
    for e in sorted(EXCLUDED_FILES):
        _log(f"    {e}")
    _log("──────────────────────────────────────────────────────────")


# ─── CLI ───────────────────────────────────────────────────────────────────────


def _prompt_yes_no(question: str, default: bool = False) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {hint}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return answer in ("y", "yes") if answer else default


def main() -> int:
    _configure_stdio_for_unicode()
    ap = argparse.ArgumentParser(
        description="CRCT system file updater — uses the GitHub Git Trees API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    _ = ap.add_argument(
        "--check", action="store_true", help="Dry-run: show what would update."
    )
    _ = ap.add_argument(
        "--force", action="store_true", help="Skip cooldown and force updates."
    )
    _ = ap.add_argument(
        "--status", action="store_true", help="Show config and last-run info."
    )
    _ = ap.add_argument(
        "--enable-auto",
        action="store_true",
        dest="enable_auto",
        help="Enable auto-update hook for dependency_processor.",
    )
    _ = ap.add_argument(
        "--disable-auto",
        action="store_true",
        dest="disable_auto",
        help="Disable auto-update hook.",
    )
    _ = ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override project root directory (default: auto-detected).",
    )
    args = ap.parse_args()

    root = args.root or _get_project_root()
    _log(f"Project root : {root}")
    _log(
        f"Platform     : {platform.system()} {platform.release()} / Python {sys.version.split()[0]}"
    )

    if args.status:
        show_status(root)
        return 0

    if args.enable_auto:
        configure_auto_update(root, True)
        return 0

    if args.disable_auto:
        configure_auto_update(root, False)
        return 0

    if args.check:
        _log("DRY RUN — no files will be written.")
        state = _load_state(root)
        _ = check_for_updates(root, state, dry_run=True, force=args.force)
        return 0

    # ── Interactive run ──────────────────────────────────────────────────────
    state = _load_state(root)

    if not args.force:
        elapsed = time.time() - state.get(CFG_LAST_RUN, 0)
        if elapsed < COOLDOWN_SECONDS and state.get(CFG_LAST_RUN, 0) > 0:
            remaining = int(COOLDOWN_SECONDS - elapsed)
            _log(f"Cooldown active: {remaining}s remaining since last check.")
            if not _prompt_yes_no("Run anyway?", default=False):
                _log("Skipped.")
                return 0

    print()
    _log("The following system paths will be checked against GitHub:")
    for p in MANAGED_PATHS:
        _log(f"  {p}/")
    _log("User data and excluded files will NOT be modified.")
    print()

    if not _prompt_yes_no("Proceed?", default=True):
        _log("Cancelled.")
        return 0

    updated = check_for_updates(root, state, dry_run=False, force=args.force)

    state[CFG_LAST_RUN] = time.time()
    if updated:
        state[CFG_LAST_UPDATED] = updated
    _save_state(root, state)

    if not state.get(CFG_AUTO_UPDATE, False):
        print()
        _log(
            "Tip: run --enable-auto to check automatically on each dependency_processor run."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
