"""
tsconfig.json / jsconfig.json handling for JS/TS import resolution.

Responsibilities (kept out of dependency_suggester, which only resolves specifiers):
  - parse JSONC config files (comments, trailing commas)
  - follow `extends` chains (relative paths, node_modules packages, TS 5 arrays)
  - decide whether a config applies to a file (`files` / `include` / `exclude`,
    with TypeScript's defaults)
  - pick the applicable configs for a source file, in a deterministic order

Caching is in-process and validated against every config file's mtime in the chain,
so editing any tsconfig in an `extends` chain is picked up without clearing caches.
"""

import fnmatch
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from cline_utils.dependency_system.utils.path_utils import normalize_path

logger = logging.getLogger(__name__)

# The merged view of one config file: see load_tsconfig().
TsConfig = Dict[str, Any]

_DEFAULT_EXCLUDES = ("node_modules", "bower_components", "jspm_packages")
_TS_SOURCE_EXTENSIONS = (".ts", ".tsx", ".d.ts", ".mts", ".cts")
_JS_SOURCE_EXTENSIONS = (".js", ".jsx", ".mjs", ".cjs")
_WILDCARD_CHARS = ("*", "?")

_raw_cache: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
_merged_cache: Dict[str, Tuple[Tuple[Tuple[str, float], ...], Optional[TsConfig]]] = {}


# --------------------------------------------------------------------------- parsing


def parse_jsonc(text: str) -> Any:
    """json.loads for JSONC: strips // and /* */ comments and trailing commas,
    leaving string contents untouched."""
    out: List[str] = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
        elif text.startswith("//", i):
            while i < n and text[i] not in "\r\n":
                i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
        elif ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            # Also skip comments between a trailing comma and the closing bracket.
            while j < n and (text.startswith("//", j) or text.startswith("/*", j)):
                if text.startswith("//", j):
                    while j < n and text[j] not in "\r\n":
                        j += 1
                else:
                    end = text.find("*/", j + 2)
                    j = n if end == -1 else end + 2
                while j < n and text[j] in " \t\r\n":
                    j += 1
            if j < n and text[j] in "}]":
                i += 1  # drop the trailing comma
            else:
                out.append(ch)
                i += 1
        else:
            out.append(ch)
            i += 1
    return json.loads("".join(out))


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return -1.0


def _read_raw(config_path: str) -> Optional[Dict[str, Any]]:
    mtime = _mtime(config_path)
    cached = _raw_cache.get(config_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    data: Optional[Dict[str, Any]] = None
    try:
        with open(config_path, "r", encoding="utf-8-sig") as fh:
            parsed = parse_jsonc(fh.read())
        data = parsed if isinstance(parsed, dict) else None
    except Exception as e:
        logger.warning(f"Could not parse {config_path}: {e}")
    _raw_cache[config_path] = (mtime, data)
    return data


def _resolve_extends(spec: str, config_dir: str) -> Optional[str]:
    """Path of an `extends` target: relative/absolute file, or a node_modules package."""
    if spec.startswith((".", "/")) or os.path.isabs(spec):
        candidate = os.path.normpath(os.path.join(config_dir, spec))
        for path in (candidate, candidate + ".json"):
            if os.path.isfile(path):
                return normalize_path(path)
        return None
    current = config_dir
    while True:
        base = os.path.join(current, "node_modules", spec)
        for path in (base, base + ".json", os.path.join(base, "tsconfig.json")):
            if os.path.isfile(path):
                return normalize_path(path)
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


# --------------------------------------------------------------------------- merging


def load_tsconfig(config_path: str) -> Optional[TsConfig]:
    """
    The effective config after following `extends`. Child values override base values;
    compilerOptions merge per option (`paths` is replaced as a whole, as in TypeScript).
    Path-valued settings remember the directory of the config that declares them:

      {"path", "dir", "compilerOptions", "base_url" (absolute or None),
       "paths", "paths_base" (baseUrl, else the declaring config's dir),
       "files"/"include"/"exclude" as (patterns, base_dir) or None,
       "references" (absolute config paths; not inherited)}
    """
    config_path = normalize_path(config_path)
    cached = _merged_cache.get(config_path)
    if cached is not None and all(_mtime(p) == m for p, m in cached[0]):
        return cached[1]
    chain: List[Tuple[str, float]] = []
    merged = _load_merged(config_path, chain, set())
    _merged_cache[config_path] = (tuple(chain), merged)
    return merged


def _load_merged(config_path: str, chain: List[Tuple[str, float]], seen: set) -> Optional[TsConfig]:
    if config_path in seen:
        logger.warning(f"Cyclic tsconfig extends at {config_path}")
        return None
    seen = seen | {config_path}
    chain.append((config_path, _mtime(config_path)))
    raw = _read_raw(config_path)
    if raw is None:
        return None
    config_dir = os.path.dirname(config_path)

    result: TsConfig = {
        "compilerOptions": {},
        "base_url": None,
        "paths": None,
        "paths_base": None,
        "files": None,
        "include": None,
        "exclude": None,
    }
    extends = raw.get("extends")
    for spec in [extends] if isinstance(extends, str) else (extends if isinstance(extends, list) else []):
        if not isinstance(spec, str):
            continue
        target = _resolve_extends(spec, config_dir)
        base = _load_merged(target, chain, seen) if target else None
        if base is None:
            logger.debug(f"{config_path}: extends '{spec}' not found or unreadable")
            continue
        result["compilerOptions"].update(base["compilerOptions"])
        for key in ("base_url", "paths", "paths_base", "files", "include", "exclude"):
            if base.get(key) is not None:
                result[key] = base[key]

    options = raw.get("compilerOptions") if isinstance(raw.get("compilerOptions"), dict) else {}
    result["compilerOptions"].update(options)
    if isinstance(options.get("baseUrl"), str):
        result["base_url"] = normalize_path(os.path.join(config_dir, options["baseUrl"]))
    if isinstance(options.get("paths"), dict):
        result["paths"] = options["paths"]
        result["paths_base"] = config_dir
    if result["paths"] is not None and result["base_url"] is not None:
        result["paths_base"] = result["base_url"]  # paths resolve against baseUrl when set
    for key in ("files", "include", "exclude"):
        if isinstance(raw.get(key), list):
            result[key] = ([p for p in raw[key] if isinstance(p, str)], config_dir)

    result["path"] = config_path
    result["dir"] = config_dir
    refs = raw.get("references") if isinstance(raw.get("references"), list) else []
    result["references"] = []
    for ref in refs:
        ref_path = ref.get("path") if isinstance(ref, dict) else None
        if isinstance(ref_path, str):
            full = os.path.normpath(os.path.join(config_dir, ref_path))
            if os.path.isdir(full):
                full = os.path.join(full, "tsconfig.json")
            result["references"].append(normalize_path(full))
    return result


# --------------------------------------------------------------------------- membership


def _glob_regex(pattern: str) -> "re.Pattern[str]":
    """TypeScript-style wildcard (`**/`, `*`, `?`) to a case-insensitive regex."""
    parts: List[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            parts.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(parts) + "$", re.IGNORECASE)


def _relative(file_path: str, base_dir: str) -> Optional[str]:
    try:
        rel = os.path.relpath(file_path, base_dir).replace("\\", "/")
    except ValueError:  # different drive
        return None
    return None if rel.startswith("../") or rel == ".." else rel


def _clean_pattern(pattern: str) -> str:
    pattern = pattern.replace("\\", "/")
    while pattern.startswith("./"):
        pattern = pattern[2:]
    return pattern.rstrip("/")


def _include_matches(pattern: str, rel: str) -> bool:
    pattern = _clean_pattern(pattern)
    last = pattern.rsplit("/", 1)[-1]
    if not any(c in pattern for c in _WILDCARD_CHARS) and "." not in last:
        pattern = f"{pattern}/**/*" if pattern else "**/*"  # a directory
    return bool(_glob_regex(pattern).match(rel))


def _exclude_matches(pattern: str, rel: str) -> bool:
    """An exclude pattern matches the file or any directory above it."""
    regex = _glob_regex(_clean_pattern(pattern))
    segments = rel.split("/")
    return any(regex.match("/".join(segments[:k])) for k in range(1, len(segments) + 1))


def tsconfig_applies_to(config: TsConfig, file_path: str) -> bool:
    """
    Whether `file_path` belongs to the config's program, with TypeScript's defaults:
    no `include` and no `files` means include `**/*`; `files` without `include` means
    only those files; default excludes are node_modules, bower_components,
    jspm_packages and outDir; JS files count only with allowJs.
    """
    file_path = normalize_path(file_path)
    lower = file_path.lower()
    options = config.get("compilerOptions", {})
    supported = _TS_SOURCE_EXTENSIONS + (_JS_SOURCE_EXTENSIONS if options.get("allowJs") else ())
    files, include, exclude = config.get("files"), config.get("include"), config.get("exclude")

    if files is not None:
        patterns, base = files
        targets = {normalize_path(os.path.join(base, p)).lower() for p in patterns}
        if lower in targets:
            return True
        if include is None:
            return False
    if not lower.endswith(supported):
        return False

    patterns, base = include if include is not None else (["**/*"], config["dir"])
    rel = _relative(file_path, base)
    if rel is None or not any(_include_matches(p, rel) for p in patterns):
        return False

    if exclude is not None:
        ex_patterns, ex_base = exclude
    else:
        ex_patterns, ex_base = list(_DEFAULT_EXCLUDES), config["dir"]
        out_dir = options.get("outDir")
        if isinstance(out_dir, str):
            ex_patterns.append(out_dir)
    ex_rel = _relative(file_path, ex_base)
    if ex_rel is not None and any(_exclude_matches(p, ex_rel) for p in ex_patterns):
        return False
    return True


# --------------------------------------------------------------------------- selection


def _candidate_config_paths(directory: str) -> List[str]:
    """Config files in one directory: tsconfig.json, its references, the other
    tsconfig.*.json files by name, then jsconfig.json."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    lower_names = {n.lower(): n for n in names}
    ordered: List[str] = []
    main = lower_names.get("tsconfig.json")
    if main:
        main_path = normalize_path(os.path.join(directory, main))
        ordered.append(main_path)
        main_cfg = load_tsconfig(main_path)
        for ref in main_cfg.get("references", []) if main_cfg else []:
            if ref not in ordered and os.path.isfile(ref):
                ordered.append(ref)
    for name in sorted(n for n in names if fnmatch.fnmatch(n.lower(), "tsconfig.*.json")):
        path = normalize_path(os.path.join(directory, name))
        if path not in ordered:
            ordered.append(path)
    if "jsconfig.json" in lower_names:
        path = normalize_path(os.path.join(directory, lower_names["jsconfig.json"]))
        if path not in ordered:
            ordered.append(path)
    return ordered


def select_tsconfigs_for_file(file_path: str, project_root: str) -> List[Tuple[str, TsConfig]]:
    """
    The configs that apply to `file_path`, as (config_path, merged_config) pairs in
    resolution order. Looks in the nearest directory (walking up from the file, not
    past project_root) that holds any config; if none of its configs includes the file,
    the first one is used as the file's context.
    """
    current = os.path.dirname(normalize_path(file_path))
    root = normalize_path(project_root).lower()
    while True:
        candidates = _candidate_config_paths(current)
        if candidates:
            loaded = [(p, c) for p in candidates for c in [load_tsconfig(p)] if c is not None]
            applicable = [(p, c) for p, c in loaded if tsconfig_applies_to(c, file_path)]
            return applicable or loaded[:1]
        if current.lower() == root or not current.lower().startswith(root):
            return []
        parent = normalize_path(os.path.dirname(current))
        if parent == current:
            return []
        current = parent
