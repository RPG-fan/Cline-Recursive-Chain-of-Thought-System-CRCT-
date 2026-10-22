"""
Bundler and package-manager resolution for JS/TS import specifiers that tsconfig
`paths` / `baseUrl` leave unresolved (see analysis.js_tsconfig for those).

Responsibilities (kept out of dependency_suggester, which only probes candidates):
  - Vite `resolve.alias`, read statically from the nearest vite.config.* (never executed)
  - SvelteKit aliases: the `$lib` default and `kit.alias`, when a svelte.config.* exists
  - workspace packages (package.json `workspaces`, pnpm-workspace.yaml) and the
    nearest package's own name, through package.json `exports`, else
    `source` / `types` / `module` / `main` / index

Every function returns base paths for dependency_suggester's candidate probe
(_js_resolution_candidates), in priority order. Only files inside the project can
match; node_modules packages stay unresolved, and an untracked `exports` target is
not guessed into a source file.

Caching is in-process and validated against each config file's mtime; reset() clears it.
"""

import glob
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from cline_utils.dependency_system.utils.path_utils import normalize_path

logger = logging.getLogger(__name__)

_VITE_CONFIG_NAMES = tuple(
    f"vite.config.{ext}" for ext in ("ts", "mts", "cts", "js", "mjs", "cjs")
)
_SVELTE_CONFIG_NAMES = tuple(f"svelte.config.{ext}" for ext in ("js", "mjs", "cjs", "ts"))
# Conditions a bundler / TS `moduleResolution: bundler` build can match; the first one
# present in an `exports` object, in the object's own key order, wins.
_EXPORT_CONDITIONS = (
    "source", "types", "typings", "import", "module", "browser", "node", "require",
    "development", "production", "default",
)
_PACKAGE_ENTRY_FIELDS = ("source", "types", "typings", "module", "main")

# path -> (mtime, parsed value)
_file_cache: Dict[str, Tuple[float, Any]] = {}
# directory -> nearest directory (or None) holding one of a set of config names
_nearest_cache: Dict[Tuple[str, Tuple[str, ...], str], Optional[str]] = {}
# workspace root -> (mtimes, {package name: package dir})
_workspace_cache: Dict[str, Tuple[Tuple[Tuple[str, float], ...], Dict[str, str]]] = {}


def reset() -> None:
    _file_cache.clear()
    _nearest_cache.clear()
    _workspace_cache.clear()


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return -1.0


def _cached(path: str, parse: Any) -> Any:
    mtime = _mtime(path)
    hit = _file_cache.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    value: Any = None
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            value = parse(fh.read(), path)
    except Exception as e:
        logger.debug(f"Could not read {path}: {e}")
    _file_cache[path] = (mtime, value)
    return value


def _nearest_dir_with(source_dir: str, project_root: str, names: Tuple[str, ...]) -> Optional[str]:
    """Walks up from source_dir (not past project_root) to the first directory that
    holds one of `names`; returns that config file's path."""
    root = normalize_path(project_root)
    key = (normalize_path(source_dir), names, root)
    if key in _nearest_cache:
        return _nearest_cache[key]
    current, found = key[0], None
    while True:
        for name in names:
            path = os.path.join(current, name)
            if os.path.isfile(path):
                found = normalize_path(path)
                break
        if found or current.lower() == root.lower() or not current.lower().startswith(root.lower()):
            break
        parent = normalize_path(os.path.dirname(current))
        if parent == current:
            break
        current = parent
    _nearest_cache[key] = found
    return found


# --------------------------------------------------------------------------- static JS config


def _parse_js_tree(text: str, _path: str) -> Any:
    from tree_sitter import Parser

    from cline_utils.dependency_system.analysis.dependency_analyzer import TS_LANGUAGE

    data = text.encode("utf-8", errors="ignore")
    return Parser(TS_LANGUAGE).parse(data), data


def _node_text(node: Any, data: bytes) -> str:
    return data[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _key_name(pair: Any, data: bytes) -> Optional[str]:
    key = pair.child_by_field_name("key")
    if key is None:
        return None
    text = _node_text(key, data)
    if key.type == "string":
        return text[1:-1]
    if key.type == "property_identifier":
        return text
    return None


def _string_value(node: Any, data: bytes) -> Optional[str]:
    if node.type == "string":
        return _node_text(node, data)[1:-1]
    if node.type == "template_string" and not any(
        c.type == "template_substitution" for c in node.named_children
    ):
        return _node_text(node, data)[1:-1]
    return None


def _eval_path(node: Any, data: bytes, config_dir: str) -> Optional[str]:
    """
    Statically evaluates the path expressions configs use for alias targets:
    string literals, __dirname, process.cwd(), path.resolve/join(...),
    fileURLToPath(new URL("./x", import.meta.url)) and `new URL(...).pathname`.
    Anything else is None (the alias is skipped). process.cwd() is taken to be the
    config's directory, where the bundler is normally run.
    """
    text = _string_value(node, data)
    if text is not None:
        return text
    if node.type == "identifier" and _node_text(node, data) == "__dirname":
        return config_dir
    if node.type == "parenthesized_expression" and node.named_children:
        return _eval_path(node.named_children[0], data, config_dir)
    if node.type == "member_expression":
        prop = node.child_by_field_name("property")
        obj = node.child_by_field_name("object")
        if prop is not None and obj is not None and _node_text(prop, data) == "pathname":
            return _eval_path(obj, data, config_dir)
        return None
    if node.type == "new_expression":
        ctor = node.child_by_field_name("constructor")
        args = node.child_by_field_name("arguments")
        if ctor is None or args is None or _node_text(ctor, data) != "URL":
            return None
        parts = args.named_children
        if len(parts) == 2 and re.sub(r"\s+", "", _node_text(parts[1], data)) == "import.meta.url":
            rel = _string_value(parts[0], data)
            return os.path.join(config_dir, rel) if rel is not None else None
        return None
    if node.type != "call_expression":
        return None
    fn = node.child_by_field_name("function")
    args = node.child_by_field_name("arguments")
    if fn is None or args is None:
        return None
    callee = re.sub(r"\s+", "", _node_text(fn, data))
    arg_nodes = args.named_children
    if callee == "process.cwd" and not arg_nodes:
        return config_dir
    if callee in ("fileURLToPath", "url.fileURLToPath") and len(arg_nodes) == 1:
        return _eval_path(arg_nodes[0], data, config_dir)
    method = callee.rsplit(".", 1)[-1]
    if method not in ("resolve", "join") or callee not in (
        method, f"path.{method}", f"path.posix.{method}", f"path.win32.{method}",
        f"nodePath.{method}",
    ):
        return None
    values = [_eval_path(a, data, config_dir) for a in arg_nodes]
    if not values or any(v is None for v in values):
        return None
    if method == "resolve":
        result = config_dir
        for v in values:
            result = os.path.join(result, v)  # an absolute segment restarts the path
        return result
    return os.path.join(*values)


def _object_pairs(obj: Any) -> List[Any]:
    return [c for c in obj.named_children if c.type == "pair"]


def _find_nested_objects(root: Any, data: bytes, outer_key: str, inner_key: str) -> List[Any]:
    """Values of every `inner_key: …` pair inside an object that is the value of an
    `outer_key: …` pair, anywhere in the file (so `defineConfig({...})`,
    `defineConfig(() => ({...}))` and `return {...}` forms are all found)."""
    found: List[Any] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "pair" and _key_name(node, data) == outer_key:
            value = node.child_by_field_name("value")
            if value is not None and value.type == "object":
                for inner in _object_pairs(value):
                    if _key_name(inner, data) == inner_key:
                        inner_value = inner.child_by_field_name("value")
                        if inner_value is not None:
                            found.append(inner_value)
        stack.extend(reversed(node.named_children))
    return found


def _alias_entries(value: Any, data: bytes, config_dir: str, root_relative: bool) -> List[Tuple[str, str]]:
    """(find, absolute replacement) pairs from an alias object or array. Regex and
    computed `find` values are skipped. A replacement starting with "/" is
    root-relative in Vite (`root_relative`), else it must evaluate to a path."""
    raw: List[Tuple[Optional[str], Any]] = []
    if value.type == "object":
        raw = [(_key_name(p, data), p.child_by_field_name("value")) for p in _object_pairs(value)]
    elif value.type == "array":
        for item in value.named_children:
            if item.type != "object":
                continue
            fields = {_key_name(p, data): p.child_by_field_name("value") for p in _object_pairs(item)}
            find_node = fields.get("find")
            find = _string_value(find_node, data) if find_node is not None else None
            raw.append((find, fields.get("replacement")))
    entries: List[Tuple[str, str]] = []
    for find, target_node in raw:
        if not find or target_node is None:
            continue
        literal = _string_value(target_node, data)
        if root_relative and literal is not None and literal.startswith("/"):
            target: Optional[str] = os.path.join(config_dir, literal.lstrip("/"))
        else:
            target = _eval_path(target_node, data, config_dir)
        if target is None:
            continue
        if not os.path.isabs(target):
            target = os.path.join(config_dir, target)
        entries.append((find, normalize_path(target)))
    return entries


def _parse_vite_aliases(text: str, path: str) -> List[Tuple[str, str]]:
    tree, data = _parse_js_tree(text, path)
    config_dir = os.path.dirname(path)
    entries: List[Tuple[str, str]] = []
    for value in _find_nested_objects(tree.root_node, data, "resolve", "alias"):
        entries.extend(_alias_entries(value, data, config_dir, root_relative=True))
    return entries


def _parse_svelte_aliases(text: str, path: str) -> List[Tuple[str, str]]:
    tree, data = _parse_js_tree(text, path)
    config_dir = os.path.dirname(path)
    entries: List[Tuple[str, str]] = []
    for value in _find_nested_objects(tree.root_node, data, "kit", "alias"):
        entries.extend(_alias_entries(value, data, config_dir, root_relative=False))
    entries.append(("$lib", normalize_path(os.path.join(config_dir, "src", "lib"))))
    return entries


def _apply_aliases(specifier: str, entries: List[Tuple[str, str]]) -> List[str]:
    """Bundler alias matching: a string `find` matches the specifier itself or the
    specifier followed by "/" (so "@" never matches "@org/pkg")."""
    bases: List[str] = []
    for find, target in entries:
        if specifier == find:
            bases.append(target)
        elif specifier.startswith(find.rstrip("/") + "/"):
            bases.append(normalize_path(os.path.join(target, specifier[len(find.rstrip("/")) + 1 :])))
    return bases


def alias_bases(specifier: str, source_dir: str, project_root: str) -> List[str]:
    """Base paths from the nearest Vite config's and svelte config's aliases."""
    bases: List[str] = []
    for names, parse in ((_VITE_CONFIG_NAMES, _parse_vite_aliases), (_SVELTE_CONFIG_NAMES, _parse_svelte_aliases)):
        config = _nearest_dir_with(source_dir, project_root, names)
        if config:
            bases.extend(_apply_aliases(specifier, _cached(config, parse) or []))
    return bases


# --------------------------------------------------------------------------- packages


def _parse_json(text: str, _path: str) -> Any:
    value = json.loads(text)
    return value if isinstance(value, dict) else None


def _parse_pnpm_workspace(text: str, _path: str) -> List[str]:
    """The `packages:` list of pnpm-workspace.yaml (plain block list; no YAML library)."""
    patterns: List[str] = []
    in_packages = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not raw[:1].isspace():
            in_packages = line.strip() == "packages:"
            continue
        item = line.strip()
        if in_packages and item.startswith("-"):
            item = item[1:].strip().strip("'\"")
            if item:
                patterns.append(item)
    return patterns


def _workspace_patterns(root_dir: str) -> List[str]:
    patterns: List[str] = []
    pkg = _cached(os.path.join(root_dir, "package.json"), _parse_json)
    if isinstance(pkg, dict):
        ws = pkg.get("workspaces")
        if isinstance(ws, dict):
            ws = ws.get("packages")
        if isinstance(ws, list):
            patterns.extend(p for p in ws if isinstance(p, str))
    pnpm = os.path.join(root_dir, "pnpm-workspace.yaml")
    if os.path.isfile(pnpm):
        patterns.extend(_cached(pnpm, _parse_pnpm_workspace) or [])
    return patterns


def _workspace_packages(root_dir: str) -> Dict[str, str]:
    """{package name: package dir} for the workspace rooted at root_dir, its own
    package included."""
    pnpm = os.path.join(root_dir, "pnpm-workspace.yaml")
    stamp_paths = [os.path.join(root_dir, "package.json"), pnpm]
    cached = _workspace_cache.get(root_dir)
    if cached is not None and all(_mtime(p) == m for p, m in cached[0]):
        return cached[1]
    packages: Dict[str, str] = {}
    dirs = [root_dir]
    excluded: set = set()
    for pattern in _workspace_patterns(root_dir):
        negated = pattern.startswith("!")
        pattern = pattern.lstrip("!").rstrip("/")
        matches = glob.glob(os.path.join(root_dir, pattern), recursive=True)
        for match in matches:
            if "node_modules" in normalize_path(match).split("/"):
                continue
            if os.path.isfile(os.path.join(match, "package.json")):
                (excluded.add if negated else dirs.append)(normalize_path(match))
    for pkg_dir in dirs:
        if pkg_dir in excluded:
            continue
        pkg_json = os.path.join(pkg_dir, "package.json")
        pkg = _cached(pkg_json, _parse_json)
        if isinstance(pkg, dict) and isinstance(pkg.get("name"), str):
            packages.setdefault(pkg["name"], normalize_path(pkg_dir))
            stamp_paths.append(pkg_json)
    _workspace_cache[root_dir] = (tuple((p, _mtime(p)) for p in stamp_paths), packages)
    return packages


def _split_package_specifier(specifier: str) -> Optional[Tuple[str, str]]:
    """"@org/pkg/sub" -> ("@org/pkg", "./sub"); "pkg" -> ("pkg", ".")."""
    parts = specifier.split("/")
    if specifier.startswith("@"):
        if len(parts) < 2 or not parts[1]:
            return None
        name, rest = "/".join(parts[:2]), parts[2:]
    else:
        name, rest = parts[0], parts[1:]
    if not name or name.startswith("."):
        return None
    return name, "./" + "/".join(rest) if rest else "."


def _export_targets(value: Any) -> List[str]:
    """Targets of one `exports` entry: a string, an array (each in order) or a
    conditions object (the bundler conditions in the object's key order)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [t for item in value for t in _export_targets(item)]
    if isinstance(value, dict):
        return [
            t for key, item in value.items() if key in _EXPORT_CONDITIONS for t in _export_targets(item)
        ]
    return []  # null blocks the subpath


def _exports_targets(exports: Any, subpath: str) -> List[str]:
    """Node's `exports` matching: exact subpath key, else the `*` pattern with the
    longest prefix (the matched text replaces `*` in the target)."""
    if not isinstance(exports, dict) or not any(k.startswith(".") for k in exports):
        return _export_targets(exports) if subpath == "." else []  # sugar: "exports": "./x.js"
    if subpath in exports:
        return _export_targets(exports[subpath])
    best: Optional[Tuple[str, str, Any]] = None
    for pattern, value in exports.items():
        if pattern.count("*") != 1:
            continue
        prefix, suffix = pattern.split("*")
        if (
            subpath.startswith(prefix)
            and subpath.endswith(suffix)
            and len(subpath) >= len(prefix) + len(suffix)
            and (best is None or len(prefix) > len(best[0]))
        ):
            best = (prefix, suffix, value)
    if best is None:
        return []
    prefix, suffix, value = best
    captured = subpath[len(prefix) : len(subpath) - len(suffix)]
    return [t.replace("*", captured) for t in _export_targets(value)]


def _package_bases(pkg_dir: str, subpath: str) -> List[str]:
    pkg = _cached(os.path.join(pkg_dir, "package.json"), _parse_json)
    if not isinstance(pkg, dict):
        return []
    if "exports" in pkg:
        targets = _exports_targets(pkg["exports"], subpath)
    elif subpath == ".":
        targets = [pkg[f] for f in _PACKAGE_ENTRY_FIELDS if isinstance(pkg.get(f), str)] + ["./index"]
    else:
        targets = [subpath]
    bases: List[str] = []
    for target in targets:
        if not target.startswith("./"):
            continue  # exports targets must be package-relative
        base = normalize_path(os.path.join(pkg_dir, target))
        if base not in bases:
            bases.append(base)
    return bases


def package_bases(specifier: str, source_dir: str, project_root: str) -> List[str]:
    """Base paths for a bare specifier naming a workspace package or the nearest
    package itself (self-reference)."""
    split = _split_package_specifier(specifier)
    if split is None:
        return []
    name, subpath = split
    candidates: Dict[str, str] = {}
    nearest = _nearest_dir_with(source_dir, project_root, ("package.json",))
    if nearest:
        pkg = _cached(nearest, _parse_json)
        if isinstance(pkg, dict) and isinstance(pkg.get("name"), str):
            candidates[pkg["name"]] = normalize_path(os.path.dirname(nearest))
    root = normalize_path(project_root)
    current = os.path.dirname(nearest) if nearest else normalize_path(source_dir)
    # The workspace root is the nearest package.json / pnpm-workspace.yaml directory,
    # from the source up to the project root, that declares workspaces.
    while True:
        current = normalize_path(current)
        if _workspace_patterns(current):
            for pkg_name, pkg_dir in _workspace_packages(current).items():
                candidates.setdefault(pkg_name, pkg_dir)
            break
        if current.lower() == root.lower() or not current.lower().startswith(root.lower()):
            break
        parent = os.path.dirname(current)
        if normalize_path(parent) == current:
            break
        current = parent
    pkg_dir = candidates.get(name)
    return _package_bases(pkg_dir, subpath) if pkg_dir else []


def resolution_bases(specifier: str, source_dir: str, project_root: str) -> List[str]:
    """All bundler / workspace base paths for a non-relative specifier, aliases first."""
    if specifier.startswith((".", "/")) or ":" in specifier.split("/", 1)[0]:
        return []  # relative, absolute, or a URL / node: builtin
    try:
        return alias_bases(specifier, source_dir, project_root) + package_bases(
            specifier, source_dir, project_root
        )
    except Exception as e:
        logger.debug(f"Bundler/package resolution failed for '{specifier}': {e}")
        return []
