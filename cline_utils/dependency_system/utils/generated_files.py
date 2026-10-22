"""
Detection of generated JavaScript (build output) during key generation.

Build output duplicates its sources in the dependency graph: a compiled `.js` beside its
`.ts` gets its own key, semantic suggestions and tracker cells, while imports already
resolve to the `.ts` source. With `exclude_generated_files` (default on), key
generation skips files this module classifies as generated:

  - twin:      a `.js` / `.jsx` / `.mjs` / `.cjs` file next to a TypeScript source of the
               same stem (`.ts`/`.tsx`, `.tsx`, `.mts`, `.cts`), or a `.d.ts` / `.d.mts` /
               `.d.cts` declaration next to its source;
  - minified:  `*.min.js` / `*.min.mjs` / `*.min.cjs`;
  - sourcemap: a JS file or declaration whose last line is a `//# sourceMappingURL=`
               comment (emitted by tsc, bundlers and minifiers).

Only the file's name, its directory listing and the last bytes of JS files are read.
"""

import os
from typing import Collection, Optional, Tuple

_TS_TWINS = {
    ".js": (".ts", ".tsx"),
    ".jsx": (".tsx",),
    ".mjs": (".mts",),
    ".cjs": (".cts",),
}
_DECLARATION_TWINS = {".d.ts": (".ts", ".tsx"), ".d.mts": (".mts",), ".d.cts": (".cts",)}
_MINIFIED_SUFFIXES = (".min.js", ".min.mjs", ".min.cjs")
_SOURCEMAP_SUFFIXES = (".js", ".mjs", ".cjs", ".jsx", ".d.ts", ".d.mts", ".d.cts")
_SOURCEMAP_MARKERS = (b"//# sourceMappingURL=", b"//@ sourceMappingURL=")
_TAIL_BYTES = 512

REASONS: Tuple[str, ...] = ("twin", "minified", "sourcemap")


def _declaration_suffix(lower_name: str) -> Optional[str]:
    return next((s for s in _DECLARATION_TWINS if lower_name.endswith(s)), None)


def _ends_with_source_map(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - _TAIL_BYTES))
            tail = fh.read()
    except OSError:
        return False
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return bool(lines) and lines[-1].startswith(_SOURCEMAP_MARKERS)


def generated_reason(file_name: str, dir_path: str, sibling_names: Collection[str]) -> Optional[str]:
    """
    Why `file_name` (in `dir_path`) is generated output, or None. `sibling_names` is the
    directory listing (case-insensitive comparison; pass lower-cased names for speed).
    """
    lower = file_name.lower()
    siblings = sibling_names
    declaration = _declaration_suffix(lower)
    if declaration is not None:
        stem = lower[: -len(declaration)]
        if any(stem + ext in siblings for ext in _DECLARATION_TWINS[declaration]):
            return "twin"
    else:
        stem, ext = os.path.splitext(lower)
        if ext in _TS_TWINS and any(stem + twin in siblings for twin in _TS_TWINS[ext]):
            return "twin"
        if lower.endswith(_MINIFIED_SUFFIXES):
            return "minified"
    if lower.endswith(_SOURCEMAP_SUFFIXES) and _ends_with_source_map(os.path.join(dir_path, file_name)):
        return "sourcemap"
    return None
