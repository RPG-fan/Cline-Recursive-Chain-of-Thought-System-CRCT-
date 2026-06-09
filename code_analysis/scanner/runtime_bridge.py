"""
Runtime bridge for CRCT report generator.
Integrates runtime metadata from the inspector with static analysis results.
"""

import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from code_analysis.scanner import heuristics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RUNTIME_SYMBOLS_PATH = os.path.join(
    "cline_utils", "dependency_system", "core", "state", "runtime_symbols.json"
)
PROJECT_SYMBOL_MAP_PATH = os.path.join(
    "cline_utils", "dependency_system", "core", "state", "project_symbol_map.json"
)


# ===========================================================================
# RuntimeIndex Class
# ===========================================================================
class RuntimeIndex:
    """In-memory index over runtime_symbols.json (or merged symbol map)."""

    def __init__(self, data: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.raw = data or {}
        # (norm_file) -> list[(start, end, symbol_dict, kind)]
        self._by_file: Dict[str, List[Tuple[int, int, Dict[str, Any], str]]] = (
            defaultdict(list)
        )
        # qualname -> list of (file, symbol_dict, kind)
        self._by_qualname: Dict[str, List[Tuple[str, Dict[str, Any], str]]] = (
            defaultdict(list)
        )
        # name -> list of files that *reference* the name (callers / accessors)
        self._refs: Dict[str, Set[str]] = defaultdict(set)
        # file -> list of __all__ entries
        self._exports: Dict[str, List[str]] = {}

        self._build()

    @staticmethod
    def norm(p: str) -> str:
        try:
            return os.path.normpath(os.path.abspath(p)).replace("\\", "/")
        except Exception:
            return p

    @staticmethod
    def _line_range(sym: Dict[str, Any]) -> Tuple[int, int]:
        ctx: Dict[str, Any] = sym.get("source_context") or {}
        rng = ctx.get("line_range")
        if isinstance(rng, (list, tuple)) and len(cast(List[Any], rng)) == 2:
            try:
                return int(cast(Any, rng[0])), int(cast(Any, rng[1]))
            except Exception:
                return (0, 0)
        if "line" in sym:
            try:
                start = int(sym["line"])
                return (start, start + 1)
            except Exception:
                return (0, 0)
        return (0, 0)

    def _record_symbol(
        self,
        file_path: str,
        sym: Dict[str, Any],
        kind: str,
        qualname: Optional[str] = None,
    ):
        start, end = self._line_range(sym)
        if start or end:
            self._by_file[file_path].append((start, end, sym, kind))
        qn = qualname or sym.get("name")
        if qn:
            self._by_qualname[qn].append((file_path, sym, kind))

    def _build(self):
        self._all_methods = defaultdict(set)  # method_name -> set(Class_name)
        self._file_defined_classes = defaultdict(set)  # file_path -> set(Class_name)

        for raw_path, finfo in self.raw.items():
            fp = self.norm(raw_path)
            classes_val = finfo.get("classes") or []
            for cls_any in classes_val:
                if isinstance(cls_any, dict):
                    cls_name = cls_any.get("name")
                    if cls_name:
                        self._file_defined_classes[fp].add(cls_name)
                        meths = cls_any.get("methods") or []
                        for meth_any in meths:
                            if isinstance(meth_any, dict):
                                method_name = meth_any.get("name")
                                if method_name:
                                    self._all_methods[method_name].add(cls_name)

        all_project_classes = set()
        for classes in self._file_defined_classes.values():
            all_project_classes.update(classes)

        for raw_path, finfo in self.raw.items():
            file_path = self.norm(raw_path)

            # Exports
            exports_val = finfo.get("exports")
            if isinstance(exports_val, dict):
                self._exports[file_path] = list(
                    cast(Dict[Any, Any], exports_val).keys()
                )
            elif isinstance(exports_val, list):
                self._exports[file_path] = list(cast(List[Any], exports_val))
            else:
                self._exports[file_path] = []

            # Determine active classes in this file
            active_classes = set(self._file_defined_classes[file_path])

            # Add imported classes
            imports_list = finfo.get("imports") or []
            for imp in imports_list:
                if isinstance(imp, dict):
                    inm = imp.get("name") or imp.get("module")
                    if inm in all_project_classes:
                        active_classes.add(inm)
                elif isinstance(imp, str):
                    for cls_name in all_project_classes:
                        if cls_name in imp:
                            active_classes.add(cls_name)

            # Helper to record a reference (using fully qualified names where active)
            def _record_ref(ref_name: str):
                self._refs[ref_name].add(file_path)
                if ref_name in self._all_methods:
                    for cls_name in self._all_methods[ref_name]:
                        if cls_name in active_classes:
                            self._refs[f"{cls_name}.{ref_name}"].add(file_path)

            # Functions
            fns = cast(List[Any], finfo.get("functions") or [])
            for fn_any in fns:
                if isinstance(fn_any, dict):
                    fn = cast(Dict[str, Any], fn_any)
                    self._record_symbol(file_path, fn, "function")
                    attrs = cast(List[Any], fn.get("attribute_accesses") or [])
                    for ref in attrs:
                        if isinstance(ref, str):
                            _record_ref(ref)
                    scope = cast(Dict[str, Any], fn.get("scope_references") or {})
                    g_list = cast(List[Any], scope.get("globals") or [])
                    for g in g_list:
                        if isinstance(g, str):
                            _record_ref(g)

            # Classes (and their methods)
            classes_val = cast(List[Any], finfo.get("classes") or [])
            for cls_any in classes_val:
                if isinstance(cls_any, dict):
                    cls = cast(Dict[str, Any], cls_any)
                    cls_qual: Optional[str] = cast(Optional[str], cls.get("name"))
                    self._record_symbol(file_path, cls, "class", qualname=cls_qual)
                    meths = cast(List[Any], cls.get("methods") or [])
                    for meth_any in meths:
                        if isinstance(meth_any, dict):
                            meth = cast(Dict[str, Any], meth_any)
                            meth_name: str = cast(str, meth.get("name") or "?")
                            qual: str = (
                                f"{cls_qual}.{meth_name}" if cls_qual else meth_name
                            )
                            self._record_symbol(
                                file_path, meth, "method", qualname=qual
                            )
                            m_attrs = cast(
                                List[Any], meth.get("attribute_accesses") or []
                            )
                            for ref in m_attrs:
                                if isinstance(ref, str):
                                    _record_ref(ref)
                            it_scope = cast(
                                Dict[str, Any], meth.get("scope_references") or {}
                            )
                            m_gs = cast(List[Any], it_scope.get("globals") or [])
                            for g in m_gs:
                                if isinstance(g, str):
                                    _record_ref(g)

            # Calls
            calls_list = cast(List[Any], finfo.get("calls") or [])
            for call in calls_list:
                n: Optional[str] = None
                if isinstance(call, dict):
                    n = cast(Optional[str], cast(Dict[Any, Any], call).get("name"))
                elif isinstance(call, str):
                    n = call
                if n:
                    _record_ref(n)

            # Imports
            imports_list = cast(List[Any], finfo.get("imports") or [])
            for imp in imports_list:
                inm: Optional[str] = None
                if isinstance(imp, dict):
                    imp_dict = cast(Dict[Any, Any], imp)
                    inm = cast(
                        Optional[str], imp_dict.get("name") or imp_dict.get("module")
                    )
                elif isinstance(imp, str):
                    inm = imp
                if inm:
                    _record_ref(inm)

        # ---- Load pre-computed connection maps from transparency_registry.json ----
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        registry_path = os.path.normpath(
            os.path.join(
                project_root,
                "cline_utils",
                "dependency_system",
                "core",
                "state",
                "transparency_registry.json",
            )
        )
        if os.path.exists(registry_path):
            try:
                print(
                    f"[runtime] Loading pre-computed connection maps from: {registry_path}"
                )
                with open(registry_path, "r", encoding="utf-8") as f:
                    reg_data = json.load(f)
                files_dict = reg_data.get("files", {})
                for fp, metadata in files_dict.items():
                    if not isinstance(metadata, dict):
                        continue
                    norm_fp = self.norm(fp)
                    conn_maps = metadata.get("connection_maps") or []
                    for rec in conn_maps:
                        if not isinstance(rec, dict):
                            continue
                        target_symbol = rec.get("target_symbol")
                        if target_symbol:
                            self._refs[target_symbol].add(norm_fp)
            except Exception as e:
                print(
                    f"[runtime] Warning: Failed to load connection maps from transparency registry: {e}"
                )

    def symbol_at(self, file_path: str, line: int) -> Optional[Dict[str, Any]]:
        """Return the smallest symbol whose line range encloses *line*."""
        nf = self.norm(file_path)
        candidates = self._by_file.get(nf, [])
        best: Optional[Tuple[int, int, Dict[str, Any], str]] = None
        for start, end, sym, kind in candidates:
            if start <= line <= end:
                if best is None or (end - start) < (best[1] - best[0]):
                    best = (start, end, sym, kind)
        if best is None:
            return None
        start, end, sym, kind = best
        out = dict(sym)
        out["_kind"] = kind
        out["_line_range"] = (start, end)
        return out

    def callers_of(self, name: str, exclude_file: Optional[str] = None) -> List[str]:
        files = self._refs.get(name, set())
        if exclude_file:
            ex = self.norm(exclude_file)
            files = {f for f in files if f != ex}
        return sorted(files)

    def enclosing_class(self, file_path: str, line: int) -> Optional[Dict[str, Any]]:
        nf = self.norm(file_path)
        candidates = self._by_file.get(nf, [])
        best: Optional[Tuple[int, int, Dict[str, Any], str]] = None
        for start, end, sym, kind in candidates:
            if kind != "class":
                continue
            if start <= line <= end:
                if best is None or (end - start) < (best[1] - best[0]):
                    best = (start, end, sym, kind)
        if best is None:
            return None
        start, end, sym, kind = best
        out = dict(sym)
        out["_kind"] = kind
        out["_line_range"] = (start, end)
        return out

    def is_exported(self, file_path: str, name: str) -> bool:
        return name in (self._exports.get(self.norm(file_path)) or [])

    @staticmethod
    def is_in_abstract_mro(sym: Dict[str, Any]) -> bool:
        inh = sym.get("inheritance")
        if not isinstance(inh, dict):
            return False
        orig = inh.get("_haystack_original")
        low = inh.get("_haystack_lower")
        if orig is None or low is None:
            bases = inh.get("bases") or []
            mro = inh.get("mro") or []
            orig = " ".join(bases) + " " + " ".join(mro)
            low = orig.lower()
            inh["_haystack_original"] = orig
            inh["_haystack_lower"] = low
        return "ABC" in str(orig) or "abc." in str(low)


# ===========================================================================
# Data Loading & Auto-Run
# ===========================================================================
def load_runtime_data(project_root_path: str) -> Dict[str, Dict[str, Any]]:
    """Locate the richest runtime data available."""
    merged_path = os.path.join(project_root_path, PROJECT_SYMBOL_MAP_PATH)
    runtime_path = os.path.join(project_root_path, RUNTIME_SYMBOLS_PATH)

    if os.path.exists(merged_path):
        try:
            with open(merged_path, "r", encoding="utf-8") as f:
                print(f"[runtime] Using merged symbol map: {merged_path}")
                return json.load(f)
        except Exception as e:
            print(f"[runtime] Failed to load merged map ({e}); falling back.")

    if os.path.exists(runtime_path):
        try:
            with open(runtime_path, "r", encoding="utf-8") as f:
                print(f"[runtime] Using runtime symbols: {runtime_path}")
                return json.load(f)
        except Exception as e:
            print(f"[runtime] Failed to load runtime symbols ({e}).")

    print("[runtime] No runtime data found; running in static-only mode.")
    return {}


def maybe_run_runtime_inspector(project_root_path: str) -> None:
    """Best-effort: run the inspector if its output is missing."""
    # Sanitize and validate the path to prevent arbitrary command execution or path traversal
    safe_root = os.path.abspath(os.path.normpath(project_root_path))
    if not os.path.isdir(safe_root):
        print(f"[runtime] Error: Invalid project root path: {project_root_path}")
        return

    if os.environ.get("CRCT_AUTO_RUNTIME") != "1":
        return
    target = os.path.join(safe_root, RUNTIME_SYMBOLS_PATH)
    if os.path.exists(target):
        return
    print("[runtime] CRCT_AUTO_RUNTIME=1; invoking runtime_inspector...")
    try:
        subprocess.run(  # nosec B603
            [
                sys.executable,
                "-m",
                "cline_utils.dependency_system.analysis.runtime_inspector",
                safe_root,
            ],
            cwd=safe_root,
            check=False,
            timeout=300,
        )
    except Exception as e:
        print(f"[runtime] Auto-run failed: {e}")


# ===========================================================================
# Suppression Logic
# ===========================================================================
def _should_suppress_issue(issue: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    owning = cast(Dict[str, Any], ctx.get("owning_symbol") or {})
    inheritance = cast(Dict[str, Any], ctx.get("inheritance") or {})
    qualname = str(owning.get("qualname") or "")
    subtype = str(issue.get("subtype") or "")
    kind = str(owning.get("kind") or "")
    file_path = str(issue.get("file") or "")

    sym_for_checks: Dict[str, Any] = {"inheritance": inheritance}
    if "." not in qualname and kind == "class":
        sym_for_checks["methods"] = []

    # 1. Suppress stubs in test files or if the name indicates a mock/noop callback
    is_test_file = (
        "test_" in file_path.lower()
        or "/tests/" in file_path.lower()
        or "\\tests\\" in file_path.lower()
    )
    name_lower = (qualname or issue.get("content") or "").lower()
    is_mock_or_noop = (
        name_lower.startswith("mock")
        or name_lower.startswith("test")
        or "mock_" in name_lower
        or "noop" in name_lower
        or "dummy" in name_lower
        or "callback" in name_lower
        or name_lower.endswith("_cb")
        or name_lower == "teardown"
        or name_lower == "setup"
    )
    if is_test_file or is_mock_or_noop:
        if subtype in {
            "Empty/Stub Function",
            "Empty/Stub Class",
            "Bare Class",
            "Concrete Stub Method",
            "Async Stub",
        }:
            return True

    # 2. Suppress abstract method/class stubs
    if ctx.get("abstract_method") or ctx.get("abstract_class"):
        if subtype in {
            "NotImplementedError",
            "Empty/Stub Function",
            "Annotated Stub",
            "Concrete Stub Method",
            "Async Stub",
        }:
            return True

    # 3. Suppress markers/exceptions
    if subtype == "Empty/Stub Class" and heuristics.is_marker_exception_class(
        sym_for_checks
    ):
        return True

    # 4. Suppress TypedDict/Enum data container classes (Bare and Empty/Stub Class)
    if subtype in {"Empty/Stub Class", "Bare Class"} and (
        ctx.get("data_container_class")
        or heuristics.is_data_container_class(sym_for_checks)
    ):
        return True

    # 5. Suppress stubs on Protocol classes
    if (
        subtype
        in {
            "Annotated Stub",
            "Concrete Stub Method",
            "Async Stub",
            "Empty/Stub Function",
        }
        and "." in qualname
        and heuristics.is_protocol_class(sym_for_checks)
    ):
        return True

    # 6. Suppress regex-based "simplified", "placeholder", "for now", "in a real" findings
    # if they occur inside a function/method/class that has a substantial (non-trivial) body.
    if subtype in {"simplified", "placeholder", "for now", "in a real"}:
        if ctx.get("has_trivial_body") is False:
            return True

    return False


# ===========================================================================
# Public Enhancement API
# ===========================================================================
def runtime_only_findings(idx: RuntimeIndex) -> List[Dict[str, Any]]:
    """Emit issues that the original static pipeline cannot detect."""
    findings: List[Dict[str, Any]] = []

    for file_path, finfo in idx.raw.items():
        nf = RuntimeIndex.norm(file_path)
        fns = cast(List[Dict[str, Any]], finfo.get("functions") or [])
        for fn in fns:
            _emit_runtime_issues_for_symbol(
                findings,
                idx,
                nf,
                fn,
                kind="function",
                qualname=cast(str, fn.get("name", "?")),
            )
        classes = cast(List[Dict[str, Any]], finfo.get("classes") or [])
        for cls in classes:
            cls_name: str = cast(str, cls.get("name") or "?")
            if heuristics.is_marker_exception_class(
                cls
            ) or heuristics.is_data_container_class(cls):
                continue
            cls_ctx: Dict[str, Any] = cast(
                Dict[str, Any], cls.get("source_context") or {}
            )
            inheritance = cast(Dict[str, Any], cls.get("inheritance") or {})
            if not (cls.get("methods") or []) and not (inheritance.get("bases") or []):
                findings.append(
                    {
                        "type": "Improper Implementation (runtime)",
                        "subtype": "Bare Class",
                        "file": nf,
                        "line": cast(List[int], cls_ctx.get("line_range", [0]))[0],
                        "content": f"class {cls_name}: (no methods, no bases)",
                    }
                )
            methods = cast(List[Dict[str, Any]], cls.get("methods") or [])
            for meth in methods:
                meth_name = cast(str, meth.get("name") or "?")
                qn = f"{cls_name}.{meth_name}"
                _emit_runtime_issues_for_symbol(
                    findings,
                    idx,
                    nf,
                    meth,
                    kind="method",
                    qualname=qn,
                    owning_class=cls,
                )
    return findings


def _emit_runtime_issues_for_symbol(
    sink: List[Dict[str, Any]],
    idx: RuntimeIndex,
    file_path: str,
    sym: Dict[str, Any],
    kind: str,
    qualname: str,
    owning_class: Optional[Dict[str, Any]] = None,
) -> None:
    name: str = cast(str, sym.get("name") or qualname)
    ctx: Dict[str, Any] = cast(Dict[str, Any], sym.get("source_context") or {})
    line: int = cast(List[int], ctx.get("line_range", [0]))[0]

    is_trivial = heuristics.has_trivial_body(sym)
    annotated_return = heuristics.annotated_non_trivial_return(sym)
    decorators = cast(List[str], sym.get("decorators") or [])
    is_async = bool(sym.get("is_async"))
    is_abstract = False
    for d in decorators:
        d_lower = str(d).lower()
        if "async" in d_lower:
            is_async = True
        if "abstract" in d_lower:
            is_abstract = True
        if is_async and is_abstract:
            break
    belongs_to_protocol = owning_class is not None and heuristics.is_protocol_class(
        owning_class
    )

    # 1. Annotated stub
    if is_trivial and annotated_return and not is_abstract and not belongs_to_protocol:
        sink.append(
            {
                "type": "Incomplete Implementation (runtime)",
                "subtype": "Annotated Stub",
                "file": file_path,
                "line": line,
                "content": f"{qualname}{sym.get('signature','')} -> {annotated_return}",
                "_runtime_only": True,
                "_qualname": qualname,
            }
        )

    # 2. Exported placeholder
    if is_trivial and idx.is_exported(file_path, name):
        sink.append(
            {
                "type": "Incomplete Implementation (runtime)",
                "subtype": "Exported Placeholder",
                "file": file_path,
                "line": line,
                "content": f"{qualname} is in __all__ but body is trivial",
                "_runtime_only": True,
                "_qualname": qualname,
            }
        )

    # 3. Async without await
    if is_async and is_trivial and not belongs_to_protocol:
        sink.append(
            {
                "type": "Improper Implementation (runtime)",
                "subtype": "Async Stub",
                "file": file_path,
                "line": line,
                "content": f"async {qualname} has trivial body",
                "_runtime_only": True,
                "_qualname": qualname,
            }
        )

    # 4. Stub method on non-abstract class
    if kind == "method" and is_trivial and owning_class is not None:
        if (
            not belongs_to_protocol
            and not RuntimeIndex.is_in_abstract_mro(owning_class)
            and not is_abstract
        ):
            sink.append(
                {
                    "type": "Improper Implementation (runtime)",
                    "subtype": "Concrete Stub Method",
                    "file": file_path,
                    "line": line,
                    "content": f"{qualname} is a stub on a non-abstract class",
                    "_runtime_only": True,
                    "_qualname": qualname,
                }
            )

    # 5. Orphan
    if name and not name.startswith("_"):
        # For methods, query using qualname (which is Class.method).
        # For other kinds, query using name.
        query_name = qualname if kind == "method" else name

        if idx.is_exported(file_path, name):
            callers = idx.callers_of(query_name, exclude_file=file_path)
            if not callers and kind == "method":
                callers = idx.callers_of(name, exclude_file=file_path)
            if not callers:
                sink.append(
                    {
                        "type": "Unused Item (runtime)",
                        "subtype": "Orphan Export",
                        "file": file_path,
                        "line": line,
                        "content": f"{qualname} exported but no other file references it",
                        "_runtime_only": True,
                        "_qualname": qualname,
                    }
                )
        else:
            # If unexported, verify it has zero callers anywhere (internal or external)
            callers = idx.callers_of(query_name, exclude_file=None)
            if not callers and kind == "method":
                callers = idx.callers_of(name, exclude_file=None)
            if not callers:
                sink.append(
                    {
                        "type": "Unused Item (runtime)",
                        "subtype": "Dead Unexported Code",
                        "file": file_path,
                        "line": line,
                        "content": f"{qualname} is unexported and has zero caller references",
                        "_runtime_only": True,
                        "_qualname": qualname,
                    }
                )


def enrich_issue(issue: Dict[str, Any], idx: RuntimeIndex) -> Dict[str, Any]:
    """Attach runtime-derived context fields to a single issue dict."""
    ctx: Dict[str, Any] = {}
    file_path: str = issue.get("file", "")
    line: int = int(issue.get("line", 0) or 0)

    sym: Optional[Dict[str, Any]] = idx.symbol_at(file_path, line) if line else None
    enclosing_class: Optional[Dict[str, Any]] = (
        idx.enclosing_class(file_path, line) if line else None
    )
    if sym:
        ctx["owning_symbol"] = {
            "name": sym.get("name"),
            "qualname": issue.get("_qualname") or sym.get("name"),
            "kind": sym.get("_kind"),
            "signature": sym.get("signature"),
            "docstring": (sym.get("docstring") or "")[:240] or None,
            "line_range": list(sym.get("_line_range", [])),
        }
        if sym.get("type_annotations"):
            ctx["type_annotations"] = sym["type_annotations"]
        if sym.get("decorators"):
            ctx["decorators"] = sym["decorators"]
        if sym.get("inheritance"):
            ctx["inheritance"] = {k: v for k, v in sym["inheritance"].items() if not k.startswith("_")}
        scope: Dict[str, Any] = sym.get("scope_references") or {}
        if scope:
            ctx["scope_references"] = scope
        if sym.get("closure_dependencies"):
            ctx["closure_dependencies"] = sym["closure_dependencies"]
        if sym.get("attribute_accesses"):
            ctx["attribute_accesses"] = list(sym["attribute_accesses"])[:25]
        ctx["abstract_method"] = heuristics.is_abstract_method(sym)
        if sym.get("_kind") == "class":
            ctx["data_container_class"] = heuristics.is_data_container_class(sym)
        ctx["has_trivial_body"] = heuristics.has_trivial_body(sym)

        # Cross-file links
        name = sym.get("name")
        if name:
            # Task 39: If kind is method and enclosing class is present, qualify target name
            if sym.get("_kind") == "method" and enclosing_class:
                class_name = enclosing_class.get("name")
                target_name = f"{class_name}.{name}" if class_name else name
            else:
                target_name = name

            callers = idx.callers_of(target_name, exclude_file=file_path)
            if not callers and sym.get("_kind") == "method":
                callers = idx.callers_of(name, exclude_file=file_path)
            if callers:
                ctx["linked_areas"] = {
                    "callers": callers[:20],
                    "caller_count": len(callers),
                }
            ctx["exported"] = idx.is_exported(file_path, name)

    if enclosing_class:
        ctx["abstract_class"] = heuristics.is_abstract_class(
            enclosing_class, runtime_idx_class=RuntimeIndex
        )
        if ctx.get("data_container_class") is None:
            ctx["data_container_class"] = heuristics.is_data_container_class(
                enclosing_class
            )
        if not ctx.get("inheritance") and enclosing_class.get("inheritance"):
            ctx["inheritance"] = {k: v for k, v in enclosing_class["inheritance"].items() if not k.startswith("_")}

    ctx["severity"] = score_severity(issue, ctx)
    issue_copy = dict(issue)
    issue_copy["context"] = ctx

    if _should_suppress_issue(issue_copy, ctx):
        issue_copy["_suppress"] = True

    return issue_copy


_SUBTYPE_SCORES = {
    "Annotated Stub": 2,
    "Exported Placeholder": 2,
    "Concrete Stub Method": 2,
    "Empty/Stub Function": 1,
    "Empty/Stub Class": 1,
    "Async Stub": 1,
}


def score_severity(issue: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    score = 0
    sub = issue.get("subtype", "")
    if "NotImplementedError" in sub:
        score += 2
    score += _SUBTYPE_SCORES.get(sub, 0)
    if ctx.get("exported"):
        score += 1
    if ctx.get("linked_areas", {}).get("caller_count", 0) > 0:
        score += 1
    if score >= 4:
        return "critical"
    if score >= 2:
        return "high"
    if score >= 1:
        return "medium"
    return "low"
