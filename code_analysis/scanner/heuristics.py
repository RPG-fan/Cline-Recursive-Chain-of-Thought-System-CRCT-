"""
Heuristics and symbol checks for CRCT code analysis.
Helps identify patterns like stubs, abstract classes, protocols, etc.
"""

import re
from typing import Any, Dict, List, Optional, cast

# Return-type annotations that are "trivial"
_TRIVIAL_RETURNS = {
    "",
    "None",
    "<class 'NoneType'>",
    "<class 'inspect._empty'>",
    "typing.Any",
    "Any",
}

# Body texts that count as "no real implementation".
_TRIVIAL_BODY_RE = re.compile(r"^\s*(pass|\.\.\.|return\s+None|return)\s*$")


def has_trivial_body(sym: Dict[str, Any]) -> bool:
    """Heuristic: function's source body is empty / pass / `...` / `return None`."""
    ctx: Dict[str, Any] = sym.get("source_context") or {}
    lines: List[str] = ctx.get("source_lines") or []
    if not lines:
        return False
    body: List[str] = []
    in_doc = False
    doc_quote: Optional[str] = None
    for ln in lines[1:]:
        s = ln.strip()
        if not s:
            continue
        if in_doc:
            if doc_quote and doc_quote in s:
                in_doc = False
            continue
        if s.startswith(('"""', "'''")):
            q = s[:3]
            if s.count(q) >= 2 and len(s) > 3:
                continue
            doc_quote = q
            in_doc = True
            continue
        if s.startswith("#"):
            continue
        body.append(s)
    if not body:
        return True
    return all(_TRIVIAL_BODY_RE.match(b) for b in body)


def annotated_non_trivial_return(sym: Dict[str, Any]) -> Optional[str]:
    ann: Dict[str, Any] = sym.get("type_annotations") or {}
    rt = ann.get("return_type") or ann.get("parameters", {}).get("return") or ""
    rt_s = str(rt).strip()
    if rt_s and rt_s not in _TRIVIAL_RETURNS and rt_s != "None":
        return rt_s
    return None


def inherits_from(sym: Dict[str, Any], *needles: str) -> bool:
    inheritance = cast(Dict[str, Any], sym.get("inheritance") or {})
    bases = cast(List[str], inheritance.get("bases") or [])
    mro = cast(List[str], inheritance.get("mro") or [])
    haystack = " ".join(str(part) for part in [*bases, *mro]).lower()
    # Optimization: avoid generator overhead in any()
    for needle in needles:
        if needle.lower() in haystack:
            return True
    return False


def source_mentions(sym: Dict[str, Any], *needles: str) -> bool:
    source_context = cast(Dict[str, Any], sym.get("source_context") or {})
    source_lines = cast(List[str], source_context.get("source_lines") or [])
    haystack = "\n".join(source_lines).lower()
    # Optimization: avoid generator overhead in any()
    for needle in needles:
        if needle.lower() in haystack:
            return True
    return False


def is_protocol_class(sym: Dict[str, Any]) -> bool:
    return inherits_from(sym, "Protocol", "typing.Protocol") or source_mentions(
        sym, "(Protocol", ", Protocol", "typing.Protocol"
    )


def is_abstract_class(sym: Dict[str, Any], runtime_idx_class: Any = None) -> bool:
    # We pass the class reference to avoid circular imports if needed,
    # but here we can just use the static method if we import it or pass the result.
    is_in_mro = False
    if runtime_idx_class:
        is_in_mro = runtime_idx_class.is_in_abstract_mro(sym)

    return is_in_mro or source_mentions(sym, "(ABC", ", ABC", "abc.ABC")


def is_exception_class(sym: Dict[str, Any]) -> bool:
    name = str(sym.get("name") or "")
    inheritance = cast(Dict[str, Any], sym.get("inheritance") or {})
    bases = cast(List[str], inheritance.get("bases") or [])
    mro = cast(List[str], inheritance.get("mro") or [])
    lineage = [name, *[str(item).split(".")[-1] for item in [*bases, *mro]]]
    # Optimization: explicitly unrolled loop instead of any() with generator expression
    for item in lineage:
        if item.endswith(("Error", "Exception")) or item in {
            "Exception",
            "BaseException",
        }:
            return True
    return False


def is_marker_exception_class(sym: Dict[str, Any]) -> bool:
    methods = cast(List[Dict[str, Any]], sym.get("methods") or [])
    return is_exception_class(sym) and not methods


def is_abstract_method(sym: Dict[str, Any]) -> bool:
    decorators = cast(List[str], sym.get("decorators") or [])
    # Optimization: explicitly unrolled loop instead of any() with generator expression
    for decorator in decorators:
        if "abstract" in str(decorator).lower():
            return True
    return source_mentions(sym, "@abstractmethod", "@abc.abstractmethod")


def is_data_container_class(sym: Dict[str, Any]) -> bool:
    name = str(sym.get("name") or "").lower()
    inheritance = cast(Dict[str, Any], sym.get("inheritance") or {})
    bases = cast(List[str], inheritance.get("bases") or [])

    # Check naming conventions
    # Optimization: explicitly unrolled loop instead of any() with generator expression
    for suffix in ("template", "schema", "dto", "enum", "dict", "model"):
        if suffix in name:
            return True

    # Check base class names
    for base in bases:
        base_lower = str(base).lower()
        # Optimization: explicitly unrolled loop instead of any() with generator expression
        for keyword in (
            "template",
            "schema",
            "dto",
            "enum",
            "dict",
            "model",
            "basemodel",
        ):
            if keyword in base_lower:
                return True

    return inherits_from(
        sym,
        "BaseModel",
        "Enum",
        "TypedDict",
    ) or source_mentions(
        sym,
        "@dataclass",
        "(BaseModel",
        "(Enum",
        "(TypedDict",
        "str, Enum",
        "int, Enum",
    )


# Patterns that indicate a data container class when reading source files
_DATA_CONTAINER_SOURCE_PATTERNS = (
    "TypedDict",
    "Enum",
    "IntEnum",
    "NamedTuple",
    "BaseModel",
    "total=False",
    "total=True",
)


def is_data_container_from_source(file_path: str, line: int) -> bool:
    """Fallback: read the source file at *line* and check if the class
    definition line (or the decorator line above it) indicates a data
    container class (TypedDict, Enum, dataclass, NamedTuple, BaseModel).

    This is used when the runtime inspector doesn't capture inheritance
    or source_context for a class, causing is_data_container_class() to
    return False even though the class is actually a data container.
    """
    if not file_path or not line:
        return False
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines_list = f.readlines()
    except Exception:
        return False
    # The runtime inspector may record a line number that's off by 1-2 lines
    # (e.g. a blank line above the class definition). Check a small window
    # around the target line for the class definition and its decorator.
    idx = line - 1  # 0-based index for the reported line
    if idx < 0 or idx >= len(lines_list):
        return False

    # Search a window of -2 to +2 lines around the reported line
    start_idx = max(0, idx - 2)
    end_idx = min(len(lines_list), idx + 3)

    for check_idx in range(start_idx, end_idx):
        check_line = lines_list[check_idx].strip()

        # Check for @dataclass decorator on the line above a class definition
        if check_line.startswith("@dataclass"):
            return True

        # Check class definition line for data container base classes
        if check_line.startswith("class "):
            class_lower = check_line.lower()
            for pattern in _DATA_CONTAINER_SOURCE_PATTERNS:
                if pattern.lower() in class_lower:
                    return True

    return False
