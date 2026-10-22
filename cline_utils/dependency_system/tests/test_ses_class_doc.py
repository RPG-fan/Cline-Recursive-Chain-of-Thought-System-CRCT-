"""
The SES class block printed a class docstring twice (once under the class line, again
after BASES / DECORATORS). It now appears once, for Python and JS/TS class records alike.
No model is loaded: generate_symbol_essence_string only formats symbol data.
"""

from pathlib import Path

import pytest

import cline_utils.dependency_system.analysis.embedding_manager as em
from cline_utils.dependency_system.analysis.embedding_manager import (
    generate_symbol_essence_string,
)


@pytest.mark.parametrize(
    "file_name, record",
    [
        ("svc.py", {"name": "Svc", "line": 1, "docstring": "Runs the service."}),
        (
            "svc.ts",
            {
                "name": "Svc", "line": 1, "kind": "class", "docstring": "Runs the service.",
                "decorators": ["Injectable"],
            },
        ),
    ],
)
def test_class_docstring_appears_once(tmp_path: Path, monkeypatch, file_name: str, record: dict) -> None:
    monkeypatch.setattr(em, "get_project_root", lambda: str(tmp_path))
    path = tmp_path / file_name
    path.write_text("", encoding="utf-8")
    ses = generate_symbol_essence_string(str(path), {"classes": [record]})
    assert ses.count("DOC: Runs the service.") == 1
    lines = ses.splitlines()
    head = lines.index("CLASS: Svc")
    assert lines[head + 1] == "  DOC: Runs the service."
    if "decorators" in record:
        assert "  DECORATORS: Injectable" in lines
