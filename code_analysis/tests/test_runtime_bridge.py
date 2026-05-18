import json
from unittest.mock import patch
from code_analysis.scanner.runtime_bridge import (
    RuntimeIndex,
    load_runtime_data,
    runtime_only_findings,
    _emit_runtime_issues_for_symbol,
    enrich_issue,
    score_severity,
    _should_suppress_issue,
    maybe_run_runtime_inspector,
)


def test_line_range_valid_context():
    sym = {"source_context": {"line_range": [10, 20]}}
    assert RuntimeIndex._line_range(sym) == (10, 20)


def test_line_range_invalid_context_type():
    # Triggers exception in int() conversion
    sym = {"source_context": {"line_range": ["abc", "def"]}}
    assert RuntimeIndex._line_range(sym) == (0, 0)


def test_line_range_wrong_length_fallback_to_line():
    sym = {"source_context": {"line_range": [10]}, "line": 5}
    assert RuntimeIndex._line_range(sym) == (5, 6)


def test_line_range_wrong_length_no_line():
    sym = {"source_context": {"line_range": [10]}}
    assert RuntimeIndex._line_range(sym) == (0, 0)


def test_line_range_valid_line_no_context():
    sym = {"line": 5}
    assert RuntimeIndex._line_range(sym) == (5, 6)


def test_line_range_invalid_line_type():
    # Triggers exception in int() conversion for "line"
    sym = {"line": "abc"}
    assert RuntimeIndex._line_range(sym) == (0, 0)


def test_line_range_missing_all():
    sym = {}
    assert RuntimeIndex._line_range(sym) == (0, 0)


def test_line_range_none_context():
    sym = {"source_context": None}
    assert RuntimeIndex._line_range(sym) == (0, 0)


# Tests for load_runtime_data


def test_load_runtime_data_merged_success(tmp_path):
    project_root = str(tmp_path)
    with patch(
        "code_analysis.scanner.runtime_bridge.PROJECT_SYMBOL_MAP_PATH", "merged.json"
    ), patch(
        "code_analysis.scanner.runtime_bridge.RUNTIME_SYMBOLS_PATH", "runtime.json"
    ):
        merged_file = tmp_path / "merged.json"
        merged_file.write_text(json.dumps({"merged": "data"}))

        result = load_runtime_data(project_root)
        assert result == {"merged": "data"}


def test_load_runtime_data_merged_invalid_fallback_runtime_success(tmp_path):
    project_root = str(tmp_path)
    with patch(
        "code_analysis.scanner.runtime_bridge.PROJECT_SYMBOL_MAP_PATH", "merged.json"
    ), patch(
        "code_analysis.scanner.runtime_bridge.RUNTIME_SYMBOLS_PATH", "runtime.json"
    ):
        merged_file = tmp_path / "merged.json"
        merged_file.write_text("invalid json")

        runtime_file = tmp_path / "runtime.json"
        runtime_file.write_text(json.dumps({"runtime": "data"}))

        result = load_runtime_data(project_root)
        assert result == {"runtime": "data"}


def test_load_runtime_data_only_runtime_success(tmp_path):
    project_root = str(tmp_path)
    with patch(
        "code_analysis.scanner.runtime_bridge.PROJECT_SYMBOL_MAP_PATH", "merged.json"
    ), patch(
        "code_analysis.scanner.runtime_bridge.RUNTIME_SYMBOLS_PATH", "runtime.json"
    ):
        runtime_file = tmp_path / "runtime.json"
        runtime_file.write_text(json.dumps({"runtime": "only"}))

        result = load_runtime_data(project_root)
        assert result == {"runtime": "only"}


def test_load_runtime_data_both_invalid(tmp_path):
    project_root = str(tmp_path)
    with patch(
        "code_analysis.scanner.runtime_bridge.PROJECT_SYMBOL_MAP_PATH", "merged.json"
    ), patch(
        "code_analysis.scanner.runtime_bridge.RUNTIME_SYMBOLS_PATH", "runtime.json"
    ):
        merged_file = tmp_path / "merged.json"
        merged_file.write_text("invalid json 1")

        runtime_file = tmp_path / "runtime.json"
        runtime_file.write_text("invalid json 2")

        result = load_runtime_data(project_root)
        assert result == {}


def test_load_runtime_data_neither_exists(tmp_path):
    project_root = str(tmp_path)
    with patch(
        "code_analysis.scanner.runtime_bridge.PROJECT_SYMBOL_MAP_PATH", "merged.json"
    ), patch(
        "code_analysis.scanner.runtime_bridge.RUNTIME_SYMBOLS_PATH", "runtime.json"
    ):
        result = load_runtime_data(project_root)
        assert result == {}


def test_load_runtime_data_merged_open_exception_fallback(tmp_path):
    project_root = str(tmp_path)
    with patch(
        "code_analysis.scanner.runtime_bridge.PROJECT_SYMBOL_MAP_PATH", "merged.json"
    ), patch(
        "code_analysis.scanner.runtime_bridge.RUNTIME_SYMBOLS_PATH", "runtime.json"
    ):

        merged_file = tmp_path / "merged.json"
        merged_file.write_text(json.dumps({"merged": "data"}))

        runtime_file = tmp_path / "runtime.json"
        runtime_file.write_text(json.dumps({"runtime": "data"}))

        original_open = open

        def mocked_open(file, *args, **kwargs):
            if "merged.json" in str(file):
                raise OSError("Mocked exception")
            return original_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=mocked_open):
            result = load_runtime_data(project_root)
            assert result == {"runtime": "data"}


def test_load_runtime_data_runtime_open_exception(tmp_path):
    project_root = str(tmp_path)
    with patch(
        "code_analysis.scanner.runtime_bridge.PROJECT_SYMBOL_MAP_PATH", "merged.json"
    ), patch(
        "code_analysis.scanner.runtime_bridge.RUNTIME_SYMBOLS_PATH", "runtime.json"
    ):

        runtime_file = tmp_path / "runtime.json"
        runtime_file.write_text(json.dumps({"runtime": "data"}))

        original_open = open

        def mocked_open(file, *args, **kwargs):
            if "runtime.json" in str(file):
                raise OSError("Mocked exception")
            return original_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=mocked_open):
            result = load_runtime_data(project_root)
            assert result == {}


# Tests for RuntimeIndex.__init__


def test_runtime_index_init_empty():
    idx1 = RuntimeIndex({})
    assert idx1._by_file == {}
    assert idx1._by_qualname == {}
    assert idx1._refs == {}
    assert idx1._exports == {}

    idx2 = RuntimeIndex(None)  # type: ignore
    assert idx2._by_file == {}
    assert idx2._by_qualname == {}
    assert idx2._refs == {}
    assert idx2._exports == {}


def test_runtime_index_init_exports():
    data = {
        "file1.py": {"exports": ["func1", "func2"]},
        "file2.py": {"exports": {"Class1": "some_value"}},
        "file3.py": {"exports": "invalid_type"},
        "file4.py": {},
    }
    idx = RuntimeIndex(data)

    nf1 = RuntimeIndex.norm("file1.py")
    nf2 = RuntimeIndex.norm("file2.py")
    nf3 = RuntimeIndex.norm("file3.py")
    nf4 = RuntimeIndex.norm("file4.py")

    assert idx._exports[nf1] == ["func1", "func2"]
    assert idx._exports[nf2] == ["Class1"]
    assert idx._exports[nf3] == []
    assert idx._exports[nf4] == []


def test_runtime_index_init_functions():
    data = {
        "main.py": {
            "functions": [
                {
                    "name": "my_func",
                    "line": 10,
                    "attribute_accesses": ["os.path", "sys.argv"],
                    "scope_references": {"globals": ["GLOBAL_VAR"]},
                },
                "invalid_function_type",  # Should be skipped
            ]
        }
    }
    idx = RuntimeIndex(data)
    nf = RuntimeIndex.norm("main.py")

    # Check _by_file
    assert nf in idx._by_file
    assert len(idx._by_file[nf]) == 1
    start, end, sym, kind = idx._by_file[nf][0]
    assert start == 10
    assert end == 11
    assert sym["name"] == "my_func"
    assert kind == "function"

    # Check _by_qualname
    assert "my_func" in idx._by_qualname
    assert len(idx._by_qualname["my_func"]) == 1
    file_path, q_sym, q_kind = idx._by_qualname["my_func"][0]
    assert file_path == nf
    assert q_sym["name"] == "my_func"
    assert q_kind == "function"

    # Check _refs
    assert nf in idx._refs["os.path"]
    assert nf in idx._refs["sys.argv"]
    assert nf in idx._refs["GLOBAL_VAR"]


def test_runtime_index_init_classes_and_methods():
    data = {
        "models.py": {
            "classes": [
                {
                    "name": "User",
                    "line": 20,
                    "methods": [
                        {
                            "name": "save",
                            "line": 25,
                            "attribute_accesses": ["db.session"],
                            "scope_references": {"globals": ["DB_COMMIT"]},
                        },
                        "invalid_method_type",  # Should be skipped
                    ],
                },
                "invalid_class_type",  # Should be skipped
            ]
        }
    }
    idx = RuntimeIndex(data)
    nf = RuntimeIndex.norm("models.py")

    # Check _by_file
    assert nf in idx._by_file
    assert len(idx._by_file[nf]) == 2  # 1 class + 1 method

    class_syms = [s for s in idx._by_file[nf] if s[3] == "class"]
    assert len(class_syms) == 1
    assert class_syms[0][2]["name"] == "User"

    method_syms = [s for s in idx._by_file[nf] if s[3] == "method"]
    assert len(method_syms) == 1
    assert method_syms[0][2]["name"] == "save"

    # Check _by_qualname for class
    assert "User" in idx._by_qualname
    assert idx._by_qualname["User"][0][1]["name"] == "User"

    # Check _by_qualname for method (should be qualified)
    assert "User.save" in idx._by_qualname
    assert idx._by_qualname["User.save"][0][1]["name"] == "save"

    # Check _refs
    assert nf in idx._refs["db.session"]
    assert nf in idx._refs["DB_COMMIT"]


def test_runtime_index_init_calls_and_imports():
    data = {
        "utils.py": {
            "calls": [
                "print",
                {"name": "os.getenv"},
                "invalid_call_type",  # Should be handled properly (ignored if invalid, but str is valid)
            ],
            "imports": [
                "sys",
                {"name": "json"},
                {"module": "typing"},
                "invalid_import_type",  # Should be handled properly
            ],
        }
    }
    idx = RuntimeIndex(data)
    nf = RuntimeIndex.norm("utils.py")

    # Check _refs for calls
    assert nf in idx._refs["print"]
    assert nf in idx._refs["os.getenv"]
    assert nf in idx._refs["invalid_call_type"]  # Because strings are valid calls

    # Check _refs for imports
    assert nf in idx._refs["sys"]
    assert nf in idx._refs["json"]
    assert nf in idx._refs["typing"]
    assert nf in idx._refs["invalid_import_type"]  # Because strings are valid imports


# Tests for RuntimeIndex methods


def test_runtime_index_symbol_at():
    data = {
        "file1.py": {
            "functions": [
                {"name": "f1", "line": 10},
                {"name": "f2", "source_context": {"line_range": [20, 25]}},
            ]
        }
    }
    idx = RuntimeIndex(data)

    # Matching exact line
    sym = idx.symbol_at("file1.py", 10)
    assert sym is not None
    assert sym["name"] == "f1"

    # Matching range
    sym2 = idx.symbol_at("file1.py", 22)
    assert sym2 is not None
    assert sym2["name"] == "f2"

    # No match
    sym3 = idx.symbol_at("file1.py", 30)
    assert sym3 is None


def test_runtime_index_callers_of():
    data = {
        "caller1.py": {
            "functions": [
                {"name": "f1", "line": 1, "attribute_accesses": ["target_func"]}
            ]
        },
        "caller2.py": {"calls": ["target_func"]},
    }
    idx = RuntimeIndex(data)
    nf1 = RuntimeIndex.norm("caller1.py")
    nf2 = RuntimeIndex.norm("caller2.py")

    callers = idx.callers_of("target_func")
    assert sorted(callers) == sorted([nf1, nf2])

    callers_exclude = idx.callers_of("target_func", exclude_file="caller1.py")
    assert callers_exclude == [nf2]

    assert idx.callers_of("unknown_func") == []


def test_runtime_index_enclosing_class():
    data = {
        "models.py": {
            "classes": [
                {
                    "name": "User",
                    "source_context": {"line_range": [10, 30]},
                    "methods": [{"name": "save", "line": 20}],
                }
            ]
        }
    }
    idx = RuntimeIndex(data)

    cls = idx.enclosing_class("models.py", 20)
    assert cls is not None
    assert cls["name"] == "User"

    cls2 = idx.enclosing_class("models.py", 40)
    assert cls2 is None


def test_runtime_index_is_exported():
    data = {"module.py": {"exports": ["my_func", "MyClass"]}}
    idx = RuntimeIndex(data)

    assert idx.is_exported("module.py", "my_func") is True
    assert idx.is_exported("module.py", "MyClass") is True
    assert idx.is_exported("module.py", "hidden_func") is False


def test_runtime_index_is_in_abstract_mro():
    sym1 = {"inheritance": {"bases": ["ABC"]}}
    assert RuntimeIndex.is_in_abstract_mro(sym1) is True

    sym2 = {"inheritance": {"mro": ["abc.ABC"]}}
    assert RuntimeIndex.is_in_abstract_mro(sym2) is True

    sym3 = {"inheritance": {"bases": ["object"]}}
    assert RuntimeIndex.is_in_abstract_mro(sym3) is False

    sym4 = {}
    assert RuntimeIndex.is_in_abstract_mro(sym4) is False


# Tests for runtime_only_findings and _emit_runtime_issues_for_symbol


def test_runtime_only_findings():
    data = {
        "file1.py": {
            "functions": [
                {
                    "name": "func_stub",
                    "line": 10,
                    "decorators": [],
                    "source_context": {"line_range": [10, 11]},
                    "body": "pass",  # trivial body -> heuristics.has_trivial_body returns true implicitly depending on implementation, but let's test bare class too
                }
            ],
            "classes": [
                {
                    "name": "BareClass",
                    "source_context": {"line_range": [20, 21]},
                    "methods": [],
                    "inheritance": {},
                }
            ],
        }
    }
    idx = RuntimeIndex(data)

    with patch("code_analysis.scanner.heuristics.has_trivial_body", return_value=False):
        findings = runtime_only_findings(idx)

    # Should only find the Bare Class, since function is not trivial with mocked False
    assert len(findings) == 1
    assert findings[0]["subtype"] == "Bare Class"
    assert findings[0]["content"] == "class BareClass: (no methods, no bases)"


def test_emit_runtime_issues_annotated_stub():
    idx = RuntimeIndex({})
    sym = {
        "name": "my_func",
        "source_context": {"line_range": [10, 11]},
        "type_annotations": {"return": "int"},
    }
    sink = []

    with patch(
        "code_analysis.scanner.heuristics.has_trivial_body", return_value=True
    ), patch(
        "code_analysis.scanner.heuristics.annotated_non_trivial_return",
        return_value="int",
    ):
        _emit_runtime_issues_for_symbol(
            sink, idx, "file.py", sym, "function", "my_func"
        )

    assert len(sink) == 1
    assert sink[0]["subtype"] == "Annotated Stub"


def test_emit_runtime_issues_exported_placeholder():
    idx = RuntimeIndex({"file.py": {"exports": ["my_func"]}})
    sym = {
        "name": "my_func",
        "source_context": {"line_range": [10, 11]},
    }
    sink = []

    with patch(
        "code_analysis.scanner.heuristics.has_trivial_body", return_value=True
    ), patch(
        "code_analysis.scanner.heuristics.annotated_non_trivial_return",
        return_value=None,
    ):
        _emit_runtime_issues_for_symbol(
            sink, idx, "file.py", sym, "function", "my_func"
        )

    assert any(s["subtype"] == "Exported Placeholder" for s in sink)


def test_emit_runtime_issues_async_stub():
    idx = RuntimeIndex({})
    sym = {
        "name": "my_func",
        "source_context": {"line_range": [10, 11]},
        "is_async": True,
    }
    sink = []

    with patch(
        "code_analysis.scanner.heuristics.has_trivial_body", return_value=True
    ), patch(
        "code_analysis.scanner.heuristics.annotated_non_trivial_return",
        return_value=None,
    ):
        _emit_runtime_issues_for_symbol(
            sink, idx, "file.py", sym, "function", "my_func"
        )

    assert len(sink) == 1
    assert sink[0]["subtype"] == "Async Stub"


def test_emit_runtime_issues_concrete_stub_method():
    idx = RuntimeIndex({})
    sym = {
        "name": "my_meth",
        "source_context": {"line_range": [10, 11]},
    }
    owning_class = {"name": "MyClass"}
    sink = []

    with patch(
        "code_analysis.scanner.heuristics.has_trivial_body", return_value=True
    ), patch(
        "code_analysis.scanner.heuristics.annotated_non_trivial_return",
        return_value=None,
    ), patch(
        "code_analysis.scanner.heuristics.is_protocol_class", return_value=False
    ):
        _emit_runtime_issues_for_symbol(
            sink,
            idx,
            "file.py",
            sym,
            "method",
            "MyClass.my_meth",
            owning_class=owning_class,
        )

    assert len(sink) == 1
    assert sink[0]["subtype"] == "Concrete Stub Method"


def test_emit_runtime_issues_orphan_export():
    idx = RuntimeIndex({"file.py": {"exports": ["my_func"]}})
    sym = {
        "name": "my_func",
        "source_context": {"line_range": [10, 11]},
    }
    sink = []

    with patch("code_analysis.scanner.heuristics.has_trivial_body", return_value=False):
        _emit_runtime_issues_for_symbol(
            sink, idx, "file.py", sym, "function", "my_func"
        )

    assert len(sink) == 1
    assert sink[0]["subtype"] == "Orphan Export"


# Tests for enrich_issue, score_severity, _should_suppress_issue


def test_enrich_issue():
    idx = RuntimeIndex(
        {
            "file.py": {
                "functions": [
                    {
                        "name": "my_func",
                        "line": 10,
                        "type_annotations": {"return": "int"},
                        "docstring": "Some docstring",
                        "decorators": ["@staticmethod"],
                    }
                ],
                "classes": [
                    {
                        "name": "MyClass",
                        "source_context": {"line_range": [5, 20]},
                        "inheritance": {"bases": ["BaseClass"]},
                    }
                ],
            }
        }
    )

    issue = {"file": "file.py", "line": 10, "_qualname": "my_func"}

    enriched = enrich_issue(issue, idx)

    assert "context" in enriched
    ctx = enriched["context"]
    assert "owning_symbol" in ctx
    assert ctx["owning_symbol"]["name"] == "my_func"
    assert ctx["type_annotations"] == {"return": "int"}
    assert ctx["decorators"] == ["@staticmethod"]
    assert ctx["owning_symbol"]["docstring"] == "Some docstring"
    assert ctx["inheritance"] == {"bases": ["BaseClass"]}  # From enclosing class


def test_score_severity():
    assert score_severity({"subtype": "NotImplementedError"}, {}) == "high"  # 2
    assert score_severity({"subtype": "Annotated Stub"}, {}) == "high"  # 2
    assert (
        score_severity({"subtype": "Exported Placeholder"}, {"exported": True})
        == "high"
    )  # 2 + 1 = 3 -> high
    assert (
        score_severity(
            {"subtype": "Annotated Stub"},
            {"exported": True, "linked_areas": {"caller_count": 5}},
        )
        == "critical"
    )  # 2 + 1 + 1 = 4 -> critical
    assert score_severity({"subtype": "Empty/Stub Function"}, {}) == "medium"  # 1
    assert score_severity({"subtype": "Other"}, {}) == "low"  # 0


def test_should_suppress_issue():
    # Abstract method
    issue = {"subtype": "Concrete Stub Method"}
    ctx = {"abstract_method": True}
    assert _should_suppress_issue(issue, ctx) is True

    # Not abstract method
    issue = {"subtype": "Concrete Stub Method"}
    ctx = {"abstract_method": False}
    assert _should_suppress_issue(issue, ctx) is False

    # Marker exception class
    with patch(
        "code_analysis.scanner.heuristics.is_marker_exception_class", return_value=True
    ):
        issue = {"subtype": "Empty/Stub Class"}
        ctx = {"owning_symbol": {"kind": "class"}}
        assert _should_suppress_issue(issue, ctx) is True

    # Data container class
    with patch(
        "code_analysis.scanner.heuristics.is_data_container_class", return_value=True
    ):
        issue = {"subtype": "Bare Class"}
        ctx = {"owning_symbol": {"kind": "class"}}
        assert _should_suppress_issue(issue, ctx) is True

    # Protocol class
    with patch("code_analysis.scanner.heuristics.is_protocol_class", return_value=True):
        issue = {"subtype": "Annotated Stub"}
        ctx = {"owning_symbol": {"qualname": "MyClass.meth"}}
        assert _should_suppress_issue(issue, ctx) is True


# Tests for maybe_run_runtime_inspector


def test_maybe_run_runtime_inspector_no_env(monkeypatch):
    monkeypatch.delenv("CRCT_AUTO_RUNTIME", raising=False)
    with patch("os.path.isdir", return_value=True), patch("subprocess.run") as mock_run:
        maybe_run_runtime_inspector("/tmp/proj")
        mock_run.assert_not_called()


def test_maybe_run_runtime_inspector_file_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("CRCT_AUTO_RUNTIME", "1")

    with patch("os.path.isdir", return_value=True), patch(
        "os.path.exists", return_value=True
    ), patch("subprocess.run") as mock_run:
        maybe_run_runtime_inspector("/tmp/proj")
        mock_run.assert_not_called()


def test_maybe_run_runtime_inspector_runs(monkeypatch):
    monkeypatch.setenv("CRCT_AUTO_RUNTIME", "1")

    with patch("os.path.isdir", return_value=True), patch(
        "os.path.exists", return_value=False
    ), patch("subprocess.run") as mock_run:
        maybe_run_runtime_inspector("/tmp/proj")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "-m" in args
        assert "cline_utils.dependency_system.analysis.runtime_inspector" in args
