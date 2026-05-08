import pytest
from unittest.mock import MagicMock
from typing import Any, Dict

from code_analysis.scanner.heuristics import (
    has_trivial_body,
    annotated_non_trivial_return,
    inherits_from,
    source_mentions,
    is_protocol_class,
    is_abstract_class,
    is_exception_class,
    is_marker_exception_class,
    is_abstract_method,
    is_data_container_class,
)

def test_has_trivial_body_empty():
    assert not has_trivial_body({})

def test_has_trivial_body_no_lines():
    assert has_trivial_body({"source_context": {"source_lines": []}}) == False
    assert has_trivial_body({"source_context": {"source_lines": ["def foo():"]}}) # Only signature, body is empty

def test_has_trivial_body_pass():
    sym = {"source_context": {"source_lines": ["def foo():", "    pass"]}}
    assert has_trivial_body(sym) == True

def test_has_trivial_body_ellipsis():
    sym = {"source_context": {"source_lines": ["def foo():", "    ..."]}}
    assert has_trivial_body(sym) == True

def test_has_trivial_body_return_none():
    sym = {"source_context": {"source_lines": ["def foo():", "    return None"]}}
    assert has_trivial_body(sym) == True

def test_has_trivial_body_return():
    sym = {"source_context": {"source_lines": ["def foo():", "    return"]}}
    assert has_trivial_body(sym) == True

def test_has_trivial_body_with_docstring():
    sym = {
        "source_context": {
            "source_lines": [
                "def foo():",
                '    """',
                '    This is a docstring.',
                '    """',
                "    pass"
            ]
        }
    }
    assert has_trivial_body(sym) == True

def test_has_trivial_body_with_single_line_docstring():
    sym = {
        "source_context": {
            "source_lines": [
                "def foo():",
                '    """Single line docstring"""',
                "    return None"
            ]
        }
    }
    assert has_trivial_body(sym) == True

def test_has_trivial_body_with_comments():
    sym = {
        "source_context": {
            "source_lines": [
                "def foo():",
                "    # This is a comment",
                "    ...",
                "    # Another comment"
            ]
        }
    }
    assert has_trivial_body(sym) == True

def test_has_trivial_body_non_trivial():
    sym = {"source_context": {"source_lines": ["def foo():", "    x = 1", "    return x"]}}
    assert has_trivial_body(sym) == False

def test_annotated_non_trivial_return_empty():
    assert annotated_non_trivial_return({}) is None

def test_annotated_non_trivial_return_trivial_none():
    sym = {"type_annotations": {"return_type": "None"}}
    assert annotated_non_trivial_return(sym) is None

def test_annotated_non_trivial_return_trivial_any():
    sym = {"type_annotations": {"return_type": "Any"}}
    assert annotated_non_trivial_return(sym) is None

def test_annotated_non_trivial_return_trivial_empty():
    sym = {"type_annotations": {"return_type": ""}}
    assert annotated_non_trivial_return(sym) is None

def test_annotated_non_trivial_return_trivial_typing_any():
    sym = {"type_annotations": {"return_type": "typing.Any"}}
    assert annotated_non_trivial_return(sym) is None

def test_annotated_non_trivial_return_non_trivial_int():
    sym = {"type_annotations": {"return_type": "int"}}
    assert annotated_non_trivial_return(sym) == "int"

def test_annotated_non_trivial_return_non_trivial_class():
    sym = {"type_annotations": {"return_type": "MyClass"}}
    assert annotated_non_trivial_return(sym) == "MyClass"

def test_annotated_non_trivial_return_from_parameters():
    sym = {"type_annotations": {"parameters": {"return": "str"}}}
    assert annotated_non_trivial_return(sym) == "str"

def test_annotated_non_trivial_return_parameters_trivial():
    sym = {"type_annotations": {"parameters": {"return": "None"}}}
    assert annotated_non_trivial_return(sym) is None

def test_inherits_from_empty():
    assert inherits_from({}, "Base") == False

def test_inherits_from_bases_match():
    sym = {"inheritance": {"bases": ["Base"]}}
    assert inherits_from(sym, "Base") == True

def test_inherits_from_bases_mismatch():
    sym = {"inheritance": {"bases": ["Other"]}}
    assert inherits_from(sym, "Base") == False

def test_inherits_from_mro_match():
    sym = {"inheritance": {"mro": ["Base"]}}
    assert inherits_from(sym, "Base") == True

def test_inherits_from_mro_mismatch():
    sym = {"inheritance": {"mro": ["Other"]}}
    assert inherits_from(sym, "Base") == False

def test_inherits_from_case_insensitive():
    sym = {"inheritance": {"bases": ["baseclass"]}}
    assert inherits_from(sym, "BaseClass") == True

def test_inherits_from_multiple_needles():
    sym = {"inheritance": {"bases": ["FirstBase"]}}
    assert inherits_from(sym, "SecondBase", "FirstBase") == True
    assert inherits_from(sym, "SecondBase", "ThirdBase") == False

def test_source_mentions_empty():
    assert source_mentions({}, "needle") == False

def test_source_mentions_match():
    sym = {"source_context": {"source_lines": ["def foo():", "    pass", "    # needle"]}}
    assert source_mentions(sym, "needle") == True

def test_source_mentions_mismatch():
    sym = {"source_context": {"source_lines": ["def foo():", "    pass"]}}
    assert source_mentions(sym, "needle") == False

def test_source_mentions_case_insensitive():
    sym = {"source_context": {"source_lines": ["def foo():", "    pass", "    # NeEdLe"]}}
    assert source_mentions(sym, "needle") == True

def test_source_mentions_multiple_needles():
    sym = {"source_context": {"source_lines": ["def foo():", "    pass", "    # needle1"]}}
    assert source_mentions(sym, "needle2", "needle1") == True
    assert source_mentions(sym, "needle2", "needle3") == False

def test_is_protocol_class_inherits_protocol():
    sym = {"inheritance": {"bases": ["Protocol"]}}
    assert is_protocol_class(sym) == True

def test_is_protocol_class_inherits_typing_protocol():
    sym = {"inheritance": {"bases": ["typing.Protocol"]}}
    assert is_protocol_class(sym) == True

def test_is_protocol_class_source_mentions():
    sym = {"source_context": {"source_lines": ["class MyProtocol(Protocol):"]}}
    assert is_protocol_class(sym) == True

def test_is_protocol_class_source_mentions_typing():
    sym = {"source_context": {"source_lines": ["class MyProtocol(typing.Protocol):"]}}
    assert is_protocol_class(sym) == True

def test_is_protocol_class_source_mentions_comma():
    sym = {"source_context": {"source_lines": ["class MyProtocol(Other, Protocol):"]}}
    assert is_protocol_class(sym) == True

def test_is_protocol_class_mismatch():
    sym = {"inheritance": {"bases": ["Other"]}, "source_context": {"source_lines": ["class MyProtocol(Other):"]}}
    assert is_protocol_class(sym) == False

def test_is_abstract_class_runtime_idx_true():
    runtime_idx = MagicMock()
    runtime_idx.is_in_abstract_mro.return_value = True
    sym = {"name": "MyClass"}
    assert is_abstract_class(sym, runtime_idx) == True
    runtime_idx.is_in_abstract_mro.assert_called_once_with(sym)

def test_is_abstract_class_runtime_idx_false_source_match():
    runtime_idx = MagicMock()
    runtime_idx.is_in_abstract_mro.return_value = False
    sym = {"source_context": {"source_lines": ["class MyClass(ABC):"]}}
    assert is_abstract_class(sym, runtime_idx) == True

def test_is_abstract_class_no_runtime_idx_source_match_comma():
    sym = {"source_context": {"source_lines": ["class MyClass(Other, ABC):"]}}
    assert is_abstract_class(sym) == True

def test_is_abstract_class_no_runtime_idx_source_match_abc():
    sym = {"source_context": {"source_lines": ["class MyClass(abc.ABC):"]}}
    assert is_abstract_class(sym) == True

def test_is_abstract_class_mismatch():
    sym = {"source_context": {"source_lines": ["class MyClass(Other):"]}}
    assert is_abstract_class(sym) == False

def test_is_exception_class_name_match_error():
    sym = {"name": "MyError"}
    assert is_exception_class(sym) == True

def test_is_exception_class_name_match_exception():
    sym = {"name": "MyException"}
    assert is_exception_class(sym) == True

def test_is_exception_class_bases_match():
    sym = {"name": "MyClass", "inheritance": {"bases": ["ValueError"]}}
    assert is_exception_class(sym) == True

def test_is_exception_class_mro_match():
    sym = {"name": "MyClass", "inheritance": {"mro": ["Exception"]}}
    assert is_exception_class(sym) == True

def test_is_exception_class_mro_match_base_exception():
    sym = {"name": "MyClass", "inheritance": {"mro": ["BaseException"]}}
    assert is_exception_class(sym) == True

def test_is_exception_class_mismatch():
    sym = {"name": "MyClass", "inheritance": {"bases": ["Base"], "mro": ["Base", "object"]}}
    assert is_exception_class(sym) == False

def test_is_exception_class_empty():
    assert is_exception_class({}) == False

def test_is_marker_exception_class_empty_methods():
    sym = {"name": "MyError", "methods": []}
    assert is_marker_exception_class(sym) == True

def test_is_marker_exception_class_no_methods_key():
    sym = {"name": "MyError"}
    assert is_marker_exception_class(sym) == True

def test_is_marker_exception_class_with_methods():
    sym = {"name": "MyError", "methods": [{"name": "foo"}]}
    assert is_marker_exception_class(sym) == False

def test_is_marker_exception_class_not_exception():
    sym = {"name": "MyClass", "methods": []}
    assert is_marker_exception_class(sym) == False

def test_is_abstract_method_decorators_match():
    sym = {"decorators": ["@abstractmethod"]}
    assert is_abstract_method(sym) == True

def test_is_abstract_method_decorators_match_abc():
    sym = {"decorators": ["@abc.abstractmethod"]}
    assert is_abstract_method(sym) == True

def test_is_abstract_method_decorators_match_case_insensitive():
    sym = {"decorators": ["@AbstractMethod"]}
    assert is_abstract_method(sym) == True

def test_is_abstract_method_source_mentions():
    sym = {"source_context": {"source_lines": ["    @abstractmethod", "    def foo(): pass"]}}
    assert is_abstract_method(sym) == True

def test_is_abstract_method_source_mentions_abc():
    sym = {"source_context": {"source_lines": ["    @abc.abstractmethod", "    def foo(): pass"]}}
    assert is_abstract_method(sym) == True

def test_is_abstract_method_mismatch():
    sym = {"decorators": ["@staticmethod"], "source_context": {"source_lines": ["    @staticmethod", "    def foo(): pass"]}}
    assert is_abstract_method(sym) == False

def test_is_abstract_method_empty():
    assert is_abstract_method({}) == False

def test_is_data_container_class_inherits_basemodel():
    sym = {"inheritance": {"bases": ["BaseModel"]}}
    assert is_data_container_class(sym) == True

def test_is_data_container_class_inherits_enum():
    sym = {"inheritance": {"bases": ["Enum"]}}
    assert is_data_container_class(sym) == True

def test_is_data_container_class_inherits_typeddict():
    sym = {"inheritance": {"bases": ["TypedDict"]}}
    assert is_data_container_class(sym) == True

def test_is_data_container_class_source_mentions_dataclass():
    sym = {"source_context": {"source_lines": ["@dataclass", "class MyClass:"]}}
    assert is_data_container_class(sym) == True

def test_is_data_container_class_source_mentions_basemodel():
    sym = {"source_context": {"source_lines": ["class MyClass(BaseModel):"]}}
    assert is_data_container_class(sym) == True

def test_is_data_container_class_source_mentions_enum():
    sym = {"source_context": {"source_lines": ["class MyClass(Enum):"]}}
    assert is_data_container_class(sym) == True

def test_is_data_container_class_source_mentions_typeddict():
    sym = {"source_context": {"source_lines": ["class MyClass(TypedDict):"]}}
    assert is_data_container_class(sym) == True

def test_is_data_container_class_source_mentions_str_enum():
    sym = {"source_context": {"source_lines": ["class MyClass(str, Enum):"]}}
    assert is_data_container_class(sym) == True

def test_is_data_container_class_source_mentions_int_enum():
    sym = {"source_context": {"source_lines": ["class MyClass(int, Enum):"]}}
    assert is_data_container_class(sym) == True

def test_is_data_container_class_mismatch():
    sym = {"inheritance": {"bases": ["Other"]}, "source_context": {"source_lines": ["class MyClass(Other):"]}}
    assert is_data_container_class(sym) == False

def test_is_data_container_class_empty():
    assert is_data_container_class({}) == False
