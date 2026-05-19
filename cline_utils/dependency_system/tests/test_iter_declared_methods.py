"""Descriptor-aware method extraction for runtime_inspector."""

import inspect
import textwrap
from pathlib import Path

from cline_utils.dependency_system.analysis.runtime_inspector import iter_declared_methods


def custom_getter_decorator(fn):
    """Decorator that exposes a custom descriptor with __get__ (not builtin property)."""

    class _CustomGetterDescriptor:
        def __init__(self, wrapped):
            self.fget = wrapped

        def __get__(self, obj, owner=None):
            return self.fget(obj) if obj is not None else self.fget(owner)

    return _CustomGetterDescriptor(fn)


class _CustomCallableDescriptor:
    def __init__(self, fn):
        self._fn = fn

    def __get__(self, obj, owner=None):
        return self._fn(obj) if obj is not None else self._fn(owner)


class DescriptorSample:
    def regular_method(self) -> str:
        """Regular method."""
        return "regular"

    @staticmethod
    def static_method() -> str:
        """Static."""
        return "static"

    @classmethod
    def class_method(cls) -> str:
        """Class method."""
        return "class"

    @property
    def builtin_property(self) -> str:
        """Builtin property."""
        return "property"

    @custom_getter_decorator
    def decorated_getter(self) -> str:
        """Custom decorated getter."""
        return "decorated"

    custom_descriptor = _CustomCallableDescriptor(lambda self: "descriptor")


def test_iter_declared_methods_includes_standard_and_custom_descriptors() -> None:
    names = {name for name, _ in iter_declared_methods(DescriptorSample)}

    assert names == {
        "regular_method",
        "static_method",
        "class_method",
        "builtin_property",
        "decorated_getter",
        "custom_descriptor",
    }


def test_loaded_module_class_exposes_custom_getter_signatures(tmp_path: Path) -> None:
    """Loaded modules with custom descriptors yield inspectable getter routines."""
    module_path = tmp_path / "descriptor_module.py"
    module_path.write_text(
        textwrap.dedent(
            """
            def custom_prop(fn):
                class _Desc:
                    def __init__(self, wrapped):
                        self.fget = wrapped

                    def __get__(self, obj, owner=None):
                        return self.fget(obj) if obj is not None else self.fget(owner)

                return _Desc(fn)

            class Indexed:
                @custom_prop
                def dynamic_getter(self) -> int:
                    return 42
            """
        ),
        encoding="utf-8",
    )

    import importlib.util

    spec = importlib.util.spec_from_file_location("descriptor_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    method_names = {name for name, func in iter_declared_methods(module.Indexed)}
    assert method_names == {"dynamic_getter"}

    dynamic = next(
        func
        for name, func in iter_declared_methods(module.Indexed)
        if name == "dynamic_getter"
    )
    assert str(inspect.signature(dynamic)) == "(self) -> int"
