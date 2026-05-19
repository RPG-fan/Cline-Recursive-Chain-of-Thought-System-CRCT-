# runtime_inspector.py
from __future__ import annotations

import inspect
import sys
import os
import json
import importlib.util
import logging
import subprocess
import tempfile
import typing
import ast
import textwrap
from collections.abc import Iterator
from types import CodeType, FunctionType, ModuleType
from typing import Any, Optional, cast

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

JSONDict = dict[str, Any]
StringList = list[str]
MODULE_INSPECT_TIMEOUT_SEC = 60
_INSPECT_MODULE_CLI_FLAG = "--inspect-module"


def get_type_annotations(obj: object) -> JSONDict:
    """Extract parameter and return type annotations."""
    try:
        return {
            "parameters": {
                k: str(v)
                for k, v in typing.get_type_hints(
                    cast(Any, obj), include_extras=True
                ).items()
            },
            "return_type": str(inspect.signature(cast(Any, obj)).return_annotation),
        }
    except Exception:
        return {}


def get_source_context(obj: object, code_roots: list[str]) -> JSONDict:
    """
    Get source file location and import context.
    Returns empty dict if source is outside code roots.
    """
    try:
        source_file = inspect.getsourcefile(cast(Any, obj))
        if not source_file:
            return {}

        # Normalize and validate against code roots
        from cline_utils.dependency_system.utils.path_utils import (
            normalize_path,
            is_subpath,
        )

        norm_source = normalize_path(source_file)

        # Check if file is within any code root
        in_code_roots = False
        for code_root in code_roots:
            norm_root = normalize_path(code_root)
            if norm_source == norm_root or is_subpath(norm_source, norm_root):
                in_code_roots = True
                break

        if not in_code_roots:
            logger.debug(f"Skipping source outside code roots: {norm_source}")
            return {}

        source_lines, start_line = inspect.getsourcelines(cast(Any, obj))
        # Strip line endings to prevent escape artifacts in JSON (improves embedding quality)
        clean_source_lines = [line.rstrip("\n").rstrip("\r") for line in source_lines]
        return {
            "file": norm_source,
            "line_range": (start_line, start_line + len(source_lines)),
            "source_lines": clean_source_lines,
        }
    except Exception:
        return {}


def get_module_exports(module: ModuleType) -> dict[str, str]:
    """Identify all exported symbols and their origins."""
    exports: dict[str, str] = {}
    exported_names = getattr(module, "__all__", None)
    if isinstance(exported_names, (list, tuple, set)):
        normalized_export_names: list[str] = []
        for raw_name in cast(list[object], list(cast(Any, exported_names))):
            if isinstance(raw_name, str):
                normalized_export_names.append(raw_name)
        for name in normalized_export_names:
            obj = getattr(module, name, None)
            if obj is None:
                continue
            module_obj = inspect.getmodule(obj)
            if module_obj is not None:
                exports[name] = module_obj.__name__
    return exports


def get_inheritance_info(cls: type[Any], code_roots: list[str]) -> JSONDict:
    """
    Extract inheritance hierarchy and method resolution order.
    Only includes bases/mro that are within code roots.
    """
    from cline_utils.dependency_system.utils.path_utils import (
        normalize_path,
        is_subpath,
    )

    try:
        bases: StringList = []
        for base in cls.__bases__:
            try:
                base_file = inspect.getsourcefile(base)
                if base_file:
                    norm_base_file = normalize_path(base_file)
                    # Check if base is in code roots
                    in_roots = any(
                        norm_base_file == normalize_path(root)
                        or is_subpath(norm_base_file, normalize_path(root))
                        for root in code_roots
                    )
                    if in_roots:
                        bases.append(base.__module__ + "." + base.__qualname__)
            except (TypeError, AttributeError):
                pass

        mro: StringList = []
        for c in inspect.getmro(cls)[1:]:  # Skip self
            try:
                c_file = inspect.getsourcefile(c)
                if c_file:
                    norm_c_file = normalize_path(c_file)
                    in_roots = any(
                        norm_c_file == normalize_path(root)
                        or is_subpath(norm_c_file, normalize_path(root))
                        for root in code_roots
                    )
                    if in_roots:
                        mro.append(c.__module__ + "." + c.__qualname__)
            except (TypeError, AttributeError):
                pass

        return {"bases": bases, "mro": mro}
    except Exception:
        return {}


def get_closure_dependencies(func: FunctionType, code_roots: list[str]) -> list[str]:
    """
    Identify variables captured in function closures.
    Only includes modules within code roots.
    """
    from cline_utils.dependency_system.utils.path_utils import (
        normalize_path,
        is_subpath,
    )

    deps: set[str] = set()
    if inspect.isfunction(func) and func.__closure__:
        for cell in func.__closure__:
            try:
                obj = cell.cell_contents
                module = inspect.getmodule(obj)
                if module:
                    try:
                        module_file = inspect.getsourcefile(module)
                        if module_file:
                            norm_module_file = normalize_path(module_file)
                            in_roots = any(
                                norm_module_file == normalize_path(root)
                                or is_subpath(norm_module_file, normalize_path(root))
                                for root in code_roots
                            )
                            if in_roots:
                                deps.add(module.__name__)
                    except (TypeError, AttributeError):
                        pass
            except Exception:
                pass
    return sorted(deps)


def _get_declaration_node(
    source_code: str,
) -> Optional[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return the primary class/function node for a source snippet."""
    try:
        dedented_source = textwrap.dedent(source_code)
        tree = ast.parse(dedented_source)
    except Exception:
        return None

    for node in tree.body:
        if isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            return node
    return None


def get_decorator_info(obj: object) -> list[str]:
    """
    Extract declared decorator expressions from source.

    This is intentionally semantic rather than dependency-scoped:
    SES needs the decorators the file actually declares, even when the
    decorator originates from the stdlib or a third-party package.
    """
    try:
        source = inspect.getsource(cast(Any, obj))
    except Exception:
        return []

    node = _get_declaration_node(source)
    if node is None:
        return []

    decorators: list[str] = []
    for decorator in node.decorator_list:
        try:
            decorator_text = ast.unparse(decorator).strip()
        except Exception:
            continue
        if decorator_text:
            decorators.append(decorator_text)
    return decorators


def get_scope_references(func: FunctionType) -> dict[str, list[str]]:
    """Extract global and nonlocal variable references."""
    try:
        code: CodeType = func.__code__
        return {"globals": list(code.co_names), "nonlocals": list(code.co_freevars)}
    except Exception:
        return {}


def get_attribute_accesses(source_code: str) -> list[str]:
    """Parse source to identify attribute access patterns."""
    accesses: set[str] = set()
    try:
        dedented_source = textwrap.dedent(source_code)
        tree = ast.parse(dedented_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                accesses.add(node.attr)
    except Exception:
        pass
    return sorted(accesses)


def _is_data_descriptor(member: object) -> bool:
    """True when member implements the descriptor protocol."""
    dunder_get = getattr(type(member), "__get__", None)
    return dunder_get is not None and callable(dunder_get)


def _unwrap_descriptor_to_function(descriptor: Any) -> Optional[FunctionType]:
    """Extract an inspectable function from a descriptor or decorator wrapper."""
    if inspect.isfunction(descriptor):
        return descriptor

    for attr in ("fget", "__func__", "func", "wrapped", "_func"):
        candidate = getattr(descriptor, attr, None)
        if candidate is None:
            continue
        if inspect.isfunction(candidate):
            return candidate
        if inspect.isroutine(candidate):
            underlying = getattr(candidate, "__func__", None)
            if inspect.isfunction(underlying):
                return underlying

    descriptor_vars = vars(descriptor) if hasattr(descriptor, "__dict__") else {}
    for value in descriptor_vars.values():
        if inspect.isfunction(value):
            return value

    return None


def iter_declared_methods(cls: type[Any]) -> Iterator[tuple[str, FunctionType]]:
    """
    Yield methods declared directly on a class, preserving descriptor-backed
    APIs such as @property, @classmethod, @staticmethod, and custom descriptors.
    """
    seen: set[str] = set()
    members = cast(dict[str, object], cls.__dict__)
    for method_name, member in members.items():
        if method_name in seen:
            continue

        func: Optional[FunctionType] = None
        if inspect.isroutine(member) and inspect.isfunction(member):
            func = member
        elif _is_data_descriptor(member):
            func = _unwrap_descriptor_to_function(member)

        if func is None:
            continue
        seen.add(method_name)
        yield method_name, func


def _resolve_project_root(file_path: str, code_roots: list[str]) -> str:
    """Best-effort project root for subprocess PYTHONPATH (must expose cline_utils)."""
    search_starts = [
        os.path.abspath(file_path),
        os.path.dirname(os.path.abspath(__file__)),
    ]
    search_starts.extend(os.path.abspath(root) for root in code_roots)

    for start in search_starts:
        path = start if os.path.isdir(start) else os.path.dirname(start)
        while True:
            if os.path.isdir(os.path.join(path, "cline_utils")):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent

    if code_roots:
        try:
            return os.path.commonpath([os.path.abspath(root) for root in code_roots])
        except ValueError:
            pass
    return os.path.dirname(os.path.abspath(file_path))


def _write_inspection_payload(out_path: str, payload: JSONDict) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _log_inspection_import_error(
    file_path: str, module_name: str, payload: JSONDict
) -> None:
    missing = payload.get("missing_module") or "unknown"
    logger.warning(
        "Runtime inspection skipped for '%s' (module %s): ImportError while loading. "
        "Missing dependency '%s'. Activate the project virtual environment, install "
        "required packages, or provide mocks for optional imports. Falling back to "
        "AST-only analysis for this file. Detail: %s",
        file_path,
        module_name,
        missing,
        payload.get("message", ""),
    )


def _log_inspection_runtime_error(
    file_path: str, module_name: str, payload: JSONDict
) -> None:
    logger.warning(
        "Runtime inspection failed for '%s' (module %s): %s: %s. "
        "The failure was isolated in a subprocess; falling back to AST-only analysis "
        "for this file.",
        file_path,
        module_name,
        payload.get("error_type", "Error"),
        payload.get("message", ""),
    )


def _log_inspection_subprocess_failure(
    file_path: str,
    module_name: str,
    completed: subprocess.CompletedProcess[str],
) -> None:
    detail = (completed.stderr or completed.stdout or "").strip()
    logger.warning(
        "Runtime inspection subprocess failed for '%s' (module %s, exit %s). "
        "Top-level module side effects were isolated from the parent process; "
        "falling back to AST-only analysis for this file.%s",
        file_path,
        module_name,
        completed.returncode,
        f" Detail: {detail}" if detail else "",
    )


def _import_module_from_path(file_path: str, module_name: str) -> ModuleType:
    """Load and execute a module from disk (runs in subprocess worker only)."""
    file_dir = os.path.dirname(file_path)
    inserted_path = False
    if file_dir and file_dir not in sys.path:
        sys.path.insert(0, file_dir)
        inserted_path = True

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            raise RuntimeError(f"No import spec/loader available for {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted_path and file_dir in sys.path:
            sys.path.remove(file_dir)


def _collect_symbols_from_module(
    module: ModuleType, module_name: str, code_roots: list[str]
) -> JSONDict:
    """Extract symbol metadata from an already-imported module."""
    runtime_classes: list[JSONDict] = []
    runtime_functions: list[JSONDict] = []
    symbols: JSONDict = {
        "classes": runtime_classes,
        "functions": runtime_functions,
        "exports": get_module_exports(module),
    }

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module_name:
            source_context = get_source_context(obj, code_roots)

            if not source_context:
                logger.debug(f"Skipping class {name} - source outside code roots")
                continue

            class_info: JSONDict = {
                "name": name,
                "docstring": inspect.getdoc(obj),
                "inheritance": get_inheritance_info(obj, code_roots),
                "decorators": get_decorator_info(obj),
                "source_context": source_context,
                "methods": [],
            }

            for method_name, method in iter_declared_methods(obj):
                method_source_context = get_source_context(method, code_roots)
                if not method_source_context:
                    logger.debug(
                        f"Skipping method {method_name} - source outside code roots"
                    )
                    continue

                try:
                    sig = str(inspect.signature(method))
                except ValueError:
                    sig = "(...)"

                try:
                    source = inspect.getsource(method)
                    attr_accesses = get_attribute_accesses(source)
                except Exception:
                    attr_accesses = []

                method_entries = cast(list[JSONDict], class_info["methods"])
                method_entries.append(
                    {
                        "name": method_name,
                        "signature": sig,
                        "docstring": inspect.getdoc(method),
                        "type_annotations": get_type_annotations(method),
                        "closure_dependencies": get_closure_dependencies(
                            method, code_roots
                        ),
                        "scope_references": get_scope_references(method),
                        "decorators": get_decorator_info(method),
                        "source_context": method_source_context,
                        "attribute_accesses": attr_accesses,
                    }
                )

            runtime_classes.append(class_info)

        elif inspect.isfunction(obj) and obj.__module__ == module_name:
            source_context = get_source_context(obj, code_roots)

            if not source_context:
                logger.debug(f"Skipping function {name} - source outside code roots")
                continue

            try:
                sig = str(inspect.signature(obj))
            except ValueError:
                sig = "(...)"

            try:
                source = inspect.getsource(obj)
                attr_accesses = get_attribute_accesses(source)
            except Exception:
                attr_accesses = []

            runtime_functions.append(
                {
                    "name": name,
                    "signature": sig,
                    "docstring": inspect.getdoc(obj),
                    "type_annotations": get_type_annotations(obj),
                    "closure_dependencies": get_closure_dependencies(obj, code_roots),
                    "scope_references": get_scope_references(obj),
                    "decorators": get_decorator_info(obj),
                    "source_context": source_context,
                    "attribute_accesses": attr_accesses,
                }
            )

    return symbols


def _load_and_collect_module_symbols(
    file_path: str, module_name: str, code_roots: list[str]
) -> JSONDict:
    """Import and inspect a module in the current process (subprocess worker)."""
    module = _import_module_from_path(file_path, module_name)
    try:
        return _collect_symbols_from_module(module, module_name, code_roots)
    finally:
        sys.modules.pop(module_name, None)


def _cli_inspect_module() -> None:
    """Subprocess entry point: inspect one module and write a JSON payload."""
    if len(sys.argv) < 6:
        print(
            "Usage: python runtime_inspector.py --inspect-module "
            "<file_path> <module_name> <out_path> <code_roots_json>",
            file=sys.stderr,
        )
        sys.exit(3)

    file_path = sys.argv[2]
    module_name = sys.argv[3]
    out_path = sys.argv[4]
    code_roots = json.loads(sys.argv[5])

    try:
        symbols = _load_and_collect_module_symbols(file_path, module_name, code_roots)
        _write_inspection_payload(out_path, {"status": "ok", "symbols": symbols})
        sys.exit(0)
    except ImportError as err:
        _write_inspection_payload(
            out_path,
            {
                "status": "import_error",
                "message": str(err),
                "missing_module": getattr(err, "name", None),
            },
        )
        sys.exit(2)
    except Exception as err:
        _write_inspection_payload(
            out_path,
            {
                "status": "error",
                "error_type": type(err).__name__,
                "message": str(err),
            },
        )
        sys.exit(1)


def get_module_info(
    file_path: str, module_name: str, code_roots: list[str]
) -> JSONDict:
    """
    Import a module in an isolated subprocess and extract symbol information.

    Top-level module side effects cannot destabilize the parent process. ImportError
    is reported explicitly; other failures fall back to AST-only analysis.
    """
    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        project_root = _resolve_project_root(file_path, code_roots)
        env = os.environ.copy()
        if project_root:
            existing_path = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{project_root}{os.pathsep}{existing_path}"
                if existing_path
                else project_root
            )

        completed = subprocess.run(
            [
                sys.executable,
                __file__,
                _INSPECT_MODULE_CLI_FLAG,
                file_path,
                module_name,
                out_path,
                json.dumps(code_roots),
            ],
            capture_output=True,
            text=True,
            timeout=MODULE_INSPECT_TIMEOUT_SEC,
            env=env,
        )

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, encoding="utf-8") as handle:
                payload = json.load(handle)

            status = payload.get("status")
            if status == "ok":
                return cast(JSONDict, payload.get("symbols", {}))
            if status == "import_error":
                _log_inspection_import_error(file_path, module_name, payload)
                return {}
            if status == "error":
                _log_inspection_runtime_error(file_path, module_name, payload)
                return {}

        _log_inspection_subprocess_failure(file_path, module_name, completed)
        return {}
    except subprocess.TimeoutExpired:
        logger.warning(
            "Runtime inspection timed out for '%s' (module %s) after %ss. "
            "Falling back to AST-only analysis for this file.",
            file_path,
            module_name,
            MODULE_INSPECT_TIMEOUT_SEC,
        )
        return {}
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def main():
    if len(sys.argv) < 2:
        print("Usage: python runtime_inspector.py <project_root>")
        sys.exit(1)

    project_root = os.path.abspath(sys.argv[1])

    # Add project root to sys.path to allow importing cline_utils
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from cline_utils.dependency_system.utils.config_manager import ConfigManager
        from cline_utils.dependency_system.utils.path_utils import normalize_path
    except ImportError as e:
        logger.error(
            f"Could not import ConfigManager: {e}. Ensure cline_utils is in python path."
        )
        sys.exit(1)

    # Initialize ConfigManager
    original_cwd = os.getcwd()
    os.chdir(project_root)

    try:
        config_manager = ConfigManager()

        # Get configuration - code_roots are already normalized by config_manager
        code_roots = config_manager.get_code_root_directories()
        excluded_dirs = set(config_manager.get_excluded_dirs())
        excluded_extensions = set(config_manager.get_excluded_extensions())
        excluded_paths = set(config_manager.get_excluded_paths())

        logger.info(f"Loaded configuration. Code roots: {code_roots}")

        # Convert relative code roots to absolute paths
        absolute_code_roots: list[str] = []
        for root_dir_rel in code_roots:
            if os.path.isabs(root_dir_rel):
                absolute_code_roots.append(normalize_path(root_dir_rel))
            else:
                absolute_code_roots.append(
                    normalize_path(os.path.join(project_root, root_dir_rel))
                )

        logger.info(f"Absolute code roots for validation: {absolute_code_roots}")

        # Save to cline_utils/dependency_system/core/runtime_symbols.json
        core_dir = os.path.join(
            project_root, "cline_utils", "dependency_system", "core"
        )
        os.makedirs(core_dir, exist_ok=True)
        output_file = os.path.join(core_dir, "runtime_symbols.json")

        all_symbols: dict[str, JSONDict] = {}

        if not code_roots:
            logger.warning(
                "No code roots defined in configuration. Skipping runtime inspection."
            )
            sys.exit(0)

        # Process each root
        for root_dir_rel in code_roots:
            # Resolve to absolute path
            if os.path.isabs(root_dir_rel):
                root_dir = root_dir_rel
            else:
                root_dir = os.path.join(project_root, root_dir_rel)

            root_dir = normalize_path(root_dir)

            if not os.path.exists(root_dir):
                logger.warning(f"Code root not found: {root_dir}")
                continue

            logger.info(f"Scanning root: {root_dir}")

            for root, dirs, files in os.walk(root_dir):
                root = normalize_path(root)

                # Modify dirs in-place to skip excluded directories
                dirs[:] = [d for d in dirs if d not in excluded_dirs]

                # Filter by path (excluded_paths)
                valid_dirs: list[str] = []
                for d in dirs:
                    dir_path = normalize_path(os.path.join(root, d))
                    if dir_path not in excluded_paths:
                        valid_dirs.append(d)
                dirs[:] = valid_dirs

                for file in files:
                    if not file.endswith(".py") or file.startswith("__"):
                        continue

                    _, ext = os.path.splitext(file)
                    if ext in excluded_extensions:
                        continue

                    file_path = normalize_path(os.path.join(root, file))
                    if file_path in excluded_paths:
                        continue

                    # Construct a module name (approximate)
                    rel_path = os.path.relpath(file_path, project_root)
                    module_name = rel_path.replace(os.sep, ".").replace(".py", "")

                    logger.info(f"Inspecting {module_name}...")

                    # Pass absolute_code_roots to get_module_info for validation
                    info = get_module_info(file_path, module_name, absolute_code_roots)
                    if info:
                        all_symbols[file_path] = info

        # Use ensure_ascii=False to prevent escape character pollution in embeddings
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_symbols, f, indent=2, ensure_ascii=False)

        logger.info(f"Runtime inspection complete. Saved to {output_file}")

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == _INSPECT_MODULE_CLI_FLAG:
        _cli_inspect_module()
    else:
        main()
