from typing import Any, cast, Dict, List

import logging

logger = logging.getLogger(__name__)


def alias_data(data: Dict[str, Any], cache_name: str) -> Dict[str, Any]:
    """Alias data before saving to disk to reduce size."""
    if cache_name != "project_symbol_map_data":
        return data

    try:
        from cline_utils.dependency_system.core.key_manager import (
            load_global_key_map,
        )

        path_to_key_map = load_global_key_map()
        if not path_to_key_map:
            return data

        aliased_data: Dict[str, Any] = {}
        for cache_key, symbol_map in data.items():
            if not isinstance(symbol_map, dict):
                aliased_data[cache_key] = symbol_map
                continue

            symbol_map_dict = cast(Dict[str, Any], symbol_map)
            aliased_symbol_map: Dict[str, Any] = {}
            for path, file_data_raw in symbol_map_dict.items():
                path = cast(str, path)
                file_data = cast(Dict[str, Any], file_data_raw)
                key_info = path_to_key_map.get(path)
                if not key_info:
                    aliased_symbol_map[path] = file_data
                    continue

                file_key = key_info.key_string

                # Store file-level metadata (metadata + non-class/func symbols)
                file_meta: Dict[str, Any] = {
                    k: v
                    for k, v in file_data.items()
                    if k not in ("classes", "functions")
                }
                aliased_symbol_map[file_key] = file_meta

                # Store classes
                classes = cast(List[Dict[str, Any]], file_data.get("classes", []))
                for cls in classes:
                    if "name" in cls:
                        cls_copy = cls.copy()
                        cls_copy["_symbol_type"] = "class"
                        aliased_symbol_map[f"{file_key}:{cls['name']}"] = cls_copy

                # Store functions
                functions = cast(List[Dict[str, Any]], file_data.get("functions", []))
                for func in functions:
                    if "name" in func:
                        func_copy = func.copy()
                        func_copy["_symbol_type"] = "function"
                        aliased_symbol_map[f"{file_key}:{func['name']}"] = func_copy

            aliased_data[cache_key] = aliased_symbol_map

        logger.debug(f"Aliased '{cache_name}' for optimized storage.")
        return aliased_data
    except Exception as e:
        logger.warning(f"Failed to alias '{cache_name}': {e}")
        return data


def dealias_data(data: Dict[str, Any], cache_name: str) -> Dict[str, Any]:
    """De-alias data after loading from disk."""
    if cache_name != "project_symbol_map_data":
        return data

    try:
        from cline_utils.dependency_system.core.key_manager import (
            load_global_key_map,
            load_old_global_key_map,
        )

        # Load current map
        path_to_key_map = load_global_key_map()

        # Create reverse map: key -> path
        key_to_path: Dict[str, str] = {}

        # Load old map first (lower precedence)
        old_path_to_key_map = load_old_global_key_map()
        if old_path_to_key_map:
            for path, info in old_path_to_key_map.items():
                key_to_path[info.key_string] = path

        # Current map takes precedence
        if path_to_key_map:
            for path, info in path_to_key_map.items():
                key_to_path[info.key_string] = path

        if not key_to_path:
            return data

        dealiased_data: Dict[str, Any] = {}
        for cache_key, aliased_symbol_map_raw in data.items():
            if not isinstance(aliased_symbol_map_raw, dict):
                dealiased_data[cache_key] = aliased_symbol_map_raw
                continue

            aliased_symbol_map = cast(Dict[str, Any], aliased_symbol_map_raw)
            full_symbol_map: Dict[str, Any] = {}
            # First pass: identify file keys and initialize full structure
            for key_raw in aliased_symbol_map.keys():
                key = cast(str, key_raw)
                if ":" not in key:
                    path = key_to_path.get(key)
                    if path:
                        full_symbol_map[path] = cast(
                            Dict[str, Any], aliased_symbol_map[key]
                        ).copy()
                        full_symbol_map[path]["classes"] = []
                        full_symbol_map[path]["functions"] = []
                    else:
                        # Keep as is if not in key map (maybe already a full path)
                        full_symbol_map[key] = aliased_symbol_map[key]

            # Second pass: distribute symbols
            for key_raw, symbol_data_raw in aliased_symbol_map.items():
                key = cast(str, key_raw)
                symbol_data = cast(Dict[str, Any], symbol_data_raw)
                if ":" in key:
                    file_key, _ = key.split(":", 1)
                    path = key_to_path.get(file_key)
                    if path and path in full_symbol_map:
                        symbol_type = symbol_data.pop("_symbol_type", None)
                        target_file = cast(Dict[str, Any], full_symbol_map[path])
                        if symbol_type == "class":
                            cast(List[Any], target_file["classes"]).append(symbol_data)
                        elif symbol_type == "function":
                            cast(List[Any], target_file["functions"]).append(
                                symbol_data
                            )
                        else:
                            # Fallback heuristic
                            if "methods" in symbol_data:
                                cast(List[Any], target_file["classes"]).append(
                                    symbol_data
                                )
                            else:
                                cast(List[Any], target_file["functions"]).append(
                                    symbol_data
                                )

            dealiased_data[cache_key] = full_symbol_map

        logger.debug(f"De-aliased '{cache_name}' after loading.")
        return dealiased_data
    except Exception as e:
        logger.warning(f"Failed to de-alias '{cache_name}': {e}")
        return data
