# analysis/symbol_map_merger.py
"""
Merges runtime_symbols.json (from runtime_inspector) with AST analysis data
to create a comprehensive project_symbol_map.json that combines the best of both:
- Runtime: Rich type info, inheritance, signatures, clean source
- AST: Imports, call graphs, file metadata
"""

import json
import logging
import os
from typing import Dict, Any, List

from cline_utils.dependency_system.utils.cache_manager import get_project_root_cached as get_project_root

logger = logging.getLogger(__name__)


def merge_runtime_and_ast(
    runtime_symbols: Dict[str, Dict[str, Any]],
    ast_analysis: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Merges runtime inspector output with AST analysis.
    
    Runtime symbols provide the foundation with:
    - Classes with inheritance, decorators, source_context
    - Methods with signatures, type_annotations, scope_references
    - Functions with full runtime metadata
    
    AST analysis adds:
    - file_type metadata
    - imports list
    - calls list with line numbers
    
    Args:
        runtime_symbols: Output from runtime_inspector.py
        ast_analysis: Output from dependency_analyzer.py (AST-based)
        
    Returns:
        Merged symbol map with runtime as primary, AST as enhancement
    """
    merged_map = {}
    
    # Start with all runtime symbols
    for file_path, runtime_data in runtime_symbols.items():
        merged = runtime_data.copy()
        
        # Get corresponding AST data
        ast_data = ast_analysis.get(file_path, {})
        
        # Add AST-specific fields that runtime can't capture
        merged["file_type"] = ast_data.get("file_type", "py")
        merged["imports"] = ast_data.get("imports", [])
        merged["calls"] = ast_data.get("calls", [])  # Call graph with line numbers

        # Preserve additional AST-derived structure for SES and cross-language analysis.
        # Runtime data remains authoritative for overlapping rich runtime fields.
        for ast_key, ast_val in ast_data.items():
            if ast_key in {
                "file_path",
                "file_type",
                "imports",
                "calls",
                "classes",
                "functions",
                "error",
                "skipped",
                "_ast_tree",
                "_ts_tree",
            }:
                continue
            if ast_key not in merged or not merged.get(ast_key):
                merged[ast_key] = ast_val
        
        # Enhance classes with AST metadata if available
        runtime_classes = merged.get("classes", [])
        ast_classes = ast_data.get("classes", [])
        
        for rt_class in runtime_classes:
            # Find matching AST class by name
            matching_ast = next(
                (c for c in ast_classes if c.get("name") == rt_class["name"]),
                None
            )
            
            if matching_ast:
                # Add AST line numbers if not in runtime
                if "line" in matching_ast and "line" not in rt_class:
                    rt_class["line"] = matching_ast["line"]
        
        # Enhance functions similarly
        runtime_functions = merged.get("functions", [])
        ast_functions = ast_data.get("functions", [])
        
        for rt_func in runtime_functions:
            matching_ast = next(
                (f for f in ast_functions if f.get("name") == rt_func["name"]),
                None
            )
            
            if matching_ast:
                if "line" in matching_ast and "line" not in rt_func:
                    rt_func["line"] = matching_ast["line"]
        
        merged_map[file_path] = merged
    
    # Add any files that AST found but runtime missed
    # (e.g., files that failed runtime inspection)
    for file_path, ast_data in ast_analysis.items():
        if file_path not in merged_map:
            logger.debug(
                f"File {file_path} in AST analysis but not in runtime symbols. "
                f"Using AST-only data."
            )
            merged_map[file_path] = ast_data
    
    return merged_map


def load_runtime_symbols(project_root: str = None) -> Dict[str, Dict[str, Any]]:
    """Load runtime_symbols.json from core directory."""
    if project_root is None:
        project_root = get_project_root()
    
    runtime_path = os.path.join(
        project_root,
        "cline_utils",
        "dependency_system",
        "core",
        "runtime_symbols.json"
    )
    
    if not os.path.exists(runtime_path):
        logger.warning(f"Runtime symbols file not found: {runtime_path}")
        return {}
    
    try:
        with open(runtime_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load runtime symbols: {e}")
        return {}


def save_merged_symbol_map(
    merged_map: Dict[str, Dict[str, Any]],
    output_path: str,
    backup_old: bool = True
) -> None:
    """
    Save merged symbol map to project_symbol_map.json.
    
    Args:
        merged_map: The merged runtime + AST symbol data
        output_path: Path to save the new project_symbol_map.json
        backup_old: Whether to backup existing file to project_symbol_map_old.json
    """
    # Backup existing file if requested
    if backup_old and os.path.exists(output_path):
        backup_path = output_path.replace(".json", "_old.json")
        try:
            import shutil
            shutil.copy2(output_path, backup_path)
            logger.info(f"Backed up old symbol map to: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to backup old symbol map: {e}")
    
    # Save merged map with clean formatting
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_map, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved merged symbol map to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save merged symbol map: {e}")
        raise


def validate_merged_output(merged_map: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Validate that merged output has expected structure.
    
    Returns:
        Dictionary categorizing validation issues by type
    """
    issues = {
        "missing_runtime_data": [],
        "missing_ast_data": [],
        "info": []
    }
    
    for file_path, file_data in merged_map.items():
        # Check for runtime fields (classes/functions)
        has_runtime = "classes" in file_data or "functions" in file_data
        has_symbols = (file_data.get("classes") or file_data.get("functions"))
        
        if not has_runtime:
            issues["missing_runtime_data"].append(
                f"{file_path}: No classes or functions (may be module-level code only)"
            )
        elif not has_symbols:
            # Has the keys but they're empty
            issues["info"].append(
                f"{file_path}: Empty classes/functions lists (may be __init__.py or config)"
            )
        
        # Check for AST enhancements
        if "imports" not in file_data:
            issues["missing_ast_data"].append(f"{file_path}: Missing imports field")
            
    return issues
