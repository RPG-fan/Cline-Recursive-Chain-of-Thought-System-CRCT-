import pytest
import logging
from typing import List, Optional

from cline_utils.dependency_system.utils.batch_processor import (
    BatchProcessor,
    process_items,
    process_with_collector,
)

def test_batch_processor_success():
    """Test that all items are successfully processed in parallel."""
    items = [1, 2, 3, 4, 5]
    def processor(x: int) -> int:
        return x * 2

    processor_obj = BatchProcessor(max_workers=2, batch_size=2, show_progress=False)
    results = processor_obj.process_items(items, processor)
    
    assert results == [2, 4, 6, 8, 10]
    assert len(results) == len(items)

def test_batch_processor_with_failures(caplog):
    """Test that failed items result in None placeholders at their exact indices and trigger the correct warning."""
    items = [1, 2, 3, 4, 5]
    
    def processor(x: int) -> int:
        if x == 3 or x == 5:
            raise ValueError(f"Failure on {x}")
        return x * 2

    processor_obj = BatchProcessor(max_workers=2, batch_size=2, show_progress=False)
    
    with caplog.at_level(logging.WARNING):
        results = processor_obj.process_items(items, processor)
        
    assert results == [2, 4, None, 8, None]
    assert len(results) == len(items)
    
    # Check that warning log was produced and contains clear description of None preservation
    warnings = [rec.message for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warnings) > 0
    assert any("Some items failed processing" in msg for msg in warnings)
    assert any("preserve 1-to-1 index correlation" in msg for msg in warnings)

def test_convenience_functions():
    """Test that top-level convenience functions work correctly under success."""
    items = ["a", "b", "c"]
    def processor(s: str) -> str:
        return s.upper()

    results = process_items(items, processor, max_workers=2, show_progress=False)
    assert results == ["A", "B", "C"]

def test_convenience_functions_with_failures(caplog):
    """Test that top-level convenience functions handle failures with None placeholders."""
    items = ["a", "b", "c"]
    def processor(s: str) -> str:
        if s == "b":
            raise RuntimeError("Failed b")
        return s.upper()

    with caplog.at_level(logging.WARNING):
        results = process_items(items, processor, max_workers=2, show_progress=False)
        
    assert results == ["A", None, "C"]
    
    warnings = [rec.message for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warnings) > 0
    assert any("preserve 1-to-1 index correlation" in msg for msg in warnings)

def test_process_with_collector():
    """Test process_with_collector forwards the results list with None placeholders to the collector."""
    items = [1, 2, 3]
    def processor(x: int) -> int:
        if x == 2:
            raise ValueError("Failure on 2")
        return x * 10

    collected_results = []
    def collector(res_list: List[Optional[int]]) -> List[Optional[int]]:
        collected_results.extend(res_list)
        return res_list

    results = process_with_collector(
        items,
        processor,
        collector,
        max_workers=2,
        show_progress=False
    )
    
    assert results == [10, None, 30]
    assert collected_results == [10, None, 30]
