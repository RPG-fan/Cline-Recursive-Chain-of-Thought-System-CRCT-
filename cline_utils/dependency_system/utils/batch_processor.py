# utils/batch_processor.py

"""
Utility module for parallel batch processing.
Provides efficient parallel execution of tasks with adaptive batch sizing.
"""

import functools  # Needed for passing kwargs
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, TypeVar

from cline_utils.dependency_system.utils.phase_tracker import PhaseTracker

# Removed cache import as caching batch processing itself is complex and often not desired
# from cline_utils.dependency_system.utils.cache_manager import cached


logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor:
    """Generic batch processor for parallel execution of tasks."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        phase_name: str = "Processing",
    ):
        """
        Initialize the batch processor.

        Args:
            max_workers: Maximum number of worker threads (defaults to CPU count * 4, capped at 64)
            batch_size: Size of batches to process (defaults to adaptive sizing)
            show_progress: Whether to show progress information (prints to stdout)
            phase_name: Name of the phase for the progress tracker
        """
        super().__init__()
        cpu_count = os.cpu_count() or 8
        # Increase parallelism: use 4x CPUs by default, cap at 48 to avoid runaway threads
        default_workers = min(4, (cpu_count * 4))
        # Ensure max_workers is at least 1
        self.max_workers = max(1, max_workers or default_workers)
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.phase_name = phase_name
        self.total_items = 0
        self.processed_items = 0
        self.start_time = 0.0

    # <<< MODIFIED: Accept **kwargs >>>
    def process_items(
        self, items: List[T], processor_func: Callable[..., R], **kwargs: Any
    ) -> List[Optional[R]]:
        """
        Process a list of items in parallel batches.
        Extra keyword arguments (**kwargs) are passed directly to the processor_func.

        Args:
            items: List of items to process
            processor_func: Function to process each item (can accept kwargs)
            **kwargs: Additional keyword arguments to pass to processor_func
        Returns:
            List of results from processing each item (order matches input items)
        """
        if not callable(processor_func):
            logger.error("processor_func must be callable")
            raise TypeError("processor_func must be a callable")  # Use TypeError

        self.total_items = len(items)
        if not self.total_items:
            logger.info("No items to process")
            return []

        self.processed_items = 0
        self.start_time = time.time()

        actual_batch_size = self._determine_batch_size()
        logger.debug(
            f"Processing {self.total_items} items with batch size: {actual_batch_size}, workers: {self.max_workers}"
        )

        # Create a results list pre-filled with None to maintain order
        results: List[Optional[R]] = [None] * self.total_items
        items_processed_in_batches = 0

        # Use PhaseTracker if progress is enabled
        self.tracker = None
        context_manager = (
            PhaseTracker(total=self.total_items, phase_name=self.phase_name)
            if self.show_progress
            else None
        )

        if context_manager:
            with context_manager as tracker:
                self.tracker = tracker
                for i in range(0, self.total_items, actual_batch_size):
                    batch_indices = range(
                        i, min(i + actual_batch_size, self.total_items)
                    )
                    batch_items = [items[idx] for idx in batch_indices]

                    if not batch_items:
                        continue

                    # Pass kwargs to the batch processing function
                    batch_results_map = self._process_batch(
                        batch_items, processor_func, **kwargs
                    )

                    # Place results back into the main list using original indices
                    for original_idx, result_value in batch_results_map.items():
                        global_index = i + original_idx
                        if 0 <= global_index < self.total_items:
                            results[global_index] = result_value
                        else:
                            logger.error(
                                f"Calculated invalid global index {global_index} from batch index {original_idx} (batch start {i})"
                            )

                    items_processed_in_batches += len(batch_items)
                    self.processed_items = items_processed_in_batches
                    tracker.update(len(batch_items))
        else:
            # No progress bar logic
            for i in range(0, self.total_items, actual_batch_size):
                batch_indices = range(i, min(i + actual_batch_size, self.total_items))
                batch_items = [items[idx] for idx in batch_indices]

                if not batch_items:
                    continue

                batch_results_map = self._process_batch(
                    batch_items, processor_func, **kwargs
                )

                for original_idx, result_value in batch_results_map.items():
                    global_index = i + original_idx
                    if 0 <= global_index < self.total_items:
                        results[global_index] = result_value
                    else:
                        logger.error(
                            f"Calculated invalid global index {global_index} from batch index {original_idx} (batch start {i})"
                        )

                items_processed_in_batches += len(batch_items)
                self.processed_items = items_processed_in_batches

        final_time = time.time() - self.start_time
        logger.debug(f"Processed {self.total_items} items in {final_time:.2f} seconds")

        # Filter out potential None values if errors occurred and weren't replaced
        # Or raise an error if None is found, depending on desired strictness
        final_results = [res for res in results if res is not None]
        if len(results) != self.total_items:
            logger.critical(
                f"Result list length ({len(results)}) does not match total items ({self.total_items}). This is an internal error."
            )

        if any(res is None for res in results):
            logger.warning(
                f"Some items failed processing ({len(results) - len(final_results)} errors). Results list contains only successful items."
            )

        # Make sure final newline is printed after progress bar
        if self.show_progress and self.total_items > 0:
            print()

        # Cast is needed because we pre-filled with None, but logic aims to replace all Nones
        return results

    # <<< MODIFIED: Accept **kwargs >>>
    def process_with_collector(
        self,
        items: List[T],
        processor_func: Callable[..., R],
        collector_func: Callable[[List[R]], Any],
        **kwargs: Any,
    ) -> Any:
        """
        Process items in batches and collect results with a collector function.
        Extra keyword arguments (**kwargs) are passed directly to the processor_func.

        Args:
            items: List of items to process
            processor_func: Function to process each item (can accept kwargs)
            collector_func: Function to collect and process ALL results at the end
            **kwargs: Additional keyword arguments to pass to processor_func
        Returns:
            Result from the collector function
        """
        # Use the modified process_items to get all results first
        all_results = self.process_items(items, processor_func, **kwargs)
        # Then pass the complete list of results to the collector
        logger.info("Calling collector function with all results...")
        return collector_func(all_results)

    def _determine_batch_size(self) -> int:
        """Determine adaptive batch size based on total items and workers."""
        if self.batch_size is not None:
            # If batch_size is explicitly set, use it (but ensure it's at least 1)
            return max(8, self.batch_size)

        # --- Adaptive sizing (more aggressive parallelism) ---
        if self.total_items == 0:
            return 8  # Handle edge case

        effective_workers = max(1, self.max_workers)

        # Favor smaller batches to keep more threads busy; minimum per worker now 8
        min_sensible_batch = 8
        if self.total_items < effective_workers * min_sensible_batch:
            min_batch = max(1, self.total_items // effective_workers)
        else:
            min_batch = min_sensible_batch

        # Increase target concurrency; aim for ~10-16 batches per worker
        target_batches_per_worker = 12
        denominator = max(4, effective_workers * target_batches_per_worker)
        calculated_batch_size = max(4, self.total_items // denominator)

        # Allow larger caps but still finite to avoid memory spikes
        max_sensible_batch = 256
        max_batch = min(self.total_items, max_sensible_batch)

        # Final batch size leaning smaller to increase utilization
        final_batch_size = min(max_batch, max(min_batch, calculated_batch_size))
        final_batch_size = max(16, final_batch_size)

        num_batches = (self.total_items + final_batch_size - 1) // final_batch_size
        logger.debug(
            f"Adaptive batch size: Total={self.total_items}, Workers={effective_workers}, "
            f"TargetBatches/Worker={target_batches_per_worker} => Calculated={calculated_batch_size}, "
            f"Min={min_batch}, Max={max_batch} -> Final={final_batch_size}, Batches={num_batches}"
        )
        return final_batch_size

    # <<< MODIFIED: Accept **kwargs and return Dict >>>
    def _process_batch(
        self, batch: List[T], processor_func: Callable[..., R], **kwargs: Any
    ) -> Dict[int, R]:
        """
        Process a single batch of items in parallel. Returns a dictionary mapping
        batch index to result to handle potential errors and maintain order linkage.

        Args:
            batch: Batch of items to process
            processor_func: Function to process each item (can accept kwargs)
            **kwargs: Additional keyword arguments to pass to processor_func
        Returns:
            Dictionary mapping batch index (0-based) to the result for that item.
            Items that failed processing will be missing from the dictionary.
        """
        if not batch:
            return {}

        batch_results_map: Dict[int, R] = {}
        # Wrap the processor_func to include kwargs using functools.partial
        # The executor will call partial_func(item)
        partial_func = functools.partial(processor_func, **kwargs)

        # Keep executor saturated; use full pool size (bounded by batch length)
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, max(1, len(batch)))
        ) as executor:
            # Map future to the index within the current batch
            future_to_idx = {
                executor.submit(partial_func, item): i for i, item in enumerate(batch)
            }

            for future in as_completed(future_to_idx):
                idx_in_batch = future_to_idx[future]
                item_info = batch[idx_in_batch]  # For logging errors
                try:
                    result = future.result()
                    batch_results_map[idx_in_batch] = result
                except Exception as e:
                    # Log the specific item that failed if possible
                    item_repr = repr(item_info)
                    if len(item_repr) > 100:
                        item_repr = item_repr[:100] + "..."
                    logger.error(
                        f"Error processing item (batch index {idx_in_batch}): {item_repr} -> {e}",
                        exc_info=True,
                    )  # Log with traceback
                    # Do not add to batch_results_map, indicating failure

        return batch_results_map

    def _show_progress(self) -> None:
        """Show progress information to stdout."""
        elapsed_time = time.time() - self.start_time
        if elapsed_time < 0.01:
            elapsed_time = 0.01  # Avoid division by zero

        items_per_second = self.processed_items / elapsed_time
        percent_complete = (
            self.processed_items / max(1, self.total_items)
        ) * 100  # Avoid division by zero total_items
        remaining_items = self.total_items - self.processed_items
        eta_seconds = remaining_items / items_per_second if items_per_second > 0 else 0

        # Format ETA nicely
        if eta_seconds > 3600:
            eta_str = f"{eta_seconds/3600:.1f}h"
        elif eta_seconds > 60:
            eta_str = f"{eta_seconds/60:.1f}m"
        else:
            eta_str = f"{eta_seconds:.1f}s"

        # Use a fixed width for progress numbers to prevent line jitter
        progress_str = (
            f"{self.processed_items:>{len(str(self.total_items))}}/{self.total_items}"
        )

        print(
            f"Progress: {progress_str} ({percent_complete:6.1f}%) | "
            f"{items_per_second:6.1f} items/s | "
            f"ETA: {eta_str:<6}",  # Left align ETA string in a fixed width
            end="\r",
            flush=True,  # Ensure it updates immediately
        )
        # No newline here, handled after the loop finishes


# --- Convenience Functions ---

# Caching these convenience functions is generally not recommended
# as the 'items' list can be large and hashing it is expensive/unreliable.


# <<< MODIFIED: Accept **kwargs >>>
def process_items(
    items: List[T],
    processor_func: Callable[..., R],
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
    phase_name: str = "Processing",
    **kwargs: Any,
) -> List[R]:
    """
    Convenience function to process items in parallel using BatchProcessor.
    Extra keyword arguments (**kwargs) are passed directly to the processor_func.
    """
    processor = BatchProcessor(
        max_workers, batch_size, show_progress, phase_name=phase_name
    )
    return processor.process_items(items, processor_func, **kwargs)


# <<< MODIFIED: Accept **kwargs >>>
def process_with_collector(
    items: List[T],
    processor_func: Callable[..., R],
    collector_func: Callable[[List[R]], Any],
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Convenience function to process items and collect results using BatchProcessor.
    Extra keyword arguments (**kwargs) are passed directly to the processor_func.
    """
    processor = BatchProcessor(max_workers, batch_size, show_progress)
    # Note: process_items used internally will handle passing kwargs to processor_func
    return processor.process_with_collector(
        items, processor_func, collector_func, **kwargs
    )


# --- End of batch_processor.py ---
