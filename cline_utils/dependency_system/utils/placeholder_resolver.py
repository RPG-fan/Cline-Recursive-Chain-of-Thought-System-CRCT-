import concurrent.futures
import itertools
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any

from cline_utils.dependency_system.analysis.local_llm_processor import LocalLLMProcessor
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.io.tracker_io import (
    update_tracker,
    PathMigrationInfo,
)
from cline_utils.dependency_system.utils.template_generator import (
    add_code_doc_dependency_to_checklist,
)
from cline_utils.dependency_system.utils.tracker_batch_collector import (
    TrackerBatchCollector,
    create_doc_tracker_update,
    create_main_tracker_update,
    create_mini_tracker_update,
)

logger = logging.getLogger(__name__)


@dataclass
class PreparedPair:
    srckey: str
    srcpath: str
    tgtkey: str
    tgtpath: str
    srcbase: str
    tgtbase: str
    srccontent: str
    tgtcontent: str
    stokens: int
    ttokens: int
    skip: bool = False
    skip_reason: str = ""


def background_commit(
    b_tracker_path: str,
    b_path_to_key_info: Dict[str, KeyInfo],
    b_tracker_type: str,
    b_suggestions: Dict[str, List[Tuple[str, str]]],
    b_accumulated_updates: Optional[List[Any]] = None,
    b_path_migration_info: Optional[PathMigrationInfo] = None,
) -> None:
    try:
        update_data = update_tracker(
            output_file_suggestion=b_tracker_path,
            path_to_key_info=b_path_to_key_info,
            tracker_type=b_tracker_type,
            suggestions_external=b_suggestions,
            return_update=True,
            force_apply_suggestions=True,
            apply_ast_overrides=False,
            path_migration_info=b_path_migration_info,
        )
        if update_data:
            t_update = None
            out_path = update_data.get("output_file", b_tracker_path)
            if b_tracker_type == "mini":
                t_update = create_mini_tracker_update(
                    output_file=out_path,
                    key_info_list=update_data["key_info_list"],
                    grid_rows=update_data["grid_rows"],
                    last_key_edit=update_data["last_key_edit"],
                    last_grid_edit=update_data["last_grid_edit"],
                    module_path=update_data.get("module_path", ""),
                    path_to_key_info=update_data.get(
                        "path_to_key_info", b_path_to_key_info
                    ),
                    existing_lines=update_data.get("existing_lines", []),
                    tracker_exists=update_data.get("tracker_exists", False),
                    ast_overrides_applied_count=update_data.get(
                        "ast_overrides_applied_count", 0
                    ),
                    suggestion_applied_count=update_data.get(
                        "suggestion_applied_count", 0
                    ),
                    structural_deps_applied_count=update_data.get(
                        "structural_deps_applied_count", 0
                    ),
                )
            elif b_tracker_type == "doc":
                t_update = create_doc_tracker_update(
                    output_file=out_path,
                    key_info_list=update_data["key_info_list"],
                    grid_rows=update_data["grid_rows"],
                    last_key_edit=update_data["last_key_edit"],
                    last_grid_edit=update_data["last_grid_edit"],
                    path_to_key_info=update_data.get(
                        "path_to_key_info", b_path_to_key_info
                    ),
                    ast_overrides_applied_count=update_data.get(
                        "ast_overrides_applied_count", 0
                    ),
                    suggestion_applied_count=update_data.get(
                        "suggestion_applied_count", 0
                    ),
                    structural_deps_applied_count=update_data.get(
                        "structural_deps_applied_count", 0
                    ),
                )
            else:  # main
                t_update = create_main_tracker_update(
                    output_file=out_path,
                    key_info_list=update_data["key_info_list"],
                    grid_rows=update_data["grid_rows"],
                    last_key_edit=update_data["last_key_edit"],
                    last_grid_edit=update_data["last_grid_edit"],
                    path_to_key_info=update_data.get(
                        "path_to_key_info", b_path_to_key_info
                    ),
                    ast_overrides_applied_count=update_data.get(
                        "ast_overrides_applied_count", 0
                    ),
                    suggestion_applied_count=update_data.get(
                        "suggestion_applied_count", 0
                    ),
                    structural_deps_applied_count=update_data.get(
                        "structural_deps_applied_count", 0
                    ),
                )

            if t_update:
                thread_collector = TrackerBatchCollector()
                thread_collector.add(t_update)
                thread_collector.commit_all(
                    skip_populate_hook=True, accumulated_updates=b_accumulated_updates
                )
            else:
                print(
                    "Error: Failed to create tracker update object in background thread."
                )
        else:
            print("Error: Failed to generate update data")
    except Exception as e:
        logger.error(f"Error processing background commit: {e}", exc_info=True)
        print(f"Error processing background commit: {e}")


class PlaceholderResolver:
    def __init__(self, processor: LocalLLMProcessor):
        super().__init__()
        self.processor = processor

    def resolve_batch(
        self,
        tasks: List[Tuple[str, str, str, str]],
        tracker_path: str,
        global_map: Dict[str, KeyInfo],
        tracker_type: str,
        prepare_func: Callable[[str, str, str, str], PreparedPair],
        accumulated_updates: Optional[List[Any]] = None,
        path_migration_info: Optional[PathMigrationInfo] = None,
    ) -> int:
        """
        Processes a list of dependency verification tasks through the local LLM.
        Returns the number of pairs processed.
        """
        if not tasks:
            return 0

        batch_suggestions: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        commit_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        commit_futures: List[concurrent.futures.Future[None]] = []

        start_time = time.time()
        PREFETCH_AHEAD = 5
        processed_count = 0
        total_processed = 0
        total_tasks = len(tasks)

        def _process_single_prepared(prepared: PreparedPair) -> None:
            nonlocal processed_count, total_processed

            total_processed += 1
            if prepared.skip:
                logger.warning(
                    f"[{total_processed}/{total_tasks}] Skipping {prepared.srckey} -> {prepared.tgtkey}: {prepared.skip_reason}"
                )
                print(
                    f"[{total_processed}/{total_tasks}] Skipping {prepared.srckey} -> {prepared.tgtkey}: {prepared.skip_reason}"
                )
                return

            print(
                f"[{total_processed}/{total_tasks}] analyzing {prepared.srckey} -> {prepared.tgtkey}..."
            )

            char, reasoning = self.processor.determine_dependency(
                source_content=prepared.srccontent,
                target_content=prepared.tgtcontent,
                source_basename=prepared.srcbase,
                target_basename=prepared.tgtbase,
                source_tokens=prepared.stokens if prepared.stokens > 0 else None,
                target_tokens=prepared.ttokens if prepared.ttokens > 0 else None,
            )

            print(f"  Result: {char}")
            print(f"--- LLM Reasoning ---\n{reasoning}\n---------------------")

            try:
                add_code_doc_dependency_to_checklist(
                    source_key_str=prepared.srckey,
                    target_key_str=prepared.tgtkey,
                    dep_type_char=char,
                    justification=reasoning.strip(),
                )
            except Exception as e:
                logger.error(
                    f"Failed to add dependency {prepared.srckey} -> {prepared.tgtkey} to checklist: {e}"
                )

            batch_suggestions[prepared.srckey].append((prepared.tgtkey, char))
            processed_count += 1

            if processed_count >= 10:
                print(
                    f"Submitting batch of {processed_count} updates to background thread..."
                )
                suggestions_copy = {k: v[:] for k, v in batch_suggestions.items()}
                future = commit_executor.submit(
                    background_commit,
                    tracker_path,
                    global_map,
                    tracker_type,
                    suggestions_copy,
                    accumulated_updates,
                    path_migration_info,
                )
                commit_futures.append(future)
                batch_suggestions.clear()
                processed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as prefetch_executor:
            task_iter = iter(tasks)

            def _submit_next(
                pair: Tuple[str, str, str, str],
            ) -> concurrent.futures.Future[PreparedPair]:
                sk, sp, tk, tp = pair
                # Call the injected prepare_func from the caller's context
                return prefetch_executor.submit(prepare_func, sk, sp, tk, tp)

            # Track source-key boundaries for per-source KV-cache optimization.
            # With the two-stage sort (source_key, -token_count) in effect, all pairs
            # for the same source are contiguous, so we can detect source changes.
            current_source: Optional[str] = None

            prefetch_queue: deque[concurrent.futures.Future[PreparedPair]] = deque()
            for pair in itertools.islice(task_iter, PREFETCH_AHEAD):
                prefetch_queue.append(_submit_next(pair))

            # Tracks the set of all source keys seen so far, used to detect
            # sort-order violations (a source key appearing after a different
            # source key has already been processed signals a broken invariant).
            seen_sources: set = set()

            def _process_with_source_tracking(prepared: PreparedPair) -> None:
                nonlocal current_source
                if prepared.srckey != current_source:
                    if current_source is not None:
                        logger.debug(
                            f"Source boundary: '{current_source}' -> '{prepared.srckey}' "
                            f"(ending source group {current_source})"
                        )
                        # Detect sort-order regression: if we've seen this source
                        # before, the contiguous-group invariant has been violated.
                        if prepared.srckey in seen_sources:
                            logger.debug(
                                f"KV-cache source-group invariant violated: "
                                f"'{prepared.srckey}' was already processed earlier. "
                                "KV-cache pinning will degrade to no-ops for this source. "
                                "Ensure tasks are sorted by (source_key, -token_count)."
                            )
                    seen_sources.add(current_source)
                    # New source — clear old pinned state (safe no-op if None)
                    self.processor.clear_pinned_state()
                    current_source = prepared.srckey
                else:
                    # Same source — restore cached context if available
                    self.processor.restore_pinned_state()
                _process_single_prepared(prepared)

            def _maybe_pin_for_next(prepared: PreparedPair) -> None:
                """Opportunistic lookahead: pin KV cache only if the next queued
                pair shares the same source key, making restoration beneficial."""
                if not prefetch_queue:
                    self.processor.clear_pinned_state()
                    return
                next_future = prefetch_queue[0]
                if not next_future.done():
                    # Not ready yet — can't peek; don't pin to be safe
                    self.processor.clear_pinned_state()
                    return
                try:
                    next_prepared = next_future.result()
                except Exception:
                    self.processor.clear_pinned_state()
                    return
                if next_prepared.srckey == prepared.srckey:
                    if not getattr(self.processor, "has_pinned_state", False):
                        self.processor.save_pinned_state()
                else:
                    self.processor.clear_pinned_state()

            for next_pair in task_iter:
                prefetch_queue.append(_submit_next(next_pair))
                future = prefetch_queue.popleft()
                try:
                    prepared = future.result()
                except Exception as e:
                    logger.error(f"Error loading pair from prefetch: {e}")
                    total_processed += 1
                    continue

                try:
                    _process_with_source_tracking(prepared)
                except Exception as e:
                    logger.error(
                        f"Error processing pair {prepared.srckey}->{prepared.tgtkey}: {e}"
                    )
                else:
                    _maybe_pin_for_next(prepared)

            while prefetch_queue:
                future = prefetch_queue.popleft()
                try:
                    prepared = future.result()
                except Exception as e:
                    logger.error(f"Error loading pair from prefetch (drain): {e}")
                    total_processed += 1
                    continue

                try:
                    _process_with_source_tracking(prepared)
                except Exception as e:
                    logger.error(
                        f"Error processing pair {prepared.srckey}->{prepared.tgtkey} (drain): {e}"
                    )
                else:
                    _maybe_pin_for_next(prepared)

        if processed_count > 0:
            print(
                f"Submitting final batch of {processed_count} updates to background thread..."
            )
            suggestions_copy = {k: v[:] for k, v in batch_suggestions.items()}
            future = commit_executor.submit(
                background_commit,
                tracker_path,
                global_map,
                tracker_type,
                suggestions_copy,
                accumulated_updates,
                path_migration_info,
            )
            commit_futures.append(future)

        if commit_futures:
            print(
                f"Waiting for {len(commit_futures)} background commits to complete..."
            )
            for future in concurrent.futures.as_completed(commit_futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in background commit: {e}")
                    logger.error(f"Error in background commit: {e}", exc_info=True)

        commit_executor.shutdown()

        elapsed = time.time() - start_time
        print(
            f"Batch processing for {os.path.basename(tracker_path)} complete. Resolved {total_processed} items in {elapsed:.2f}s."
        )

        return total_processed
