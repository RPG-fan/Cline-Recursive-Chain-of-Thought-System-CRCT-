import threading
import time
from unittest.mock import patch, MagicMock
from cline_utils.dependency_system.analysis.local_llm_processor import LocalLLMProcessor

def test_local_llm_thread_safety_serialization():
    # Keep track of concurrent execution in the Llama constructor
    active_loads = 0
    max_concurrent_loads = 0
    load_lock = threading.Lock()

    def mock_llama_init(*args, **kwargs):
        nonlocal active_loads, max_concurrent_loads
        with load_lock:
            active_loads += 1
            if active_loads > max_concurrent_loads:
                max_concurrent_loads = active_loads
        
        # Sleep to simulate time spent loading the heavy model and allow concurrency
        time.sleep(0.1)
        
        with load_lock:
            active_loads -= 1
        
        # Return a mock model object
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1, 2, 3]
        return mock_model

    # Use a dummy model path
    processor = LocalLLMProcessor("models/dummy.gguf")

    # Patch the Llama class in local_llm_processor module
    with patch("cline_utils.dependency_system.analysis.local_llm_processor.Llama") as mock_llama_class:
        # Make sure Llama is not treated as None in the code
        mock_llama_class.side_effect = mock_llama_init
        
        # Spawn multiple thread workers to concurrently call _load_model
        threads = []
        errors = []

        def worker(thread_id):
            try:
                # n_gpu_layers=0 to skip dynamic VRAM calculation to keep it fast/pure
                # required_ctx is varied to force reloading/closing
                ctx = 1024 + thread_id * 512
                processor._load_model(required_ctx=ctx, n_gpu_layers=0)
            except Exception as e:
                errors.append(e)

        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Workers encountered errors: {errors}"
        # Verify that allocations were completely serialized (max concurrent loads is exactly 1)
        assert max_concurrent_loads == 1, f"Expected 1 concurrent load at most, but got {max_concurrent_loads}"
