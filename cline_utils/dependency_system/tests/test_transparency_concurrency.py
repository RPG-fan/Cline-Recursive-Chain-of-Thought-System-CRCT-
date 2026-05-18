import os
import threading
import time
from pathlib import Path
import pytest
from cline_utils.dependency_system.io.transparency_manager import TransparencyManager

def test_transparency_concurrency_stress_test(tmp_path):
    """
    Spawns 30 threads concurrently reading and writing to the TransparencyManager
    to ensure thread safety and resilience against WinError 5 PermissionError.
    """
    registry_file = tmp_path / "transparency_registry.json"
    manager = TransparencyManager(str(registry_file))

    num_threads = 30
    num_iterations = 20
    errors = []

    def worker(thread_idx):
        try:
            # Create a dedicated temp file for this thread's file I/O operations
            thread_file = tmp_path / f"file_thread_{thread_idx}.py"
            
            for iteration in range(num_iterations):
                # 1. Update file metadata
                content = f"# --- STATION_HEADER: TH_{thread_idx} ---\ndef func_{thread_idx}():\n    pass\n"
                thread_file.write_text(content, encoding="utf-8")
                
                manager.update_file_metadata(
                    file_path=str(thread_file),
                    sections={
                        "STATION_HEADER": {
                            "range": [1, 1],
                            "anchors": ["# --- STATION_HEADER: TH_", "def func_"],
                        }
                    },
                    content=content
                )

                # 2. Virtualize connection maps
                map_content = (
                    f"# --- CONNECTION_MAP: 1B(target:7) {{d}} --- func_{thread_idx} [AUTO]\n"
                    f"def func_{thread_idx}():\n"
                    f"    pass\n"
                )
                thread_file.write_text(map_content, encoding="utf-8")
                manager.virtualize_connection_maps(str(thread_file))

                # 3. Read metadata
                metadata = manager.get_file_metadata(str(thread_file))
                assert metadata is not None
                assert "connection_maps" in metadata

                # 4. Check drift
                assert not manager.check_drift(str(thread_file), thread_file.read_text(encoding="utf-8"))

                # 5. Remove and restore markers
                marked_content = (
                    f"---TAGS_START---\n"
                    f"tag1, tag2\n"
                    f"---TAGS_END---\n"
                    f"def func_{thread_idx}():\n"
                    f"    pass\n"
                )
                thread_file.write_text(marked_content, encoding="utf-8")
                manager.remove_markers(str(thread_file))
                manager.restore_markers(str(thread_file))

                # Sleep briefly to introduce scheduling interleaving
                time.sleep(0.01)

        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify no errors occurred across all concurrent execution threads
    assert len(errors) == 0, f"Encountered {len(errors)} errors during concurrency test: {errors}"
    
    # Verify the registry file was successfully saved and is loadable
    assert os.path.exists(registry_file)
    with open(registry_file, "r", encoding="utf-8") as f:
        import json
        data = json.load(f)
        assert "files" in data
        # We expect entries for files to be present
        assert len(data["files"]) > 0
