# test_transparency_marker_placement.py

import os
import unittest
from pathlib import Path
from cline_utils.dependency_system.io.transparency_manager import TransparencyManager


class TestTransparencyMarkerPlacement(unittest.TestCase):
    """
    Verifies that transparency markers are placed at the correct positions
    during remove and restore cycles, especially around section borders.
    """

    def setUp(self):
        self.tmp_dir = Path("H:/Projects/Cline-Recursive-Chain-of-Thought-System-CRCT-/scratch/test_markers")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.tmp_dir / "transparency_registry.json"
        if self.registry_path.exists():
            self.registry_path.unlink()
        self.manager = TransparencyManager(str(self.registry_path))

    def tearDown(self):
        # Clean up files
        for file in self.tmp_dir.glob("*"):
            try:
                if file.is_file():
                    file.unlink()
            except Exception:
                pass
        try:
            self.tmp_dir.rmdir()
        except Exception:
            pass

    def test_marker_placement_cycle(self):
        # 1. Create a sample markdown file with correct compliance markers
        file_path = self.tmp_dir / "sample_doc.md"
        content_with_markers = (
            "# Sample Doc Title\n"
            "\n"
            "---CONTEXT_START---\n"
            "## Context\n"
            "This is context content line 1.\n"
            "This is context content line 2.\n"
            "---CONTEXT_END---\n"
            "\n"
            "---OVERVIEW_START---\n"
            "## Overview\n"
            "This is overview content line 1.\n"
            "---OVERVIEW_END---\n"
            "\n"
            "---DETAILS_START---\n"
            "## Details\n"
            "This is details content line 1.\n"
            "This is details content line 2.\n"
            "---DETAILS_END---\n"
        )
        file_path.write_text(content_with_markers, encoding="utf-8")

        # 2. Call remove_markers to move them to the transparency registry
        success = self.manager.remove_markers(str(file_path))
        self.assertTrue(success)

        # Verify the file on disk is clean (no markers)
        clean_content = file_path.read_text(encoding="utf-8")
        self.assertNotIn("---CONTEXT_START---", clean_content)
        self.assertNotIn("---CONTEXT_END---", clean_content)
        self.assertNotIn("---OVERVIEW_START---", clean_content)
        self.assertNotIn("---OVERVIEW_END---", clean_content)

        # 3. Call restore_markers to restore them to the file
        success_restore = self.manager.restore_markers(str(file_path))
        self.assertTrue(success_restore)

        # Read restored content
        restored_content = file_path.read_text(encoding="utf-8")

        # Verify exact positioning of the restored content
        self.assertEqual(restored_content, content_with_markers)

    def test_marker_placement_with_drift_recovery(self):
        # 1. Create a sample markdown file with correct compliance markers
        file_path = self.tmp_dir / "sample_doc.md"
        content_with_markers = (
            "# Sample Doc Title\n"
            "\n"
            "---CONTEXT_START---\n"
            "## Context\n"
            "This is context content line 1.\n"
            "This is context content line 2.\n"
            "---CONTEXT_END---\n"
            "\n"
            "---OVERVIEW_START---\n"
            "## Overview\n"
            "This is overview content line 1.\n"
            "---OVERVIEW_END---\n"
            "\n"
            "---DETAILS_START---\n"
            "## Details\n"
            "This is details content line 1.\n"
            "This is details content line 2.\n"
            "---DETAILS_END---\n"
        )
        file_path.write_text(content_with_markers, encoding="utf-8")

        # 2. Call remove_markers to move them to the transparency registry
        success = self.manager.remove_markers(str(file_path))
        self.assertTrue(success)

        # 3. Simulate drift on clean file: prepend a comment block on disk
        clean_content = file_path.read_text(encoding="utf-8")
        drifted_content = "# Prepend Comment Line 1\n# Prepend Comment Line 2\n" + clean_content
        file_path.write_text(drifted_content, encoding="utf-8")

        # 4. Call restore_markers which should recover alignment dynamically and place markers
        success_restore = self.manager.restore_markers(str(file_path))
        self.assertTrue(success_restore)

        # Read restored content
        restored_content = file_path.read_text(encoding="utf-8")

        # Verify that markers are placed correctly relative to sections (last line included)
        expected_restored = (
            "# Prepend Comment Line 1\n"
            "# Prepend Comment Line 2\n"
            "# Sample Doc Title\n"
            "\n"
            "---CONTEXT_START---\n"
            "## Context\n"
            "This is context content line 1.\n"
            "This is context content line 2.\n"
            "---CONTEXT_END---\n"
            "\n"
            "---OVERVIEW_START---\n"
            "## Overview\n"
            "This is overview content line 1.\n"
            "---OVERVIEW_END---\n"
            "\n"
            "---DETAILS_START---\n"
            "## Details\n"
            "This is details content line 1.\n"
            "This is details content line 2.\n"
            "---DETAILS_END---\n"
        )
        self.maxDiff = None
        try:
            self.assertEqual(restored_content, expected_restored)
        except AssertionError as e:
            print("\n--- ACTUAL RESTORED CONTENT ---")
            print(repr(restored_content))
            print("\n--- EXPECTED RESTORED CONTENT ---")
            print(repr(expected_restored))
            print("\n--- METADATA ---")
            import json
            print(json.dumps(self.manager.get_raw_file_metadata(str(file_path)), indent=2))
            raise e

    def test_marker_placement_with_incorrect_registry_range_correction(self):
        # 1. Create a sample markdown file with correct compliance markers
        file_path = self.tmp_dir / "sample_doc.md"
        content_with_markers = (
            "# Sample Doc Title\n"
            "\n"
            "---CONTEXT_START---\n"
            "## Context\n"
            "This is context content line 1.\n"
            "This is context content line 2.\n"
            "---CONTEXT_END---\n"
        )
        file_path.write_text(content_with_markers, encoding="utf-8")

        # 2. Call remove_markers to move them to the transparency registry
        success = self.manager.remove_markers(str(file_path))
        self.assertTrue(success)

        # 3. Deliberately corrupt the range in the registry (e.g. shift range from [3, 6] to [2, 5])
        # The file content remains clean (no drift), but registry ranges are wrong.
        from cline_utils.dependency_system.utils.path_utils import normalize_path
        norm_path = normalize_path(str(file_path))
        self.manager._registry["files"][norm_path]["sections"]["CONTEXT"]["range"] = [2, 5]
        self.manager._save()

        # 4. Call restore_markers which should check anchors, see mismatch, and auto-correct range back to [3, 6]
        success_restore = self.manager.restore_markers(str(file_path))
        self.assertTrue(success_restore)

        # Read restored content and assert it was correctly wrapped
        restored_content = file_path.read_text(encoding="utf-8")
        self.assertEqual(restored_content, content_with_markers)


if __name__ == "__main__":
    unittest.main()


