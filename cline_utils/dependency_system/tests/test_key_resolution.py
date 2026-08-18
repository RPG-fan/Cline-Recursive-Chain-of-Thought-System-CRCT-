import argparse
import unittest
from io import StringIO
from unittest.mock import patch

from cline_utils.dependency_system.core.key_manager import (
    KeyInfo,
    _apply_global_instance_suffixes,
)
from cline_utils.dependency_system.dependency_processor import (
    handle_show_dependencies,
)
from cline_utils.dependency_system.utils.tracker_utils import (
    get_globally_resolved_key_info_for_cli,
)


class TestKeyResolution(unittest.TestCase):
    """Tests for get_globally_resolved_key_info_for_cli and handle_show_dependencies."""

    def setUp(self):
        self.mock_global_map = {
            "H:/path/doc.md": KeyInfo(
                key_string="2A5#1",
                norm_path="H:/path/doc.md",
                parent_path="H:/path",
                tier=2,
                is_directory=False,
            ),
            "H:/path/state.py": KeyInfo(
                key_string="2A5#3",
                norm_path="H:/path/state.py",
                parent_path="H:/path",
                tier=2,
                is_directory=False,
            ),
            "H:/path/SettingsModal.svelte": KeyInfo(
                key_string="2A5#4",
                norm_path="H:/path/SettingsModal.svelte",
                parent_path="H:/path",
                tier=2,
                is_directory=False,
            ),
            "H:/path/event_mgr.py": KeyInfo(
                key_string="1Bg3",
                norm_path="H:/path/event_mgr.py",
                parent_path="H:/path",
                tier=1,
                is_directory=False,
            ),
            "H:/path/prefix1.py": KeyInfo(
                key_string="2A1#1",
                norm_path="H:/path/prefix1.py",
                parent_path="H:/path",
                tier=2,
                is_directory=False,
            ),
            "H:/path/prefix10.py": KeyInfo(
                key_string="2A10",
                norm_path="H:/path/prefix10.py",
                parent_path="H:/path",
                tier=2,
                is_directory=False,
            ),
            "H:/path/prefix11.py": KeyInfo(
                key_string="2A11",
                norm_path="H:/path/prefix11.py",
                parent_path="H:/path",
                tier=2,
                is_directory=False,
            ),
            "H:/path/single_inst.py": KeyInfo(
                key_string="3X1#1",
                norm_path="H:/path/single_inst.py",
                parent_path="H:/path",
                tier=3,
                is_directory=False,
            ),
        }

    def test_exact_resolution_non_contiguous_instance_3(self):
        ki = get_globally_resolved_key_info_for_cli(
            "2A5", 3, self.mock_global_map, "source"
        )
        self.assertIsNotNone(ki)
        self.assertEqual(ki.key_string, "2A5#3")
        self.assertEqual(ki.norm_path, "H:/path/state.py")

    def test_exact_resolution_non_contiguous_instance_4(self):
        ki = get_globally_resolved_key_info_for_cli(
            "2A5", 4, self.mock_global_map, "source"
        )
        self.assertIsNotNone(ki)
        self.assertEqual(ki.key_string, "2A5#4")
        self.assertEqual(ki.norm_path, "H:/path/SettingsModal.svelte")

    def test_invalid_instance_number_shows_available(self):
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            ki = get_globally_resolved_key_info_for_cli(
                "2A5", 2, self.mock_global_map, "source"
            )
            self.assertIsNone(ki)
            output = mock_out.getvalue()
            self.assertIn("specifies an invalid global instance number", output)
            self.assertIn("2A5#1", output)
            self.assertIn("2A5#3", output)
            self.assertIn("2A5#4", output)

    def test_ambiguous_base_key_without_instance(self):
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            ki = get_globally_resolved_key_info_for_cli(
                "2A5", None, self.mock_global_map, "source"
            )
            self.assertIsNone(ki)
            output = mock_out.getvalue()
            self.assertIn("is globally ambiguous", output)
            self.assertIn("2A5#1", output)
            self.assertIn("2A5#3", output)
            self.assertIn("2A5#4", output)

    def test_exact_base_key_isolation_no_prefix_overlap(self):
        # 2A1 should not match 2A10 or 2A11
        matching = [
            info
            for info in self.mock_global_map.values()
            if info.key_string.split("#")[0] == "2A1"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].key_string, "2A1#1")

    def test_single_item_with_instance_queried_as_base_key(self):
        ki = get_globally_resolved_key_info_for_cli(
            "3X1", None, self.mock_global_map, "source"
        )
        self.assertIsNotNone(ki)
        self.assertEqual(ki.key_string, "3X1#1")

    def test_unique_key_without_instance(self):
        ki = get_globally_resolved_key_info_for_cli(
            "1Bg3", None, self.mock_global_map, "source"
        )
        self.assertIsNotNone(ki)
        self.assertEqual(ki.key_string, "1Bg3")

    def test_nonexistent_base_key(self):
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            ki = get_globally_resolved_key_info_for_cli(
                "NONEXISTENT", None, self.mock_global_map, "source"
            )
            self.assertIsNone(ki)
            output = mock_out.getvalue()
            self.assertIn("Base source key 'NONEXISTENT' not found", output)


class TestShowDependenciesIntegration(unittest.TestCase):
    """Integration tests for handle_show_dependencies."""

    def setUp(self):
        self.mock_global_map = {
            "H:/Projects/cline_LLMRPG/docs/development_guidance/AI_and_LLM_Operations/System_Forbidden_Tags_Negative_Space.md": KeyInfo(
                key_string="2A5#1",
                norm_path="H:/Projects/cline_LLMRPG/docs/development_guidance/AI_and_LLM_Operations/System_Forbidden_Tags_Negative_Space.md",
                parent_path="H:/Projects/cline_LLMRPG/docs/development_guidance/AI_and_LLM_Operations",
                tier=2,
                is_directory=False,
            ),
            "H:/Projects/cline_LLMRPG/src/game_loop/creation/state.py": KeyInfo(
                key_string="2A5#3",
                norm_path="H:/Projects/cline_LLMRPG/src/game_loop/creation/state.py",
                parent_path="H:/Projects/cline_LLMRPG/src/game_loop/creation",
                tier=2,
                is_directory=False,
            ),
            "H:/Projects/cline_LLMRPG/src/ui/components/SettingsModal.svelte": KeyInfo(
                key_string="2A5#4",
                norm_path="H:/Projects/cline_LLMRPG/src/ui/components/SettingsModal.svelte",
                parent_path="H:/Projects/cline_LLMRPG/src/ui/components",
                tier=2,
                is_directory=False,
            ),
        }

    @patch(
        "cline_utils.dependency_system.dependency_processor._load_global_map_or_exit"
    )
    @patch(
        "cline_utils.dependency_system.dependency_processor.find_all_tracker_paths",
        return_value=[],
    )
    @patch(
        "cline_utils.dependency_system.dependency_processor._load_token_metadata",
        return_value={},
    )
    def test_show_dependencies_2A5_3(self, _mock_tokens, _mock_trackers, mock_load_map):
        mock_load_map.return_value = self.mock_global_map
        args = argparse.Namespace(key="2A5#3")
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            ret = handle_show_dependencies(args)
            self.assertEqual(ret, 0)
            output = mock_out.getvalue()
            self.assertIn("--- Dependencies for: 2A5#3", output)
            self.assertIn("src/game_loop/creation/state.py", output)

    @patch(
        "cline_utils.dependency_system.dependency_processor._load_global_map_or_exit"
    )
    @patch(
        "cline_utils.dependency_system.dependency_processor.find_all_tracker_paths",
        return_value=[],
    )
    @patch(
        "cline_utils.dependency_system.dependency_processor._load_token_metadata",
        return_value={},
    )
    def test_show_dependencies_2A5_4(self, _mock_tokens, _mock_trackers, mock_load_map):
        mock_load_map.return_value = self.mock_global_map
        args = argparse.Namespace(key="2A5#4")
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            ret = handle_show_dependencies(args)
            self.assertEqual(ret, 0)
            output = mock_out.getvalue()
            self.assertIn("--- Dependencies for: 2A5#4", output)
            self.assertIn("src/ui/components/SettingsModal.svelte", output)

    @patch(
        "cline_utils.dependency_system.dependency_processor._load_global_map_or_exit"
    )
    @patch(
        "cline_utils.dependency_system.dependency_processor.find_all_tracker_paths",
        return_value=[],
    )
    @patch(
        "cline_utils.dependency_system.dependency_processor._load_token_metadata",
        return_value={},
    )
    def test_show_dependencies_2A5_2_invalid(
        self, _mock_tokens, _mock_trackers, mock_load_map
    ):
        mock_load_map.return_value = self.mock_global_map
        args = argparse.Namespace(key="2A5#2")
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            ret = handle_show_dependencies(args)
            self.assertEqual(ret, 1)
            output = mock_out.getvalue()
            self.assertIn("specifies an invalid global instance number", output)
            self.assertIn("2A5#1", output)
            self.assertIn("2A5#3", output)
            self.assertIn("2A5#4", output)


class TestApplyGlobalInstanceSuffixes(unittest.TestCase):
    """Tests for sequential 1..N instance number assignment in _apply_global_instance_suffixes."""

    def test_strictly_sequential_instances_no_gaps(self):
        input_map = {
            "H:/path/c_state.py": KeyInfo(
                key_string="2A5",
                norm_path="H:/path/c_state.py",
                parent_path="H:/path",
                tier=2,
                is_directory=False,
            ),
            "H:/path/a_doc.md": KeyInfo(
                key_string="2A5",
                norm_path="H:/path/a_doc.md",
                parent_path="H:/path",
                tier=2,
                is_directory=False,
            ),
            "H:/path/z_modal.svelte": KeyInfo(
                key_string="2A5",
                norm_path="H:/path/z_modal.svelte",
                parent_path="H:/path",
                tier=2,
                is_directory=False,
            ),
        }
        old_map_with_gaps = {
            "H:/path/a_doc.md": KeyInfo(
                "2A5#1", "H:/path/a_doc.md", "H:/path", 2, False
            ),
            "H:/path/c_state.py": KeyInfo(
                "2A5#3", "H:/path/c_state.py", "H:/path", 2, False
            ),
            "H:/path/z_modal.svelte": KeyInfo(
                "2A5#4", "H:/path/z_modal.svelte", "H:/path", 2, False
            ),
        }
        # Even if old_map had gaps (#1, #3, #4), the new map MUST be contiguous (1, 2, 3)
        result = _apply_global_instance_suffixes(input_map, old_map_with_gaps)
        self.assertEqual(result["H:/path/a_doc.md"].key_string, "2A5#1")
        self.assertEqual(result["H:/path/c_state.py"].key_string, "2A5#2")
        self.assertEqual(result["H:/path/z_modal.svelte"].key_string, "2A5#3")

    def test_unique_base_key_has_no_instance_suffix(self):
        input_map = {
            "H:/path/unique.py": KeyInfo(
                key_string="1Bg3#1",
                norm_path="H:/path/unique.py",
                parent_path="H:/path",
                tier=1,
                is_directory=False,
            ),
        }
        result = _apply_global_instance_suffixes(input_map)
        self.assertEqual(result["H:/path/unique.py"].key_string, "1Bg3")
