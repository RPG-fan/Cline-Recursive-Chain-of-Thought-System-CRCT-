# test_context_packager.py

import unittest
from unittest.mock import patch, MagicMock
from cline_utils.dependency_system.io.context_packager import ContextPackager
from cline_utils.dependency_system.core.key_manager import KeyInfo


class TestContextPackager(unittest.TestCase):
    """
    Tests context packaging, tier priorities, and token-based fallback allocation.
    """

    def setUp(self):
        self.packager = ContextPackager(project_root="/mock/project")

    @patch("cline_utils.dependency_system.io.context_packager.get_transparency_manager")
    @patch("cline_utils.dependency_system.io.context_packager.load_global_key_map")
    @patch("cline_utils.dependency_system.io.context_packager._load_token_metadata")
    @patch(
        "cline_utils.dependency_system.io.context_packager.aggregate_all_dependencies"
    )
    @patch("cline_utils.dependency_system.io.context_packager.find_all_tracker_paths")
    @patch("cline_utils.dependency_system.io.context_packager.build_path_migration_map")
    def test_build_package_basic(
        self,
        mock_migration_map,
        mock_tracker_paths,
        mock_agg_deps,
        mock_token_meta,
        mock_key_map,
        mock_get_tm,
    ):
        from cline_utils.dependency_system.utils.path_utils import normalize_path

        path1 = normalize_path("/mock/project/file1.py")
        path2 = normalize_path("/mock/project/file2.py")

        # 1. Setup mock keymap and KeyInfo
        mock_key_map.return_value = {
            path1: KeyInfo(
                key_string="1A1",
                norm_path=path1,
                parent_path=normalize_path("/mock/project"),
                tier=1,
                is_directory=False,
            ),
            path2: KeyInfo(
                key_string="1A2",
                norm_path=path2,
                parent_path=normalize_path("/mock/project"),
                tier=1,
                is_directory=False,
            ),
        }

        # 2. Setup mock token metadata
        mock_token_meta.return_value = {
            path1: {"full_tokens": 100, "ses_tokens": 30},
            path2: {"full_tokens": 200, "ses_tokens": 50},
        }

        # 3. Setup mock aggregated dependencies (T1 mutual x)
        mock_agg_deps.return_value = {
            ("1A1", "1A2"): ("x", {normalize_path("/mock/project/tracker.md")})
        }

        # 4. Mock transparency manager connection maps
        mock_tm = MagicMock()
        mock_tm.get_file_metadata.return_value = {
            "connection_maps": [
                {
                    "target_key": "1A2",
                    "target_symbol": "func",
                    "target_line": 5,
                }
            ]
        }
        mock_tm._registry = {"files": {}}
        mock_get_tm.return_value = mock_tm

        # 5. Mock file content overlay and static essence / slicing
        self.packager.get_overlaid_content = MagicMock(
            return_value="print('file_content')"
        )
        self.packager._extract_sliced_block_from_overlaid = MagicMock(
            return_value="def func():\n    pass\n"
        )

        # Execute
        result = self.packager.build_package(target_keys=["1A1"], max_tokens=1000)

        # Asserts
        self.assertIn("# TARGET CORE LOGIC", result)
        self.assertIn("1A1", result)
        self.assertIn("Tier 1: Mutual dependencies (x)", result)
        self.assertIn("1A2", result)
        self.assertIn("Sliced Symbols: func", result)
        self.assertIn("def func():", result)
        self.packager.get_overlaid_content.assert_any_call(
            path1, include_connection_maps=False
        )

    @patch("cline_utils.dependency_system.io.context_packager.get_transparency_manager")
    @patch("cline_utils.dependency_system.io.context_packager.load_global_key_map")
    @patch("cline_utils.dependency_system.io.context_packager._load_token_metadata")
    @patch(
        "cline_utils.dependency_system.io.context_packager.aggregate_all_dependencies"
    )
    @patch("cline_utils.dependency_system.io.context_packager.find_all_tracker_paths")
    @patch("cline_utils.dependency_system.io.context_packager.build_path_migration_map")
    def test_build_package_budget_capping(
        self,
        mock_migration_map,
        mock_tracker_paths,
        mock_agg_deps,
        mock_token_meta,
        mock_key_map,
        mock_get_tm,
    ):
        from cline_utils.dependency_system.utils.path_utils import normalize_path

        path1 = normalize_path("/mock/project/file1.py")
        path2 = normalize_path("/mock/project/file2.py")

        # Setup mock keymap and KeyInfo
        mock_key_map.return_value = {
            path1: KeyInfo(
                key_string="1A1",
                norm_path=path1,
                parent_path=normalize_path("/mock/project"),
                tier=1,
                is_directory=False,
            ),
            path2: KeyInfo(
                key_string="1A2",
                norm_path=path2,
                parent_path=normalize_path("/mock/project"),
                tier=1,
                is_directory=False,
            ),
        }

        # Setup mock token metadata
        mock_token_meta.return_value = {
            path1: {"full_tokens": 100, "ses_tokens": 30},
            path2: {"full_tokens": 200, "ses_tokens": 50},
        }

        # Setup mock aggregated dependencies
        mock_agg_deps.return_value = {
            ("1A1", "1A2"): ("x", {normalize_path("/mock/project/tracker.md")})
        }

        # Mock transparency manager connection maps
        mock_tm = MagicMock()
        mock_tm.get_file_metadata.return_value = {
            "connection_maps": [
                {
                    "target_key": "1A2",
                    "target_symbol": "func",
                    "target_line": 5,
                }
            ]
        }
        mock_tm._registry = {"files": {}}
        mock_get_tm.return_value = mock_tm

        # Mock content and slicing: slice content length is 40 chars (~10 tokens)
        self.packager.get_overlaid_content = MagicMock(return_value="print('target')")
        self.packager._extract_sliced_block_from_overlaid = MagicMock(
            return_value="def func_slice():\n    pass\n"
        )

        # Set max_tokens to 105. target takes 100 tokens, leaving 5 tokens budget.
        # Sliced content of 10 tokens exceeds remaining 5 token budget!
        result = self.packager.build_package(target_keys=["1A1"], max_tokens=105)

        # Asserts
        self.assertIn("# TARGET CORE LOGIC", result)
        self.assertIn("1A1", result)
        # Should not include 1A2 and should mention ceiling reached
        self.assertNotIn("1A2", result)
        self.assertIn("Context ceiling of 105 tokens reached", result)

    @patch("cline_utils.dependency_system.io.context_packager.get_transparency_manager")
    @patch("cline_utils.dependency_system.io.context_packager.load_global_key_map")
    @patch("cline_utils.dependency_system.io.context_packager._load_token_metadata")
    @patch(
        "cline_utils.dependency_system.io.context_packager.aggregate_all_dependencies"
    )
    @patch("cline_utils.dependency_system.io.context_packager.find_all_tracker_paths")
    @patch("cline_utils.dependency_system.io.context_packager.build_path_migration_map")
    def test_build_package_omits_directories(
        self,
        mock_migration_map,
        mock_tracker_paths,
        mock_agg_deps,
        mock_token_meta,
        mock_key_map,
        mock_get_tm,
    ):
        from cline_utils.dependency_system.utils.path_utils import normalize_path

        path1 = normalize_path("/mock/project/file1.py")
        path_dir = normalize_path("/mock/project/docs")

        # Setup mock keymap and KeyInfo
        mock_key_map.return_value = {
            path1: KeyInfo(
                key_string="1A1",
                norm_path=path1,
                parent_path=normalize_path("/mock/project"),
                tier=1,
                is_directory=False,
            ),
            path_dir: KeyInfo(
                key_string="1A",
                norm_path=path_dir,
                parent_path=normalize_path("/mock/project"),
                tier=1,
                is_directory=True,
            ),
        }

        # Setup mock token metadata
        mock_token_meta.return_value = {
            path1: {"full_tokens": 100, "ses_tokens": 30},
            path_dir: {"full_tokens": 100, "ses_tokens": 30},
        }

        # Setup mock aggregated dependencies
        mock_agg_deps.return_value = {
            ("1A1", "1A"): ("x", {normalize_path("/mock/project/tracker.md")})
        }

        # Mock transparency manager to return empty metadata
        mock_tm = MagicMock()
        mock_tm.get_file_metadata.return_value = {}
        mock_tm._registry = {"files": {}}
        mock_get_tm.return_value = mock_tm

        # Mock overlaid content
        self.packager.get_overlaid_content = MagicMock(return_value="print('full')")
        self.packager.get_ses_content = MagicMock(return_value="def func_ses(): ...")

        result = self.packager.build_package(target_keys=["1A1", "1A"], max_tokens=1000)

        # Asserts: 1A should be completely omitted from TARGET CORE LOGIC and tiers
        self.assertIn("1A1", result)
        self.assertNotIn("## [1A] docs (Full Content)", result)
        self.assertNotIn("Tier 1: Mutual dependencies (x)", result)


if __name__ == "__main__":
    unittest.main()
