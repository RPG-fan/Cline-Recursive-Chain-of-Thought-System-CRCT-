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

    @patch("cline_utils.dependency_system.io.context_packager.load_global_key_map")
    @patch("cline_utils.dependency_system.io.context_packager._load_token_metadata")
    @patch("cline_utils.dependency_system.io.context_packager.aggregate_all_dependencies")
    @patch("cline_utils.dependency_system.io.context_packager.find_all_tracker_paths")
    @patch("cline_utils.dependency_system.io.context_packager.build_path_migration_map")
    def test_build_package_basic(
        self,
        mock_migration_map,
        mock_tracker_paths,
        mock_agg_deps,
        mock_token_meta,
        mock_key_map,
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

        # 4. Mock file content overlay and static essence
        self.packager.get_overlaid_content = MagicMock(return_value="print('file_content')")
        self.packager.get_ses_content = MagicMock(return_value="def func(): ...")

        # Execute
        result = self.packager.build_package(target_keys=["1A1"], max_tokens=1000)

        # Asserts
        self.assertIn("# TARGET CORE LOGIC", result)
        self.assertIn("1A1", result)
        self.assertIn("Tier 1: Mutual dependencies (x)", result)
        self.assertIn("1A2", result)
        self.packager.get_overlaid_content.assert_any_call(path1)

    @patch("cline_utils.dependency_system.io.context_packager.load_global_key_map")
    @patch("cline_utils.dependency_system.io.context_packager._load_token_metadata")
    @patch("cline_utils.dependency_system.io.context_packager.aggregate_all_dependencies")
    @patch("cline_utils.dependency_system.io.context_packager.find_all_tracker_paths")
    @patch("cline_utils.dependency_system.io.context_packager.build_path_migration_map")
    def test_build_package_ses_fallback(
        self,
        mock_migration_map,
        mock_tracker_paths,
        mock_agg_deps,
        mock_token_meta,
        mock_key_map,
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

        # Setup mock token metadata (file1 full is 100, file2 full is 500 but ses is 50)
        mock_token_meta.return_value = {
            path1: {"full_tokens": 100, "ses_tokens": 30},
            path2: {"full_tokens": 500, "ses_tokens": 50},
        }

        # Setup mock aggregated dependencies (T1 mutual x)
        mock_agg_deps.return_value = {
            ("1A1", "1A2"): ("x", {normalize_path("/mock/project/tracker.md")})
        }

        # Mock overlaid content and SES representation
        self.packager.get_overlaid_content = MagicMock(return_value="print('full')")
        self.packager.get_ses_content = MagicMock(return_value="def func_ses(): ...")

        # Set max_tokens to 200. target 1A1 takes 100, leaving 100 budget.
        # file2.py full is 500 (does not fit), but ses is 50 (fits!)
        result = self.packager.build_package(target_keys=["1A1"], max_tokens=200)

        # Asserts
        self.assertIn("# TARGET CORE LOGIC", result)
        self.assertIn("1A1", result)
        self.assertIn("1A2", result)
        self.assertIn("def func_ses(): ...", result)
        self.assertIn("SES Signatures Only", result)


if __name__ == "__main__":
    unittest.main()
