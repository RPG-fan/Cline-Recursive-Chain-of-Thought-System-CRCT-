# test_normalize_docs_resilience.py

import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from cline_utils.dependency_system.io.normalize_docs import prepare_normalization


class TestNormalizeDocsResilience(unittest.TestCase):
    """
    Verifies that prepare_normalization in normalize_docs.py is resilient to existing/misplaced markers.
    """

    def test_prepare_normalization_extracts_context_and_overview_with_existing_markers(self):
        # 1. Setup sample markdown content containing existing/misplaced markers
        content = (
            "# Sample Title\n\n"
            "---CONTEXT_START---\n"
            "## Context\n"
            "This is the actual context paragraph that should be extracted.\n"
            "---CONTEXT_END---\n\n"
            "---OVERVIEW_START---\n"
            "## Overview\n"
            "This is the actual overview paragraph that should be extracted.\n"
            "---OVERVIEW_END---\n\n"
            "## Details\n"
            "Some detailed content here.\n"
        )

        # 2. Mock the processor and its methods
        mock_processor = MagicMock()
        mock_processor.get_token_count.return_value = 100
        mock_processor.generate.return_value = (
            '{\n'
            '  "tags": ["test"],\n'
            '  "related_tags": ["unit-test"],\n'
            '  "role": "Verify resilience of normalization.",\n'
            '  "layer": "Practices & Guidelines",\n'
            '  "context": "Stub context.",\n'
            '  "overview": "Stub overview."\n'
            '}'
        )

        # Create temporary file containing the content
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_path = os.path.join(tmp_dir, "test_doc.md")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # 3. Execute prepare_normalization
            result = prepare_normalization(
                filepath=temp_file_path,
                processor=mock_processor,
                project_root=tmp_dir
            )

            # 4. Asserts
            self.assertIsNotNone(result)
            
            # Check that prompt contains the correctly extracted existing context and overview
            called_args, called_kwargs = mock_processor.generate.call_args
            prompt = called_args[0]
            
            self.assertIn("Existing Context: This is the actual context paragraph that should be extracted.", prompt)
            self.assertIn("Existing Overview: This is the actual overview paragraph that should be extracted.", prompt)


if __name__ == "__main__":
    unittest.main()
