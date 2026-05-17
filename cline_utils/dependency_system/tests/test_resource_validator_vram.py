import unittest
from typing import Generator
from unittest.mock import MagicMock, patch
from cline_utils.dependency_system.utils.resource_validator import ResourceValidator


class TestResourceValidatorVRAM(unittest.TestCase):
    validator: ResourceValidator

    def setUp(self) -> None:
        self.validator = ResourceValidator()

    @patch(
        "cline_utils.dependency_system.utils.resource_validator.TORCH_AVAILABLE", True
    )
    @patch("cline_utils.dependency_system.utils.resource_validator.torch")
    def test_wait_for_vram_release_success(self, mock_torch: MagicMock) -> None:
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        # Simulate VRAM growth: 100 -> 150 -> 200 (target)
        mock_torch.cuda.mem_get_info.side_effect = [
            (100 * 1024 * 1024, 0),  # prev_free_mb query
            (150 * 1024 * 1024, 0),  # first poll
            (200 * 1024 * 1024, 0),  # second poll (target reached)
        ]

        result = self.validator.wait_for_vram_release(
            target_free_mb=200, poll_interval=0.01, tolerance_mb=0.0
        )
        self.assertTrue(result)
        self.assertEqual(mock_torch.cuda.mem_get_info.call_count, 3)

    @patch(
        "cline_utils.dependency_system.utils.resource_validator.TORCH_AVAILABLE", True
    )
    @patch("cline_utils.dependency_system.utils.resource_validator.torch")
    def test_wait_for_vram_release_stall(self, mock_torch: MagicMock) -> None:
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        # Simulate VRAM stall: 100 -> 110 -> 110 -> 110 (stall)
        mock_torch.cuda.mem_get_info.side_effect = [
            (100 * 1024 * 1024, 0),  # prev_free_mb
            (110 * 1024 * 1024, 0),  # poll 1 (growth)
            (110 * 1024 * 1024, 0),  # poll 2 (no growth, stall_count=1)
            (110 * 1024 * 1024, 0),  # poll 3 (no growth, stall_count=2)
            (
                110 * 1024 * 1024,
                0,
            ),  # poll 4 (no growth, stall_count=3 -> stall triggered)
        ]

        # We use stall_tolerance=3 (default)
        result = self.validator.wait_for_vram_release(
            target_free_mb=200, poll_interval=0.01
        )
        self.assertFalse(result)
        # Should call mem_get_info 5 times (initial + 4 polls)
        self.assertEqual(mock_torch.cuda.mem_get_info.call_count, 5)

    @patch(
        "cline_utils.dependency_system.utils.resource_validator.TORCH_AVAILABLE", True
    )
    @patch("cline_utils.dependency_system.utils.resource_validator.torch")
    def test_wait_for_vram_release_hard_cap(self, mock_torch: MagicMock) -> None:
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True

        # Always return growth to avoid stall, but never reach target
        def growth_generator() -> Generator[tuple[int, int], None, None]:
            val = 100
            while True:
                yield (val * 1024 * 1024, 0)
                val += 1

        mock_torch.cuda.mem_get_info.side_effect = growth_generator()

        # Set a very low hard cap
        result = self.validator.wait_for_vram_release(
            target_free_mb=1000, poll_interval=0.01, hard_cap_seconds=0.05
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
