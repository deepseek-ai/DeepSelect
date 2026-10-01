"""Host argument/allocation tests; the native top-k implementation is not executed."""

import importlib.util
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import torch


class TopkValidationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(__file__).resolve().parents[1]
        self.backend = ModuleType("_fake_topk_backend")
        self.backend.get_alignment_requirement = lambda: (16, 32)
        self.backend.topk = Mock()
        package = ModuleType("deep_select")
        package.__path__ = []
        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "deep_select": package,
                    "deep_select.deep_select_cuda": self.backend,
                    "deep_select.deep_select_npu": self.backend,
                },
            )
        )
        spec = importlib.util.spec_from_file_location(
            "deep_select.interface", root / "deep_select/interface.py"
        )
        self.interface = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.interface)
        self.input = torch.ones((2, 64), dtype=torch.float32)

    def assert_rejected_before_allocation(self, k, **kwargs):
        with (
            patch.object(
                torch,
                "empty",
                side_effect=AssertionError("output allocation attempted"),
            ) as allocate,
            self.assertRaisesRegex(ValueError, "topk"),
        ):
            self.interface.topk(self.input, k, **kwargs)
        allocate.assert_not_called()
        self.backend.topk.assert_not_called()

    def test_zero_is_rejected_before_allocation(self):
        self.assert_rejected_before_allocation(0)

    def test_negative_is_rejected_before_allocation(self):
        self.assert_rejected_before_allocation(-1)

    def test_above_limit_is_rejected_before_allocation(self):
        self.assert_rejected_before_allocation(4097)

    def test_huge_k_is_rejected_without_allocating(self):
        self.assert_rejected_before_allocation(2**31)

    def test_indices_only_still_rejects_invalid_k_early(self):
        self.assert_rejected_before_allocation(4097, return_value=False)

    def test_small_valid_k_keeps_output_alignment(self):
        values, indices = self.interface.topk(self.input, 1)
        self.assertEqual(values.shape, (2, 1))
        self.assertEqual(indices.shape, (2, 1))
        self.assertEqual(values.stride(0) * values.element_size() % 32, 0)
        self.assertEqual(indices.stride(0) * indices.element_size() % 32, 0)
        self.backend.topk.assert_called_once()

    def test_limit_and_k_larger_than_vocab_remain_allowed(self):
        values, indices = self.interface.topk(self.input, 4096)
        self.assertEqual(values.shape, (2, 4096))
        self.assertEqual(indices.shape, (2, 4096))
        self.backend.topk.assert_called_once()

    def test_caller_provided_output_is_preserved(self):
        supplied = torch.empty((2, 4), dtype=torch.int64)
        values, indices = self.interface.topk(
            self.input, 4, return_value=False, output_idx=supplied
        )
        self.assertIsNone(values)
        self.assertIs(indices, supplied)


if __name__ == "__main__":
    unittest.main()
