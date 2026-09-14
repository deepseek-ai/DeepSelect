"""CPU-only tests for Python interface validation.

Run with ``python -O`` to ensure public validation does not depend on removable
assertions. The CUDA extension is replaced with a tiny fake, but real PyTorch
tensors exercise the public function's argument handling.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

import torch


class InterfaceValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend_calls = 0
        module_names = (
            "deep_select",
            "deep_select.deep_select_cuda",
            "deep_select.interface",
        )
        cls.saved_modules = {name: sys.modules.get(name) for name in module_names}
        backend = types.ModuleType("deep_select.deep_select_cuda")
        backend.get_alignment_requirement = lambda: (16, 32)

        def topk(*_args: object) -> None:
            cls.backend_calls += 1

        backend.topk = topk
        package = types.ModuleType("deep_select")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "deep_select")]
        sys.modules["deep_select"] = package
        sys.modules["deep_select.deep_select_cuda"] = backend
        cls.interface = importlib.import_module("deep_select.interface")

    @classmethod
    def tearDownClass(cls) -> None:
        for name, module in cls.saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def setUp(self) -> None:
        self.__class__.backend_calls = 0
        self.input = torch.empty((1, 32), dtype=torch.float32)

    def assert_rejected_before_backend(
        self, error_type: type[Exception], **kwargs: object
    ) -> None:
        with self.assertRaises(error_type):
            self.interface.topk(self.input, 4, return_value=False, **kwargs)
        self.assertEqual(self.backend_calls, 0)

    def test_begin_is_rejected_explicitly(self) -> None:
        self.assert_rejected_before_backend(
            ValueError, begin=torch.empty(1, dtype=torch.int32)
        )

    def test_hint_is_rejected_explicitly(self) -> None:
        self.assert_rejected_before_backend(
            ValueError, hint=torch.empty(1, dtype=torch.int32)
        )

    def test_preallocated_output_dtype_must_match(self) -> None:
        self.assert_rejected_before_backend(
            TypeError,
            output_idx=torch.empty((1, 4), dtype=torch.int32),
            indices_type=torch.int64,
        )

    def test_valid_preallocated_output_reaches_backend(self) -> None:
        output_idx = torch.empty((1, 4), dtype=torch.int64)
        output_value, returned_idx = self.interface.topk(
            self.input,
            4,
            return_value=False,
            output_idx=output_idx,
            indices_type=torch.int64,
        )
        self.assertIsNone(output_value)
        self.assertIs(returned_idx, output_idx)
        self.assertEqual(self.backend_calls, 1)


if __name__ == "__main__":
    unittest.main()
