"""Exercise platform detection without requiring PyTorch, GPUs, or PCI tools."""

from contextlib import ExitStack
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.torch = types.ModuleType("torch")
        self.cuda_available = Mock(return_value=False)
        self.torch.cuda = types.SimpleNamespace(is_available=self.cuda_available)
        self.torch.version = types.SimpleNamespace(cuda="13.4")
        with patch.dict(sys.modules, {"torch": self.torch}):
            self.platform = types.SimpleNamespace(**runpy.run_path(
                str(ROOT / "tests/kernelkit/platform.py"),
            ))
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.device_exists = stack.enter_context(patch("os.path.exists", return_value=False))
        self.pci_probe = stack.enter_context(patch(
            "subprocess.check_output",
            side_effect=AssertionError("Platform detection must not invoke lspci"),
        ))

    def test_cuda_detection_without_lspci(self):
        self.cuda_available.return_value = True
        for os_name in ["nt", "posix"]:
            with self.subTest(os_name=os_name), patch.object(os, "name", os_name):
                self.platform.get_current_platform.cache_clear()
                self.assertEqual(self.platform.get_current_platform(), self.platform.Platform.CUDA)
        self.pci_probe.assert_not_called()

    def test_cpu_detection_without_lspci(self):
        for os_name in ["nt", "posix"]:
            with self.subTest(os_name=os_name), patch.object(os, "name", os_name):
                self.platform.get_current_platform.cache_clear()
                self.assertEqual(self.platform.get_current_platform(), self.platform.Platform.CPU_ONLY)
        self.pci_probe.assert_not_called()

    def test_ascend_takes_precedence(self):
        self.device_exists.return_value = True
        self.cuda_available.return_value = True
        self.assertEqual(self.platform.get_current_platform(), self.platform.Platform.ASCEND)
        self.device_exists.assert_called_once_with("/dev/davinci_manager")
        self.cuda_available.assert_not_called()
        self.pci_probe.assert_not_called()

    def test_non_cuda_pytorch_is_not_classified_as_cuda(self):
        self.torch.version.cuda = None
        self.cuda_available.return_value = True  # ROCm also exposes torch.cuda.
        self.assertEqual(self.platform.get_current_platform(), self.platform.Platform.CPU_ONLY)
        self.cuda_available.assert_not_called()

    def test_detection_is_cached(self):
        self.cuda_available.return_value = True
        self.assertEqual(self.platform.get_current_platform(), self.platform.Platform.CUDA)
        self.cuda_available.return_value = False
        self.assertTrue(self.platform.is_on_cuda_platform())
        self.assertFalse(self.platform.is_on_cpu_only_platform())
        self.assertFalse(self.platform.is_on_ascend_platform())
        self.cuda_available.assert_called_once_with()
        self.device_exists.assert_called_once()

    def test_cuda_errors_are_not_hidden(self):
        self.cuda_available.side_effect = RuntimeError("CUDA probe failed")
        with self.assertRaisesRegex(RuntimeError, "CUDA probe failed"):
            self.platform.get_current_platform()

    def test_build_override_does_not_claim_runtime_cuda_support(self):
        with patch.dict(os.environ, {"DEEP_SELECT_BUILD_TARGET_PLATFORM": "CUDA"}):
            self.assertEqual(self.platform.get_current_platform(), self.platform.Platform.CPU_ONLY)


if __name__ == "__main__":
    unittest.main()
