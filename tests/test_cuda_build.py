"""Check platform-specific setup arguments without importing PyTorch."""

from contextlib import ExitStack
import enum
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class Platform(enum.Enum):
    CUDA = "CUDA"
    ASCEND = "ASCEND"
    CPU_ONLY = "CPU_ONLY"


class CudaBuildTests(unittest.TestCase):
    def load_setup(self, os_name, cutlass_exists=True, detected_platform=Platform.CUDA):
        os_module = types.ModuleType("os")
        os_module.__dict__.update(vars(os))
        os_module.name = os_name

        kernelkit = types.ModuleType("tests.kernelkit")
        kernelkit.Platform = Platform
        kernelkit.get_current_platform = lambda: detected_platform
        tests = types.ModuleType("tests")
        tests.kernelkit = kernelkit

        extension = types.ModuleType("torch.utils.cpp_extension")
        extension.CUDA_HOME = str(ROOT / "test-cuda-toolkit")
        extension.BuildExtension = type("BuildExtension", (), {})
        extension.CUDAExtension = Mock(side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs))

        setuptools = types.ModuleType("setuptools")
        setuptools.setup = Mock()
        setuptools.find_packages = Mock(return_value=["deep_select"])

        modules = {
            "os": os_module,
            "tests": tests,
            "tests.kernelkit": kernelkit,
            "torch.utils.cpp_extension": extension,
            "setuptools": setuptools,
        }
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.dict(os.environ, {"DEEP_SELECT_BUILD_TARGET_PLATFORM": "CUDA"}))
            check_output = stack.enter_context(patch("subprocess.check_output", side_effect=[
                b"test-revision\n", b"Cuda compilation tools, release 12.9, V12.9.86\n",
            ]))
            stack.enter_context(patch.object(Path, "is_file", return_value=cutlass_exists))
            stack.enter_context(patch("builtins.print"))
            runpy.run_path(str(ROOT / "setup.py"), run_name="__main__")
        self.assertEqual(check_output.call_args.args[0], [
            str(Path(extension.CUDA_HOME) / "bin" / ("nvcc.exe" if os_name == "nt" else "nvcc")),
            "--version",
        ])
        return extension.CUDAExtension.call_args.kwargs

    def test_windows_arguments(self):
        args = self.load_setup("nt")
        cxx = args["extra_compile_args"]["cxx"]
        self.assertIn("/O2", cxx)
        self.assertIn("/std:c++20", cxx)
        self.assertIn("/Zc:preprocessor", cxx)
        self.assertIn("/DKERUTILS_IS_BUILD_ON_CUDA", cxx)
        self.assertIn("/DDEEP_SELECT_IS_BUILD_ON_CUDA", cxx)
        self.assertFalse(any(flag.startswith("-") for flag in cxx))
        self.assertIn("-Xcompiler=/Zc:preprocessor", args["extra_compile_args"]["nvcc"])
        self.assertEqual(args["libraries"], ["cuda"])
        self.assertEqual(args["library_dirs"], [])
        self.assertIn(ROOT / "test-cuda-toolkit/include/cccl", args["include_dirs"])

    def test_linux_arguments(self):
        args = self.load_setup("posix")
        self.assertIn("-O3", args["extra_compile_args"]["cxx"])
        self.assertIn("-std=c++20", args["extra_compile_args"]["cxx"])
        self.assertNotIn("-Xcompiler=/Zc:preprocessor", args["extra_compile_args"]["nvcc"])
        self.assertEqual(args["libraries"], ["cuda"])
        self.assertEqual(args["library_dirs"], [
            str(ROOT / "test-cuda-toolkit/targets/x86_64-linux/lib/stubs"),
            str(ROOT / "test-cuda-toolkit/targets/sbsa-linux/lib/stubs"),
        ])

    def test_missing_cutlass(self):
        with self.assertRaisesRegex(RuntimeError, "git submodule update --init --recursive"):
            self.load_setup("nt", cutlass_exists=False)

    def test_explicit_cuda_target_without_gpu(self):
        args = self.load_setup("nt", detected_platform=Platform.CPU_ONLY)
        self.assertEqual(args["name"], "deep_select.deep_select_cuda")
        self.assertIn("/DDEEP_SELECT_IS_BUILD_ON_CUDA", args["extra_compile_args"]["cxx"])

    def test_spill_checker_tool_paths(self):
        platform = types.ModuleType("tests.kernelkit.platform")
        platform.Platform = Platform
        platform.get_current_platform = lambda: Platform.CPU_ONLY
        platform.requires_platform = lambda supported: lambda function: function
        extension = types.ModuleType("torch.utils.cpp_extension")
        extension.CUDA_HOME = str(ROOT / "test-cuda-toolkit")
        for os_name, executable in [("nt", "cuobjdump.exe"), ("posix", "cuobjdump")]:
            with self.subTest(os_name=os_name), ExitStack() as stack:
                os_module = types.ModuleType("os")
                os_module.__dict__.update(vars(os))
                os_module.name = os_name
                stack.enter_context(patch.dict(sys.modules, {
                    "os": os_module,
                    "tests.kernelkit.platform": platform,
                    "torch.utils.cpp_extension": extension,
                }))
                exists = stack.enter_context(patch("os.path.exists", return_value=True))
                run = stack.enter_context(patch("subprocess.run", return_value=types.SimpleNamespace(
                    returncode=0, stdout="Function kernel:\nREG:32 STACK:8 SHARED:0 LOCAL:0\n",
                    stderr="",
                )))
                module = runpy.run_path(
                    str(ROOT / "tests/kernelkit/build.py"), run_name="tests.kernelkit.build",
                )
                checker = module["check_kernel_reg_spill_in_artifact"]
                self.assertEqual(checker("kernel.obj", stack_baseline=8, quiet=True), [])
                expected_path = str(Path(extension.CUDA_HOME) / "bin" / executable)
                exists.assert_called_once_with(expected_path)
                self.assertEqual(run.call_args.args[0], [expected_path, "-res-usage", "kernel.obj"])
                run.return_value.stdout = "Function kernel:\nREG:32 STACK:8 SHARED:0 LOCAL:16\n"
                self.assertEqual(checker("kernel.obj", stack_baseline=8, quiet=True), ["kernel"])


if __name__ == "__main__":
    unittest.main()
