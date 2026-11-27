"""CUDA compilation and host-side payload checks; no PyTorch or GPU required."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CudaCompilationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nvcc = shutil.which("nvcc")
        if cls.nvcc is None:
            raise unittest.SkipTest("nvcc must be on PATH")
        if not (ROOT / "csrc/3rdparty/cutlass/include/cutlass/cutlass.h").is_file():
            raise RuntimeError("Initialize CUTLASS with git submodule update --init --recursive")

    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.output_dir = Path(temporary_directory.name)

    def run_command(self, args):
        result = subprocess.run(args, cwd=self.output_dir, capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, f"{args}\n{result.stdout}\n{result.stderr}")

    def compile(self, source, output, compile_only=False):
        args = [
            self.nvcc, str(source), "-o", str(output),
            "-std=c++20", "-O3", "-DDEEP_SELECT_IS_BUILD_ON_CUDA",
            "-Wno-deprecated-declarations",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr", "--expt-extended-lambda",
            "--use_fast_math", "--ftz=false",
            "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-double-precision-use",
            "-lineinfo", "--source-in-ptx",
            "-gencode", "arch=compute_100a,code=sm_100a",
            "-gencode", "arch=compute_103a,code=sm_103a",
            f"-I{ROOT / 'csrc'}",
            f"-I{ROOT / 'csrc/3rdparty/cutlass/include'}",
            f"-I{ROOT / 'csrc/3rdparty/kerutils/include'}",
        ]
        if os.name == "nt":
            args.append("-Xcompiler=/Zc:preprocessor")
        if compile_only:
            args.append("-c")
        self.run_command(args)

    def test_payloads(self):
        output = self.output_dir / ("payloads.exe" if os.name == "nt" else "payloads")
        self.compile(ROOT / "tests/cuda_portability.cu", output)
        self.run_command([str(output)])

    def test_kernel_instantiations(self):
        sources = [
            ("v3", "value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_512_"
             "numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu"),
            ("v3_cluster", "value_bf16_outidx_i32_sv0_si0_rv1_maxtopk_1024_"
             "numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu"),
            ("v3_fp32", "value_fp32_outidx_i64_sv1_si0_rv1_maxtopk_512_"
             "numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu"),
        ]
        for variant, filename in sources:
            with self.subTest(variant=variant):
                source = ROOT / "csrc/cuda_kernels" / variant / "instantiations" / filename
                output = self.output_dir / (variant + (".obj" if os.name == "nt" else ".o"))
                self.compile(source, output, compile_only=True)


if __name__ == "__main__":
    unittest.main()
