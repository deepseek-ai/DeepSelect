import os
import subprocess
from datetime import datetime
from pathlib import Path

from setuptools import setup, find_packages

import tests.kernelkit as kk

exec(open("deep_select/__version__.py").read())

CUDA_SOURCES = [
    "csrc/api.cpp",

    # Generated via `scripts/generate_instantiations.py`

"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv1_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv1_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si0_rv1_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv0_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv0_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv0_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv1_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv1_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i32_sv0_si1_rv1_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv0_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv0_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv0_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv1_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv1_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si0_rv1_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv0_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv0_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv0_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv1_maxtopk_512_numthreads_256_occupancy_2_b_4096_b2_4096_tma_4.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv1_maxtopk_1024_numthreads_256_occupancy_2_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_5.cu",
"csrc/cuda_kernels/v3/instantiations/value_bf16_outidx_i64_sv0_si1_rv1_maxtopk_4096_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",

"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si0_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si0_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si0_rv0_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si0_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si0_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si0_rv1_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si1_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si1_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si1_rv0_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si1_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si1_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv0_si1_rv1_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv1_si0_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv1_si0_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i32_sv1_si0_rv1_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si0_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si0_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si0_rv0_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si0_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si0_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si0_rv1_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si1_rv0_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si1_rv0_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si1_rv0_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si1_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si1_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv0_si1_rv1_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv1_si0_rv1_maxtopk_512_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv1_si0_rv1_maxtopk_1024_numthreads_512_occupancy_1_b_8192_b2_4096_tma_3.cu",
"csrc/cuda_kernels/v3_fp32/instantiations/value_fp32_outidx_i64_sv1_si0_rv1_maxtopk_4096_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu",

"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i32_sv0_si0_rv1_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i32_sv0_si1_rv0_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i32_sv0_si1_rv1_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i64_sv0_si0_rv0_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i64_sv0_si0_rv1_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i64_sv0_si1_rv0_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",
"csrc/cuda_kernels/v3_cluster/instantiations/value_bf16_outidx_i64_sv0_si1_rv1_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_16_cluster_16.cu",

]

def build_on_cuda_platform():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"

    def append_nvcc_threads(nvcc_extra_args):
        nvcc_threads = os.getenv("NVCC_THREADS") or "16"
        return nvcc_extra_args + ["--threads", nvcc_threads]

    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"],
        stderr=subprocess.STDOUT,
    ).decode("utf-8")
    nvcc_version_number = nvcc_version.split("release ")[1].split(",")[0].strip()
    major, minor = map(int, nvcc_version_number.split("."))
    print(f"Compiling using NVCC {major}.{minor}")
    cc_flag = [
        # Hopper (H800/H100) support: kernels have explicit sm90 fallback paths
        "-gencode", "arch=compute_90a,code=sm_90a",
    ]
    if major > 12 or (major == 12 and minor >= 9):
        # sm100/sm103 need CUTLASS headers (cute/atom/copy_traits_sm100.hpp);
        # enable with DEEP_SELECT_BUILD_SM100=1 on Blackwell hosts.
        if os.getenv("DEEP_SELECT_BUILD_SM100", "0") == "1":
            cc_flag += [
                "-gencode", "arch=compute_100a,code=sm_100a",
                "-gencode", "arch=compute_103a,code=sm_103a",
            ]
        else:
            print("DEEP_SELECT_BUILD_SM100=0: building Hopper (sm_90a) only")
    else:
        print("NVCC < 12.9: skipping sm100/sm103 gencode (Hopper-only build)")

    this_dir = os.path.dirname(os.path.abspath(__file__))

    ext_modules = [CUDAExtension(
        name="deep_select.deep_select_cuda",
        sources=CUDA_SOURCES,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations", "-DKERUTILS_IS_BUILD_ON_CUDA", "-DDEEP_SELECT_IS_BUILD_ON_CUDA"],
            "nvcc": append_nvcc_threads([
                "-O3",
                "-std=c++20",
                "-DDEEP_SELECT_IS_BUILD_ON_CUDA",
                # pip-shipped nvcc (e.g. 13.4) is stricter than the bundled CTK
                # headers (CUDART 13.0) in cccl's compat assert; relax it here.
                "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
                "-Wno-deprecated-declarations",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "--ftz=false",  # Don't flush subnormals to zero, since we're going to use float addition to simulate integer addition (in order to be faster)
                "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-double-precision-use",
                "-lineinfo",
                "--source-in-ptx",
            ] + cc_flag)
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "3rdparty" / "cutlass" / "include",
            Path(this_dir) / "csrc" / "3rdparty" / "kerutils" / "include",
            *[p for p in (os.getenv("DEEP_SELECT_CCCL_INCLUDE"),) if p],
            Path(CUDA_HOME) / "targets" / "x86_64-linux" / "include" / "cccl",
            Path(CUDA_HOME) / "targets" / "sbsa-linux" / "include" / "cccl",
        ],
        extra_link_args=[
            f'-L{Path(CUDA_HOME) / "targets" / "x86_64-linux" / "lib" / "stubs"}',
            f'-L{Path(CUDA_HOME) / "targets" / "sbsa-linux" / "lib" / "stubs"}',
            "-lcuda",
        ],
    )]

    class SpillCheckBuildExtension(BuildExtension):
        STACK_BASELINE = 8  # Because of we're using `printf`

        def run(self):
            super().run()
            for ext in self.extensions:
                so_path = self.get_ext_fullpath(ext.name)
                spills = kk.check_kernel_reg_spill_in_artifact(
                    so_path,
                    stack_baseline=self.STACK_BASELINE,
                    suppress_checking_env_var="DEEP_SELECT_DISABLE_REG_SPILL_CHECK",
                )
                if spills:
                    raise RuntimeError("Register spilling detected. Build failed!")

    return (ext_modules, SpillCheckBuildExtension)


def build_on_ascend_platform():
    import torch
    import torch_npu
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    this_dir = os.path.dirname(os.path.abspath(__file__))

    arch = os.environ.get("ASCEND_NPU_ARCH", "dav-3510")
    asc_home = os.environ.get(
        "ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"
    )
    if not os.path.isdir(asc_home):
        raise RuntimeError(
            f"Ascend toolkit not found at '{asc_home}'. "
            "Please set ASCEND_HOME_PATH to the root of your CANN installation "
            "(e.g. /usr/local/Ascend/cann-9.2.0)."
        )

    class AscendBuildExtension(BuildExtension):
        def build_extensions(self):
            self.compiler.src_extensions += [".asc"]
            os.environ["CXX"] = "bisheng"
            os.environ["CC"] = "bisheng"
            super().build_extensions()

    torch_npu_root = Path(torch_npu.__file__).resolve().parent
    ext_modules = [
        CppExtension(
            name="deep_select.deep_select_npu",
            sources=[
                "csrc/api.cpp",
                "csrc/ascend_kernels/kernel.asc",
            ],
            include_dirs=[
                Path(this_dir) / "csrc",
                Path(this_dir) / "csrc/ascend_kernels",
                Path(this_dir) / "csrc" / "3rdparty" / "kerutils" / "include",
                Path(asc_home) / "include",
            ],
            libraries=["ascendcl", "torch_npu"],
            library_dirs=[str(Path(asc_home) / "lib64"), str(torch_npu_root / "lib")],
            extra_compile_args={
                "cxx": [
                    "-O2",
                    "-std=c++20",
                    f"--npu-arch={arch}",
                    "-DDEEP_SELECT_IS_BUILD_ON_ASCEND",
                    "-Wno-deprecated-declarations",
                    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
                    # `kernel_operator.h` lives here; kerutils' common.h detects it to flip
                    # KERUTILS_IS_BUILD_ON_ASCEND on.
                    "-isystem", str(Path(asc_home) / "aarch64-linux" / "asc" / "include"),
                    "-isystem", str(torch_npu_root / "include"),
                    "-isystem", str(torch_npu_root / "include/third_party/acl/inc"),
                    # Disable argument preload at the beginning of kernels to avoid a hardware bug
                    "-mllvm", "-cce-aicore-dcpreload-args=false",
                ],
            },
            extra_link_args=[
                "-Wl,-Bsymbolic-functions",
                # Otherwise the linker complains about "undefined symbol: ReportAscendProf" in CANN 9.2.0
                str(Path(asc_home) / "aarch64-linux" / "lib64" / "libascendc_runtime.a"),
            ],
        )
    ]

    return (ext_modules, AscendBuildExtension)


build_target_platform = kk.get_current_platform()
overrided_platform = os.environ.get('DEEP_SELECT_BUILD_TARGET_PLATFORM', None)
if overrided_platform is not None:
    overrided_platform_dict = {
        'CUDA': kk.Platform.CUDA,
        'ASCEND': kk.Platform.ASCEND,
    }
    if overrided_platform not in overrided_platform_dict:
        raise ValueError(f"Invalid `DEEP_SELECT_BUILD_TARGET_PLATFORM`: {overrided_platform}. Available values are {list(overrided_platform_dict.keys())}")
    build_target_platform = overrided_platform_dict[overrided_platform]
print(f"Build target: {build_target_platform}")

try:
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    git_rev = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode('ascii').rstrip()
except Exception:
    # e.g. building from a source tarball that has no `.git`. Keep the version
    # a valid PEP 440 local version in that case.
    git_rev = "unknown"

datetime_rev = datetime.now().strftime("%Y%m%d.%H%M%S")

if build_target_platform == kk.Platform.CUDA:
    ext_modules, build_ext = build_on_cuda_platform()
elif build_target_platform == kk.Platform.ASCEND:
    ext_modules, build_ext = build_on_ascend_platform()
else:
    raise RuntimeError(f"Unsupported platform: {build_target_platform}.\n(You may use `DEEP_SELECT_BUILD_TARGET_PLATFORM` to explicitly set the target platform for building)")


setup(
    name="deep_select",
    version=f"{__version__}+{git_rev}.{datetime_rev}",
    packages=find_packages(include=['deep_select']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
