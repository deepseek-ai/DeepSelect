"""
Generate explicit template instantiation .cu files for a topk kernel template

Usage:
    Run from the root directory:
        python3 scripts/generate_instantiations.py <relative/path/to/the/instantiations/directory>

    For example,
        python3 scripts/generate_instantiations.py csrc/cuda_kernels/v3/instantiations
        python3 scripts/generate_instantiations.py csrc/cuda_kernels/v3_fp32/instantiations
        python3 scripts/generate_instantiations.py csrc/cuda_kernels/v3_cluster/instantiations

    The script will:
    1. Create .cu files in the target directory, one per config
    2. Print the paths (relative to repo root) for inclusion in setup.py's
       `sources` list

    After running, manually copy the printed paths into `setup.py`.
"""

import dataclasses
import os
import shutil
import sys
from typing import List

@dataclasses.dataclass
class TopkSelectConfigs:
    ValueT: str
    OutIdxT: str
    sorted_value: bool
    sorted_index: bool
    return_value: bool
    max_topk: int
    num_threads: int
    target_occupancy: int
    elements_per_round: int
    reconstruct_threshold: int
    tma_buffer_depth: int
    cluster: int = 1

    def check_validity(self):
        assert self.ValueT in ["nv_bfloat16", "float"], "Invalid `ValueT`"
        assert self.OutIdxT in ["int32_t", "int64_t"], "Invalid `OutIdxT`"
        if self.sorted_value:
            assert self.return_value, "`return_value` must be `True` for `sorted_value`"
            assert not self.sorted_index, "`sorted_value` and `sorted_index` cannot be specified at the same time"
        if self.ValueT == "nv_bfloat16":
            assert not self.sorted_value, "`sorted_value` is fp32-only"
        assert self.max_topk in [512, 1024, 4096]
        assert self.elements_per_round == self.num_threads * 16, "contract ABI: B = NUM_THREADS * 16"
        assert self.reconstruct_threshold >= self.max_topk, "RECONSTRUCT_THRESHOLD >= MAX_TOPK"
        assert 1 <= self.cluster <= 16, "Invalid `cluster`"


def generate_instantiation_file(instantiation_dir: str, namespace: str, config: TopkSelectConfigs) -> str:
    file_content = \
f"""#include "../topk_select.cuh"

namespace {namespace} {{

template
void run_topk_select_kernel<
    TopkSelectConfig<{config.ValueT}, {config.OutIdxT}, {str(config.sorted_value).lower()}, {str(config.sorted_index).lower()}, {str(config.return_value).lower()}, {config.max_topk}, {config.num_threads}, {config.target_occupancy}, {config.elements_per_round}, {config.reconstruct_threshold}, {config.tma_buffer_depth}, 512, {config.cluster}>
>(const TopkSelectArgs &args);

}}   // {namespace}
"""
    value_abbrev = {"nv_bfloat16": "bf16", "float": "fp32"}[config.ValueT]
    outidx_abbrev = {"int32_t": "i32", "int64_t": "i64"}[config.OutIdxT]
    file_name = f"value_{value_abbrev}_outidx_{outidx_abbrev}_sv{int(config.sorted_value)}_si{int(config.sorted_index)}_rv{int(config.return_value)}_maxtopk_{config.max_topk}_numthreads_{config.num_threads}_occupancy_{config.target_occupancy}"
    file_name += f"_b_{config.elements_per_round}_b2_{config.reconstruct_threshold}_tma_{config.tma_buffer_depth}"
    if config.cluster != 1:
        file_name += f"_cluster_{config.cluster}"
    file_name += ".cu"
    file_path = os.path.join(instantiation_dir, file_name)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(file_content)
    
    return file_path


def generate_instantiations(instantiation_dir: str, namespace: str, configs: List[TopkSelectConfigs]):
    for config in configs:
        config.check_validity()
    setup_py_source_files = []
    for config in configs:
        setup_py_source_files.append(generate_instantiation_file(instantiation_dir, namespace, config))
    for t in setup_py_source_files:
        print(f"\"{t}\",")

def main(instantiation_dir: str):
    def remove_and_remake_dir():
        if os.path.exists(instantiation_dir):
            shutil.rmtree(instantiation_dir)
        os.makedirs(instantiation_dir, exist_ok=True)
    if instantiation_dir == "csrc/cuda_kernels/v3/instantiations":
        remove_and_remake_dir()
        # bf16: sv0 x si{0,1} x rv{0,1}
        valid_si_rv_combinations = [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ]
        # TMA4 costs 109.6 KB of smem per CTA, which fits twice per SM only for max_topk <= 512.
        fast_path_tuples_by_max_topk = {
            512:  [(256, 2, 4096, 4096, 4),     # flagship (num_waves >= 2)
                   (512, 1, 8192, 4096, 5)],    # wave1
            1024: [(256, 2, 4096, 4096, 3),     # flagship (num_waves >= 2)
                   (512, 1, 8192, 4096, 5)],    # wave1
        }
        # Single coverage tuple for topk in (1024, 4096]: correctness-only
        big_topk_tuple = (512, 1, 8192, 4096, 3)
        configs = []
        for out_idx_t in ["int32_t", "int64_t"]:
            for si, rv in valid_si_rv_combinations:
                for max_topk, fast_path_tuples in fast_path_tuples_by_max_topk.items():
                    for num_threads, occ, b, b2, tma in fast_path_tuples:
                        configs.append(TopkSelectConfigs("nv_bfloat16", out_idx_t, False, si, rv, max_topk, num_threads, occ, b, b2, tma))
                num_threads, occ, b, b2, tma = big_topk_tuple
                configs.append(TopkSelectConfigs("nv_bfloat16", out_idx_t, False, si, rv, 4096, num_threads, occ, b, b2, tma))
        generate_instantiations(instantiation_dir, "topk_select_bf16_normal", configs)
    elif instantiation_dir == "csrc/cuda_kernels/v3_fp32/instantiations":
        remove_and_remake_dir()
        # fp32: sv0 x si{0,1} x rv{0,1} + sv1_si0_rv1
        valid_sv_si_rv_combinations_fp32 = [
            (False, False, False),
            (False, False, True),
            (False, True, False),
            (False, True, True),
            (True, False, True),
        ]
        tuple_by_max_topk = {
            512: (512, 1, 8192, 4096, 3),
            1024: (512, 1, 8192, 4096, 3),
            4096: (256, 1, 4096, 4096, 3),
        }
        configs = []
        for out_idx_t in ["int32_t", "int64_t"]:
            for sv, si, rv in valid_sv_si_rv_combinations_fp32:
                for max_topk, (num_threads, occ, b, b2, tma) in tuple_by_max_topk.items():
                    configs.append(TopkSelectConfigs("float", out_idx_t, sv, si, rv, max_topk, num_threads, occ, b, b2, tma))
        generate_instantiations(instantiation_dir, "topk_select_fp32", configs)
    elif instantiation_dir == "csrc/cuda_kernels/v3_cluster/instantiations":
        remove_and_remake_dir()
        # The cluster tier is bf16 + mk1024 only and always uses a 16-CTA cluster.
        configs = []
        for out_idx_t in ["int32_t", "int64_t"]:
            for si in [False, True]:
                for rv in [False, True]:
                    configs.append(TopkSelectConfigs("nv_bfloat16", out_idx_t, False, si, rv, 1024, 256, 1, 4096, 4096, 16, 16))
        generate_instantiations(instantiation_dir, "topk_select_bf16_cluster", configs)
    else:
        raise ValueError(f"Invalid `instantiation_dir: {instantiation_dir}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <instantiation_dir>")
        sys.exit(1)
    instantiation_dir = sys.argv[1]
    main(instantiation_dir)
