import argparse
import dataclasses
import os
import time
from typing import Optional, List
import copy
import sys

import torch
import kernelkit as kk
import random

import deep_select

import lib
from lib import TestParam, Testcase, UniformUIntDistribution, NormalFloatDistribution, UintDistributionWithHotspotAndSpecifiedPivot, get_fp_config

def get_space(t: torch.Tensor):
    return t.numel() * t.element_size()

_counter = kk.Counter()
@torch.inference_mode()
def run_testcase(p: TestParam):
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()

    print(f"Running on {p}")

    t = lib.generate_testcase(p)

    def run_topk_select(testcase: Testcase = t, start_batch_idx: int = 0, end_batch_idx: Optional[int] = None):
        if end_batch_idx is None:
            end_batch_idx = testcase.input.size(0)
        return deep_select.topk(
            testcase.input[start_batch_idx: end_batch_idx],
            p.topk,
            sorted=p.sorted_value,
            begin=None,
            end=testcase.end,
            indices_type=p.out_idx_dtype,
            sorted_index=p.sorted_index,
            hint=None,
            output_idx=None,
            output_idx_offset=testcase.output_idx_offset[start_batch_idx: end_batch_idx] if testcase.output_idx_offset is not None else None,
            idx_oob_fill_value=p.idx_oob_fill_value,
            value_oob_fill_value=p.value_oob_fill_value,
            return_value=p.return_value,
            do_check_nan=p.do_check_nan,
            abort_when_nan_found=False
        )

    ans_topk_value, ans_topk_index = run_topk_select()
    batch_topk_value = ans_topk_value.clone() if ans_topk_value is not None else None
    batch_topk_index = ans_topk_index.clone()
    if p.return_value:
        assert ans_topk_value is not None
    else:
        assert ans_topk_value is None
    assert ans_topk_index.dtype == p.out_idx_dtype
        
    is_correct = True
    if p.check_correctness:
        has_nan_mask = torch.zeros((p.batch_size,), dtype=torch.bool)   # Whether a batch contains NaN
        if t.input.isnan().any().item():
            has_nan_mask = t.input.isnan()
            if t.end is not None:
                lib.row_wise_masked_fill_(has_nan_mask, t.end, False)
            has_nan_mask = has_nan_mask.any(dim=-1) # [batch_size]

        selected_counts = (
            torch.full((p.batch_size,), min(p.vocab_size, p.topk), dtype=torch.int32)
            if t.end is None
            else torch.clamp(t.end, max=p.topk)
        )   # How many numbers are selected for each batch
        selected_mask = torch.arange(p.topk).unsqueeze(0) < selected_counts.unsqueeze(1)    # Whether a position should be a valid output number

        # Assert: index[0] = 0x3F3F3F3F for rows contain NaN
        valid_nan_guard = ans_topk_index[has_nan_mask, 0] == 0x3F3F3F3F
        valid_nan_guard |= p.vocab_size <= p.topk if t.end is None else t.end[has_nan_mask] <= p.topk
        valid_nan_guard |= not p.do_check_nan    
        is_correct &= kk.check_is_bitwise_equal("NaN guard", valid_nan_guard, torch.ones_like(valid_nan_guard))

        selected_mask &= ~has_nan_mask.unsqueeze(1)
        selected_index = ans_topk_index.to(torch.int64)
        if t.output_idx_offset is not None:
            selected_index -= t.output_idx_offset.to(torch.int64).unsqueeze(1)

        # Assert: 0 <= index < valid_len
        valid_len = torch.full((p.batch_size,), p.vocab_size, dtype=torch.int64) if t.end is None else t.end.to(torch.int64)
        index_in_range = (selected_index >= 0) & (selected_index < valid_len.unsqueeze(1))
        valid_index_mask = index_in_range | ~selected_mask
        is_correct &= kk.check_is_bitwise_equal("index range", valid_index_mask, torch.ones_like(valid_index_mask))

        # Assert: index[i] != index[j] (i != j)
        sorted_selected_index = torch.sort(
            torch.where(selected_mask, selected_index, torch.iinfo(torch.int64).max), dim=1
        ).values
        duplicate_mask = sorted_selected_index[:, 1:] == sorted_selected_index[:, :-1]
        duplicate_mask &= selected_mask[:, 1:]
        is_correct &= kk.check_is_bitwise_equal("unique index", duplicate_mask, torch.zeros_like(duplicate_mask))

        safe_selected_index = torch.where(selected_mask & index_in_range, selected_index, 0)
        gathered_value = t.input.gather(1, safe_selected_index)
        valid_row_mask = ~has_nan_mask
        if bool(torch.any(valid_row_mask).item()):
            valid_selected_mask = selected_mask[valid_row_mask]
            valid_gathered_value = gathered_value[valid_row_mask]

            # Assert: value_i = input[index_i]
            if ans_topk_value is not None:
                expected_value = valid_gathered_value.masked_fill(~valid_selected_mask, p.value_oob_fill_value)
                is_correct &= kk.check_is_bitwise_equal(
                    "topk gathered value", ans_topk_value[valid_row_mask], expected_value
                )

            # Assert: min(selected) >= max(unselected)
            selected_min = valid_gathered_value.masked_fill(~valid_selected_mask, float("inf")).amin(dim=1)
            unselected_input = t.input[valid_row_mask].clone()
            unselected_input.scatter_(1, safe_selected_index[valid_row_mask], float("-inf"))
            lib.row_wise_masked_fill_(unselected_input, valid_len[valid_row_mask], float("-inf"))
            topk_condition = selected_min >= unselected_input.amax(dim=1)
            is_correct &= kk.check_is_bitwise_equal("topk condition", topk_condition, torch.ones_like(topk_condition))

        # Assert: index[i] <= index[i+1] if sorted_index
        ordered_mask = selected_mask[:, 1:]
        if p.sorted_index:
            index_ordered = batch_topk_index[:, 1:] >= batch_topk_index[:, :-1]
            index_ordered |= ~ordered_mask
            is_correct &= kk.check_is_bitwise_equal("sorted index", index_ordered, torch.ones_like(index_ordered))

        # Assert: value[i] >= value[i+1] if sorted_value
        if p.sorted_value:
            assert batch_topk_value is not None
            value_ordered = batch_topk_value[:, :-1] >= batch_topk_value[:, 1:]
            value_ordered |= ~ordered_mask
            is_correct &= kk.check_is_bitwise_equal("sorted value", value_ordered, torch.ones_like(value_ordered))

    if p.num_runs > 0:
        total_size = (t.end.sum() if t.end is not None else p.batch_size * p.vocab_size) * t.input.element_size() + (get_space(ans_topk_value) if ans_topk_value is not None else 0) + get_space(ans_topk_index)
        bench_result = kk.bench(run_topk_select, p.num_runs)
        kernel_names = [s for s in bench_result.get_kernel_names() if "topk" in s]
        if len(kernel_names) == 1:
            time_usage = bench_result.get_kernel_time(kernel_names[0])
        else:
            time_usage = bench_result.get_e2e_time(kernel_names)
        print(f"topk           : {time_usage * 1e6:9.3f} us, {total_size / time_usage / 1e12:.3f} TB/s")

        if t.end is None and t.output_idx_offset is None and p.vocab_size >= p.topk:
            def run_torch_topk():
                return torch.topk(t.input, p.topk, dim=1, sorted=p.sorted_value)
            torch_bench_result = kk.bench(run_torch_topk, p.num_runs)
            # torch.topk is a multi-kernel op (`mbtopk`), so measure the span over its kernels.
            torch_kernel_names = [s for s in torch_bench_result.get_kernel_names() if "topk" in s]
            if len(torch_kernel_names) == 1:
                torch_time = torch_bench_result.get_kernel_time(torch_kernel_names[0])
            elif torch_kernel_names:
                torch_time = torch_bench_result.get_e2e_time(torch_kernel_names)
            else:
                torch_time = 0
            if torch_time > 0:
                print(f"torch.topk     : {torch_time * 1e6:9.3f} us, {total_size / torch_time / 1e12:.3f} TB/s  (speedup {torch_time / time_usage:.2f}x)")

    return is_correct

if __name__ == '__main__':
    torch.set_default_device("cuda")

    parser = argparse.ArgumentParser()
    lib.stick_unit_test_args(parser)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default=None,
                        help="Only run testcases whose input dtype matches")
    parser.add_argument("--perf-only", action="store_true",
                        help="Only run performance testcases (num_runs > 0)")
    args = parser.parse_args()

    valid_sv_si_rv_combinations = [ # sv_si_rv: sorted_value, sorted_index, return_value
        (False, False, False),
        (False, False, True),
        (True, False, True),
        (False, True, True),
        (False, True, False)
    ]

    correctness_cases = []
    for dtype in [torch.float, torch.bfloat16]:
        for out_idx_dtype in [torch.int32, torch.int64]:
            for sv, si, rv in valid_sv_si_rv_combinations:
                if sv and dtype == torch.bfloat16:
                    # sorted_value is fp32-only
                    continue
                for b in [random.randint(1, 20), random.randint(1, 100), 121, 512, 4096]:
                    for vocab_size in [
                        1, random.randint(2, 500), 1602, 32768, 123245, 225467, 262144, 418673, 682965, 998123, 1048576, (1 << 23) - 1
                    ]:
                        if b == 4096 and vocab_size > 1048576:
                            continue    # To avoid OOM
                        for topk in [
                            1, random.randint(2, 500), 512, 1024, 1231, 2048, 2132, 2333, 4096
                        ]:
                            enable_end_position = topk%2 == 1   # Use this method instead of enumerating it to save testcase count
                            for distrib in [
                                NormalFloatDistribution(post_proc_func)
                                for post_proc_func in [
                                    None,
                                    lambda x: x + (100 if enable_end_position else -100),
                                    lambda x: x * (100 if enable_end_position else 1/100),
                                    lambda x: x * (1000 if enable_end_position else 1/1000),
                                    lambda x: x * (10000 if enable_end_position else 1/10000),
                                ]
                            ] + [
                                UniformUIntDistribution(
                                    get_fp_config(dtype).negative_0_as_int if enable_end_position else get_fp_config(dtype).positive_0_as_int,
                                    get_fp_config(dtype).negative_inf_as_int if enable_end_position else get_fp_config(dtype).positive_inf_as_int
                                ),
                                UniformUIntDistribution(0x0, 0x1),
                                UniformUIntDistribution(0x0, 0x20),
                                UniformUIntDistribution(0x0, 0x1000),
                            ] + [
                                UintDistributionWithHotspotAndSpecifiedPivot(None, topk, [(get_fp_config(dtype).positive_nan, min(8, vocab_size))], True),
                                UintDistributionWithHotspotAndSpecifiedPivot(None, topk, [(get_fp_config(dtype).negative_nan, min(8, vocab_size))], True),
                            ]:
                                enable_output_idx_offset = b%2 == 1
                                cur_case = TestParam(b, vocab_size, topk, sv, si, rv, dtype, out_idx_dtype, enable_end_position, enable_output_idx_offset, num_runs=0, idx_oob_fill_value=-2000000+vocab_size, input_distrib=distrib)
                                correctness_cases.append(cur_case)

    performance_cases = [
        # Lightning Indexer
        TestParam(b, compressed_seqlen, topk, False, False, False, torch.bfloat16, torch.int32, num_runs=10)
        for topk in [512, 1024]
        for b in [
            6,      # RL rollout
            256,    # Decoding
            512,
            768,
            4096    # Prefill
        ]
        for compressed_seqlen in [256, 1024, 4096, 16384, 65536, 131072, 262144, 524288, 1048576]
    ] + [
        # Sampler
        TestParam(b, vocab_size, 512, True, False, True, torch.float, torch.int64, num_runs=10)
        for b in [6, 256, 512, 768, 4096]
        for vocab_size in [129280]
    ]

    testcases = correctness_cases + performance_cases

    if args.dtype is not None:
        wanted_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
        testcases = [t for t in testcases if t.dtype == wanted_dtype]
    if args.perf_only:
        testcases = [t for t in testcases if t.num_runs > 0]

    # Use the following testcases to compare the kernel performance under different batch size & sequence lengths
    # testcases = [
    #     TestParam(b, l, topk, False, si, False, torch.bfloat16, torch.int32, num_runs=10, input_distrib=NormalFloatDistribution(lambda f: f*1000))
    #     for topk in [1024]
    #     for b in [6, 128, 256, 384, 512, 768, 1024, 4096]
    #     for l in [2048, 4096, 6144] + list(range(8192, 262144+1, 8192))
    #     for si in [True]
    # ]

    failed_cases = []
    for test_idx, test in enumerate(testcases):
        if test != testcases[0] and test.num_runs > 0 and not args.no_cooldown:
            time.sleep(0.2)
        print("================")
        print(f"[{test_idx+1:6d}/{len(testcases):6d}, {test_idx/len(testcases)*100:2.0f}%] ", end="")
        is_correct = run_testcase(test)
        if not is_correct:
            failed_cases.append(test)
            if not args.run_to_finish:
                sys.exit(1)
    
    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {len(testcases)} cases failed:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
        sys.exit(1)
    else:
        print(f"\033[32m\033[1mAll {len(testcases)} cases passed!\033[0m")
