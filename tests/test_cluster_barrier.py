"""CPU-oracle cluster regression; run on SM100/SM103 with DeepSelect installed.

python tests/test_cluster_barrier.py
compute-sanitizer --tool memcheck --error-exitcode 99 python tests/test_cluster_barrier.py
compute-sanitizer --tool racecheck --error-exitcode 99 python tests/test_cluster_barrier.py
"""

import itertools

import torch
import deep_select
from deep_select import deep_select_cuda

from lib import TestParam


def make_fixture(p: TestParam, scenario: str):
    generator = torch.Generator(device="cpu").manual_seed(2026)
    x = torch.randn((p.batch_size, p.vocab_size), generator=generator,
                    dtype=torch.float32, device="cpu").to(p.dtype)
    lengths = [p.vocab_size, p.vocab_size - 12345] if p.enable_end_position else [p.vocab_size] * p.batch_size
    offsets = torch.tensor([37, 103], dtype=torch.int32, device="cpu") if p.enable_output_idx_offset else None
    if p.enable_end_position:
        for row, length in enumerate(lengths):
            x[row, length:] = float("nan")
    if scenario == "nan":
        # The first permuted segment visited by CTA1, not CTA0's tail or visit interval.
        perm_segs = (p.vocab_size // 512 - 1) // 8 * 8
        cta1_start = (p.vocab_size // 16 - 4096) // 512
        nan_segment = (cta1_start * (0xB559EB75 % perm_segs) + 0x22262226) % perm_segs
        x[0, nan_segment * 512] = float("nan")
    expected = [None if scenario == "nan" and row == 0 else
                torch.topk(x[row, :length].float(), p.topk).values
                for row, length in enumerate(lengths)]
    end = torch.tensor(lengths, dtype=torch.int32, device="cpu") if p.enable_end_position else None
    return x, end, offsets, lengths, expected


def check_result(p, scenario, x, offsets, lengths, expected, values, indices):
    assert indices.shape == (p.batch_size, p.topk)
    assert indices.dtype == p.out_idx_dtype
    assert (values is not None) == p.return_value
    if values is not None:
        assert values.shape == indices.shape and values.dtype == p.dtype
        values = values.cpu()
    indices = indices.cpu()
    for row, length in enumerate(lengths):
        if scenario == "nan" and row == 0:
            assert indices[row, 0].item() == 0x3F3F3F3F
            continue
        selected = indices[row].to(torch.int64)
        if offsets is not None:
            selected -= offsets[row]
        assert bool(((selected >= 0) & (selected < length)).all())
        assert selected.unique().numel() == p.topk
        if p.sorted_index:
            assert bool((selected[1:] > selected[:-1]).all())
        gathered = x[row, selected]
        assert torch.equal(gathered.float().sort(descending=True).values, expected[row])
        if values is not None:
            assert torch.equal(values[row].view(torch.int16), gathered.view(torch.int16))


@torch.inference_mode()
def run_testcase(p: TestParam, scenario: str):
    x, end, offsets, lengths, expected = make_fixture(p, scenario)
    device_x = x.to("cuda")
    device_end = end.to("cuda") if end is not None else None
    device_offsets = offsets.to("cuda") if offsets is not None else None
    for api in ["public", "pybind"]:
        print(f"Running {scenario}, {api}: {p}", flush=True)
        if api == "public":
            values, indices = deep_select.topk(
                device_x, p.topk, end=device_end, indices_type=p.out_idx_dtype,
                sorted_index=p.sorted_index, output_idx_offset=device_offsets,
                return_value=p.return_value, abort_when_nan_found=False,
            )
        else:
            indices = torch.empty((p.batch_size, p.topk), dtype=p.out_idx_dtype, device="cuda")
            values = torch.empty((p.batch_size, p.topk), dtype=p.dtype, device="cuda") if p.return_value else None
            deep_select_cuda.topk(
                device_x, p.topk, None, device_end, False, p.sorted_index,
                values, indices, device_offsets, p.idx_oob_fill_value,
                p.value_oob_fill_value, p.return_value, False,
            )
        torch.cuda.synchronize()
        check_result(p, scenario, x, offsets, lengths, expected, values, indices)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "This regression requires a CUDA device"
    assert torch.cuda.get_device_capability() in [(10, 0), (10, 3)], "Run on SM100/SM103"
    for indices_type, sorted_index, return_value in itertools.product(
        [torch.int32, torch.int64], [False, True], [False, True]
    ):
        for scenario in ["plain", "varlen", "nan"]:
            p = TestParam(
                2, 524288, 512, False, sorted_index, return_value,
                torch.bfloat16, indices_type,
                enable_end_position=scenario == "varlen",
                enable_output_idx_offset=scenario != "plain", num_runs=0,
            )
            run_testcase(p, scenario)
    print("All 48 cluster barrier regression calls passed!", flush=True)
