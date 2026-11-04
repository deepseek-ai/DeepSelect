"""Finite DeepSelect boundary checks against a CPU torch.topk reference."""

import torch

import deep_select


def check(name, batch, width, k, dtype, index_dtype, *, sorted_value=False,
          sorted_index=False, return_value=True, lengths=None, offsets=None, ties=False):
    generator = torch.Generator(device="cpu").manual_seed(412)
    if ties:
        host = torch.randint(-16, 17, (batch, width), generator=generator).to(dtype)
    else:
        host = torch.randn((batch, width), generator=generator).to(dtype)
    x = host.to("cuda")
    end = None if lengths is None else torch.tensor(lengths, dtype=torch.int32, device="cuda")
    offset = None if offsets is None else torch.tensor(offsets, dtype=torch.int32, device="cuda")
    value, index = deep_select.topk(
        x, k, sorted=sorted_value, sorted_index=sorted_index, return_value=return_value,
        indices_type=index_dtype, end=end, output_idx_offset=offset,
        idx_oob_fill_value=-1234567, value_oob_fill_value=float("-inf"),
    )
    torch.cuda.synchronize()
    index = index.cpu().to(torch.int64)
    value = None if value is None else value.cpu()
    assert (value is not None) == return_value, name
    for row in range(batch):
        length = width if lengths is None else lengths[row]
        valid = min(k, length)
        offset_value = 0 if offsets is None else offsets[row]
        selected = index[row, :valid] - offset_value
        assert torch.all((0 <= selected) & (selected < length)).item(), name
        assert selected.unique().numel() == valid, name
        assert torch.all(index[row, valid:] == -1234567).item(), name
        if sorted_index and valid > 1:
            assert torch.all(selected[1:] >= selected[:-1]).item(), name
        picked = host[row].gather(0, selected)
        if value is not None:
            assert torch.equal(value[row, :valid], picked), name
            assert torch.isneginf(value[row, valid:]).all().item(), name
            if sorted_value and valid > 1:
                assert torch.all(value[row, 1:valid] <= value[row, :valid-1]).item(), name
        if valid:
            expected = torch.topk(host[row, :length], valid, sorted=True).values
            assert torch.equal(torch.sort(picked, descending=True).values, expected), name
    print(f"{name}: PASS", flush=True)


torch.set_num_threads(1)
check("fp32-k1-ties", 2, 4096, 1, torch.float32, torch.int32, sorted_value=True, ties=True)
check("fp32-k511-short", 3, 4096, 511, torch.float32, torch.int64,
      sorted_value=True, lengths=[0, 510, 512], offsets=[-17, 0, 21])
check("fp32-k513-ties", 2, 4096, 513, torch.float32, torch.int32, sorted_index=True, ties=True)
check("fp32-k1025", 2, 8192, 1025, torch.float32, torch.int64, sorted_value=True)
check("fp32-k4095", 2, 8192, 4095, torch.float32, torch.int32, lengths=[4094, 8192])
check("fp32-k4096", 1, 8192, 4096, torch.float32, torch.int64, sorted_index=True)
check("bf16-k1-short", 3, 4096, 1, torch.bfloat16, torch.int32,
      sorted_index=True, lengths=[0, 1, 2], offsets=[0, -3, 7])
check("bf16-k511", 2, 4096, 511, torch.bfloat16, torch.int64, sorted_index=True, ties=True)
check("bf16-k513-index-only", 2, 4096, 513, torch.bfloat16, torch.int32,
      sorted_index=True, return_value=False, offsets=[-19, 31])
check("bf16-k1025-short", 2, 8192, 1025, torch.bfloat16, torch.int64, lengths=[1024, 1026])
check("bf16-k4095", 1, 8192, 4095, torch.bfloat16, torch.int32, sorted_index=True)
check("bf16-k4096", 1, 8192, 4096, torch.bfloat16, torch.int64)
