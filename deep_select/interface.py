import functools
import importlib
import os

import torch

from typing import Optional, Tuple

# Platform probe
_IS_ASCEND = os.path.exists("/dev/davinci_manager")

def _import_backend():
    name = "deep_select_npu" if _IS_ASCEND else "deep_select_cuda"
    try:
        return importlib.import_module(f".{name}", __package__)
    except ImportError as e:
        target = "ASCEND" if _IS_ASCEND else "CUDA"
        raise ImportError(
            f"DeepSelect: failed to load the backend `{__package__}.{name}`. "
            f"Build it with DEEP_SELECT_BUILD_TARGET_PLATFORM={target}."
        ) from e

_backend = _import_backend()


@functools.lru_cache(maxsize=1)
def get_stride_requirement() -> Tuple[int, int]:
    """
    Returns the stride requirement for input / output tensors, in bytes
    """
    return _backend.get_alignment_requirement()


def topk(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    begin: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
    indices_type: torch.dtype = torch.int64,
    sorted_index: bool = False,
    hint: Optional[torch.Tensor] = None,
    output_idx: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Arguments:
        input: (b, vocab_size), dtype=torch.bfloat16/torch.float. Ascend only supports torch.bfloat16. stride(0) must be a multiple of `deep_select.get_stride_requirement()[0]` bytes, and stride(1) must be 1.
        topk: int. Select topk elements for each row.
        sorted: bool. Whether to return sorted **output_val**. Only supports fp32. Ascend only supports False.
        begin(optional): (b,), dtype=int32. CURRENTLY NOT SUPPORTED. The left(inclusive) range for input row, default is 0.
        end(optional): (b,), dtype=int32. The right(exclusive) range for input row, default is vocab_size. The stride of this tensor must be 1.
                       Note when end[i] <= topk, valid elements will be gathered at the beginning of values and indices returned. The rest of `values` will be filled with `value_oob_fill_value`, while the rest of `indices` will be filled with `idx_oob_fill_value` (won't be plused by `output_idx_offset`).
                       `end` <= `vocab_size` must be held
        indices_type: torch.dtype. The output indices dtype, only support torch.int32 and torch.int64. Ascend only supports torch.int32.
        sorted_index: bool. Whether to return sorted **output_idx**.
        hint(optional): CURRENTLY NOT SUPPORTED
        output_idx(optional): (b, topk), dtype=indices_type. A contiguous tensor to store output.
        output_idx_offset(optional): (b,), dtype=int32. If provided, all output_idx (`idx_oob_fill_value` not included) will += output_idx_offset.
        idx_oob_fill_value: int. See comments above when end[i]-begin[i]<topk.
        return_value: bool. If False, only return indices without values to accelerate the kernel. The return value is still a Tuple, but the first element will be None.
        abort_when_nan_found: bool. When a NaN is found, if True, aborts the whole kernel; if False, writes 0x3F3F3F3F into the output. The NaN check itself is always enabled.
                The exact fallback differs by backend: on CUDA only `output_idx[batch_idx][0]` is set, and rows whose length <= topk skip the check entirely;
                on Ascend the whole `output_idx[batch_idx]` row is filled with 0x3F3F3F3F, and short rows are checked as well.

    Return:
        output_val: (b, topk), dtype=input.dtype.
        output_idx: (b, topk), dtype=indices_type.
                    The output tensors may not be contiguous, when topk * sizeof(input.dtype or indices_dtype) is not a multiple of 32 Bytes
    """

    if begin is not None:
        raise ValueError("`begin` is not supported now")
    if hint is not None:
        raise ValueError("`hint` is not supported now")
    if output_idx is not None and output_idx.dtype != indices_type:
        raise TypeError(
            f"`output_idx` dtype ({output_idx.dtype}) must match `indices_type` ({indices_type})"
        )

    N = input.shape[0]

    def get_empty_and_aligned_tensor(dim0: int, dim1: int, device: torch.device, dtype: torch.dtype):
        """
        Return a tensor with shape (dim0, dim1), and stride (X, 1), where X is a multiple of 32B
        """
        output_stride_requirement_bytes = get_stride_requirement()[1]
        output_stride_requirement = output_stride_requirement_bytes // dtype.itemsize
        assert output_stride_requirement > 0
        dim1_rounded = (dim1+output_stride_requirement-1) // output_stride_requirement * output_stride_requirement
        return torch.empty((dim0, dim1_rounded), device=device, dtype=dtype)[:, :dim1]
    
    output_val = get_empty_and_aligned_tensor(N, topk, device=input.device, dtype=input.dtype) if return_value else None
    if output_idx is None:
        output_idx = get_empty_and_aligned_tensor(N, topk, device=input.device, dtype=indices_type)

    backend_args = (
        input,
        topk,
        begin, end,
        sorted, sorted_index,
        output_val, output_idx,
        output_idx_offset,
        idx_oob_fill_value,
        value_oob_fill_value,
        return_value,
        abort_when_nan_found,
    )
    _backend.topk(*backend_args)
    return output_val, output_idx
