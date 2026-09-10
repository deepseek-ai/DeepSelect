import argparse
import dataclasses
import math
import random
from typing import Optional, Callable, List, Tuple
import abc

import torch
import kernelkit as kk
import deep_select

@dataclasses.dataclass
class FPConfig:
    positive_0_as_int: int
    negative_0_as_int: int
    positive_inf_as_int: int
    negative_inf_as_int: int
    positive_nan: int
    negative_nan: int
    smallest_positive_norm: int
    full_bitmask: int

def get_fp_config(dtype: torch.dtype) -> FPConfig:
    if dtype == torch.bfloat16:
        return FPConfig(
            0x0000, 0x8000,
            0x7F80, 0xFF80,
            0x7F81, 0xFF81,
            0x0100,
            0xFFFF
        )
    elif dtype == torch.float:
        return FPConfig(
            0x00000000, 0x80000000,
            0x7F800000, 0xFF800000,
            0x7F800001, 0xFF800001,
            0x00800000,
            0xFFFFFFFF
        )
    else:
        assert False, f"Unknown dtype: {dtype}"


class Distribution(abc.ABC):
    """
    Distribution - The base class for all distributions
    """

    def __init__(self):
        pass

    def dtype2uint_dtype(self, dtype: torch.dtype):
        """
        Convert a dtype to its corresponding (equal-width) integer dtype
        """
        return {
            torch.bfloat16: torch.uint16,
            torch.float: torch.uint32
        }[dtype]
        
    def generate(self, result: torch.Tensor):
        """
        Generate the `input` for `deep_select.topk`, according to a specific distribution

        `result` is a tensor of shape (batch_size, vocab_size).
        """
        raise NotImplementedError()


class NormalFloatDistribution(Distribution):
    """
    A uniform distribution (in terms of floating datatype)
    """

    def __init__(self, post_proc_func: Optional[Callable] = None):
        self.post_proc_func = post_proc_func

    def __repr__(self):
        if self.post_proc_func is None:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}(post_proc_func={self.post_proc_func.__name__})"

    def generate(self, result: torch.Tensor):
        data = torch.randn_like(result, dtype=result.dtype)
        if self.post_proc_func is not None:
            data = self.post_proc_func(data)
        result.copy_(data)


class UniformUIntDistribution(Distribution):
    """
    A uniform distribution (in terms of byte representation)
    """

    def __init__(self, lower: int, upper: int):
        assert lower >= 0 and upper >= 0
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        return f"{self.__class__.__name__}(lower=0x{self.lower:08x}, upper=0x{self.upper:08x})"

    def generate(self, result: torch.Tensor):
        uint_dtype = self.dtype2uint_dtype(result.dtype)
        data = torch.randint(self.lower, self.upper, result.shape, dtype=torch.long).to(uint_dtype)
        result.view(uint_dtype).copy_(data)


class UintDistributionWithHotspotAndSpecifiedPivot(Distribution):
    """
    A distribution that:
    - Is able to specify hotspot ((number, frequency), which specifies numbers and their frequency)
    - Is able to specify a particular k-th value
    """

    def __init__(self, pivot: int | None, topk: int, hotspots: list[tuple[int, int]], allow_nan: bool):
        self.pivot = pivot
        self.topk = topk
        self.hotspots = hotspots
        self.allow_nan = allow_nan
    
    def __repr__(self):
        return f"{self.__class__.__name__}(pivot={f'0x{self.pivot:08x}' if self.pivot is not None else 'None'}, topk={self.topk}, hotspots={self.hotspots}, allow_nan={self.allow_nan})"
    
    def generate(self, result: torch.Tensor):
        fp_cfg = get_fp_config(result.dtype)
        uint_dtype = self.dtype2uint_dtype(result.dtype)
        signed_dtype = {
            torch.uint16: torch.int16,
            torch.uint32: torch.int32,
        }[uint_dtype]
        result_as_signed = result.view(signed_dtype)
        batch_size, vocab_size = result.shape

        indices_perm = torch.sort(torch.randn(batch_size, vocab_size, dtype=torch.float), dim=-1).indices   # A batched randperm of shape (batch_size, vocab_size)

        acc_freq = 0
        def put_values(freq: int, value: torch.Tensor | int):
            """
            Put `value` (can be a `torch.Tensor` or `int`) to `result_as_signed[:, indices[:, acc_freq: acc_freq + freq]]`
            """
            if freq == 0:
                return
            assert freq > 0
            nonlocal acc_freq
            cur_indices = indices_perm[:, acc_freq: acc_freq + freq]
            if isinstance(value, int):
                value = torch.full_like(cur_indices, value).to(signed_dtype)
            else:
                value = value.view(signed_dtype)
            result_as_signed.scatter_(dim=-1, index=cur_indices, src=value)
            acc_freq += freq

        # Put `hotspots`
        for (value, freq) in self.hotspots:
            put_values(freq, value)

        if self.pivot is not None:
            # Put values smaller / larger than pivot
            larger_values_bound = [-1, fp_cfg.positive_nan if self.allow_nan else fp_cfg.positive_inf_as_int]  # Inclusive
            smaller_values_bound = [-1, fp_cfg.negative_nan if self.allow_nan else fp_cfg.negative_inf_as_int] # Inclusive
            if (self.pivot & fp_cfg.negative_0_as_int) == 0:
                # pivot >= 0
                larger_values_bound[0] = self.pivot
                smaller_values_bound[0] = fp_cfg.negative_0_as_int
            else:
                # pivot < 0
                larger_values_bound[0] = fp_cfg.positive_0_as_int
                smaller_values_bound[0] = self.pivot
                
            num_pivots = min(random.randint(1, int(self.topk*1.1)), vocab_size - acc_freq)
            num_larger_values = min(max(self.topk - num_pivots, 0), vocab_size - acc_freq - num_pivots)
            num_smaller_values = vocab_size - acc_freq - num_pivots - num_larger_values
            assert num_larger_values >= 0
            assert num_smaller_values >= 0
            put_values(num_larger_values, torch.randint(larger_values_bound[0], larger_values_bound[1]+1, (batch_size, num_larger_values), dtype=torch.long).to(uint_dtype))
            put_values(num_smaller_values, torch.randint(smaller_values_bound[0], smaller_values_bound[1]+1, (batch_size, num_smaller_values), dtype=torch.long).to(uint_dtype))
            put_values(num_pivots, self.pivot)

        else:
            num_remaining_elems = vocab_size - acc_freq
            num_positive_values = random.randint(0, num_remaining_elems)
            num_negative_values = num_remaining_elems - num_positive_values
            put_values(num_positive_values, torch.randint(fp_cfg.positive_0_as_int, fp_cfg.positive_inf_as_int+1+(1 if self.allow_nan else 0), (batch_size, num_positive_values), dtype=torch.long).to(uint_dtype))
            put_values(num_negative_values, torch.randint(fp_cfg.negative_0_as_int, fp_cfg.negative_inf_as_int+1+(1 if self.allow_nan else 0), (batch_size, num_negative_values), dtype=torch.long).to(uint_dtype))

        assert acc_freq == vocab_size


@dataclasses.dataclass
class TestParam:
    batch_size: int
    vocab_size: int
    topk: int
    sorted_value: bool
    sorted_index: bool
    return_value: bool
    dtype: torch.dtype
    out_idx_dtype: torch.dtype
    enable_end_position: bool = False
    enable_output_idx_offset: bool = False
    input_distrib: Distribution = NormalFloatDistribution()
    idx_oob_fill_value: int = -2147483647
    value_oob_fill_value: float = -1234123412341234
    do_check_nan: bool = True

    seed: int = -1
    check_correctness: bool = True
    num_runs: int = 10

@dataclasses.dataclass
class Testcase:
    input: torch.Tensor
    end: Optional[torch.Tensor]
    output_idx_offset: Optional[torch.Tensor]

def generate_testcase(p: TestParam):
    kk.set_random_seed(p.seed)

    stride_alignment = deep_select.get_stride_requirement()[0] // p.dtype.itemsize
    vocab_size_rounded = int(math.ceil(p.vocab_size / stride_alignment)) * stride_alignment
    input = torch.empty((p.batch_size, vocab_size_rounded), dtype=p.dtype)
    input = input[:, :p.vocab_size]
    p.input_distrib.generate(input)
    end = torch.randint(0, p.vocab_size, (p.batch_size, ), dtype=torch.int) if p.enable_end_position else None
    output_idx_offset = torch.randint(-2**30, 2**30, (p.batch_size, ), dtype=torch.int) if p.enable_output_idx_offset else None

    return Testcase(input, end, output_idx_offset)

def stick_unit_test_args(parser: argparse.ArgumentParser):
    parser.add_argument("-nc", "--no-cooldown", action="store_true", help="Don't call time.sleep() before performance testcases")
    parser.add_argument("-rf", "--run-to-finish", action="store_true", help="Don't exit when a testcase is failed")

def row_wise_masked_fill_(tensor: torch.Tensor, bound: Optional[torch.Tensor], value):
    """
    tensor[:, bound:] = value
    """
    if bound is None:
        return
    mask = torch.arange(0, tensor.shape[1]).unsqueeze(0).broadcast_to(tensor.shape) >= bound.unsqueeze(-1).broadcast_to(tensor.shape)
    tensor[mask] = value
