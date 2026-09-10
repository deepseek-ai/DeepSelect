# Algorithm Design Background

In DeepSeek Sparse Attention (DSA), TopK is used to select the most relevant positions from a large number of context tokens. During sampling, TopK is also used to select candidate tokens from a large vocabulary. To improve end-to-end inference performance, we designed a new TopK algorithm.

# Algorithm

The pseudocode below captures the core flow of the algorithm. The actual implementation requires hardware-specific optimization.

The algorithm keeps a top-$k$ threshold $T$, initialized to $-\infty$, and repeatedly performs three simple steps:

1. **Scan**: Process the input one block of size $B$ at a time, following a random block order.
2. **Filter**: Filter elements using the current top-$k$ threshold $T$, and append only those that may still enter the top-$k$ to `topk_candidate` (i.e. $> T$).
3. **Compact**: When the candidate buffer becomes large or at the end of the algorithm, run radix-select-based TopK in shared memory (or another TopK algorithm) to reduce it back to $k$ elements, and set $T$ to the smallest element among them.

In the pseudocode below, $B$ denotes the input block size, while $B_2$ controls when `topk_candidate` is compacted. A reasonable configuration for $k=512$ is $B=B_2=1024$.

```text
DeepSelectTopk(x[0:N), k, B, B2):
    require 1 <= k <= N and B >= 1 and B2 >= 1

    topk_candidate = []                     # shared memory
    topk_threshold = -inf                   # current top-k threshold
    p = random_permutation(ceil_div(N, B))  # independent of x

    for block in p:
        lo, hi = block * B, min((block + 1) * B, N)

        # Keep only elements that may still enter the top-k.
        for j in range(lo, hi):
            if x[j] > topk_threshold:
                topk_candidate.append((x[j], j))

        # Periodically compact the candidate buffer.
        if len(topk_candidate) >= k + B2:
            topk_candidate = RadixSelectTopK(topk_candidate, k)
            topk_threshold = min(v for v, _ in topk_candidate)

    return RadixSelectTopK(topk_candidate, k)
```

Initially, the threshold is `-inf`, so all elements are accepted before the first compaction. Afterwards, `topk_threshold` is the current $k$-th largest value and never decreases as the scan proceeds. Thus, later elements tend to be less likely to pass the threshold, causing the candidate buffer to grow increasingly slowly.

To avoid performance degradation on unfavorable inputs, blocks are processed in a random order. Below, we establish an expected performance bound under a uniform random block permutation.

Notes:

- At any time, `len(topk_candidate) <= k + B2 + B` holds
- Every element of $x$ is read exactly once, with accesses performed in contiguous blocks of size $B$.
- Excluding the random block permutation, the algorithm requires only $O(k+B+B_2)$ additional space, which can reside in a smaller and faster memory space such as shared memory.

# Expected Upper Bound on the Total Elements Processed by RadixSelectTopK

With a fixed input, the only randomness in the following analysis comes from a uniform random block permutation independent of the input.

For simplicity, assume that all elements are distinct. Equal-valued elements do not worsen the performance bound and are omitted from the discussion.

Let $\mathcal B_i$ denote the $i$-th block in the random processing order, $C_i$ the contents of `topk_candidate` at the start of processing $\mathcal B_i$, $A_i$ the number of elements appended while processing $\mathcal B_i$, and $W$ the sum of the input sizes of all `TopK` calls, including the final one. Define

$$
m=\left\lceil \frac NB\right\rceil,\qquad
L=k+B+B_2,\qquad
H_m=\sum_{i=1}^m\frac1i.
$$

Since $|C_i|<k+B_2$, and every previously processed element outside $C_i$ is no larger than the current threshold, any element appended from $\mathcal B_i$ has rank at most $|C_i|+|\mathcal B_i|\le |C_i|+B<L$ among the elements in the first $i$ processed blocks.

Now consider the set of the first $i$ processed blocks, $U_i=\{\mathcal B_1,\ldots,\mathcal B_i\}.$

Conditioned on $U_i$, the block $\mathcal B_i$ is uniformly distributed over the $i$ blocks in $U_i$. Let $T_i$ be the top $L$ elements contained in $U_i$, or all elements if $U_i$ contains fewer than $L$. By the rank bound above, every element appended from $\mathcal B_i$ belongs to $T_i$. Therefore,

```math
\mathbb E[A_i\mid U_i]
\le \frac1i\sum_{b\in U_i}|b\cap T_i|
= \frac{|T_i|}{i}
\le \frac Li.
```

Let $A=\sum_i A_i$. By linearity of expectation, $\mathbb E[A]\le LH_m.$

Let $R$ be the number of in-loop `TopK` calls. Each such call removes at least $B_2$ elements, so $RB_2\le A.$

Across all `TopK` calls, each appended element contributes once to an input, while each in-loop call retains $k$ elements that are carried over into a later call. Hence, $W=A+kR.$

Therefore,

```math
\boxed{
\mathbb E[W]
\le \left(1+\frac{k}{B_2}\right)LH_m
= \left(1+\frac{k}{B_2}\right)L\bigl(\ln m+O(1)\bigr).
}
```

# Performance Summary

Each input element is read from global memory exactly once, with accesses performed in contiguous blocks. The remaining computation is performed mainly on a candidate set of size $O(k+B+B_2)$. Under a random block order, the expected total number of elements processed by all TopK calls is

$$
O\left(\left(1+\frac{k}{B_2}\right)(k+B+B_2)\log\frac NB\right).
$$

When $B,B_2=\Theta(k)$, the additional computation required by the TopK stage is only $O(k\log(N/k))$, which is much smaller than the input size $N$. With careful hardware-specific optimization, the algorithm can achieve a substantial speedup over `torch.topk` in DSA workloads.

# Implementation

We implement this DeepSelect kernel in CUDA C++ and make extensive use of low-level PTX optimization techniques and bit manipulation techniques to reduce constant factors, lower runtime, and improve GPU bandwidth utilization.

Normally, in the Filter step, each CUDA thread needs to scan the input twice: the first scan counts how many of the numbers read by each thread are greater than the threshold, so that each thread knows where to write its results in the `topk_candidate` array; the second scan goes through each element again and places elements greater than the threshold in the correct positions in `topk_candidate`. Even if shared memory is used here to avoid repeatedly reading global memory, the overhead of CUDA Core operations still limits the performance of the entire kernel. To address this, while counting the number of elements greater than the threshold in the first scan, we generate a `uint32_t` mask whose `i`-th bit records whether the `i`-th element is greater than the threshold. When placing elements later, we can use [`__ffs(mask)`](../csrc/cuda_kernels/common_parts.cuh#L1466) to get the index of the lowest set bit in `mask`, and then clear the lowest set bit with `mask &= mask - 1u`, thereby quickly finding all elements greater than the threshold. In addition, we use low-level PTX instructions such as `set.s32.bf16x2`, `prmt.b32`, and `dp4a.s32.s32` to accelerate [`mask`](../csrc/cuda_kernels/common_parts.cuh#L1431) generation.

In addition, there are a very large number of comparison and bitwise instructions in this kernel. Based on our observations, such instructions contend with integer addition instructions for some hardware resources, reducing kernel performance. Even if integer additions are replaced with integer multiply-add (`mad`) instructions that do not conflict with comparison and bitwise instructions, their throughput is only half that of floating-point multiply-add, making it a performance bottleneck. We observed that when `x` and `y` satisfy `0 <= x, y <= 2^22`, `x+y = __float_as_uint(__uint_as_float(x) + __uint_as_float(y))` holds; that is, the result of adding two non-negative integers no greater than $2^{22}$ is the same as treating their binary representations as `float32` (in which case both should be denormal numbers) and then performing floating-point addition. Therefore, we can [use floating-point addition to replace some integer additions](../csrc/cuda_kernels/common_parts.cuh#L1051) to improve throughput.

In addition to the above two points, we also use low-level PTX instructions and bitwise operations to accelerate other operations, including [warp-level prefix sum](../csrc/cuda_kernels/common_parts.cuh#L1478), [byte extraction](../csrc/cuda_kernels/common_parts.cuh#L955), [histogram construction](../csrc/cuda_kernels/common_parts.cuh#L940), etc. See the code for details.

For scenarios with a small batch size (or scenarios where the number of active SMs in the last wave is small due to wave quantization), we use Threadblock Cluster to split the computation of a sequence across multiple Streaming Multiprocessors (SMs). Specifically, each sequence is divided into several parts. For each part, a CTA (CUDA Threadblock) first computes the top-k largest elements in that part. Then, each CTA sends these top-k elements to CTA0 in its Threadblock Cluster, and CTA0 selects the top k largest elements among them.
