# 算法设计背景

在 DeepSeek Sparse Attention（DSA）中，需要使用 TopK 算子从大量 context token 中选出最相关的位置。在采样阶段，也需要使用 TopK 从较大的词表中选出候选 token。为了提升模型推理的端到端性能，我们设计了一种新的 TopK 算法。

# 算法

下面的伪代码描述了算法的核心流程。实际实现还需要针对具体硬件进行优化。

算法维护一个初值为 $-\infty$ 的 top-$`k`$ 阈值 $T$，并循环执行三个简单步骤：

1. **扫描（Scan）**：按照随机顺序，每次扫描一个输入序列中大小为 $B$ 的块。
2. **筛选（Filter）**：根据当前的 top-$`k`$ 阈值 $T$ 进行筛选，只将仍有可能进入 top-$`k`$ 的元素（即大于 $T$ 的元素）加入 `topk_candidate`。
3. **压缩（Compact）**：当候选缓冲区变大或算法结束时，在 shared memory 中执行基于 radix-select 的 TopK（或其他 TopK 算法），将其重新精简到 $k$ 个元素，并将 $T$ 更新为其中最小元素的值。

下方伪代码中，$B$ 表示输入块大小，$B_2$ 用于控制 `topk_candidate` 触发压缩时的大小。对于 $k=512$，一组较为合理的参数配置是 $B=B_2=1024$。

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

初始时，阈值为 `-inf`，因此在第一次压缩之前，所有元素都会被接受。此后，`topk_threshold` 表示当前第 $k$ 大元素的值。随着不断扫描，`topk_threshold` 只会单调增大。因此大体上来说，后续元素通过阈值筛选的概率会逐渐降低，候选缓冲区的增长速度也会越来越慢。

为了避免极端输入下的性能退化，块处理顺序必须是随机的。我们在下方也补充了在随机块顺序下的性能保证。

Notes:

- 任意时候，我们都有：`len(topk_candidate) <= k + B2 + B`。
- $x$ 中的每个元素都恰好读取一次，并且以大小为 $B$ 的连续块进行访问。
- 除去随机排列，算法所需要的额外空间为 $O(k+B+B_2)$，因此可以存放在容量更小但速度更快的存储空间（例如 shared memory）中。

# RadixSelectTopK 总处理元素数的期望上界

固定输入，下方分析中的随机性仅来自一个与输入独立的均匀随机块排列。

在证明中，不妨设所有元素互不相同（存在相同元素时，算法不会产生性能下降，这里不仔细讨论）。

令 $\mathcal B_i$ 表示随机处理顺序中的第 $i$ 个块，$C_i$ 表示开始处理 $\mathcal B_i$ 时的 `topk_candidate`，$A_i$ 表示处理 $\mathcal B_i$ 时加入候选缓冲区的元素数量，$W$ 表示所有 `TopK` 调用的输入长度之和，包括最后一次调用。定义

$$
m=\left\lceil \frac NB\right\rceil,\qquad
L=k+B+B_2,\qquad
H_m=\sum_{i=1}^m\frac1i.
$$

由于 $|C_i|<k+B_2$，并且此前已经处理但不在 $C_i$ 中的所有元素都不大于当前阈值，因此，从 $\mathcal B_i$ 中加入 `topk_candidate` 的任意元素，在前 $i$ 个已处理块中的排名至多为 $|C_i|+|\mathcal B_i|\le |C_i|+B<L$。

现在考虑前 $i$ 个已处理块构成的集合 $U_i=\{\mathcal B_1,\ldots,\mathcal B_i\}$。

在给定 $U_i$ 的条件下，$\mathcal B_i$ 在 $U_i$ 中的 $i$ 个块之间是均匀分布。令 $T_i$ 表示 $U_i$ 中所有元素里的前 $L$ 大元素；若元素总数不足 $L$，则取全部元素。根据上面的界，所有被加入的元素都属于 $T_i$，因此

```math
\mathbb E[A_i\mid U_i]
\le \frac1i\sum_{b\in U_i}|b\cap T_i|
= \frac{|T_i|}{i}
\le \frac Li.
```

于是，令 $A=\sum_i A_i$，根据期望的线性性，我们可以得出 $\mathbb E[A]\le LH_m$。

令 $R$ 表示循环内部调用 `TopK` 的次数。每次这样的调用至少会删除 $B_2$ 个元素，因此 $RB_2\le A$。

每个被加入的元素都会被处理一次，而每次循环内部 `TopK` 保留下来的 $k$ 个元素之后还会再被处理一次。因此 $W=A+kR$。

从而得到

```math
\boxed{
\mathbb E[W]
\le \left(1+\frac{k}{B_2}\right)LH_m
= \left(1+\frac{k}{B_2}\right)L\bigl(\ln m+O(1)\bigr).
}
```

# 性能综述

该算法对输入中的每个元素只从全局内存读取一次，并以连续块的方式进行访问；后续计算主要在大小为 $O(k+B+B_2)$ 的候选集合上完成。随机块顺序下，所有 TopK 调用处理的元素总数期望为

$$
O\left(\left(1+\frac{k}{B_2}\right)(k+B+B_2)\log\frac NB\right).
$$

当 $B,B_2=\Theta(k)$ 时，TopK 部分的额外计算量为 $O(k\log(N/k))$，远小于输入规模 $N$。经过针对硬件的细致优化后，该算法在 DSA 场景下相较于 `torch.topk` 可以实现显著加速。

# 实现
我们使用 CUDA C\+\+ 实现这个 DeepSelect kernel，并大量应用底层 PTX 优化技巧与位运算技巧减少常数、降低耗时、提高显卡的带宽使用率。

正常来说，在筛选（Filter）那一步中，每个 CUDA 线程需要扫描两遍输入：第一遍统计出每个线程读到的数字中有多少个大于阈值的元素，以此得知每个线程应该把结果写入到 `topk_candidate` 数组中的哪个地方；第二遍再次扫描每个元素，并将大于阈值的元素放置在`topk_candidate` 中的正确位置。此处哪怕使用了 shared memory 避免反复读取显存，其 CUDA Core 操作的开销也会限制整个算子的性能。为此，我们在第一遍扫描中统计大于阈值的元素的数量的同时，会生成一个 `uint32_t` 格式的 mask，其第 `i` 个 bit 记录了第 `i` 个元素是否大于阈值。在后续放置元素时，我们可以使用 [`__ffs(mask)`](../csrc/cuda_kernels/common_parts.cuh#L1466) 来获取 `mask` 中最低的 1 的下标，然后用 `mask &= mask - 1u` 消去最低的 1，以此快速找到所有大于阈值的元素。此外，我们使用 `set.s32.bf16x2`、`prmt.b32`、`dp4a.s32.s32` 等底层 PTX 指令以加速 [`mask`](../csrc/cuda_kernels/common_parts.cuh#L1431) 生成。

此外，该算子中的比较指令和位运算指令数量极多。根据我们的观察，此类指令会和整数加法指令抢占部分硬件资源，造成算子性能下降。哪怕将整数加法换成与比较、位运算指令没有冲突的整数乘加（`mad`）指令，其吞吐也只有浮点乘加的一半，成为性能瓶颈。观察到，当 `x` 和 `y` 满足 `0 <= x, y <= 2^22` 时，`x+y = __float_as_uint(__uint_as_float(x) + __uint_as_float(y))`，也即，两个不超过 $2^{22}$ 的正整数的加法结果，与将其二进制表示视作 `float32`（此时二者应为 denormal number）然后做浮点加法的结果是一致的。因此我们可以[使用浮点加法来替换部分整数加法](../csrc/cuda_kernels/common_parts.cuh#L1051)，以提高吞吐。

除了上述两点之外，我们还使用底层 PTX 指令与位运算操作加速了其它操作，包括[warp 级前缀和](../csrc/cuda_kernels/common_parts.cuh#L1478)、[字节提取](../csrc/cuda_kernels/common_parts.cuh#L955)、[直方图构建](../csrc/cuda_kernels/common_parts.cuh#L940)等，详见代码。

对于 batch size 较小的场景（或者 wave quantization 效应下最后一个 wave 中活跃 SM 数量较少的场景），我们使用 Threadblock Cluster 将一个序列的计算切分至多个 Stream Multiprocessor \(SM\) 上。具体来说，每条序列会被切成若干份，每份首先经由一个 CTA \(CUDA Threadblock\) 求出这一份中 top\-k 大的元素。紧接着，每个 CTA 会将这 top\-k 个元素发送至其所在的 Threadblock Cluster 的 CTA0，再由 CTA0 取出这些元素中的前 k 大元素。
