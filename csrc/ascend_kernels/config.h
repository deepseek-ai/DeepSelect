#pragma once

#include "kernel.h"

#include <bit>
#include <cstdint>
#include <type_traits>

#include <kernel_operator.h>

namespace topk_select_ascend {

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------

inline constexpr uint32_t TOPK_COMPACT_PADDING = 256;
inline constexpr uint32_t TOPK_HISTOGRAM_BINS = 256;
inline constexpr uint32_t TOPK_UB_LIMIT_BYTES = 248 * 1024;
inline constexpr uint32_t VST_VLD_ALIGN_BYTES = 32;
inline constexpr uint32_t VST_VLD_ALIGN_ELEMENTS = VST_VLD_ALIGN_BYTES / sizeof(uint16_t);

template <typename T1, typename T2>
__aicore__ constexpr auto ceil_div(T1 a, T2 b) -> decltype(a + b) {
    return (a + b - 1) / b;
}

template <typename T1, typename T2>
__aicore__ constexpr auto aligned(T1 a, T2 b) -> decltype(a + b) {
    return ceil_div(a, b) * b;
}

struct TopkPage {
    uint32_t start_offset;
    uint32_t len;
};

// Map each logical scan page to a physical input range. Short inputs are scanned sequentially.
// For longer inputs, full pages are visited in a randomized order, while the physical tail is
// split at a 128-element boundary. The last tail segment is placed at the end of warmup, and the
// preceding segment is visited last, keeping page starts aligned and avoiding out-of-bounds loads.
template <uint32_t kWarmupPages, uint32_t kShufflePageSize>
struct TopkPermGenerator {
    uint32_t valid_len;
    uint32_t num_pages;
    uint32_t size;
    uint32_t seed0;
    uint32_t seed1;

    __aicore__ TopkPermGenerator(uint32_t valid_len, uint64_t seed)
        : valid_len(valid_len),
          num_pages(ceil_div(valid_len, kShufflePageSize)),
          size(num_pages > 1 ? num_pages - 2 : 0),
          seed0(static_cast<uint16_t>(seed)),
          seed1(static_cast<uint16_t>(seed >> 16) % (size == 0 ? 1 : size)) {}

    __aicore__ inline uint32_t reflect(uint32_t value, uint32_t pivot, uint32_t size) const {
        return value <= pivot ? pivot - value : size + pivot - value;
    }

    __aicore__ inline TopkPage get_page_id(uint32_t logical_page_id) const {
        if (num_pages == 1)
            return {0, valid_len};

        if (num_pages <= kWarmupPages) {
            const uint32_t start_offset = logical_page_id * kShufflePageSize;
            const uint32_t len = start_offset + kShufflePageSize <= valid_len
                                     ? kShufflePageSize
                                     : valid_len - start_offset;
            return {start_offset, len};
        }

        const uint32_t last_start = aligned(valid_len - kShufflePageSize, 128u);
        if (logical_page_id == kWarmupPages - 1)
            return {last_start, valid_len - last_start};
        if (logical_page_id == num_pages - 1) {
            const uint32_t penultimate_start = (num_pages - 2) * kShufflePageSize;
            return {penultimate_start, last_start - penultimate_start};
        }

        uint32_t value = logical_page_id < kWarmupPages - 1 ? logical_page_id : logical_page_id - 1;

        value = (value * 65537u + seed0) % size;
        value = reflect(value, seed1, size);
        const uint32_t page_id = (value * 65521u + seed1) % size;
        return {page_id * kShufflePageSize, kShufflePageSize};
    }
};

// ---------------------------------------------------------------------------
// Algorithm parameters
// ---------------------------------------------------------------------------
// 算法流程：
// 1. 将输入序列划分为大小为 shuffle_page_size 的页，并随机打乱各页的处理顺序。
// 2. 取乱序后的前 warmup_pages 页作为预热数据，通过 radix selection 求出其中的
//    topk，并将 topk 中的最小值记为 threshold。后续仅保留大于 threshold 的元素，
//    因为不大于该阈值的元素不可能进入当前 topk。
// 3. 将剩余页按 batch 处理，每个 batch 包含若干完整页，其元素总数由 block_size 指定。
//    处理每个 batch 时，将大于 threshold 的元素依次写入候选区；当累计候选元素数达到
//    filtered_capacity 时，将候选元素与当前 topk 合并，再通过一轮 radix selection 得到
//    新的 topk，并以新 topk 的最小值更新 threshold。重复该过程直至处理完整个序列。
//
// 保存执行 topk selection 算法所需的一组编译期参数。
template <uint32_t kK = 4 * 1024, uint32_t kFilteredCapacity = 4096,
          uint32_t kWarmupPages = 3, uint32_t kNumStages = 3,
          uint32_t kBlockSize = 16 * 1024, uint32_t kShufflePageSize = 2 * 1024,
          uint32_t kBlockSizeBits = 14>
struct TopkSelectConfig {
    static constexpr uint32_t topk_capacity = kK;
    static constexpr uint32_t filtered_capacity = kFilteredCapacity;
    static constexpr uint32_t warmup_pages = kWarmupPages;
    static constexpr uint32_t block_size = kBlockSize;
    static constexpr uint32_t shuffle_page_size = kShufflePageSize;
    static constexpr uint32_t shuffle_page_size_bits =
        std::countr_zero(kShufflePageSize);  // 以 2 为底的页大小对数。
    static constexpr uint32_t block_size_bits = kBlockSizeBits;  // 以 2 为底的 batch 大小对数。
    static constexpr uint32_t num_stages = kNumStages;  // kernel 采用的软件流水线级数。
};

// 根据输入序列长度选择算法参数：长度小于或等于 pivot 时使用 LowerConfig，长度大于
// pivot 时使用 UpperConfig。两组配置采用相同的算法流程，并针对不同规模的问题使用
// 更优的参数执行算法。
template <uint32_t kPivot, typename kLowerConfig, typename kUpperConfig>
struct TopkSelectPolicyConfig {
    static constexpr uint32_t pivot = kPivot;
    using LowerConfig = kLowerConfig;
    using UpperConfig = kUpperConfig;
};

using Topk512Config = TopkSelectPolicyConfig<
    32832, TopkSelectConfig<512, 4096, 2>, TopkSelectConfig<512, 1536, 2>>;
using Topk1024Config = TopkSelectPolicyConfig<
    36800, TopkSelectConfig<1024, 3072, 3>, TopkSelectConfig<1024, 2560, 3>>;
using Topk4096Config = TopkSelectPolicyConfig<
    10432, TopkSelectConfig<4096, 2560, 4>, TopkSelectConfig<4096, 3072, 4>>;

// ---------------------------------------------------------------------------
// Derived compile-time configuration of one `Config` instantiation
// ---------------------------------------------------------------------------
template <Config CONFIG>
struct Kernel {
    static constexpr uint32_t max_topk = CONFIG.MAX_TOPK;
    static constexpr bool abort_when_nan_found = CONFIG.ABORT_WHEN_NAN_FOUND;
    static constexpr bool sorted_index = CONFIG.SORTED_INDEX;

    static_assert(max_topk == 512 || max_topk == 1024 || max_topk == 4096,
                  "MAX_TOPK must be one of 512 / 1024 / 4096");

    // The scan-page algorithm is parameterized by a pivot: inputs no longer than
    // the pivot use `LowerConfig`, longer ones use `UpperConfig`. The pivot is a
    // runtime value, so the choice is made inside `run`.
    using PolicyConfig = std::conditional_t<
        (max_topk <= 512), Topk512Config,
        std::conditional_t<(max_topk <= 1024), Topk1024Config, Topk4096Config>>;

    // Entry point of the kernel; defined in kernel.asc.
    static __aicore__ void run(const TopkSelectArgs args);
};

// ---------------------------------------------------------------------------
// Unified-buffer (UB) memory plan
// ---------------------------------------------------------------------------
// Every offset below is a compile-time constant derived from the algorithm
// parameters. The plan is keyed by the algorithm config instead of
// `Kernel<CONFIG>` because the runtime pivot selects between the lower and
// upper config.
template <typename AlgoConfig, bool kSortedIndex, typename value_t>
struct UnifiedBufferMemoryPlan {
    static constexpr uint32_t kTopkCapacity = AlgoConfig::topk_capacity;
    static constexpr uint32_t kFilteredCapacity = AlgoConfig::filtered_capacity;
    static constexpr uint32_t kWarmupPages = AlgoConfig::warmup_pages;
    static constexpr uint32_t kBlockSize = AlgoConfig::block_size;
    static constexpr uint32_t kShufflePageSize = AlgoConfig::shuffle_page_size;
    static constexpr uint32_t kBlockSizeBits = AlgoConfig::block_size_bits;
    static constexpr uint32_t kNumStages = AlgoConfig::num_stages;

    static_assert(kWarmupPages > 0 && kNumStages > 0);
    static_assert(kNumStages <= 8, "num_stages exceeds the available event IDs");
    static_assert(kWarmupPages * kShufflePageSize - 128 >= kTopkCapacity,
                  "warmup pages cannot hold top-k candidates");
    static_assert(kBlockSize == (uint32_t(1) << kBlockSizeBits),
                  "block size and block size bits must match");
    static_assert(kBlockSize % kShufflePageSize == 0,
                  "block size must contain whole shuffle pages");
    static_assert((kShufflePageSize & (kShufflePageSize - 1)) == 0,
                  "shuffle page size must be a power of two");
    static constexpr uint32_t kShufflePageCapacity = MAX_VOCAB_SIZE / kShufflePageSize;
    static_assert(uint64_t(kShufflePageCapacity) * kShufflePageSize <= (uint64_t(1) << 32),
                  "shuffle page ids cannot reconstruct uint32 indices");
    static_assert(TOPK_COMPACT_PADDING >= VST_VLD_ALIGN_ELEMENTS,
                  "compact padding must cover filtered-count alignment");

    static constexpr uint32_t kWarmupElements = kWarmupPages * kShufflePageSize;
    static constexpr uint32_t kWarmupBytes = kWarmupElements * sizeof(value_t);
    static constexpr uint32_t kCandidateCapacity =
        kTopkCapacity + kBlockSize + kFilteredCapacity;
    static constexpr uint32_t kCandidatePartBytes =
        aligned((kCandidateCapacity + TOPK_COMPACT_PADDING) * sizeof(uint16_t), 32u);
    static constexpr uint32_t kCandidateBytes = 3 * kCandidatePartBytes;
    static constexpr uint32_t kReuseBytes =
        kWarmupBytes > kCandidateBytes ? kWarmupBytes : kCandidateBytes;
    static constexpr uint32_t kInputBytes = kNumStages * kBlockSize * sizeof(value_t);
    static constexpr uint32_t kHistogramBytes = TOPK_HISTOGRAM_BINS * sizeof(uint16_t);
    static constexpr uint32_t kTemporaryPositionBytes =
        aligned(kTopkCapacity * sizeof(uint16_t), 32u);
    static constexpr uint32_t kThresholdBytes = 32;
    static constexpr uint32_t kStateBytes = 32 + 64 * sizeof(uint32_t);
    static constexpr uint32_t kShufflePageIdBytes = kShufflePageCapacity * sizeof(uint16_t);

    // The leading region is deliberately reused by two phases: warmup owns it
    // first; candidate value/low/high own it after warmup selection starts.
    static constexpr uintptr_t kCandidateValueOffset = 0;
    static constexpr uintptr_t kCandidateIndexLowOffset =
        kCandidateValueOffset + kCandidatePartBytes;
    static constexpr uintptr_t kCandidateIndexHighOffset =
        kCandidateIndexLowOffset + kCandidatePartBytes;
    static constexpr uintptr_t kHistogramOffset = kReuseBytes;
    static constexpr uintptr_t kHighHistogramOffset = kHistogramOffset + kHistogramBytes;
    static constexpr uintptr_t kThresholdOffset = kHighHistogramOffset + kHistogramBytes;
    static constexpr uintptr_t kStateOffset = kThresholdOffset + kThresholdBytes;
    static constexpr uintptr_t kTemporaryPositionOffset = kStateOffset + kStateBytes;
    static constexpr uintptr_t kShufflePageIdOffset =
        kTemporaryPositionOffset + kTemporaryPositionBytes;
    static constexpr uintptr_t kMetadataEnd = kShufflePageIdOffset + kShufflePageIdBytes;
    static constexpr uintptr_t kMinInputOffset = (uint32_t(1) << 16) * sizeof(value_t);
    static constexpr uintptr_t kInputOffset =
        kMetadataEnd > kMinInputOffset ? kMetadataEnd : kMinInputOffset;
    static constexpr uintptr_t kUbEnd = kInputOffset + kInputBytes;

    static constexpr uint32_t kSortIndexBytes = kTopkCapacity * sizeof(uint32_t);
    static constexpr uint32_t kSortTemporaryBytes = 512 + 11 * kTopkCapacity;
    static constexpr uintptr_t kSortedIndexOffset = kInputOffset + kSortIndexBytes;
    static constexpr uintptr_t kSortPermutationOffset = kSortedIndexOffset + kSortIndexBytes;
    static constexpr uintptr_t kSortTemporaryOffset = kSortPermutationOffset + kSortIndexBytes;
    static constexpr uintptr_t kSortEnd = kSortTemporaryOffset + kSortTemporaryBytes;

    // Sorting reuses the input staging region after all loads end.
    static_assert(kUbEnd <= TOPK_UB_LIMIT_BYTES, "topk_select UB layout overflow");
    static_assert(!kSortedIndex || kSortIndexBytes <= kInputBytes,
                  "sorted-index source does not fit in reused input staging");
    static_assert(!kSortedIndex || kSortEnd <= TOPK_UB_LIMIT_BYTES,
                  "sorted-index workspace exceeds UB capacity");
};

}  // namespace topk_select_ascend
