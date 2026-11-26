#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "kernel.h"

#include <kernel_operator.h>

template <typename T1, typename T2>
__aicore__ constexpr auto ceil_div(T1 a, T2 b) -> decltype(a + b) {
    return (a + b - 1) / b;
}

template <typename T1, typename T2>
__aicore__ constexpr auto aligned(T1 a, T2 b) -> decltype(a + b) {
    return ceil_div(a, b) * b;
}

namespace topk_select_ascend_fp32 {

inline constexpr uint32_t VEC_BYTES = 256;
inline constexpr uint32_t NUM_FP32_PER_VEC = VEC_BYTES / sizeof(float);  // 64 lanes
inline constexpr uint32_t NUM_U8_PER_VEC = VEC_BYTES;                    // 256 lanes
// One byte-histogram round consumes four fp32 vectors, i.e. 256 elements, one byte each.
inline constexpr uint32_t HIST_ELEMS_PER_ROUND = NUM_U8_PER_VEC;  // 256

inline constexpr uint32_t TOPK_UB_LIMIT_BYTES = 248 * 1024;
inline constexpr uint32_t VST_VLD_ALIGN_BYTES = 32;
inline constexpr uint32_t VST_VLD_ALIGN_ELEMENTS = VST_VLD_ALIGN_BYTES / sizeof(float);
// Padding after the candidate region: it lets the compacted filtered-count store
// round up to a whole vector and makes the rebuild scan's over-read safe.
inline constexpr uint32_t TOPK_COMPACT_PADDING = 256;
inline constexpr uint32_t TOPK_HISTOGRAM_BINS = 256;
// Bytes of fp32 payload per radix pass; the monotone key is 32 bit, so four passes cover it.
inline constexpr uint32_t RADIX_PASSES = 4;
// AscendC::Sort's merge path (used for sorted_value) only accepts a count that is a multiple of
// 32; the radix path has no such requirement.
inline constexpr uint32_t SORT_ELEMENT_ALIGN = 32;

struct ShufflePage {
    uint32_t start_offset;
    uint32_t len;
};

// Map each logical scan page to a physical input range. Short inputs are scanned
// sequentially. For longer inputs, full pages are visited in a randomized order,
// while the physical tail is split at a 128-element boundary; the last tail segment
// is placed at the end of warmup and the preceding segment is visited last, keeping
// page starts aligned and avoiding out-of-bounds loads.
template <uint32_t NUM_WARMUP_PAGES, uint32_t SHUFFLE_PAGE_SIZE>
struct ShufflePermutationGenerator {
    uint32_t valid_len;
    uint32_t num_pages;
    uint32_t size;
    uint32_t seed0;
    uint32_t seed1;

    __aicore__ ShufflePermutationGenerator(uint32_t valid_len, uint64_t seed)
        : valid_len(valid_len),
          num_pages(ceil_div(valid_len, SHUFFLE_PAGE_SIZE)),
          size(num_pages > 1 ? num_pages - 2 : 0),
          seed0(static_cast<uint16_t>(seed)),
          seed1(static_cast<uint16_t>(seed >> 16) % (size == 0 ? 1 : size)) {}

    __aicore__ inline uint32_t reflect(uint32_t value, uint32_t pivot, uint32_t size) const {
        return value <= pivot ? pivot - value : size + pivot - value;
    }

    __aicore__ inline ShufflePage get_page_id(uint32_t logical_page_id) const {
        if (num_pages == 1)
            return {0, valid_len};

        if (num_pages <= NUM_WARMUP_PAGES) {
            const uint32_t start_offset = logical_page_id * SHUFFLE_PAGE_SIZE;
            const uint32_t len = start_offset + SHUFFLE_PAGE_SIZE <= valid_len
                                     ? SHUFFLE_PAGE_SIZE
                                     : valid_len - start_offset;
            return {start_offset, len};
        }

        const uint32_t last_start = aligned(valid_len - SHUFFLE_PAGE_SIZE, 128u);
        if (logical_page_id == NUM_WARMUP_PAGES - 1)
            return {last_start, valid_len - last_start};
        if (logical_page_id == num_pages - 1) {
            const uint32_t penultimate_start = (num_pages - 2) * SHUFFLE_PAGE_SIZE;
            return {penultimate_start, last_start - penultimate_start};
        }

        uint32_t value = logical_page_id < NUM_WARMUP_PAGES - 1 ? logical_page_id : logical_page_id - 1;

        value = (value * 65537u + seed0) % size;
        value = reflect(value, seed1, size);
        const uint32_t page_id = (value * 65521u + seed1) % size;
        return {page_id * SHUFFLE_PAGE_SIZE, SHUFFLE_PAGE_SIZE};
    }
};

// ---------------------------------------------------------------------------
// Algorithm parameters: one AlgoConfig per topk tier
// ---------------------------------------------------------------------------
// Same family as the Ascend bf16 kernel (see csrc/ascend_kernels/bf16/config.h for the
// full walkthrough of the permutation / warmup / streaming / rebuild structure) and as the
// CUDA fp32 kernel (csrc/cuda_kernels/v3_fp32/topk_select.cuh). The differences are:
//  * a candidate is an 8 B pair {fp32 value, u32 real row index}, and the index is materialized
//    when the value is appended (the CUDA pair layout), so no deferred index reconstruction;
//  * the bf16 kernel's two 8-bit radix passes become four, because an fp32 key is 32 bit wide.
//
// Algorithm outline:
// 1. Split the row into pages of SHUFFLE_PAGE_SIZE elements and visit them in a pseudo-random
//    order, so that the warmup sample is not biased by adversarial value distributions.
// 2. The first NUM_WARMUP_PAGES visited pages are the sample; radix selection on them yields
//    the k-th largest value as `threshold`.
// 3. Scan the remaining pages in blocks of BLOCK_SIZE elements, appending the values
//    > threshold to the candidate buffer. When the candidate count exceeds
//    MAX_TOPK + FILTERED_CAPACITY, run radix selection again to tighten `threshold` and compact
//    the buffer back to exactly k candidates.
//
// FILTERED_CAPACITY and WARMUP_PAGES are starting points borrowed from the CUDA fp32 tuning
// (reconstruct_threshold = 4096) and from the bf16 kernel; they are not yet profiled for
// fp32 on Ascend. The hard constraints are the static_asserts in Kernel and the UB budget:
// the candidate region costs 2 * (MAX_TOPK + BLOCK_SIZE + FILTERED_CAPACITY + padding) * 4 B.
struct AlgoConfig {
    uint32_t MAX_TOPK;           // Must equal Config::MAX_TOPK
    uint32_t FILTERED_CAPACITY;  // Rebuild when candidates exceed MAX_TOPK + this
    uint32_t WARMUP_PAGES;       // Pages sampled before the streaming scan starts
    uint32_t NUM_STAGES;         // Software-pipeline depth of the input staging
    uint32_t BLOCK_SIZE;         // Elements loaded per streaming iteration
    uint32_t SHUFFLE_PAGE_SIZE;  // Elements per scan page (must be a power of two)
};

inline constexpr AlgoConfig ALGO_512 = {
    .MAX_TOPK = 512, .FILTERED_CAPACITY = 4096, .WARMUP_PAGES = 2,
    .NUM_STAGES = 3, .BLOCK_SIZE = 8 * 1024, .SHUFFLE_PAGE_SIZE = 2 * 1024};
inline constexpr AlgoConfig ALGO_1024 = {
    .MAX_TOPK = 1024, .FILTERED_CAPACITY = 4096, .WARMUP_PAGES = 3,
    .NUM_STAGES = 3, .BLOCK_SIZE = 8 * 1024, .SHUFFLE_PAGE_SIZE = 2 * 1024};
inline constexpr AlgoConfig ALGO_4096 = {
    .MAX_TOPK = 4096, .FILTERED_CAPACITY = 3072, .WARMUP_PAGES = 4,
    .NUM_STAGES = 3, .BLOCK_SIZE = 8 * 1024, .SHUFFLE_PAGE_SIZE = 2 * 1024};

template <Config CONFIG>
inline constexpr AlgoConfig ALGO_CONFIG =
    CONFIG.MAX_TOPK <= 512 ? ALGO_512 : CONFIG.MAX_TOPK <= 1024 ? ALGO_1024 : ALGO_4096;

// Per-row runtime arguments of one kernel launch. Defined in kernel_body.h because
// its pointer members need the device-only `__gm__` qualifier.
struct RowArgs;

template <Config CONFIG>
struct Kernel {
    // ================= 1. Compile-time knobs (from CONFIG) =================
    static constexpr uint32_t MAX_TOPK = CONFIG.MAX_TOPK;
    static constexpr bool SORTED_INDEX = CONFIG.SORTED_INDEX;
    static constexpr bool SORTED_VALUE = CONFIG.SORTED_VALUE;
    static constexpr bool OUT_IDX_INT64 = CONFIG.OUT_IDX_INT64;
    static_assert(MAX_TOPK == 512 || MAX_TOPK == 1024 || MAX_TOPK == 4096,
                  "MAX_TOPK must be one of 512 / 1024 / 4096");
    static_assert(!(SORTED_INDEX && SORTED_VALUE), "sorting by index and by value are exclusive");

    // Output index type of this tier. Everything inside the kernel (candidate indices, Sort, the
    // offset) stays 32 bit; only the final GM store differs, see `store_index`.
    using OutIdxT = std::conditional_t<OUT_IDX_INT64, int64_t, int32_t>;

    // ================= 2. Algorithm parameters (from ALGO_CONFIG) =================
    static constexpr AlgoConfig ALGO = ALGO_CONFIG<CONFIG>;
    static_assert(ALGO.MAX_TOPK == MAX_TOPK, "the algorithm variant does not match CONFIG.MAX_TOPK");
    static constexpr uint32_t FILTERED_CAPACITY = ALGO.FILTERED_CAPACITY;
    static constexpr uint32_t NUM_WARMUP_PAGES = ALGO.WARMUP_PAGES;
    static constexpr uint32_t NUM_STAGES = ALGO.NUM_STAGES;
    static constexpr uint32_t BLOCK_SIZE = ALGO.BLOCK_SIZE;
    static constexpr uint32_t SHUFFLE_PAGE_SIZE = ALGO.SHUFFLE_PAGE_SIZE;
    static constexpr uint32_t SHUFFLE_PAGE_SIZE_BITS = std::countr_zero(SHUFFLE_PAGE_SIZE);

    // ================= 3. Structural constraints =================
    static constexpr uint32_t PAGES_PER_BLOCK = BLOCK_SIZE / SHUFFLE_PAGE_SIZE;
    static constexpr uint32_t MAX_SHUFFLE_PAGES = ceil_div(MAX_VOCAB_SIZE, SHUFFLE_PAGE_SIZE);
    // Page-map ids: warmup takes [0, NUM_WARMUP_PAGES) and the block-aligned scan pages start at
    // PAGES_PER_BLOCK, so the last id is num_pages - NUM_WARMUP_PAGES + PAGES_PER_BLOCK - 1.
    // The map holds physical page start offsets as u32 so that the append step can index it with
    // u32 values; it is rounded up to whole vectors to keep the following metadata aligned.
    static constexpr uint32_t SHUFFLE_PAGE_CAPACITY =
        aligned(MAX_SHUFFLE_PAGES - NUM_WARMUP_PAGES + PAGES_PER_BLOCK, VST_VLD_ALIGN_ELEMENTS);
    static_assert((SHUFFLE_PAGE_CAPACITY * sizeof(uint32_t)) % VST_VLD_ALIGN_BYTES == 0,
                  "the page map must be vector sized to keep the following metadata aligned");
    static_assert(NUM_WARMUP_PAGES > 0 && NUM_STAGES > 0);
    // The warmup slots and the first block's slots share one page map: warmup occupies
    // [0, NUM_WARMUP_PAGES) while the first stable block starts at PAGES_PER_BLOCK. A config with
    // more warmup pages than pages per block would overlap them and silently produce wrong
    // indices, so the relation is asserted instead of documented.
    static_assert(NUM_WARMUP_PAGES <= PAGES_PER_BLOCK,
                  "warmup pages must fit in front of the first stable block's page-map slots");
    static_assert(NUM_STAGES <= 8, "num_stages exceeds the available event IDs");
    static_assert(NUM_WARMUP_PAGES * SHUFFLE_PAGE_SIZE - 128 >= MAX_TOPK,
                  "warmup pages cannot hold top-k candidates");
    static_assert(BLOCK_SIZE % SHUFFLE_PAGE_SIZE == 0,
                  "block size must contain whole shuffle pages");
    static_assert((SHUFFLE_PAGE_SIZE & (SHUFFLE_PAGE_SIZE - 1)) == 0,
                  "shuffle page size must be a power of two");
    static_assert(SHUFFLE_PAGE_SIZE % NUM_FP32_PER_VEC == 0,
                  "streaming page lengths must be vector-sized (the scan loop has no tail mask)");
    static_assert((uint64_t(1) << 16) % BLOCK_SIZE == 0,
                  "BLOCK_SIZE must divide 64K so a block's virtual page ids stay block aligned");
    static_assert(uint64_t(SHUFFLE_PAGE_CAPACITY) * SHUFFLE_PAGE_SIZE <= (uint64_t(1) << 32),
                  "page offsets cannot reconstruct uint32 indices");
    static_assert(TOPK_COMPACT_PADDING >= VST_VLD_ALIGN_ELEMENTS,
                  "compact padding must cover filtered-count alignment");

    // ================= 4. Element / byte counts of each region =================
    static constexpr uint32_t WARMUP_ELEMENTS = NUM_WARMUP_PAGES * SHUFFLE_PAGE_SIZE;
    static constexpr uint32_t WARMUP_BYTES = WARMUP_ELEMENTS * sizeof(float);

    // Candidate region: k selected + at most one new block + filtered capacity, plus the padding
    // that lets the filtered-count store round up to a whole vector. A candidate is a value and
    // its index, so the region is two arrays of CANDIDATE_ELEMENTS.
    static constexpr uint32_t CANDIDATE_CAPACITY = MAX_TOPK + BLOCK_SIZE + FILTERED_CAPACITY;
    static constexpr uint32_t CANDIDATE_ELEMENTS = CANDIDATE_CAPACITY + TOPK_COMPACT_PADDING;
    static constexpr uint32_t CANDIDATE_ARRAY_BYTES = CANDIDATE_ELEMENTS * sizeof(float);
    static constexpr uint32_t CANDIDATE_BYTES = 2 * CANDIDATE_ARRAY_BYTES;
    static_assert(CANDIDATE_ARRAY_BYTES % VST_VLD_ALIGN_BYTES == 0,
                  "candidate arrays must be 32B aligned");

    static constexpr uint32_t INPUT_BYTES = NUM_STAGES * BLOCK_SIZE * sizeof(float);
    static constexpr uint32_t HISTOGRAM_BYTES = TOPK_HISTOGRAM_BINS * sizeof(uint16_t);
    static constexpr uint32_t THRESHOLD_BYTES = VST_VLD_ALIGN_BYTES;
    static constexpr uint32_t THRESHOLD_ELEMENTS = THRESHOLD_BYTES / sizeof(float);
    static constexpr uint32_t STATE_BYTES = VST_VLD_ALIGN_BYTES + 64 * sizeof(uint32_t);
    static constexpr uint32_t TEMPORARY_POSITION_BYTES =
        aligned(MAX_TOPK * sizeof(uint32_t), VST_VLD_ALIGN_BYTES);
    static constexpr uint32_t TEMPORARY_POSITION_ELEMENTS =
        TEMPORARY_POSITION_BYTES / sizeof(uint32_t);
    static constexpr uint32_t SHUFFLE_PAGE_ID_BYTES = SHUFFLE_PAGE_CAPACITY * sizeof(uint32_t);

    // ================= 5. Layout of the metadata block and of the input staging =================
    static constexpr uint32_t METADATA_END =
        CANDIDATE_BYTES + 2 * HISTOGRAM_BYTES + THRESHOLD_BYTES + STATE_BYTES +
        TEMPORARY_POSITION_BYTES + SHUFFLE_PAGE_ID_BYTES;
    static constexpr uint32_t INPUT_OFFSET = METADATA_END;
    static constexpr uint32_t UB_END = INPUT_OFFSET + INPUT_BYTES;
    static_assert(INPUT_OFFSET % VST_VLD_ALIGN_BYTES == 0, "input staging must be 32B aligned");

    // ================= 6. Sort workspace (aliased onto the input staging) =================
    // The two output modes use different AscendC::Sort paths (see the sort block in run):
    //  * sorted_index sorts the int32 row indices with the 4-pass LSD radix path, which returns a
    //    permutation; a second gather turns that into the sorted indices. Its temporary needs
    //    512 + 15 * AlignUp(count, 32) bytes.
    //  * sorted_value sorts the fp32 values with the merge path, which needs a 32-aligned count and
    //    a buffer for the interleaved (value, index) pairs, i.e. two arrays of SORT_COUNT elements.
    //    It returns the sorted indices directly and has no scatter, which makes it about twice as
    //    fast at this scale; its temporary only needs 8 * SORT_COUNT bytes.
    static constexpr uint32_t SORT_COUNT = aligned(MAX_TOPK, SORT_ELEMENT_ALIGN);
    static constexpr uint32_t SORT_VALUE_ELEMENTS = SORTED_VALUE ? 2 * SORT_COUNT : MAX_TOPK;
    static constexpr uint32_t SORT_INDEX_ELEMENTS = SORTED_VALUE ? SORT_COUNT : MAX_TOPK;
    static constexpr uint32_t SORT_ARRAY0_BYTES = SORT_VALUE_ELEMENTS * sizeof(float);
    static constexpr uint32_t SORT_ARRAY1_BYTES = SORT_INDEX_ELEMENTS * sizeof(uint32_t);
    static constexpr uint32_t SORT_TEMPORARY_BYTES = SORTED_VALUE ? 8 * SORT_COUNT : 512 + 15 * MAX_TOPK;
    static constexpr uint32_t SORT_WORKSPACE_BYTES =
        SORT_ARRAY0_BYTES + SORT_ARRAY1_BYTES + SORT_TEMPORARY_BYTES;

    // ================= 7. Unified-buffer memory plan =================
    // A lens over raw UB: the base address is modelled as 0, so a member's byte offset *is* its UB
    // address (`(Ub*)0` is used by the driver). Field declaration order below == address order, and
    // the plan must never contain a pointer (the vector frontend would treat it as UB payload).
    struct alignas(VST_VLD_ALIGN_BYTES) UnifiedBufferMemoryPlan {
        static_assert(sizeof(float) == sizeof(uint32_t), "UB plan assumes a 4-byte value type");

        // ---- nested types ----
        // State that must survive across radix rebuilds. `_pad0` puts `nan_lane_count` at +32 so
        // that it can be read with one 32B-aligned load.
        struct State {
            uint32_t result_count;  // number of live candidates (read as result_count[0])
            uint8_t _pad0[VST_VLD_ALIGN_BYTES - sizeof(uint32_t)];
            uint32_t nan_lane_count;  // number of NaN values seen (only != 0 is tested)
            uint8_t _pad1[STATE_BYTES - VST_VLD_ALIGN_BYTES - sizeof(uint32_t)];
        };
        static_assert(sizeof(State) == STATE_BYTES, "state size must be frozen");

        // Input staging and the sort workspace are live in disjoint phases: every GM->UB load has
        // completed before the final index preparation / Sort runs.
        union Overlay {
            alignas(VST_VLD_ALIGN_BYTES) float input_staging[NUM_STAGES * BLOCK_SIZE];
            // Sort scratch. Both modes put their destination and their companion array at the
            // same two slots, which is why one member covers both:
            //  * sorted_index: [sorted indices][permutation], source is `candidate_index`;
            //  * sorted_value: [sorted values][source positions], source is `candidate_value`.
            struct Sort {
                alignas(VST_VLD_ALIGN_BYTES) float array0[SORT_VALUE_ELEMENTS];
                alignas(VST_VLD_ALIGN_BYTES) uint32_t array1[SORT_INDEX_ELEMENTS];
                alignas(VST_VLD_ALIGN_BYTES) uint8_t temporary[SORT_TEMPORARY_BYTES];
            } sort;
            Overlay() {}
        };
        static_assert(sizeof(Overlay) == INPUT_BYTES, "the overlay must be sized by the staging");

        // ---- members: declaration order == UB address order ----
        // Candidate fp32 values. The warmup sampler uses [0, WARMUP_ELEMENTS) of this same field;
        // after the warmup selection [0, k) holds the current top-k values.
        float candidate_value[CANDIDATE_ELEMENTS];
        // Real row index of each candidate (`output_idx_offset` is added when it is written out).
        uint32_t candidate_index[CANDIDATE_ELEMENTS];
        // Scratch byte histogram of passes 2..4 / AR counter scratch, rebuilt every round.
        uint16_t histogram[TOPK_HISTOGRAM_BINS];
        // Top-byte histogram of the descending key, accumulated over every value ever scanned;
        // survives rebuilds (append-only, clamped to k+1).
        uint16_t high_histogram[TOPK_HISTOGRAM_BINS];
        // Current threshold, raw fp32 bit pattern in lane 0 (broadcast when read).
        uint32_t threshold[THRESHOLD_ELEMENTS];
        // Result count / NaN count (see State).
        State state;
        // Positions of the selected candidates (u32 stream, length k).
        uint32_t temporary_position[TEMPORARY_POSITION_ELEMENTS];
        // Block-aligned virtual page id -> physical page start offset; valid from warmup on.
        uint32_t shuffle_page_ids[SHUFFLE_PAGE_CAPACITY];
        // Streaming input staging <-> final Sort workspace (see Overlay).
        alignas(VST_VLD_ALIGN_BYTES) Overlay overlay;

        // ---- byte offsets needed by AscendC::Sort ----
        // Sort() takes byte offsets into UB instead of field references.
        static constexpr uintptr_t SORT_ARRAY0_OFFSET = INPUT_OFFSET;
        static constexpr uintptr_t SORT_ARRAY1_OFFSET = SORT_ARRAY0_OFFSET + SORT_ARRAY0_BYTES;
        static constexpr uintptr_t SORT_TEMPORARY_OFFSET = SORT_ARRAY1_OFFSET + SORT_ARRAY1_BYTES;
        static constexpr uintptr_t SORT_END = SORT_TEMPORARY_OFFSET + SORT_TEMPORARY_BYTES;
    };

    using Ub = UnifiedBufferMemoryPlan;

    // ================= 8. Capacity limits of the plan =================
    static_assert(sizeof(Ub) == UB_END, "UB plan layout does not match the byte-count plan");
    static_assert(alignof(Ub) == VST_VLD_ALIGN_BYTES, "UB plan must be 32B aligned");
    static_assert(UB_END <= TOPK_UB_LIMIT_BYTES, "topk_select UB layout overflow");
    static_assert(SORT_WORKSPACE_BYTES <= INPUT_BYTES,
                  "sort workspace does not fit in the reused input staging");
    static_assert(Ub::SORT_END <= TOPK_UB_LIMIT_BYTES, "sort workspace exceeds UB capacity");
    static_assert(MAX_TOPK <= CANDIDATE_ELEMENTS, "top-k candidates must fit in the candidate region");
    static_assert(WARMUP_BYTES <= CANDIDATE_ARRAY_BYTES,
                  "warmup sampling must alias the candidate value array");
    static_assert(SORT_ARRAY0_BYTES % VST_VLD_ALIGN_BYTES == 0 &&
                      SORT_ARRAY1_BYTES % VST_VLD_ALIGN_BYTES == 0,
                  "sort arrays must be 32B aligned");
    // An int64 output needs eight bytes per index. Rather than reserving a second top-k sized
    // array in the plan, the widening writes into the Sort temporary buffer: it is dead once Sort
    // has produced its result and untouched by the tiers that do not sort, and it starts after the
    // sorted arrays, so it never overlaps the int32 source either.
    static constexpr uint32_t OUT_IDX_WIDEN_BYTES = MAX_TOPK * sizeof(int64_t);
    static_assert(!OUT_IDX_INT64 || OUT_IDX_WIDEN_BYTES <= SORT_TEMPORARY_BYTES,
                  "the int64 widening borrows the Sort temporary buffer");

    // ================= 9. Named event (flag) IDs of the vector-core driver =================
    // On dav-3510 an event ID is a 3-bit value (0..7) and the ID space is independent per
    // (src, dst) pipe pair, so every group below is a disjoint range on its own pipe pair.
    struct Flags {
        // V -> MTE2: the input stage buffer has been consumed and MTE2 may refill it.
        static constexpr uint32_t INPUT_STAGE_FREE_BASE = 0;  // [0, NUM_STAGES)
        // MTE2 -> V: stage data has landed. Warmup pages use a disjoint range on the same pipe pair.
        static constexpr uint32_t INPUT_STAGE_FULL_BASE = 0;  // [0, NUM_STAGES)
        static constexpr uint32_t WARMUP_PAGE_FULL_BASE = 4;  // [4, 4 + WARMUP_PAGES)
        // V -> MTE3: the output value/index block is ready.
        static constexpr uint32_t OUTPUT_READY = 0;
        // V -> S: the NaN lane counter (and the result count) are visible to the scalar core.
        static constexpr uint32_t NAN_LANE_COUNT_READY = 0;
    };
    static_assert(NUM_STAGES <= Flags::WARMUP_PAGE_FULL_BASE,
                  "too many stages: the free flag IDs overlap the warmup page IDs");
    static_assert(Flags::WARMUP_PAGE_FULL_BASE + NUM_WARMUP_PAGES <= 8,
                  "warmup page flag IDs out of range");
    static_assert(Flags::INPUT_STAGE_FREE_BASE + NUM_STAGES <= 8,
                  "input stage free flag IDs out of range");
    static_assert(Flags::INPUT_STAGE_FULL_BASE + NUM_STAGES <= 8,
                  "input stage full flag IDs out of range");

    // ================= 10. Device functions (defined in kernel_body.h) =================
    // Every device function takes the UB plan as its first argument and scalars after it.
    template <bool IS_INITIALIZE>
    static __simd_callee__ void accumulate_nan_lane_count(__ubuf__ Ub& ub, __ubuf__ float* values,
                                                          uint32_t count);

    // Add one byte per element into the two halves of a 256-bin cumulative histogram:
    // `hist[i] += #{fed byte <= i}`. `IS_MSB` is radix pass 1: the fed byte is the top byte of the
    // descending key and every element counts. Otherwise the raw byte `BYTE_INDEX` (0 = least
    // significant) is fed, inverted on the negative lanes, and only elements whose higher bytes
    // equal `prefix_bits` count.
    template <bool IS_MSB, uint32_t BYTE_INDEX>
    static __simd_callee__ void accumulate_byte_histogram(vector_uint16_t& hist_low,
                                                          vector_uint16_t& hist_high,
                                                          __ubuf__ float* values, uint32_t count,
                                                          uint32_t prefix_bits, uint32_t prefix_mask,
                                                          bool negative);

    // Turn the compacted block-local positions of appended candidates into real row indices:
    // `index = page_start(pos >> SHUFFLE_PAGE_SIZE_BITS) + (pos % SHUFFLE_PAGE_SIZE)`.
    static __simd_callee__ void materialize_index(__ubuf__ Ub& ub, vector_uint32_t position,
                                                  uint32_t page_base, vector_uint32_t& index,
                                                  vector_bool mask);

    // Add `output_idx_offset` in place to the first `valid_count` candidate indices; the lanes
    // that hold the out-of-bounds fill value are left alone (see the oob rule in `topk_select`).
    static __simd_vf__ void prepare_index_vf(__ubuf__ Ub& ub, uint32_t count, uint32_t valid_count,
                                             int32_t output_idx_offset);

    static __simd_vf__ void fill_nan_sentinel_vf(__ubuf__ Ub& ub, __ubuf__ int32_t* destination,
                                                 uint32_t count);

    template <bool IS_INITIALIZE>
    static __simd_vf__ void build_high_histogram_vf(__ubuf__ Ub& ub, __ubuf__ float* values,
                                                     uint32_t count, uint32_t keep_count);

    // The warmup sample is loaded straight into `candidate_value` instead of being appended, so its
    // real row indices have to be materialized separately. The sample occupies positions
    // [0, count) of the buffer, which is the same as a scan block whose page base is 0.
    static __simd_vf__ void build_warmup_index_vf(__ubuf__ Ub& ub, uint32_t count);

    // sorted_index: reorder the values according to the Sort permutation, so that they line up
    // with the sorted indices (the permutation array is read and overwritten in place).
    static __simd_vf__ void get_sorted_value_vf(__ubuf__ Ub& ub, uint32_t topk);

    // sorted_value: fill [topk, SORT_COUNT) with -inf, so that the merge path's 32-aligned count
    // never brings a padded lane into the first k outputs.
    static __simd_vf__ void pad_sort_tail_vf(__ubuf__ Ub& ub, uint32_t topk,
                                             uint32_t sort_count);

    // sorted_value on a short row: only the valid elements are sorted, so the fill lanes from
    // [valid_count, topk) have to be restored in both output arrays afterwards.
    static __simd_vf__ void restore_short_row_fill_vf(__ubuf__ Ub& ub, uint32_t valid_count,
                                                      uint32_t topk, float value_oob_fill,
                                                      int32_t idx_oob_fill);

    /*
    This VF has three call sites:
    - During warmup, select the initial top-k candidates and establish the threshold.
    - In regular scan rounds, append values satisfying !(value <= threshold) to the filtered
      buffer, and reduce the buffer back to k candidates (updating the threshold) when it
      exceeds MAX_TOPK + FILTERED_CAPACITY.
    - On the last scan round, do the same and additionally finalize the selection.

    This deliberately selects every NaN so that it is retained and detected.
    */
    template <bool IS_WARMUP, bool IS_FINALIZE = false>
    static __simd_vf__ void select_candidates_vf(__ubuf__ Ub& ub, uint32_t keep_count,
                                                 uint32_t warmup_source_count,
                                                 uint32_t filter_source_count,
                                                 uint32_t filter_page_base, uint32_t filter_stage);

    template <bool RETURN_VALUE>
    static __simd_vf__ void prepare_short_row_vf(__ubuf__ Ub& ub, uint32_t valid_count,
                                                 uint32_t topk, float value_oob_fill,
                                                 int32_t idx_oob_fill);

    // Only instantiated for `OUT_IDX_INT64` tiers: sign extend `count` int32 indices into int64.
    static __simd_vf__ void widen_index_vf(__ubuf__ Ub& ub, __ubuf__ int32_t* source,
                                           __ubuf__ int64_t* destination, uint32_t count);

    // Scalar-side wrapper that emits the prepared int32 indices to the row's index output,
    // widening them first when the tier asks for int64 (see OUT_IDX_WIDEN_BYTES above). It owns
    // the V -> MTE3 barrier, because for int64 the copy has to wait for the widening VF.
    static __aicore__ void store_index(__ubuf__ Ub& ub, __ubuf__ int32_t* source,
                                       __gm__ void* destination, uint32_t count);

    // ================= 11. Driver (defined in kernel_body.h) =================
    // The kernel entry point for one row.
    static __aicore__ void run(const RowArgs& row);
};

}  // namespace topk_select_ascend_fp32
