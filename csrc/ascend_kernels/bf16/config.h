#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>

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

namespace topk_select_ascend {

inline constexpr uint32_t VEC_BYTES = 256;
inline constexpr uint32_t NUM_BF16_PER_VEC = VEC_BYTES / sizeof(uint16_t);  // 128 lanes
inline constexpr uint32_t NUM_U32_PER_VEC = VEC_BYTES / sizeof(uint32_t);   // 64 lanes
// The byte-histogram pass consumes two bf16 vectors (256 byte-lanes) per round.
inline constexpr uint32_t HIST_BYTES_PER_ROUND = 2 * NUM_BF16_PER_VEC;  // 256

inline constexpr uint32_t TOPK_UB_LIMIT_BYTES = 248 * 1024;
inline constexpr uint32_t VST_VLD_ALIGN_BYTES = 32;
inline constexpr uint32_t VST_VLD_ALIGN_ELEMENTS = VST_VLD_ALIGN_BYTES / sizeof(uint16_t);
// Padding after the candidate region: it lets the compacted filtered-count store
// round up to a whole vector and makes the rebuild scan's 128-lane over-read safe.
inline constexpr uint32_t TOPK_COMPACT_PADDING = 256;
inline constexpr uint32_t TOPK_HISTOGRAM_BINS = 256;

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
// Algorithm parameters: one AlgoConfig per (tier, variant)
// ---------------------------------------------------------------------------
// Algorithm outline:
// 1. Split the row into pages of SHUFFLE_PAGE_SIZE elements and visit them in a
//    pseudo-random order (the permutation mixes rows with adversarial value
//    distributions so that the warmup threshold is unbiased).
// 2. The first WARMUP_PAGES visited pages are the warmup sample. Radix selection on
//    them gives the k-th largest value as `threshold`; afterwards only values
//    > threshold are kept, because smaller ones can never enter the current top-k.
// 3. Scan the remaining pages in blocks of BLOCK_SIZE elements, appending the values
//    > threshold to the candidate buffer. When the candidate count exceeds
//    MAX_TOPK + FILTERED_CAPACITY, merge them with the current top-k, run another
//    radix selection and update `threshold`. Repeat until the whole row is scanned.
//
// FILTERED_CAPACITY and WARMUP_PAGES were tuned per tier (and per variant). Hard
// constraints are the static_asserts in Kernel and the UB budget: the candidate
// region costs 3 * (MAX_TOPK + BLOCK_SIZE + FILTERED_CAPACITY + padding) * 2 B.
struct AlgoConfig {
    uint32_t MAX_TOPK;           // Must equal Config::MAX_TOPK
    uint32_t FILTERED_CAPACITY;  // Rebuild when candidates exceed MAX_TOPK + this
    uint32_t WARMUP_PAGES;       // Pages sampled before the streaming scan starts
    uint32_t NUM_STAGES;         // Software-pipeline depth of the input staging
    uint32_t BLOCK_SIZE;         // Elements loaded per streaming iteration
    uint32_t SHUFFLE_PAGE_SIZE;  // Elements per scan page (must be a power of two)
};

inline constexpr AlgoConfig ALGO_512_SHORT_SEQ = {
    .MAX_TOPK = 512, .FILTERED_CAPACITY = 4096, .WARMUP_PAGES = 2,
    .NUM_STAGES = 3, .BLOCK_SIZE = 16 * 1024, .SHUFFLE_PAGE_SIZE = 2 * 1024};
inline constexpr AlgoConfig ALGO_512_LONG_SEQ = {
    .MAX_TOPK = 512, .FILTERED_CAPACITY = 1536, .WARMUP_PAGES = 2,
    .NUM_STAGES = 3, .BLOCK_SIZE = 16 * 1024, .SHUFFLE_PAGE_SIZE = 2 * 1024};
inline constexpr AlgoConfig ALGO_1024_SHORT_SEQ = {
    .MAX_TOPK = 1024, .FILTERED_CAPACITY = 3072, .WARMUP_PAGES = 3,
    .NUM_STAGES = 3, .BLOCK_SIZE = 16 * 1024, .SHUFFLE_PAGE_SIZE = 2 * 1024};
inline constexpr AlgoConfig ALGO_1024_LONG_SEQ = {
    .MAX_TOPK = 1024, .FILTERED_CAPACITY = 2560, .WARMUP_PAGES = 3,
    .NUM_STAGES = 3, .BLOCK_SIZE = 16 * 1024, .SHUFFLE_PAGE_SIZE = 2 * 1024};
inline constexpr AlgoConfig ALGO_4096_SHORT_SEQ = {
    .MAX_TOPK = 4096, .FILTERED_CAPACITY = 2560, .WARMUP_PAGES = 4,
    .NUM_STAGES = 3, .BLOCK_SIZE = 16 * 1024, .SHUFFLE_PAGE_SIZE = 2 * 1024};
inline constexpr AlgoConfig ALGO_4096_LONG_SEQ = {
    .MAX_TOPK = 4096, .FILTERED_CAPACITY = 3072, .WARMUP_PAGES = 4,
    .NUM_STAGES = 3, .BLOCK_SIZE = 16 * 1024, .SHUFFLE_PAGE_SIZE = 2 * 1024};

// Rows no longer than SHORT_LONG_SEQ_SPLIT_POINT use the SHORT_SEQ variant, longer ones
// use LONG_SEQ. Both run the same algorithm; only the parameters differ, which is
// cheaper than paying for the fixed overhead of the long-row tuning on short rows.
inline constexpr uint32_t SHORT_LONG_SEQ_SPLIT_POINT_512 = 32832;
inline constexpr uint32_t SHORT_LONG_SEQ_SPLIT_POINT_1024 = 36800;
inline constexpr uint32_t SHORT_LONG_SEQ_SPLIT_POINT_4096 = 10432;

// Per-`Config` selection. The per-row dispatch in kernel_body.h reads SHORT_LONG_SEQ_SPLIT_POINT and
// dispatches to Kernel<CONFIG, SHORT_SEQ_ALGO<CONFIG>> or Kernel<CONFIG, LONG_SEQ_ALGO<CONFIG>>.
template <Config CONFIG>
inline constexpr uint32_t SHORT_LONG_SEQ_SPLIT_POINT =
    CONFIG.MAX_TOPK <= 512 ? SHORT_LONG_SEQ_SPLIT_POINT_512
    : CONFIG.MAX_TOPK <= 1024 ? SHORT_LONG_SEQ_SPLIT_POINT_1024
    : SHORT_LONG_SEQ_SPLIT_POINT_4096;
template <Config CONFIG>
inline constexpr AlgoConfig SHORT_SEQ_ALGO =
    CONFIG.MAX_TOPK <= 512 ? ALGO_512_SHORT_SEQ
    : CONFIG.MAX_TOPK <= 1024 ? ALGO_1024_SHORT_SEQ
    : ALGO_4096_SHORT_SEQ;
template <Config CONFIG>
inline constexpr AlgoConfig LONG_SEQ_ALGO =
    CONFIG.MAX_TOPK <= 512 ? ALGO_512_LONG_SEQ
    : CONFIG.MAX_TOPK <= 1024 ? ALGO_1024_LONG_SEQ
    : ALGO_4096_LONG_SEQ;

// Per-row runtime arguments of one kernel launch. Defined in kernel_body.h because
// its pointer members need the device-only `__gm__` qualifier.
struct RowArgs;

template <Config CONFIG, AlgoConfig ALGO_CONFIG>
struct Kernel {
    // ================= 1. Compile-time knobs (from CONFIG) =================
    static constexpr uint32_t MAX_TOPK = CONFIG.MAX_TOPK;
    static constexpr bool SORTED_INDEX = CONFIG.SORTED_INDEX;
    static_assert(MAX_TOPK == 512 || MAX_TOPK == 1024 || MAX_TOPK == 4096,
                  "MAX_TOPK must be one of 512 / 1024 / 4096");
    static_assert(ALGO_CONFIG.MAX_TOPK == MAX_TOPK,
                  "the algorithm variant does not match CONFIG.MAX_TOPK");

    // ================= 2. Algorithm parameters (from ALGO_CONFIG) =================
    static constexpr uint32_t FILTERED_CAPACITY = ALGO_CONFIG.FILTERED_CAPACITY;
    static constexpr uint32_t NUM_WARMUP_PAGES = ALGO_CONFIG.WARMUP_PAGES;
    static constexpr uint32_t NUM_STAGES = ALGO_CONFIG.NUM_STAGES;
    static constexpr uint32_t BLOCK_SIZE = ALGO_CONFIG.BLOCK_SIZE;
    static constexpr uint32_t SHUFFLE_PAGE_SIZE = ALGO_CONFIG.SHUFFLE_PAGE_SIZE;
    static constexpr uint32_t SHUFFLE_PAGE_SIZE_BITS = std::countr_zero(SHUFFLE_PAGE_SIZE);

    // ================= 3. Structural constraints =================
    static constexpr uint32_t PAGES_PER_BLOCK = BLOCK_SIZE / SHUFFLE_PAGE_SIZE;
    static constexpr uint32_t MAX_SHUFFLE_PAGES = ceil_div(MAX_VOCAB_SIZE, SHUFFLE_PAGE_SIZE);
    // Page-map ids: warmup takes [0, NUM_WARMUP_PAGES) and the block-aligned scan pages start at
    // PAGES_PER_BLOCK, so the last id is num_pages - NUM_WARMUP_PAGES + PAGES_PER_BLOCK - 1.
    // Rounded up to whole vectors to keep the metadata that follows 32B aligned.
    static constexpr uint32_t SHUFFLE_PAGE_CAPACITY =
        aligned(MAX_SHUFFLE_PAGES - NUM_WARMUP_PAGES + PAGES_PER_BLOCK, VST_VLD_ALIGN_ELEMENTS);
    static_assert((SHUFFLE_PAGE_CAPACITY * sizeof(uint16_t)) % VST_VLD_ALIGN_BYTES == 0,
                  "the page map must be vector sized to keep the following metadata aligned");
    static_assert(NUM_WARMUP_PAGES > 0 && NUM_STAGES > 0);
    static_assert(NUM_STAGES <= 8, "num_stages exceeds the available event IDs");
    static_assert(NUM_WARMUP_PAGES * SHUFFLE_PAGE_SIZE - 128 >= MAX_TOPK,
                  "warmup pages cannot hold top-k candidates");
    static_assert(BLOCK_SIZE % SHUFFLE_PAGE_SIZE == 0,
                  "block size must contain whole shuffle pages");
    static_assert((SHUFFLE_PAGE_SIZE & (SHUFFLE_PAGE_SIZE - 1)) == 0,
                  "shuffle page size must be a power of two");
    static_assert(SHUFFLE_PAGE_SIZE % NUM_BF16_PER_VEC == 0,
                  "streaming page lengths must be vector-sized (the scan loop has no tail mask)");
    static_assert((uint64_t(1) << 16) % BLOCK_SIZE == 0,
                  "BLOCK_SIZE must divide 64K so the virtual index high half is block-constant");
    static_assert(uint64_t(SHUFFLE_PAGE_CAPACITY) * SHUFFLE_PAGE_SIZE <= (uint64_t(1) << 32),
                  "shuffle page ids cannot reconstruct uint32 indices");
    static_assert(TOPK_COMPACT_PADDING >= VST_VLD_ALIGN_ELEMENTS,
                  "compact padding must cover filtered-count alignment");

    // ================= 4. Element / byte counts of each region =================
    static constexpr uint32_t WARMUP_ELEMENTS = NUM_WARMUP_PAGES * SHUFFLE_PAGE_SIZE;
    static constexpr uint32_t WARMUP_BYTES = WARMUP_ELEMENTS * sizeof(bfloat16_t);

    // Candidate region: k selected + at most one new block + filtered capacity,
    // plus the padding that lets the filtered-count store round up to a whole vector.
    static constexpr uint32_t CANDIDATE_CAPACITY = MAX_TOPK + BLOCK_SIZE + FILTERED_CAPACITY;
    static constexpr uint32_t CANDIDATE_ELEMENTS = CANDIDATE_CAPACITY + TOPK_COMPACT_PADDING;
    static constexpr uint32_t CANDIDATE_PART_BYTES = CANDIDATE_ELEMENTS * sizeof(uint16_t);
    static constexpr uint32_t CANDIDATE_BYTES = 3 * CANDIDATE_PART_BYTES;
    static_assert(CANDIDATE_PART_BYTES == aligned(CANDIDATE_PART_BYTES, VST_VLD_ALIGN_BYTES),
                  "candidate part must be 32B aligned");
    static_assert(WARMUP_BYTES <= CANDIDATE_BYTES,
                  "warmup sampling must alias the candidate value part");

    static constexpr uint32_t INPUT_BYTES = NUM_STAGES * BLOCK_SIZE * sizeof(bfloat16_t);
    static constexpr uint32_t HISTOGRAM_BYTES = TOPK_HISTOGRAM_BINS * sizeof(uint16_t);
    static constexpr uint32_t THRESHOLD_BYTES = VST_VLD_ALIGN_BYTES;
    static constexpr uint32_t THRESHOLD_ELEMENTS = THRESHOLD_BYTES / sizeof(uint16_t);
    static constexpr uint32_t STATE_BYTES = VST_VLD_ALIGN_BYTES + 64 * sizeof(uint32_t);
    static constexpr uint32_t TEMPORARY_POSITION_BYTES =
        aligned(MAX_TOPK * sizeof(uint16_t), VST_VLD_ALIGN_BYTES);
    static constexpr uint32_t TEMPORARY_POSITION_ELEMENTS =
        TEMPORARY_POSITION_BYTES / sizeof(uint16_t);
    static constexpr uint32_t SHUFFLE_PAGE_ID_BYTES = SHUFFLE_PAGE_CAPACITY * sizeof(uint16_t);

    // ================= 5. Metadata end / explicit pad / input staging start =================
    static constexpr uint32_t SHUFFLE_PAGE_ID_END =
        CANDIDATE_BYTES + 2 * HISTOGRAM_BYTES + THRESHOLD_BYTES + STATE_BYTES +
        TEMPORARY_POSITION_BYTES + SHUFFLE_PAGE_ID_BYTES;
    static constexpr uint32_t METADATA_END = SHUFFLE_PAGE_ID_END;
    // `gather_base = filter_source - (uint16_t)virtual_index_base` may subtract up to
    // 3 blocks (49152 elements), so the input staging has to stay above the u16 range.
    static constexpr uint32_t MIN_INPUT_OFFSET = (uint32_t(1) << 16) * sizeof(bfloat16_t);
    static constexpr uint32_t RESERVED_BYTES =
        MIN_INPUT_OFFSET > METADATA_END ? MIN_INPUT_OFFSET - METADATA_END : 0;
    static constexpr uint32_t INPUT_OFFSET = METADATA_END + RESERVED_BYTES;
    static constexpr uint32_t UB_END = INPUT_OFFSET + INPUT_BYTES;
    static_assert(INPUT_OFFSET >= MIN_INPUT_OFFSET,
                  "input staging must be reachable by the u16 gather offset");
    static_assert(INPUT_OFFSET >= METADATA_END, "input staging overlaps metadata");

    // ================= 6. Sort workspace (aliased onto the input staging) =================
    static constexpr uint32_t SORT_INDEX_BYTES = MAX_TOPK * sizeof(uint32_t);
    // Empirical AscendC::Sort temporary size: 512 + 11 * k (matches SortImpl).
    static constexpr uint32_t SORT_TEMPORARY_BYTES = 512 + 11 * MAX_TOPK;
    static constexpr uint32_t SORT_WORKSPACE_BYTES =
        3 * SORT_INDEX_BYTES + SORT_TEMPORARY_BYTES;

    // ================= 7. Unified-buffer memory plan =================
    // A lens over raw UB: the base address is modelled as 0, so a member's byte
    // offset *is* its UB address (`(Ub*)0` is used by the driver). Field declaration
    // order below == address order, and the plan must never contain a pointer (the
    // vector frontend would treat it as UB payload).
    //
    //   [ candidate value | index_low | index_high | metadata | reserved | overlay ]
    struct alignas(VST_VLD_ALIGN_BYTES) UnifiedBufferMemoryPlan {
        static_assert(sizeof(bfloat16_t) == sizeof(uint16_t),
                      "UB plan assumes a 2-byte value type (bf16 / u16)");

        // ---- nested types ----
        // State that must survive across radix rebuilds. `_pad0` puts
        // `nan_lane_count` at +32 so it can be read with one 32B-aligned load.
        struct State {
            uint32_t result_count;  // number of live candidates (read as result_count[0])
            uint8_t _pad0[VST_VLD_ALIGN_BYTES - sizeof(uint32_t)];
            uint32_t nan_lane_count;  // number of NaN values seen (only != 0 is tested)
            uint8_t _pad1[STATE_BYTES - VST_VLD_ALIGN_BYTES - sizeof(uint32_t)];
        };
        static_assert(sizeof(State) == STATE_BYTES, "state size must be frozen");

        // Input staging and the sort workspace are live in disjoint phases: every
        // GM->UB load has completed before the final index preparation / Sort runs.
        union Overlay {
            alignas(VST_VLD_ALIGN_BYTES) bfloat16_t input_staging[NUM_STAGES * BLOCK_SIZE];
            struct Sort {
                alignas(VST_VLD_ALIGN_BYTES) int32_t source_index[MAX_TOPK];
                alignas(VST_VLD_ALIGN_BYTES) int32_t sorted_index[MAX_TOPK];
                alignas(VST_VLD_ALIGN_BYTES) uint32_t permutation[MAX_TOPK];
                alignas(VST_VLD_ALIGN_BYTES) uint8_t temporary[SORT_TEMPORARY_BYTES];
            } sort;
            Overlay() {}
        };

        // ---- members: declaration order == UB address order ----
        // Candidate bf16 values. The warmup sampler uses [0, WARMUP_ELEMENTS) of this
        // same field; after the warmup selection [0, k) holds the current top-k values.
        bfloat16_t candidate_value[CANDIDATE_ELEMENTS];
        // Candidate virtual index, low 16 bits. The short-row path borrows the whole
        // field as an int32 output buffer (valid_len <= topk).
        uint16_t candidate_index_low[CANDIDATE_ELEMENTS];
        // Candidate virtual index, high 16 bits (constant within one block). The
        // sorted-index path borrows the whole field to hold the sorted values.
        uint16_t candidate_index_high[CANDIDATE_ELEMENTS];
        // Low-byte histogram scratch / AR counter scratch, rebuilt every round.
        uint16_t histogram[TOPK_HISTOGRAM_BINS];
        // High-byte histogram accumulated over every value ever scanned; survives rebuilds.
        uint16_t high_histogram[TOPK_HISTOGRAM_BINS];
        // Current threshold, raw bf16 bit pattern in lane 0 (broadcast when read).
        uint16_t threshold[THRESHOLD_ELEMENTS];
        // Result count / NaN count (see State).
        State state;
        // Positions of the selected candidates (u16 stream, length k).
        uint16_t temporary_position[TEMPORARY_POSITION_ELEMENTS];
        // Virtual page id -> physical page start (offset / 128); valid from warmup on.
        uint16_t shuffle_page_ids[SHUFFLE_PAGE_CAPACITY];
        // Explicit pad raising the metadata end to MIN_INPUT_OFFSET (0 / 1184 / 4256 / 14496 B).
        uint8_t reserved_to_min_input_offset[RESERVED_BYTES];
        // Streaming input staging <-> final Sort workspace (see Overlay).
        alignas(VST_VLD_ALIGN_BYTES) Overlay overlay;

        // ---- byte offsets needed by AscendC::Sort ----
        // Sort() takes byte offsets into UB instead of field references. Its workspace
        // overlays the input staging, so `source_index` already starts at INPUT_OFFSET;
        // only the members after it need an explicit offset here. Every array of
        // Overlay::Sort is 32B-aligned, so the natural struct layout matches these
        // prefix sums (asserted below).
        static_assert(SORT_INDEX_BYTES % VST_VLD_ALIGN_BYTES == 0,
                      "sort arrays must be 32B-aligned so these offsets match Overlay::Sort");
        static constexpr uintptr_t SORTED_INDEX_OFFSET = INPUT_OFFSET + SORT_INDEX_BYTES;
        static constexpr uintptr_t SORT_PERMUTATION_OFFSET =
            SORTED_INDEX_OFFSET + SORT_INDEX_BYTES;
        static constexpr uintptr_t SORT_TEMPORARY_OFFSET =
            SORT_PERMUTATION_OFFSET + SORT_INDEX_BYTES;
        static constexpr uintptr_t SORT_END = SORT_TEMPORARY_OFFSET + SORT_TEMPORARY_BYTES;
    };

    using Ub = UnifiedBufferMemoryPlan;

    // ================= 8. Capacity limits of the plan =================
    static_assert(sizeof(Ub) == UB_END, "UB plan layout does not match the byte-count plan");
    static_assert(alignof(Ub) == VST_VLD_ALIGN_BYTES, "UB plan must be 32B aligned");
    static_assert(UB_END <= TOPK_UB_LIMIT_BYTES, "topk_select UB layout overflow");
    static_assert(SORT_WORKSPACE_BYTES <= INPUT_BYTES,
                  "sorted-index workspace does not fit in the reused input staging");
    static_assert(!SORTED_INDEX || Ub::SORT_END <= TOPK_UB_LIMIT_BYTES,
                  "sorted-index workspace exceeds UB capacity");
    static_assert(!SORTED_INDEX || MAX_TOPK <= CANDIDATE_ELEMENTS,
                  "top-k candidates must fit in the candidate value part");
    static_assert(CANDIDATE_BYTES % VST_VLD_ALIGN_BYTES == 0,
                  "candidate region must be 32B aligned");
    static_assert(INPUT_OFFSET % VST_VLD_ALIGN_BYTES == 0,
                  "input staging must be 32B aligned");

    // ================= 9. Named event (flag) IDs of the vector-core driver =================
    // On dav-3510 an event ID is a 3-bit value (0..7) and the ID space is independent
    // per (src, dst) pipe pair, so every group below is a disjoint range on its own
    // pipe pair rather than a global slot.
    struct Flags {
        // V -> MTE2: the input stage buffer has been consumed and MTE2 may refill it.
        static constexpr uint32_t INPUT_STAGE_FREE_BASE = 0;  // [0, NUM_STAGES)
        // MTE2 -> V: stage data has landed. Warmup pages use a disjoint range on the
        // same pipe pair so that the two sources never share an ID.
        static constexpr uint32_t INPUT_STAGE_FULL_BASE = 0;  // [0, NUM_STAGES)
        static constexpr uint32_t WARMUP_PAGE_FULL_BASE = 4;  // [4, 4 + WARMUP_PAGES)
        // V -> MTE3 / MTE3 -> V: the output value/index block is ready / drained.
        static constexpr uint32_t OUTPUT_READY = 0;
        static constexpr uint32_t OUTPUT_DRAINED = 0;
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
    // Every device function takes the UB plan as its first argument and scalars after
    // it; the plan is the only piece of kernel state that is not a compile-time fact.
    template <bool IS_INITIALIZE>
    static __simd_callee__ void accumulate_nan_lane_count(__ubuf__ Ub& ub, __ubuf__ bfloat16_t* values, uint32_t count);

    static __simd_callee__ void restore_index(__ubuf__ Ub& ub, vector_uint16_t index_low, vector_uint16_t index_high, uint32_t offset, int32_t output_idx_offset, vector_bool mask);

    template <bool DO_RESTORE_INDEX>
    static __simd_vf__ void prepare_index_vf(__ubuf__ Ub& ub, uint32_t topk, uint32_t valid_count, int32_t output_idx_offset);

    static __simd_vf__ void fill_nan_sentinel_vf(__ubuf__ Ub& ub, __ubuf__ int32_t* destination, uint32_t count);

    template <bool IS_INITIALIZE>
    static __simd_vf__ void build_high_histogram_vf(__ubuf__ Ub& ub, __ubuf__ bfloat16_t* values, uint32_t count, uint32_t keep_count);

    static __simd_vf__ void get_sorted_value_vf(__ubuf__ Ub& ub, uint32_t topk);

    template <bool IS_WARMUP, bool GENERATE_INDEX = false, uint32_t GENERATED_INDEX_CAPACITY = 0, bool IS_FINALIZE = false>
    static __simd_vf__ void select_candidates_vf(__ubuf__ Ub& ub, uint32_t keep_count, uint32_t warmup_source_count, uint32_t filter_source_count, uint32_t filter_index_base, uint32_t filter_stage);

    template <bool RETURN_VALUE>
    static __simd_vf__ void prepare_short_row_vf(__ubuf__ Ub& ub, uint32_t valid_count, uint32_t topk, uint16_t value_fill_bits, int32_t index_fill);

    // ================= 11. Driver (defined in kernel_body.h) =================
    // The kernel entry point for one row. Defined out of class so that the free
    // per-row dispatch in kernel_body.h can pick the SHORT_SEQ/LONG_SEQ variant at runtime.
    static __aicore__ void run(const RowArgs& row);
};

}  // namespace topk_select_ascend
