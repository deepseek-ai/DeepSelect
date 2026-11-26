#pragma once

#include "kernel.h"

#include <cstdint>

#include <acl/acl.h>
#include <adv_api/sort/sort.h>
#include <c_api/asc_simd.h>
#include <kernel_operator.h>

#include <kerutils/kerutils_for_ascend_npu.h>

#include "config.h"

using namespace AscendC;

namespace topk_select_ascend {

/*
Accumulate a 256-bin bf16 byte histogram into two 128-lane vectors.
For the low byte, only values whose raw high byte matches `prefix` are counted;
`prefix` should be a distorted number. `DESCENDING` controls the bucket order:
- IS_HIGH_BYTE && DESCENDING: hist[i] += #[high_byte(distort(number)) >= ~i]
- !IS_HIGH_BYTE && !DESCENDING: hist[i] += #[low_byte(distort(number)) <= i]
*/
template <bool IS_HIGH_BYTE, bool DESCENDING = false>
__simd_callee__ inline void accumulate_histogram(vector_uint16_t& hist_low,
                                                 vector_uint16_t& hist_high,
                                                 __ubuf__ bfloat16_t* values, uint32_t count,
                                                 vector_uint8_t prefix) {

    vector_bool mask_u8 = asc_create_mask_b8(PAT_ALL);
    vector_uint8_t positive_flip, negative_flip;
    asc_duplicate_scalar(positive_flip, uint8_t(DESCENDING ? 0x7f : 0x80), mask_u8);
    asc_duplicate_scalar(negative_flip, uint8_t(DESCENDING ? 0x00 : 0xff), mask_u8);
    vector_uint8_t raw_prefix;
    vector_bool negative_prefix;
    if constexpr (!IS_HIGH_BYTE) {
        static_assert(DESCENDING == false);
        asc_lt_scalar(negative_prefix, prefix, uint8_t(0x80), mask_u8);
        vector_uint8_t prefix_flip;
        asc_select(prefix_flip, negative_flip, positive_flip, negative_prefix);
        asc_xor(raw_prefix, prefix, prefix_flip, mask_u8);
    } else {
        static_assert(DESCENDING == true);
    }
    uint32_t remaining = count;
    const uint32_t rounds = ceil_div(count, HIST_BYTES_PER_ROUND);
    auto* source = reinterpret_cast<__ubuf__ uint16_t*>(values);
    #pragma unroll 1
    for (uint32_t round = 0; round < rounds; ++round) {
        vector_uint8_t low_byte, high_byte;
        asc_loadalign_deintlv_postupdate(
            low_byte, high_byte,
            reinterpret_cast<__ubuf__ uint8_t*&>(source),
            2*HIST_BYTES_PER_ROUND
        );

        vector_bool active = asc_update_mask_b8(remaining);
        if constexpr (IS_HIGH_BYTE) {
            // DESCENDING must be True
            vector_bool nonnegative;
            asc_ge_scalar(nonnegative, reinterpret_cast<vector_int8_t&>(high_byte), int8_t(0),
                        mask_u8);
            // No asc_* C-API exposes MODE_MERGING for vxor; keep the raw intrinsic.
            vxor(high_byte, high_byte, positive_flip, nonnegative, MODE_MERGING);
            asc_cumulative_histogram_bin0(hist_low, high_byte, active);
            asc_cumulative_histogram_bin1(hist_high, high_byte, active);
        } else {
            // DESCENDING must be False
            // No asc_* C-API exposes MODE_MERGING for vxor; keep the raw intrinsic.
            vxor(low_byte, low_byte, negative_flip, negative_prefix, MODE_MERGING);
            vector_bool selected;
            asc_eq(selected, high_byte, raw_prefix, active);
            asc_cumulative_histogram_bin0(hist_low, low_byte, selected);
            asc_cumulative_histogram_bin1(hist_high, low_byte, selected);
        }
    }
}

/*
Find the first bucket whose cumulative histogram count reaches `rank`, then broadcast its index
i.e. `bucket` is filled with the smallest i such that (hist_low .concat. hist_high)[i] >= rank.
`rank` should be a "unified" (all positions having the same value) vector register
*/
[[maybe_unused]]
__simd_callee__ inline void find_bucket(vector_uint16_t& bucket, vector_uint16_t hist_low,
                                        vector_uint16_t hist_high, vector_uint16_t rank) {
    vector_bool mask = asc_create_mask_b16(PAT_ALL);
    vector_bool found_low, found_high;
    asc_ge(found_low, hist_low, rank, mask);
    asc_ge(found_high, hist_high, rank, mask);

    vector_int16_t index_low_s16, index_high_s16;
    asc_arange(index_low_s16, int16_t(0));
    asc_arange(index_high_s16, int16_t(128));
    vector_uint16_t not_found, candidate_low, candidate_high, candidate;
    asc_duplicate_scalar(not_found, uint16_t(0xffff), mask);
    asc_select(candidate_low, reinterpret_cast<vector_uint16_t&>(index_low_s16), not_found, found_low);
    asc_select(candidate_high, reinterpret_cast<vector_uint16_t&>(index_high_s16), not_found, found_high);
    asc_min(candidate, candidate_low, candidate_high, mask);
    asc_reduce_min(bucket, candidate, mask);
    asc_duplicate(bucket, bucket, mask);
}


// Per-row runtime arguments
// `TopkSelectArgs` is row-agnostic. The per-row pointers and scalars are derived once
// in `run_topk_select_row()` and passed down as one value instead of nine parameters.
struct RowArgs {
    __gm__ bfloat16_t* row_in;
    __gm__ bfloat16_t* row_out_value;
    __gm__ int32_t* row_out_index;
    int32_t output_idx_offset;  // already read from args.output_idx_offset; 0 if absent
    uint32_t valid_len;
    uint32_t topk;
    bool return_value;
    bool abort_when_nan_found;
    int32_t idx_oob_fill;
    uint16_t value_oob_fill_bits;
};


/*
Count the lanes that contain a NaN in [values, values + count) and accumulate them into
`ub.state.nan_lane_count`. `count` may be any element count: the last vector is
masked with the remaining lanes.
*/
template <Config CONFIG, AlgoConfig ALGO_CONFIG>
template <bool IS_INITIALIZE>
__simd_callee__ void Kernel<CONFIG, ALGO_CONFIG>::accumulate_nan_lane_count(
    __ubuf__ Ub& ub, __ubuf__ bfloat16_t* values, uint32_t count) {
    __ubuf__ uint32_t* nan_lane_count = &ub.state.nan_lane_count;
    vector_bool full_mask = asc_create_mask_b16(PAT_ALL);
    vector_bool acc_is_nan_mask = asc_create_mask_b16(PAT_ALLF);
    uint32_t remaining = count;
    auto* values_ptr = values;
    #pragma unroll 1
    for (uint32_t offset = 0; offset < count; offset += NUM_BF16_PER_VEC) {
        vector_bool active_mask = asc_update_mask_b16(remaining);
        vector_bfloat16_t value;
        asc_loadalign_postupdate(value, values_ptr, static_cast<int32_t>(NUM_BF16_PER_VEC));
        vector_bool is_nan_mask;
        asc_ne(is_nan_mask, value, value, active_mask);
        asc_or(acc_is_nan_mask, acc_is_nan_mask, is_nan_mask, full_mask);
    }
    vector_uint16_t nan_as_u16;
    asc_duplicate_scalar(nan_as_u16, uint16_t(1), acc_is_nan_mask);
    vector_uint32_t current_nan_lane_count;
    asc_reduce_sum(current_nan_lane_count, nan_as_u16, full_mask);
    if constexpr (!IS_INITIALIZE) {
        vector_uint32_t old_nan_lane_count;
        asc_loadalign(old_nan_lane_count, nan_lane_count);
        asc_add(current_nan_lane_count, current_nan_lane_count, old_nan_lane_count, full_mask);
    }
    asc_storealign_1st(nan_lane_count, current_nan_lane_count);
}

// Reconstruct the position within the input row from the split virtual index, then
// apply output_idx_offset to produce the final output index:
//   x = index_low | (index_high << 16)
//   input_row_index = shuffle_page_ids[x >> SHUFFLE_PAGE_SIZE_BITS] * 128
//                     + x % shuffle_page_size
//   output_index = input_row_index + output_idx_offset
template <Config CONFIG, AlgoConfig ALGO_CONFIG>
__simd_callee__ void Kernel<CONFIG, ALGO_CONFIG>::restore_index(__ubuf__ Ub& ub,
                                                                vector_uint16_t index_low,
                                                                vector_uint16_t index_high,
                                                                uint32_t offset,
                                                                int32_t output_idx_offset,
                                                                vector_bool mask) {
    constexpr int16_t PAGE_ID_HIGH_SHIFT = 16 - SHUFFLE_PAGE_SIZE_BITS;
    __ubuf__ uint16_t* shuffle_page_ids = ub.shuffle_page_ids;
    __ubuf__ int32_t* destination = reinterpret_cast<__ubuf__ int32_t*>(ub.overlay.input_staging);
    vector_uint16_t page_in_low, page_from_high, virtual_page_id, physical_page_id;
    asc_shiftright_scalar(page_in_low, index_low,
                          static_cast<int16_t>(SHUFFLE_PAGE_SIZE_BITS), mask);
    asc_shiftleft_scalar(page_from_high, index_high, PAGE_ID_HIGH_SHIFT, mask);
    asc_or(virtual_page_id, page_from_high, page_in_low, mask);
    asc_gather(physical_page_id, shuffle_page_ids, virtual_page_id, mask);
    asc_shiftleft_scalar(index_low, index_low, PAGE_ID_HIGH_SHIFT, mask);
    asc_shiftright_scalar(index_low, index_low, PAGE_ID_HIGH_SHIFT, mask);

    vector_uint16_t zero, offset0, offset1, base0, base1;
    asc_duplicate_scalar(zero, uint16_t(0), mask);
    asc_intlv(offset0, offset1, index_low, zero);
    asc_intlv(base0, base1, physical_page_id, zero);

    vector_uint32_t result0, result1;
    asc_shiftleft_scalar(result0, reinterpret_cast<vector_uint32_t&>(base0), int16_t(7), asc_create_mask_b32(PAT_ALL));
    asc_shiftleft_scalar(result1, reinterpret_cast<vector_uint32_t&>(base1), int16_t(7), asc_create_mask_b32(PAT_ALL));
    asc_add(result0, result0, reinterpret_cast<vector_uint32_t&>(offset0), asc_create_mask_b32(PAT_ALL));
    asc_add(result1, result1, reinterpret_cast<vector_uint32_t&>(offset1), asc_create_mask_b32(PAT_ALL));
    vector_int32_t index_offset;
    asc_duplicate_scalar(index_offset, output_idx_offset, asc_create_mask_b32(PAT_ALL));
    asc_add(reinterpret_cast<vector_int32_t&>(result0), reinterpret_cast<vector_int32_t&>(result0),
         index_offset, asc_create_mask_b32(PAT_ALL));
    asc_add(reinterpret_cast<vector_int32_t&>(result1), reinterpret_cast<vector_int32_t&>(result1),
         index_offset, asc_create_mask_b32(PAT_ALL));
    vector_bool mask0, mask1;
    asc_intlv_b16(mask0, mask1, mask, asc_create_mask_b16(PAT_ALLF));
    asc_storealign(destination + offset, reinterpret_cast<vector_int32_t&>(result0), mask0);
    asc_storealign(destination + offset + 64, reinterpret_cast<vector_int32_t&>(result1), mask1);
}

template <Config CONFIG, AlgoConfig ALGO_CONFIG>
template <bool DO_RESTORE_INDEX>
__simd_vf__ void Kernel<CONFIG, ALGO_CONFIG>::prepare_index_vf(__ubuf__ Ub& ub, uint32_t topk,
                                                               uint32_t valid_count,
                                                               int32_t output_idx_offset) {
    __ubuf__ uint16_t* index_low = ub.candidate_index_low;
    __ubuf__ uint16_t* index_high = ub.candidate_index_high;
    // The short-row path leaves the offset-free prepared indices in the low-index field.
    __ubuf__ int32_t* prepared_index = reinterpret_cast<__ubuf__ int32_t*>(ub.candidate_index_low);
    // The input staging is idle by now, so the prepared indices can be staged there.
    __ubuf__ int32_t* destination = reinterpret_cast<__ubuf__ int32_t*>(ub.overlay.input_staging);
    if constexpr (DO_RESTORE_INDEX) {
        uint32_t remaining = topk;
        auto* index_low_ptr = index_low;
        auto* index_high_ptr = index_high;
        #pragma unroll 1
        for (uint32_t offset = 0; offset < topk; offset += NUM_BF16_PER_VEC) {
            vector_bool active = asc_update_mask_b16(remaining);
            vector_uint16_t low, high;
            asc_loadalign_postupdate(low, index_low_ptr, static_cast<int32_t>(NUM_BF16_PER_VEC));
            asc_loadalign_postupdate(high, index_high_ptr, static_cast<int32_t>(NUM_BF16_PER_VEC));
            restore_index(ub, low, high, offset, output_idx_offset, active);
        }
    } else {
        uint32_t remaining = topk;
        uint32_t valid_remaining = valid_count;
        auto* prepared_index_ptr = prepared_index;
        #pragma unroll 1
        for (uint32_t offset = 0; offset < topk; offset += NUM_U32_PER_VEC) {
            vector_bool active_mask = asc_update_mask_b32(remaining);
            vector_bool valid_mask = asc_update_mask_b32(valid_remaining);
            vector_int32_t index;
            asc_loadalign_postupdate(index, prepared_index_ptr,
                static_cast<int32_t>(NUM_U32_PER_VEC));
            vector_int32_t index_offset;
            asc_duplicate_scalar(index_offset, output_idx_offset, active_mask);
            // No asc_* C-API exposes MODE_MERGING for vadd; keep the raw intrinsic.
            vadd(index, index, index_offset, valid_mask, MODE_MERGING);
            asc_storealign_postupdate(destination, index, NUM_U32_PER_VEC, active_mask);
        }
    }
}

// Write the NaN sentinel index into every output lane. Only used when NaNs were detected
// and the caller asked not to abort, so both this VF and the extra UB->GM copy are
// exception-path costs; the common path never runs them.
template <Config CONFIG, AlgoConfig ALGO_CONFIG>
__simd_vf__ void Kernel<CONFIG, ALGO_CONFIG>::fill_nan_sentinel_vf(
    __ubuf__ Ub& ub, __ubuf__ int32_t* destination, uint32_t count) {
    vector_int32_t sentinel;
    vector_bool mask = asc_create_mask_b32(PAT_ALL);
    asc_duplicate_scalar(sentinel, int32_t(0x3F3F3F3F), mask);
    uint32_t remaining = count;
    #pragma unroll 1
    for (uint32_t offset = 0; offset < count; offset += NUM_U32_PER_VEC) {
        vector_bool active_mask = asc_update_mask_b32(remaining);
        asc_storealign_postupdate(destination, sentinel, NUM_U32_PER_VEC, active_mask);
    }
}

// Accumulate every value that has entered the buffer into the high-byte histogram
// without clearing it, which still yields the correct high-byte bucket. On each rebuild,
// only newly added values in the filtered buffer are accumulated. Clamp the counters to
// avoid overflow.
template <Config CONFIG, AlgoConfig ALGO_CONFIG>
template <bool IS_INITIALIZE>
__simd_vf__ void Kernel<CONFIG, ALGO_CONFIG>::build_high_histogram_vf(
    __ubuf__ Ub& ub, __ubuf__ bfloat16_t* values, uint32_t count, uint32_t keep_count) {
    __ubuf__ uint16_t* histogram = ub.high_histogram;
    vector_bool mask_u16 = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t hist_low, hist_high;
    if constexpr (IS_INITIALIZE) {
        asc_duplicate_scalar(hist_low, uint16_t(0), mask_u16);
        asc_duplicate_scalar(hist_high, uint16_t(0), mask_u16);
    } else {
        asc_loadalign(hist_low, histogram);
        asc_loadalign(hist_high, histogram, 128);
    }
    vector_uint8_t unused_prefix;
    accumulate_histogram<true, true>(hist_low, hist_high, values, count, unused_prefix);
    accumulate_nan_lane_count<IS_INITIALIZE>(ub, values, count);
    asc_min_scalar(hist_low, hist_low, static_cast<uint16_t>(keep_count + 1), mask_u16);
    asc_min_scalar(hist_high, hist_high, static_cast<uint16_t>(keep_count + 1), mask_u16);
    asc_storealign(histogram, hist_low, mask_u16);
    asc_storealign(histogram, hist_high, 128, mask_u16);
}

// Only instantiated for `SORTED_INDEX == true` tiers.
// Reorder the values according to the AscendC::Sort permutation, so that they line up
// with the sorted indices.
template <Config CONFIG, AlgoConfig ALGO_CONFIG>
__simd_vf__ void Kernel<CONFIG, ALGO_CONFIG>::get_sorted_value_vf(__ubuf__ Ub& ub, uint32_t topk) {
    __ubuf__ bfloat16_t* destination_value =
        reinterpret_cast<__ubuf__ bfloat16_t*>(ub.candidate_index_high);
    __ubuf__ bfloat16_t* source_value = ub.candidate_value;
    __ubuf__ uint16_t* permutation_ptr =
        reinterpret_cast<__ubuf__ uint16_t*>(ub.overlay.sort.permutation);
    uint32_t remaining = topk;
    #pragma unroll 1
    for (uint32_t offset = 0; offset < topk; offset += NUM_BF16_PER_VEC) {
        vector_bool mask = asc_update_mask_b16(remaining);
        vector_uint16_t position, unused_high_half;
        asc_loadalign_deintlv_postupdate(position, unused_high_half, permutation_ptr,
                                         2 * NUM_BF16_PER_VEC);
        vector_bfloat16_t value;
        asc_gather(value, source_value, position, mask);
        asc_storealign_postupdate(destination_value, value, NUM_BF16_PER_VEC, mask);
    }
}

/*
This VF has three call sites:
- During warmup, select the initial top-k candidates and establish the threshold.
- In regular scan rounds, append values satisfying !(value <= threshold) to the filtered
  buffer, and reduce the buffer back to k candidates (updating the threshold) when it
  exceeds FILTER_OVERFLOW_TRIGGER.
- On the last scan round, do the same and additionally finalize the selection.

This deliberately selects every NaN so it is retained and detected.
Split virtual indices are carried with the candidates; prepare_index_vf restores the real indices later.
*/
template <Config CONFIG, AlgoConfig ALGO_CONFIG>
template <bool IS_WARMUP, bool GENERATE_INDEX, uint32_t GENERATED_INDEX_CAPACITY, bool IS_FINALIZE>
__simd_vf__ void Kernel<CONFIG, ALGO_CONFIG>::select_candidates_vf(
    __ubuf__ Ub& ub, uint32_t keep_count, uint32_t warmup_source_count,
    uint32_t filter_source_count, uint32_t filter_index_base, uint32_t filter_stage) {
    constexpr uint32_t FILTER_OVERFLOW_TRIGGER = MAX_TOPK + FILTERED_CAPACITY;
    // Warmup and streaming rounds share the same source and destination: the candidate
    // value/low/high fields (during warmup the sampled pages live in the value field).
    __ubuf__ bfloat16_t* source_value = ub.candidate_value;
    __ubuf__ bfloat16_t* destination_value = ub.candidate_value;
    __ubuf__ uint16_t* destination_index_low = ub.candidate_index_low;
    __ubuf__ uint16_t* destination_index_high = ub.candidate_index_high;
    __ubuf__ uint16_t* temporary_position = ub.temporary_position;
    vector_bool mask_u16 = asc_create_mask_b16(PAT_ALL);

    uint32_t source_count = warmup_source_count;
    if constexpr (!IS_WARMUP) {
        source_count = ub.state.result_count;
    }
    // A scan round first appends the low indices (the low 2 bytes of each index) of values above
    // the threshold to the filtered buffer. All values in the same block share one high index, so
    // gather their values by low index, fill in that high index, and pad the result to 32B for
    // subsequent vector loads/stores.
    if constexpr (!IS_WARMUP) {
        const uint32_t destination_count = source_count;
        auto* index_low_output = destination_index_low + destination_count;
        vector_store_unalign index_low_state;
        asc_squeeze_and_storeunalign_init();
        vector_uint16_t threshold_bits;
        asc_loadalign_brc(threshold_bits, ub.threshold);
        __ubuf__ bfloat16_t* filter_source = ub.overlay.input_staging + filter_stage * BLOCK_SIZE;
        auto* filter_source_ptr = filter_source;

        vector_uint16_t index_low, packed_low;
        // `filter_index_base` is truncated to its low 16 bits on purpose: the high bits
        // travel separately in `index_high`, so the 16-bit arange wraps like the old code.
        asc_arange((vector_int16_t&)index_low, static_cast<int16_t>(filter_index_base));
        #pragma unroll 1
        for (uint32_t offset = 0; offset < filter_source_count; offset += NUM_BF16_PER_VEC) {
            vector_bfloat16_t value;
            asc_loadalign_postupdate(value, filter_source_ptr, NUM_BF16_PER_VEC);
            vector_bool selected;
            vector_bool less_equal;
            asc_le(less_equal, value, reinterpret_cast<vector_bfloat16_t&>(threshold_bits), mask_u16);
            asc_not(selected, less_equal, mask_u16);
            asc_squeeze_and_storeunalign(index_low_output, packed_low, index_low_state, index_low,
                                          selected);
            asc_add_scalar(index_low, index_low, NUM_BF16_PER_VEC, mask_u16);
        }
        asc_squeeze_and_storeunalign_finalize(index_low_output, index_low_state);
        asc_mem_bar(VST_VLD);
        auto* count_scratch = reinterpret_cast<__ubuf__ uint32_t*>(ub.histogram);
        // `asc_get_squeeze_status()` is `__aicore__`-only (PIPE_S), so the AR count is spilled
        // to UB by the VF-side wrapper and read back by the scalar code after the barriers.
        asc_storealign_squeeze_status(count_scratch);
        asc_mem_bar(VST_LD);

        source_count = destination_count + count_scratch[0] / sizeof(uint16_t);
        uint32_t remaining = source_count - destination_count;
        auto* gather_base = filter_source - static_cast<uint16_t>(filter_index_base);
        vector_uint16_t index_high;
        asc_duplicate_scalar(index_high, static_cast<uint16_t>(filter_index_base >> 16), mask_u16);

        auto* index_low_in_ptr = destination_index_low + destination_count;
        auto* destination_value_out_ptr = destination_value + destination_count;
        auto* destination_index_high_out_ptr = destination_index_high + destination_count;
        uint32_t offset = destination_count;
        for (; remaining >= NUM_BF16_PER_VEC;
             offset += NUM_BF16_PER_VEC, remaining -= NUM_BF16_PER_VEC) {
            vector_uint16_t current_index_low;
            vector_bfloat16_t gathered_value;
            asc_loadalign_postupdate(current_index_low, index_low_in_ptr, NUM_BF16_PER_VEC);
            asc_gather(gathered_value, gather_base, current_index_low, mask_u16);
            asc_storealign_postupdate(destination_value_out_ptr, gathered_value, NUM_BF16_PER_VEC, mask_u16);
            asc_storealign_postupdate(destination_index_high_out_ptr, index_high, NUM_BF16_PER_VEC, mask_u16);
        }
        if (remaining != 0) {
            uint32_t aligned_remaining = aligned(remaining, VST_VLD_ALIGN_ELEMENTS);
            vector_bool tail = asc_update_mask_b16(remaining);
            vector_bool aligned_tail = asc_update_mask_b16(aligned_remaining);
            vector_uint16_t current_index_low;
            vector_bfloat16_t gathered_value;
            asc_loadalign(current_index_low, destination_index_low, offset);
            asc_gather(gathered_value, gather_base, current_index_low, tail);
            vector_uint16_t padding_value;
            asc_duplicate_scalar(padding_value, uint16_t(0xff80), mask_u16);
            asc_select(reinterpret_cast<vector_uint16_t&>(gathered_value),
                 reinterpret_cast<vector_uint16_t&>(gathered_value), padding_value, tail);
            asc_storealign(destination_value, gathered_value, offset, aligned_tail);
            asc_storealign(destination_index_high, index_high, offset, tail);
            source_count = aligned(source_count, VST_VLD_ALIGN_ELEMENTS);
        }
    }
    if (IS_WARMUP || source_count > FILTER_OVERFLOW_TRIGGER || IS_FINALIZE) {
        if constexpr (!IS_WARMUP)
            asc_mem_bar(VST_VLD);
        vector_uint16_t hist_low, hist_high;
        // For radix selection, maintain a clamped high-byte histogram of all values scanned so far.
        // Each rebuild adds only the newly appended filtered values; values discarded from the
        // candidate buffer are intentionally not removed. This historical histogram determines the
        // global high-byte threshold bucket.
        asc_loadalign(hist_low, ub.high_histogram);
        asc_loadalign(hist_high, ub.high_histogram, TOPK_HISTOGRAM_BINS / 2);
        if constexpr (!IS_WARMUP) {
            const uint32_t histogram_begin = aligned(keep_count, VST_VLD_ALIGN_ELEMENTS);
            if (source_count > histogram_begin) {
                vector_uint8_t unused_prefix;
                accumulate_histogram<true, true>(
                    hist_low, hist_high, source_value + histogram_begin,
                    source_count - histogram_begin, unused_prefix);
                accumulate_nan_lane_count<false>(ub, source_value + histogram_begin,
                                                 source_count - histogram_begin);
                vector_uint16_t counter_limit;
                asc_duplicate_scalar(counter_limit, static_cast<uint16_t>(keep_count + 1), mask_u16);
                asc_min(hist_low, hist_low, counter_limit, mask_u16);
                asc_min(hist_high, hist_high, counter_limit, mask_u16);
                asc_storealign(ub.high_histogram, hist_low, mask_u16);
                asc_storealign(ub.high_histogram, hist_high, TOPK_HISTOGRAM_BINS / 2, mask_u16);
            }
        }

        // Radix-select the BF16 threshold from the high- and low-byte histograms, then compute how
        // many values equal to it must be kept. This runs only once per rebuild, so these SIMD
        // operations are not aggressively optimized.
        vector_uint16_t rank, high_bucket;
        asc_duplicate_scalar(rank, static_cast<uint16_t>(keep_count), mask_u16);
        find_bucket(high_bucket, hist_low, hist_high, rank);

        vector_uint16_t previous_count, zero;
        asc_duplicate_scalar(zero, uint16_t(0), mask_u16);
        vector_uint16_t one, lane_mask, scratch, bucket_lane;
        asc_duplicate_scalar(one, uint16_t(1), mask_u16);
        asc_duplicate_scalar(lane_mask, uint16_t(127), mask_u16);
        // Locate high_bucket - 1 in the two 128-lane histogram vectors.
        asc_add_scalar(bucket_lane, high_bucket, uint16_t(-1), mask_u16);
        asc_and(bucket_lane, bucket_lane, lane_mask, mask_u16);
        vector_bool bucket_condition;
        // Select its half and extract the cumulative count from its lane.
        asc_le_scalar(bucket_condition, high_bucket, uint16_t(128), mask_u16);
        asc_select(scratch, hist_low, hist_high, bucket_condition);
        asc_gather(previous_count, scratch, bucket_lane);
        // previous_count = high_bucket == 0 ? 0 : histogram[high_bucket - 1]
        asc_eq_scalar(bucket_condition, high_bucket, uint16_t(0), mask_u16);
        asc_select(previous_count, zero, previous_count, bucket_condition);
        // rank -= previous_count
        asc_sub(rank, rank, previous_count, mask_u16);
        // high_prefix = 0xff - high_bucket.  Note that `high_bucket` (the
        // descending rank key D_high) is rewritten in place into the ascending
        // key A_high = 0xff - D_high from here on.
        asc_duplicate_scalar(scratch, uint16_t(0xff), mask_u16);
        asc_sub(high_bucket, scratch, high_bucket, mask_u16);

        vector_uint8_t high_prefix, unused_high_byte;
        asc_deintlv(high_prefix, unused_high_byte, reinterpret_cast<vector_uint8_t&>(high_bucket),
                reinterpret_cast<vector_uint8_t&>(high_bucket));
        asc_xor(hist_low, hist_low, hist_low, mask_u16);    // clear hist_low and hist_high
        asc_xor(hist_high, hist_high, hist_high, mask_u16);
        accumulate_histogram<false>(hist_low, hist_high, source_value, source_count,
                                         high_prefix);

        vector_uint16_t low_bucket, threshold_key;
        vector_uint16_t high_bucket_count;
        // low_rank = hist_high[127] - rank + 1, where hist_high[127] is the
        // number of elements in the pivot high-byte bucket.
        // low_bucket = first b where the low-byte cumulative histogram >= low_rank
        asc_duplicate_highest(high_bucket_count, hist_high, mask_u16);
        asc_sub(rank, high_bucket_count, rank, mask_u16);
        asc_add(rank, rank, one, mask_u16);
        find_bucket(low_bucket, hist_low, hist_high, rank);
        vector_uint16_t pivot_low_count, greater_count, equal_limit;
        // greater_count = previous_count
        //               + (hist_high[127] - hist[low_bucket])   // strictly greater in the bucket
        // equal_limit = keep_count - greater_count
        asc_and(bucket_lane, low_bucket, lane_mask, mask_u16);
        asc_lt_scalar(bucket_condition, low_bucket, uint16_t(128), mask_u16);
        asc_select(scratch, hist_low, hist_high, bucket_condition);
        asc_gather(pivot_low_count, scratch, bucket_lane);

        asc_sub(rank, high_bucket_count, pivot_low_count, mask_u16);
        asc_add(greater_count, previous_count, rank, mask_u16);
        asc_duplicate_scalar(rank, static_cast<uint16_t>(keep_count), mask_u16);
        asc_sub(equal_limit, rank, greater_count, mask_u16);

        // threshold_key = (A_high << 8) | A_low, then un-distort the ascending
        // key back to the raw bf16 bit pattern: xor 0x8000 for a non-negative
        // threshold, 0xffff for a negative one.
        asc_shiftleft_scalar(threshold_key, high_bucket, int16_t(8), mask_u16);
        asc_or(threshold_key, threshold_key, low_bucket, mask_u16);
        asc_ge_scalar(bucket_condition, high_bucket, uint16_t(0x80), mask_u16);
        asc_duplicate_scalar(lane_mask, uint16_t(0x8000), mask_u16);
        asc_duplicate_scalar(scratch, uint16_t(0xffff), mask_u16);
        asc_select(scratch, lane_mask, scratch, bucket_condition);
        asc_xor(threshold_key, threshold_key, scratch, mask_u16);
        asc_storealign_1st(ub.threshold, threshold_key);

        vector_bool threshold_negative;
        asc_lt_scalar(threshold_negative, reinterpret_cast<vector_int16_t&>(threshold_key), int16_t(0),
                 mask_u16);
        vector_uint16_t compare_mask, negative_compare_mask;
        asc_duplicate_scalar(compare_mask, uint16_t(0), mask_u16);
        asc_duplicate_scalar(negative_compare_mask, uint16_t(0x7fff), mask_u16);
        asc_select(compare_mask, negative_compare_mask, compare_mask, threshold_negative);
        vector_uint16_t threshold_compare;
        asc_xor(threshold_compare, threshold_key, compare_mask, mask_u16);

        // Collect the positions of k candidates from the top-k/filtered buffer: all values above
        // the threshold and the required number equal to it. These positions are then used to
        // gather the values and their low/high indices.
        vector_uint16_t equal_seen = zero;
        auto* position_output = temporary_position;
        vector_store_unalign position_state;
        asc_squeeze_and_storeunalign_init();
        uint32_t remaining = source_count;
        auto* source_value_ptr = source_value;
        vector_uint16_t position;
        asc_arange(reinterpret_cast<vector_int16_t&>(position), static_cast<int16_t>(0));
        #pragma unroll 1
        for (uint32_t offset = 0; offset < source_count; offset += NUM_BF16_PER_VEC) {
            vector_bool active = asc_update_mask_b16(remaining);
            vector_bfloat16_t value;
            asc_loadalign_postupdate(value, source_value_ptr,
                                     static_cast<int32_t>(NUM_BF16_PER_VEC));
            vector_uint16_t value_compare;
            // Compare value > threshold (with +0 > -0). Let mask = threshold >= 0 ? 0 : 0x7fff,
            // value_compare = reinterpret<int16>(value) ^ mask, and
            // threshold_compare = reinterpret<int16>(threshold) ^ mask. For finite BF16 values,
            // value > threshold iff value_compare > threshold_compare, with +0 ordered above -0.
            asc_xor(value_compare, reinterpret_cast<vector_uint16_t&>(value), compare_mask, active);
            vector_bool is_greater, is_equal;
            asc_gt(is_greater, reinterpret_cast<vector_int16_t&>(value_compare),
                    reinterpret_cast<vector_int16_t&>(threshold_compare), active);
            asc_eq(is_equal, reinterpret_cast<vector_int16_t&>(value_compare),
                    reinterpret_cast<vector_int16_t&>(threshold_compare), active);
            vector_uint16_t equal_prefix = zero;
            asc_unsqueeze(equal_prefix, is_equal);
            // No asc_* C-API exposes MODE_MERGING for vadd; keep the raw intrinsic.
            vadd(equal_prefix, equal_prefix, one, is_equal, MODE_MERGING);
            asc_add(equal_prefix, equal_prefix, equal_seen, mask_u16);
            vector_bool take_equal;
            asc_ge(take_equal, equal_limit, equal_prefix, is_equal);
            vector_bool selected;
            asc_or(selected, is_greater, take_equal, active);
            vector_uint16_t packed_position;
            asc_squeeze_and_storeunalign(position_output, packed_position, position_state, position,
                                          selected);
            asc_duplicate_highest(equal_seen, equal_prefix, mask_u16);
            asc_add_scalar(position, position, NUM_BF16_PER_VEC, mask_u16);
        }
        asc_squeeze_and_storeunalign_finalize(position_output, position_state);
        asc_mem_bar(VST_VLD);

        remaining = keep_count;
        vector_uint16_t index_high;
        if constexpr (GENERATE_INDEX) {
            static_assert(GENERATED_INDEX_CAPACITY <= (uint32_t(1) << 16),
                          "generated virtual index high must be zero");
            asc_duplicate_scalar(index_high, uint16_t(0), mask_u16);
        }
        auto* temporary_position_ptr = temporary_position;
        auto* keep_destination_value_ptr = destination_value;
        auto* keep_destination_index_low_ptr = destination_index_low;
        auto* keep_destination_index_high_ptr = destination_index_high;
        for (uint32_t offset = 0; offset < keep_count; offset += NUM_BF16_PER_VEC) {
            vector_bool mask = asc_update_mask_b16(remaining);
            vector_uint16_t index_low;
            asc_loadalign_postupdate(index_low, temporary_position_ptr,
                                     static_cast<int32_t>(NUM_BF16_PER_VEC));
            vector_bfloat16_t value;
            asc_gather(value, source_value, index_low, mask);
            if (offset + NUM_BF16_PER_VEC > keep_count) {
                vector_int16_t lane;
                asc_arange(lane, static_cast<int16_t>(offset));
                vector_bool out_of_bounds;
                asc_ge_scalar(out_of_bounds, lane, static_cast<int16_t>(keep_count), mask_u16);
                vector_uint16_t padding;
                asc_duplicate_scalar(padding, uint16_t(0x7fff), mask_u16);
                asc_select(reinterpret_cast<vector_uint16_t&>(value), padding,
                     reinterpret_cast<vector_uint16_t&>(value), out_of_bounds);
            }
            asc_storealign_postupdate(keep_destination_value_ptr, value,
                                      static_cast<int32_t>(NUM_BF16_PER_VEC), mask_u16);
            if constexpr (!GENERATE_INDEX) {
                asc_gather(index_high, ub.candidate_index_high, index_low, mask);
                asc_gather(index_low, ub.candidate_index_low, index_low, mask);
            }
            asc_storealign_postupdate(keep_destination_index_low_ptr, index_low,
                                      static_cast<int32_t>(NUM_BF16_PER_VEC), mask);
            asc_storealign_postupdate(keep_destination_index_high_ptr, index_high,
                                      static_cast<int32_t>(NUM_BF16_PER_VEC), mask);
        }
        const uint32_t aligned_keep_count = aligned(keep_count, VST_VLD_ALIGN_ELEMENTS);
        const uint32_t aligned_offset =
            keep_count / VST_VLD_ALIGN_ELEMENTS * VST_VLD_ALIGN_ELEMENTS;
        uint32_t valid_count = keep_count - aligned_offset;
        uint32_t store_count = aligned_keep_count - aligned_offset;
        vector_bool valid_mask = asc_update_mask_b16(valid_count);
        vector_bool store_mask = asc_update_mask_b16(store_count);
        vector_uint16_t value, padding_value;
        asc_mem_bar(VST_VLD);
        asc_loadalign(value, reinterpret_cast<__ubuf__ uint16_t*>(destination_value) + aligned_offset);
        asc_duplicate_scalar(padding_value, uint16_t(0xffff), asc_create_mask_b16(PAT_ALL));
        asc_select(value, value, padding_value, valid_mask);
        asc_storealign(reinterpret_cast<__ubuf__ uint16_t*>(destination_value) + aligned_offset, value, store_mask);
        if constexpr (!IS_WARMUP)
            source_count = aligned_keep_count;
    }
    if constexpr (!IS_WARMUP)
        ub.state.result_count = source_count;
}

// Prepare the output directly when the number of valid elements does not exceed top-k.
template <Config CONFIG, AlgoConfig ALGO_CONFIG>
template <bool RETURN_VALUE>
__simd_vf__ void Kernel<CONFIG, ALGO_CONFIG>::prepare_short_row_vf(__ubuf__ Ub& ub,
                                                                   uint32_t valid_count,
                                                                   uint32_t topk,
                                                                   uint16_t value_fill_bits,
                                                                   int32_t index_fill) {
    vector_bool full_mask = asc_create_mask_b16(PAT_ALL);
    accumulate_nan_lane_count<true>(ub, ub.overlay.input_staging, valid_count);
    if constexpr (RETURN_VALUE) {
        __ubuf__ bfloat16_t* source_ptr = ub.overlay.input_staging;
        __ubuf__ bfloat16_t* destination_value_out_ptr = ub.candidate_value;
        vector_uint16_t index;
        vector_bfloat16_t full_fill_value_vec;
        asc_arange((vector_int16_t&)index, 0);
        asc_duplicate_scalar((vector_uint16_t&)full_fill_value_vec, value_fill_bits);
        #pragma unroll 1
        for (uint32_t offset = 0; offset < topk; offset += NUM_BF16_PER_VEC) {
            vector_bfloat16_t value;
            asc_loadalign_postupdate(value, source_ptr, NUM_BF16_PER_VEC);
            vector_bool need_fill_mask;
            asc_ge_scalar(need_fill_mask, index, valid_count, full_mask);
            asc_add_scalar(index, index, NUM_BF16_PER_VEC, full_mask);
            asc_select(value, full_fill_value_vec, value, need_fill_mask);
            asc_storealign_postupdate(destination_value_out_ptr, value,
                                      NUM_BF16_PER_VEC, full_mask);
        }
    }

    __ubuf__ int32_t* destination_index =
        reinterpret_cast<__ubuf__ int32_t*>(ub.candidate_index_low);
    auto* destination_index_out_ptr = destination_index;
    vector_bool full_mask_b32 = asc_create_mask_b32(PAT_ALL);
    vector_int32_t index_lane;
    vector_int32_t full_fill_index_vec;
    asc_arange(index_lane, 0);
    asc_duplicate_scalar(full_fill_index_vec, index_fill, full_mask_b32);
    #pragma unroll 1
    for (uint32_t offset = 0; offset < topk; offset += NUM_U32_PER_VEC) {
        vector_int32_t index;
        vector_bool need_fill_mask;
        asc_ge_scalar(need_fill_mask, index_lane, static_cast<int32_t>(valid_count),
                      full_mask_b32);
        asc_select(index, full_fill_index_vec, index_lane, need_fill_mask);
        asc_add_scalar(index_lane, index_lane, static_cast<int32_t>(NUM_U32_PER_VEC),
                       full_mask_b32);
        asc_storealign_postupdate(destination_index_out_ptr, index,
                                  static_cast<int32_t>(NUM_U32_PER_VEC), full_mask_b32);
    }
}


template <Config CONFIG, AlgoConfig ALGO_CONFIG>
__aicore__ void Kernel<CONFIG, ALGO_CONFIG>::run(const RowArgs& row) {
    // Unpack the per-row arguments once, so the body below reads like the algorithm.
    const uint32_t valid_len = row.valid_len;
    const uint32_t topk = row.topk;
    __gm__ bfloat16_t* row_in = row.row_in;
    __gm__ bfloat16_t* row_out_value = row.row_out_value;
    __gm__ int32_t* row_out_index = row.row_out_index;
    const int32_t output_idx_offset = row.output_idx_offset;
    const bool return_value = row.return_value;
    const int32_t idx_oob_fill = row.idx_oob_fill;
    const uint16_t value_oob_fill_bits = row.value_oob_fill_bits;

    // The UB plan is a lens over raw UB: base address 0, so offsets are addresses.
    // Two steps instead of one so the compiler does not warn about the null
    // dereference: the "pointer" is never really dereferenced, it only carries the type.
    __ubuf__ Ub* ub_ptr = reinterpret_cast<__ubuf__ Ub*>(0);
    __ubuf__ Ub& ub = *ub_ptr;

    // V -> MTE2: the stage buffer has been consumed and MTE2 may refill it.
    event_t stage_free_events[NUM_STAGES];
    event_t warmup_events[NUM_WARMUP_PAGES];
    for (uint32_t stage = 0; stage < NUM_STAGES; ++stage) {
        stage_free_events[stage] = event_t(Flags::INPUT_STAGE_FREE_BASE + stage);
        // Prime the free counters so the first round's wait_flag() can pair with a set.
        set_flag(PIPE_V, PIPE_MTE2, stage_free_events[stage]);
    }
    // MTE2 -> V: one named ID per warmup page, in a range disjoint from the stages.
    for (uint32_t page = 0; page < NUM_WARMUP_PAGES; ++page)
        warmup_events[page] = event_t(Flags::WARMUP_PAGE_FULL_BASE + page);

    if (valid_len <= topk) {
        if (valid_len != 0) {
            static_assert(MAX_TOPK * sizeof(bfloat16_t) <= sizeof(ub.overlay.input_staging));
            wait_flag(PIPE_V, PIPE_MTE2, stage_free_events[0]);
            copy_gm_to_ubuf_align_v2(ub.overlay.input_staging, row_in, 0, 1,
                                     valid_len * sizeof(bfloat16_t), 0, 0, false,
                                     static_cast<uint8_t>(
                                         LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN),
                                     valid_len * sizeof(bfloat16_t),
                                     valid_len * sizeof(bfloat16_t));
            event_t stage_full = event_t(Flags::INPUT_STAGE_FULL_BASE);
            set_flag(PIPE_MTE2, PIPE_V, stage_full);
            wait_flag(PIPE_MTE2, PIPE_V, stage_full);
        }
        if (return_value)
            prepare_short_row_vf<true>(ub, valid_len, topk, value_oob_fill_bits,
                                            static_cast<int32_t>(idx_oob_fill));
        else
            prepare_short_row_vf<false>(ub, valid_len, topk, value_oob_fill_bits,
                                             static_cast<int32_t>(idx_oob_fill));
        if (valid_len != 0)
            set_flag(PIPE_V, PIPE_MTE2, stage_free_events[0]);
    } else {
        const uint32_t num_shuffle_pages = ceil_div(valid_len, SHUFFLE_PAGE_SIZE);
        ascendc_assert(num_shuffle_pages <= MAX_SHUFFLE_PAGES, "topk shuffle page map overflow");
        const uint64_t input_hash = ld_dev(reinterpret_cast<__gm__ uint64_t*>(row_in), int16_t(0));
        const uint64_t permutation_seed = input_hash ^ uint64_t(valid_len);
        const ShufflePermutationGenerator<NUM_WARMUP_PAGES, SHUFFLE_PAGE_SIZE> permutation{
            valid_len, permutation_seed};
        const uint32_t num_warmup_pages =
            num_shuffle_pages < NUM_WARMUP_PAGES ? num_shuffle_pages : NUM_WARMUP_PAGES;

        uint32_t num_warmup_elements = 0;
        #pragma unroll 1
        for (uint32_t logical_page_id = 0; logical_page_id < num_warmup_pages; ++logical_page_id) {
            const ShufflePage physical_page = permutation.get_page_id(logical_page_id);
            const uint32_t page_offset = physical_page.start_offset;
            const uint32_t page_count = physical_page.len;
            auto* warmup_destination = ub.candidate_value + logical_page_id * SHUFFLE_PAGE_SIZE;
            copy_gm_to_ubuf_align_v2(warmup_destination, row_in + page_offset, 0, 1,
                                     page_count * sizeof(bfloat16_t), 0, 0, false,
                                     static_cast<uint8_t>(
                                         LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN),
                                     page_count * sizeof(bfloat16_t),
                                     page_count * sizeof(bfloat16_t));
            set_flag(PIPE_MTE2, PIPE_V, warmup_events[logical_page_id]);
            wait_flag(PIPE_MTE2, PIPE_V, warmup_events[logical_page_id]);
            if (logical_page_id == 0)
                build_high_histogram_vf<true>(ub, warmup_destination, page_count, topk);
            else
                build_high_histogram_vf<false>(ub, warmup_destination, page_count, topk);
            ub.shuffle_page_ids[logical_page_id] = static_cast<uint16_t>(page_offset / 128);
            num_warmup_elements += page_count;
        }

        const uint32_t stable_pages = num_shuffle_pages - num_warmup_pages;
        const uint32_t stable_blocks = ceil_div(stable_pages, PAGES_PER_BLOCK);
        // `IS_WARMUP` already forces the radix rebuild below, so there is no
        // separate "finalize" variant to pick here.
        select_candidates_vf<true, true, WARMUP_ELEMENTS>(ub, topk, num_warmup_elements,
                                                                    0, 0, 0);
        ub.state.result_count = aligned(topk, VST_VLD_ALIGN_ELEMENTS);
        for (uint32_t stable_idx = 0; stable_idx < stable_blocks; ++stable_idx) {
            const uint32_t stage = stable_idx % NUM_STAGES;
            wait_flag(PIPE_V, PIPE_MTE2, stage_free_events[stage]);
            auto* block_destination = ub.overlay.input_staging + stage * BLOCK_SIZE;
            uint32_t count = 0;
            for (uint32_t page = 0; page < PAGES_PER_BLOCK; ++page) {
                const uint32_t logical_page_id =
                    num_warmup_pages + stable_idx * PAGES_PER_BLOCK + page;
                if (logical_page_id >= num_shuffle_pages)
                    break;
                const ShufflePage physical_page = permutation.get_page_id(logical_page_id);
                const uint32_t page_offset = physical_page.start_offset;
                const uint32_t page_count = physical_page.len;
                copy_gm_to_ubuf_align_v2(
                    block_destination + page * SHUFFLE_PAGE_SIZE, row_in + page_offset, 0, 1,
                    page_count * sizeof(bfloat16_t), 0, 0, false,
                    static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN),
                    page_count * sizeof(bfloat16_t), page_count * sizeof(bfloat16_t));
                // Block-aligned virtual page id (see SHUFFLE_PAGE_CAPACITY).
                const uint32_t virtual_page_id = (stable_idx + 1) * PAGES_PER_BLOCK + page;
                ub.shuffle_page_ids[virtual_page_id] =
                    static_cast<uint16_t>(page_offset / 128);
                count += page_count;
            }
            event_t stage_full = event_t(Flags::INPUT_STAGE_FULL_BASE + stage);
            set_flag(PIPE_MTE2, PIPE_V, stage_full);
            wait_flag(PIPE_MTE2, PIPE_V, stage_full);
            // Keep virtual indices block-aligned so one filtered block never crosses
            // a 64K boundary and can share one index-high value.
            const uint32_t index_base = (stable_idx + 1) * BLOCK_SIZE;
            const bool finalize = stable_idx + 1 == stable_blocks;
            if (finalize)
                select_candidates_vf<false, false, 0, true>(ub, topk, 0, count, index_base,
                                                                stage);
            else
                select_candidates_vf<false, false, 0, false>(ub, topk, 0, count, index_base, stage);
            set_flag(PIPE_V, PIPE_MTE2, stage_free_events[stage]);
        }
    }

    const bool do_restore_index = valid_len > topk;
    const bool sort_index = topk > 1 && valid_len > topk &&
                            ceil_div(valid_len, SHUFFLE_PAGE_SIZE) > NUM_WARMUP_PAGES;
    // Prepared indices are staged in the (now idle) input staging region.
    __ubuf__ int32_t* output_index =
        reinterpret_cast<__ubuf__ int32_t*>(ub.overlay.input_staging);
    if (do_restore_index)
        prepare_index_vf<true>(ub, topk, min(valid_len, topk), output_idx_offset);
    else
        prepare_index_vf<false>(ub, topk, min(valid_len, topk), output_idx_offset);

    // The candidate value field is what gets written out; sorted-index tiers replace it
    // with the values gathered in index order (into the borrowed high-index field).
    __ubuf__ bfloat16_t* output_value = ub.candidate_value;
    if constexpr (SORTED_INDEX) {
        if (sort_index) {
            LocalTensor<int32_t> source_index_tensor(TPosition::VECCALC, INPUT_OFFSET,
                                                     MAX_TOPK);
            LocalTensor<int32_t> sorted_index_tensor(TPosition::VECCALC, Ub::SORTED_INDEX_OFFSET,
                                                     MAX_TOPK);
            LocalTensor<uint32_t> permutation_tensor(TPosition::VECCALC,
                                                     Ub::SORT_PERMUTATION_OFFSET,
                                                     MAX_TOPK);
            LocalTensor<uint8_t> temporary_tensor(TPosition::VECCALC, Ub::SORT_TEMPORARY_OFFSET,
                                                  SORT_TEMPORARY_BYTES);
            AscendC::Sort<int32_t, true, DEFAULT_SORT_CONFIG>(
                sorted_index_tensor, permutation_tensor, source_index_tensor, temporary_tensor,
                topk);
            if (return_value) {
                get_sorted_value_vf(ub, topk);
                output_value = reinterpret_cast<__ubuf__ bfloat16_t*>(ub.candidate_index_high);
            }
            output_index = reinterpret_cast<__ubuf__ int32_t*>(Ub::SORTED_INDEX_OFFSET);
        }
    }

    event_t output_ready = event_t(Flags::OUTPUT_READY);
    set_flag(PIPE_V, PIPE_MTE3, output_ready);
    wait_flag(PIPE_V, PIPE_MTE3, output_ready);
    if (return_value) {
        copy_ubuf_to_gm_align_v2(
            row_out_value, output_value, 0, 1, topk * sizeof(bfloat16_t),
            static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
            topk * sizeof(bfloat16_t), topk * sizeof(bfloat16_t));
    }
    copy_ubuf_to_gm_align_v2(
        row_out_index, output_index, 0, 1, topk * sizeof(int32_t),
        static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
        topk * sizeof(int32_t), topk * sizeof(int32_t));

    // Decide on NaN only after every launch has been issued: the scalar core has nothing
    // left to submit, so the V->S wait does not leave the vector pipeline idle. Checking
    // earlier would serialize the common path behind the wait for ~4% on small rows.
    event_t nan_lane_count_ready = event_t(Flags::NAN_LANE_COUNT_READY);
    set_flag(PIPE_V, PIPE_S, nan_lane_count_ready);
    wait_flag(PIPE_V, PIPE_S, nan_lane_count_ready);
    const bool has_nan = ub.state.nan_lane_count != 0;
    if (has_nan) {
        if (row.abort_when_nan_found) {
            printf("[topk_select] NaN detected at batch index %u\n", get_block_idx());
            trap();
        } else {
            // Exception path: the index output in GM was already written with the regular
            // indices, so overwrite it with the NaN sentinel and copy it again. The first copy may
            // still be reading `output_index`, so drain every pipe before rewriting it; the
            // barrier sits inside this branch, so the common path never pays for it.
            PipeBarrier<PIPE_ALL>();
            fill_nan_sentinel_vf(ub, output_index, topk);
            event_t sentinel_ready = event_t(Flags::OUTPUT_READY);
            set_flag(PIPE_V, PIPE_MTE3, sentinel_ready);
            wait_flag(PIPE_V, PIPE_MTE3, sentinel_ready);
            copy_ubuf_to_gm_align_v2(
                row_out_index, output_index, 0, 1, topk * sizeof(int32_t),
                static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
                topk * sizeof(int32_t), topk * sizeof(int32_t));
        }
    }
}

// ---------------------------------------------------------------------------
// Per-row dispatch, __global__ wrapper and host-side launch
// ---------------------------------------------------------------------------
// One row = one block. The row is routed to the SHORT_SEQ or LONG_SEQ algorithm variant
// by SHORT_LONG_SEQ_SPLIT_POINT; both run the same algorithm with different parameters.
template <Config CONFIG>
__aicore__ inline void run_topk_select_row(const TopkSelectArgs& args) {
    AscendC::InitSocState();

    const uint32_t row_idx = get_block_idx();
    RowArgs row;
    row.valid_len =
        args.end_ptr == nullptr ? args.vocab_size : static_cast<uint32_t>(args.end_ptr[row_idx]);
    row.topk = args.topk;
    row.abort_when_nan_found = args.abort_when_nan_found;
    row.row_in =
        reinterpret_cast<__gm__ bfloat16_t*>(args.input) + row_idx * args.stride_input_batch;
    row.row_out_value =
        args.return_value
            ? reinterpret_cast<__gm__ bfloat16_t*>(args.output_value) +
                  row_idx * args.stride_output_value_batch
            : nullptr;
    row.row_out_index = reinterpret_cast<__gm__ int32_t*>(args.output_index) +
                        row_idx * args.stride_output_index_batch;
    row.output_idx_offset = args.output_idx_offset == nullptr ? 0 : args.output_idx_offset[row_idx];
    row.return_value = args.return_value;
    row.idx_oob_fill = args.idx_oob_fill_value;
    row.value_oob_fill_bits = args.value_oob_fill_bits;

    if (row.valid_len <= SHORT_LONG_SEQ_SPLIT_POINT<CONFIG>)
        Kernel<CONFIG, SHORT_SEQ_ALGO<CONFIG>>::run(row);
    else
        Kernel<CONFIG, LONG_SEQ_ALGO<CONFIG>>::run(row);
}

template <Config CONFIG>
__global__ __vector__ __aicore__ void topk_select_kernel(const TopkSelectArgs args) {
    run_topk_select_row<CONFIG>(args);
}

// Host-side launch. Runtime invariants that would otherwise silently produce wrong
// output (a mismatched dispatch, for instance) are checked here, not in the kernel body.
template <Config CONFIG>
void run_topk_select_kernel(const TopkSelectArgs& args) {
    KU_ASSERT_NPU(args.topk <= CONFIG.MAX_TOPK,
                  "topk (%u) exceeds the compiled MAX_TOPK (%u): dispatch failure", args.topk,
                  CONFIG.MAX_TOPK);
    topk_select_kernel<CONFIG><<<args.batch_size, TOPK_UB_LIMIT_BYTES, args.stream>>>(args);
    KU_ACLRT_CHECK_NPU(aclrtGetLastError(aclrtLastErrLevel::ACL_RT_THREAD_LEVEL));
}

}  // namespace topk_select_ascend
