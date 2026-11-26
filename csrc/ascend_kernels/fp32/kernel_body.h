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

namespace topk_select_ascend_fp32 {

// `Sort` needs the descending configuration as a compile-time template argument. The merge path
// natively sorts descending and has no scatter pass, so it is the cheaper choice for the fp32
// values; the radix path is kept for the int32 index sort, which the merge path does not accept.
inline constexpr SortConfig SORT_MERGE_DESCENDING{SortType::MERGE_SORT, /*isDescend=*/true};

// Pad the candidate buffer with -inf: it is below every real value, so no radix pass can
// select a padded lane. (A NaN bit pattern would be preferred by the total order on the
// negative side, and it would also have to be kept out of the NaN count.)
inline constexpr uint32_t VALUE_PADDING_BITS = 0xff800000u;

/*
Radix digits of an fp32 bit pattern.

The monotone key of an fp32 value is
    A(bits) = bits ^ (sign ? 0xffffffff : 0x80000000)      // ascending key
which orders -inf < ... < -0 < +0 < ... < +inf. The radix select only ever needs one byte
of the key at a time, and inside one pivot bucket every candidate has the same sign, so each
pass works on a raw byte plane with a single xor:

- pass 1 feeds the *descending* top byte D3 = 0xff - A3, i.e. `bits3 ^ (nonnegative ? 0x7f : 0)`,
  so that the cumulative histogram directly counts "#candidates whose bucket >= i", and
  `find_bucket` with rank k yields the bucket holding the k-th largest;
- passes 2..4 feed A_byte = `bits_byte ^ (negative ? 0xff : 0)`, so the cumulative histogram
  counts "#candidates in the pivot bucket whose byte <= i" and the rank has to be mirrored
  (`fold_radix_pass`).

The CUDA original of this decomposition is `compute_pivot_and_quota` in
csrc/cuda_kernels/v3_fp32/topk_select.cuh.
*/

// Split the two 16-bit halves of 256 elements into byte planes. One `asc_deintlv` on the u8 view
// of two u16 registers separates their even bytes (the low byte of every element) from their odd
// bytes (the high byte), so one instruction yields both planes of a 16-bit half.
__simd_callee__ inline void split_byte_planes(vector_uint8_t& low_byte, vector_uint8_t& high_byte,
                                              vector_uint16_t pair0, vector_uint16_t pair1) {
    asc_deintlv(low_byte, high_byte, reinterpret_cast<vector_uint8_t&>(pair0),
                reinterpret_cast<vector_uint8_t&>(pair1));
}

// Smallest bucket index whose cumulative histogram count reaches `rank`, broadcast to every
// lane: `bucket = min { i : hist[i] >= rank }`. `hist_low`/`hist_high` are the two 128-lane
// halves of the 256-bin histogram, in bin order.
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

// Broadcast `hist[bucket]` of a 256-bin cumulative histogram held in two 128-lane halves.
// `bucket` must be a broadcast register.
__simd_callee__ inline void histogram_lookup(vector_uint16_t& count, vector_uint16_t hist_low,
                                            vector_uint16_t hist_high, vector_uint16_t bucket) {
    vector_bool mask = asc_create_mask_b16(PAT_ALL);
    vector_bool in_low_half;
    asc_lt_scalar(in_low_half, bucket, uint16_t(128), mask);
    vector_uint16_t half, lane_mask, lane;
    asc_select(half, hist_low, hist_high, in_low_half);
    asc_duplicate_scalar(lane_mask, uint16_t(127), mask);
    asc_and(lane, bucket, lane_mask, mask);
    asc_gather(count, half, lane);
}

/*
Fold one radix refinement pass (radix passes 2..4) into the pivot state.

`hist_*` is the cumulative histogram of the pivot-prefix bucket's byte for this pass, `rank` is
the number of candidates still to be taken from the current prefix bucket counted from the
largest, and `prefix_bits` accumulates one raw byte per call until it holds the exact bit
pattern of the k-th largest value.

Two quantities come out of the pass:
- `greater_count` grows by the candidates this byte proves to be strictly above the bucket;
- `rank` is re-derived as `keep_count - greater_count`, i.e. the number of candidates still
  missing below the pivot. After the last pass it *is* the equal-element quota.
*/
__simd_callee__ inline void fold_radix_pass(__ubuf__ uint16_t* pivot_scratch,
                                            vector_uint16_t hist_low, vector_uint16_t hist_high,
                                            bool negative, uint16_t keep_count, uint32_t byte_shift,
                                            uint32_t& prefix_bits, vector_uint16_t& rank,
                                            vector_uint16_t& greater_count) {
    vector_bool mask = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t bucket_count, keep, low_rank, bucket, pivot_count, delta, raw_byte;
    asc_duplicate_scalar(keep, keep_count, mask);
    // hist[255] = number of candidates that share the pivot prefix, i.e. the bucket size.
    asc_duplicate_highest(bucket_count, hist_high, mask);
    // The low bytes are histogrammed as ascending keys, so the rank counted from the largest
    // becomes a rank counted from the smallest inside the pivot bucket.
    asc_sub(low_rank, bucket_count, rank, mask);
    asc_add_scalar(low_rank, low_rank, uint16_t(1), mask);
    find_bucket(bucket, hist_low, hist_high, low_rank);
    histogram_lookup(pivot_count, hist_low, hist_high, bucket);
    asc_sub(delta, bucket_count, pivot_count, mask);
    asc_add(greater_count, greater_count, delta, mask);
    asc_sub(rank, keep, greater_count, mask);
    // `bucket` is the ascending byte key; undo the negative-lane inversion to get the raw byte.
    vector_uint16_t negative_flip;
    asc_duplicate_scalar(negative_flip, static_cast<uint16_t>(negative ? 0xff : 0x00), mask);
    asc_xor(raw_byte, bucket, negative_flip, mask);
    // The next pass takes the prefix as a scalar, so spill lane 0 and read it back. A
    // store->load barrier is enough: this is the same pattern as the AR-status read below.
    asc_storealign_1st(pivot_scratch, raw_byte);
    asc_mem_bar(VST_LD);
    prefix_bits |= static_cast<uint32_t>(pivot_scratch[0] & 0xff) << byte_shift;
}

// Per-row runtime arguments.
// `TopkSelectArgs` is row-agnostic. The per-row pointers and scalars are derived once in
// `run_topk_select_row()` and passed down as one value instead of nine parameters.
struct RowArgs {
    __gm__ float* row_in;
    __gm__ float* row_out_value;
    __gm__ void* row_out_index;  // int32 or int64, decided by Config::OUT_IDX_INT64
    int32_t output_idx_offset;  // already read from args.output_idx_offset; 0 if absent
    uint32_t valid_len;
    uint32_t topk;
    bool return_value;
    bool abort_when_nan_found;
    int32_t idx_oob_fill;
    float value_oob_fill;
};

/*
Count the lanes that contain a NaN in [values, values + count) and accumulate them into
`ub.state.nan_lane_count`. `count` may be any element count: the last vector is masked with
the remaining lanes.
*/
template <Config CONFIG>
template <bool IS_INITIALIZE>
__simd_callee__ void Kernel<CONFIG>::accumulate_nan_lane_count(__ubuf__ Ub& ub,
                                                               __ubuf__ float* values,
                                                               uint32_t count) {
    __ubuf__ uint32_t* nan_lane_count = &ub.state.nan_lane_count;
    vector_bool full_mask = asc_create_mask_b32(PAT_ALL);
    vector_bool acc_is_nan_mask = asc_create_mask_b32(PAT_ALLF);
    uint32_t remaining = count;
    auto* values_ptr = values;
    #pragma unroll 1
    for (uint32_t offset = 0; offset < count; offset += NUM_FP32_PER_VEC) {
        vector_bool active_mask = asc_update_mask_b32(remaining);
        vector_float value;
        asc_loadalign_postupdate(value, values_ptr, static_cast<int32_t>(NUM_FP32_PER_VEC));
        vector_bool is_nan_mask;
        asc_ne(is_nan_mask, value, value, active_mask);
        asc_or(acc_is_nan_mask, acc_is_nan_mask, is_nan_mask, full_mask);
    }
    vector_uint32_t nan_as_u32;
    asc_duplicate_scalar(nan_as_u32, uint32_t(1), acc_is_nan_mask);
    vector_uint32_t current_nan_lane_count;
    asc_reduce_sum(current_nan_lane_count, nan_as_u32, full_mask);
    if constexpr (!IS_INITIALIZE) {
        vector_uint32_t old_nan_lane_count;
        asc_loadalign(old_nan_lane_count, nan_lane_count);
        asc_add(current_nan_lane_count, current_nan_lane_count, old_nan_lane_count, full_mask);
    }
    asc_storealign_1st(nan_lane_count, current_nan_lane_count);
}

/*
Add one byte per element into the two halves of a 256-bin cumulative histogram, i.e.
`hist[i] += #{fed byte <= i}` (see the digit derivation at the top of this file).

`IS_MSB` is radix pass 1: the fed byte is the top byte of the descending key and every element
counts. Otherwise only the elements whose higher raw bytes equal the pivot prefix count, and
the fed byte is `BYTE_INDEX` inverted on the negative lanes.

The prefix comparison runs on the u8 byte planes rather than on whole 32-bit lanes because
`chistv2` takes a b8 mask (one bit per u8 lane); the planes are already available, so each
extra prefix byte costs one `asc_eq`.
*/
template <Config CONFIG>
template <bool IS_MSB, uint32_t BYTE_INDEX>
__simd_callee__ void Kernel<CONFIG>::accumulate_byte_histogram(
    vector_uint16_t& hist_low, vector_uint16_t& hist_high, __ubuf__ float* values, uint32_t count,
    uint32_t prefix_bits, uint32_t prefix_mask, bool negative) {
    static_assert(IS_MSB ? BYTE_INDEX == 3 : BYTE_INDEX < 3,
                  "pass 1 feeds the top byte, the refinement passes the lower ones");
    constexpr uint32_t DIGIT_BYTE = IS_MSB ? 3 : BYTE_INDEX;
    vector_bool mask_u8 = asc_create_mask_b8(PAT_ALL);

    // Per-lane sign flip, loop invariant. The prefix bytes are compared as scalars below.
    vector_uint8_t negative_flip, descending_flip;
    if constexpr (IS_MSB) {
        asc_duplicate_scalar(descending_flip, uint8_t(0x7f), mask_u8);
    } else {
        // Only the bytes above BYTE_INDEX take part in the comparison below, so the effective
        // prefix is the high part of `prefix_bits`.
        prefix_bits &= prefix_mask;
        asc_duplicate_scalar(negative_flip, static_cast<uint8_t>(negative ? 0xff : 0x00), mask_u8);
    }

    uint32_t remaining = count;
    const uint32_t rounds = ceil_div(count, HIST_ELEMS_PER_ROUND);
    auto* source = reinterpret_cast<__ubuf__ uint16_t*>(values);
    #pragma unroll 1
    for (uint32_t round = 0; round < rounds; ++round) {
        // 512 B per call: the low/high 16 bits of 128 fp32 elements each.
        vector_uint16_t low0, high0, low1, high1;
        asc_loadalign_deintlv_postupdate(low0, high0, source, static_cast<int32_t>(NUM_U8_PER_VEC));
        asc_loadalign_deintlv_postupdate(low1, high1, source, static_cast<int32_t>(NUM_U8_PER_VEC));

        // Both byte planes of the top half are needed by every pass, the bottom ones only by the
        // passes that compare or histogram a low byte.
        vector_uint8_t byte2, byte3;
        split_byte_planes(byte2, byte3, high0, high1);
        vector_uint8_t byte0, byte1;
        if constexpr (!IS_MSB && BYTE_INDEX < 2) {
            split_byte_planes(byte0, byte1, low0, low1);
        }

        vector_bool active = asc_update_mask_b8(remaining);
        vector_bool selected = active;
        if constexpr (!IS_MSB) {
            if constexpr (BYTE_INDEX < 3)
                asc_eq_scalar(selected, byte3, static_cast<uint8_t>(prefix_bits >> 24), selected);
            if constexpr (BYTE_INDEX < 2)
                asc_eq_scalar(selected, byte2, static_cast<uint8_t>(prefix_bits >> 16), selected);
            if constexpr (BYTE_INDEX < 1)
                asc_eq_scalar(selected, byte1, static_cast<uint8_t>(prefix_bits >> 8), selected);
        }

        vector_uint8_t digit;
        if constexpr (DIGIT_BYTE == 3) {
            digit = byte3;
        } else if constexpr (DIGIT_BYTE == 2) {
            digit = byte2;
        } else if constexpr (DIGIT_BYTE == 1) {
            digit = byte1;
        } else {
            digit = byte0;
        }
        if constexpr (IS_MSB) {
            // Non-negative lanes see 0x7f, negative lanes 0x00 (the descending top-byte key).
            vector_bool nonnegative;
            asc_ge_scalar(nonnegative, reinterpret_cast<vector_int8_t&>(digit), int8_t(0), mask_u8);
            // No asc_* C-API exposes MODE_MERGING for vxor; keep the raw intrinsic.
            vxor(digit, digit, descending_flip, nonnegative, MODE_MERGING);
        } else {
            asc_xor(digit, digit, negative_flip, mask_u8);
        }
        asc_cumulative_histogram_bin0(hist_low, digit, selected);
        asc_cumulative_histogram_bin1(hist_high, digit, selected);
    }
}

/*
Turn the block-local position of an appended candidate into its real row index:
    index = page_start(position >> SHUFFLE_PAGE_SIZE_BITS) + (position % SHUFFLE_PAGE_SIZE)
where `page_start` is the physical element offset of that page (see the page map in config.h).
`page_base` is the page-map slot the scanned block starts at.
*/
template <Config CONFIG>
__simd_callee__ void Kernel<CONFIG>::materialize_index(__ubuf__ Ub& ub, vector_uint32_t position,
                                                       uint32_t page_base, vector_uint32_t& index,
                                                       vector_bool mask) {
    __ubuf__ uint32_t* page_ids = ub.shuffle_page_ids + page_base;
    constexpr int16_t PAGE_BITS = SHUFFLE_PAGE_SIZE_BITS;
    vector_uint32_t page_in_block, page_offset, page_start;
    asc_shiftright_scalar(page_in_block, position, PAGE_BITS, mask);
    asc_shiftleft_scalar(page_offset, page_in_block, PAGE_BITS, mask);
    asc_sub(page_offset, position, page_offset, mask);
    asc_gather(page_start, page_ids, page_in_block, mask);
    asc_add(index, page_start, page_offset, mask);
}

// Add `output_idx_offset` to the first `valid_count` candidate indices. The out-of-bounds fill
// indices of a short row must keep their fill value, hence the masked add.
template <Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::prepare_index_vf(__ubuf__ Ub& ub, uint32_t count,
                                                  uint32_t valid_count,
                                                  int32_t output_idx_offset) {
    vector_bool full_mask = asc_create_mask_b32(PAT_ALL);
    // The offset may be negative; adding its two's-complement pattern is the same operation as
    // the signed add the index deserves.
    const uint32_t offset = static_cast<uint32_t>(output_idx_offset);
    auto* index_in_ptr = ub.candidate_index;
    auto* index_out_ptr = ub.candidate_index;
    uint32_t remaining = count;
    uint32_t valid_remaining = valid_count;
    #pragma unroll 1
    for (uint32_t offset_elem = 0; offset_elem < count; offset_elem += NUM_FP32_PER_VEC) {
        vector_bool active = asc_update_mask_b32(remaining);
        vector_bool valid = asc_update_mask_b32(valid_remaining);
        vector_uint32_t current;
        asc_loadalign_postupdate(current, index_in_ptr, static_cast<int32_t>(NUM_FP32_PER_VEC));
        vector_uint32_t shifted;
        asc_add_scalar(shifted, current, offset, full_mask);
        asc_select(current, shifted, current, valid);
        asc_storealign_postupdate(index_out_ptr, current, static_cast<int32_t>(NUM_FP32_PER_VEC),
                                  active);
    }
}

// Write the NaN sentinel index into every output lane. Only used when NaNs were detected and
// the caller asked not to abort, so both this VF and the extra UB->GM copy are exception-path
// costs; the common path never runs them.
template <Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::fill_nan_sentinel_vf(__ubuf__ Ub& ub, __ubuf__ int32_t* destination,
                                                      uint32_t count) {
    vector_int32_t sentinel;
    vector_bool mask = asc_create_mask_b32(PAT_ALL);
    asc_duplicate_scalar(sentinel, int32_t(0x3F3F3F3F), mask);
    uint32_t remaining = count;
    #pragma unroll 1
    for (uint32_t offset = 0; offset < count; offset += NUM_FP32_PER_VEC) {
        vector_bool active = asc_update_mask_b32(remaining);
        asc_storealign_postupdate(destination, sentinel, static_cast<int32_t>(NUM_FP32_PER_VEC),
                                  active);
    }
}

// Accumulate every value that has entered the buffer into the top-byte histogram without
// clearing it; on each rebuild only the newly added values are accumulated. The counters are
// clamped so that a stale (over-counted) entry can never push the pivot below the k-th value.
template <Config CONFIG>
template <bool IS_INITIALIZE>
__simd_vf__ void Kernel<CONFIG>::build_high_histogram_vf(__ubuf__ Ub& ub, __ubuf__ float* values,
                                                         uint32_t count, uint32_t keep_count) {
    __ubuf__ uint16_t* histogram = ub.high_histogram;
    vector_bool mask_u16 = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t hist_low, hist_high;
    if constexpr (IS_INITIALIZE) {
        asc_duplicate_scalar(hist_low, uint16_t(0), mask_u16);
        asc_duplicate_scalar(hist_high, uint16_t(0), mask_u16);
    } else {
        asc_loadalign(hist_low, histogram);
        asc_loadalign(hist_high, histogram, TOPK_HISTOGRAM_BINS / 2);
    }
    accumulate_byte_histogram<true, 3>(hist_low, hist_high, values, count, 0, 0, false);
    accumulate_nan_lane_count<IS_INITIALIZE>(ub, values, count);
    asc_min_scalar(hist_low, hist_low, static_cast<uint16_t>(keep_count + 1), mask_u16);
    asc_min_scalar(hist_high, hist_high, static_cast<uint16_t>(keep_count + 1), mask_u16);
    asc_storealign(histogram, hist_low, mask_u16);
    asc_storealign(histogram, hist_high, TOPK_HISTOGRAM_BINS / 2, mask_u16);
}

// The warmup sample is copied straight into `candidate_value` instead of being appended, so its
// real row indices have to be materialized once before the first radix rebuild. The sample sits
// at positions [0, count), which is exactly what a scan block with page base 0 looks like.
template <Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::build_warmup_index_vf(__ubuf__ Ub& ub, uint32_t count) {
    vector_bool full_mask = asc_create_mask_b32(PAT_ALL);
    __ubuf__ uint32_t* index_out_ptr = ub.candidate_index;
    uint32_t remaining = count;
    vector_uint32_t position;
    asc_arange(reinterpret_cast<vector_int32_t&>(position), int32_t(0));
    #pragma unroll 1
    for (uint32_t offset = 0; offset < count; offset += NUM_FP32_PER_VEC) {
        vector_bool active = asc_update_mask_b32(remaining);
        vector_uint32_t index;
        materialize_index(ub, position, 0, index, active);
        asc_storealign_postupdate(index_out_ptr, index, static_cast<int32_t>(NUM_FP32_PER_VEC),
                                  active);
        asc_add_scalar(position, position, uint32_t(NUM_FP32_PER_VEC), full_mask);
    }
}

// Only instantiated for `SORTED_INDEX` tiers.
// Reorder the values according to the Sort permutation, so that they line up with the sorted
// indices. The permutation array is read and overwritten in place: each window is read before
// it is written, and no later window reads a position that an earlier one wrote.
template <Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::get_sorted_value_vf(__ubuf__ Ub& ub, uint32_t topk) {
    __ubuf__ float* source_value = ub.candidate_value;
    auto* permutation = reinterpret_cast<__ubuf__ float*>(ub.overlay.sort.array1);
    auto* position_in_ptr = reinterpret_cast<__ubuf__ uint32_t*>(permutation);
    auto* value_out_ptr = permutation;
    uint32_t remaining = topk;
    #pragma unroll 1
    for (uint32_t offset = 0; offset < topk; offset += NUM_FP32_PER_VEC) {
        vector_bool active = asc_update_mask_b32(remaining);
        vector_uint32_t position;
        asc_loadalign_postupdate(position, position_in_ptr, static_cast<int32_t>(NUM_FP32_PER_VEC));
        vector_float value;
        asc_gather(value, source_value, position, active);
        asc_storealign_postupdate(value_out_ptr, value, static_cast<int32_t>(NUM_FP32_PER_VEC),
                                  active);
    }
}

// Only instantiated for `SORTED_VALUE` tiers. A short row sorts only its valid elements, so the
// lanes from `valid_count` to `topk` keep the fill values the short-row preparation wrote: they
// must be restored in the sorted output arrays, because the sort window carried -inf padding over
// part of that range.
template <Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::restore_short_row_fill_vf(__ubuf__ Ub& ub, uint32_t valid_count,
                                                           uint32_t topk, float value_oob_fill,
                                                           int32_t idx_oob_fill) {
    vector_bool mask = asc_create_mask_b32(PAT_ALL);
    vector_uint32_t value_fill, index_fill;
    asc_duplicate_scalar(value_fill, reinterpret_cast<const uint32_t&>(value_oob_fill), mask);
    asc_duplicate_scalar(index_fill, static_cast<uint32_t>(idx_oob_fill), mask);
    constexpr int32_t STEP = static_cast<int32_t>(NUM_FP32_PER_VEC);
    // The window that holds `valid_count` starts at a 64-lane boundary, so the first store is
    // masked to keep the sorted elements intact; every following store fills a whole vector.
    const uint32_t start = valid_count & ~(NUM_FP32_PER_VEC - 1);
    uint32_t kept = valid_count - start;
    vector_bool kept_prefix = asc_update_mask_b32(kept);
    vector_bool beyond;
    asc_not(beyond, kept_prefix, mask);
    auto* value_ptr = reinterpret_cast<__ubuf__ uint32_t*>(ub.overlay.sort.array0) + start;
    auto* index_ptr = ub.overlay.sort.array1 + start;
    if (start < topk) {
        asc_storealign_postupdate(value_ptr, value_fill, STEP, beyond);
        asc_storealign_postupdate(index_ptr, index_fill, STEP, beyond);
    }
    // Both pointers advance in the store instruction itself, so the loop body is just two stores.
    for (uint32_t filled = start + NUM_FP32_PER_VEC; filled < topk; filled += NUM_FP32_PER_VEC) {
        asc_storealign_postupdate(value_ptr, value_fill, STEP, mask);
        asc_storealign_postupdate(index_ptr, index_fill, STEP, mask);
    }
}

// Only instantiated for `SORTED_VALUE` tiers. The merge sort path needs the element count to be a
// multiple of SORT_ELEMENT_ALIGN, so the lanes between topk and SORT_COUNT are filled with -inf:
// they sort behind every real value and never reach the first k outputs.
template <Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::pad_sort_tail_vf(__ubuf__ Ub& ub, uint32_t topk,
                                                  uint32_t sort_count) {
    vector_bool mask = asc_create_mask_b32(PAT_ALL);
    vector_uint32_t padding;
    asc_duplicate_scalar(padding, VALUE_PADDING_BITS, mask);
    auto* value_ptr = reinterpret_cast<__ubuf__ uint32_t*>(ub.candidate_value);
    // The window that holds `topk` starts at a 64-lane boundary, so the store is masked to keep
    // the real candidates below `topk` intact.
    const uint32_t start = topk & ~(NUM_FP32_PER_VEC - 1);
    uint32_t kept = topk - start;
    vector_bool kept_prefix = asc_update_mask_b32(kept);
    vector_bool beyond;
    asc_not(beyond, kept_prefix, mask);
    if (start < sort_count) asc_storealign(value_ptr + start, padding, beyond);
    for (uint32_t offset = start + NUM_FP32_PER_VEC; offset < sort_count;
         offset += NUM_FP32_PER_VEC) {
        asc_storealign(value_ptr + offset, padding, mask);
    }
}

/*
This VF has three call sites:
- During warmup, select the initial top-k candidates and establish the threshold.
- In regular scan rounds, append the values above the threshold to the candidate buffer and
  reduce the buffer back to k candidates (updating the threshold) when it exceeds
  MAX_TOPK + FILTERED_CAPACITY.
- On the last scan round, do the same and additionally finalize the selection.

This deliberately selects every NaN so that it is retained and detected.
*/
template <Config CONFIG>
template <bool IS_WARMUP, bool IS_FINALIZE>
__simd_vf__ void Kernel<CONFIG>::select_candidates_vf(__ubuf__ Ub& ub, uint32_t keep_count,
                                                      uint32_t warmup_source_count,
                                                      uint32_t filter_source_count,
                                                      uint32_t filter_page_base,
                                                      uint32_t filter_stage) {
    constexpr uint32_t FILTER_OVERFLOW_TRIGGER = MAX_TOPK + FILTERED_CAPACITY;
    __ubuf__ float* source_value = ub.candidate_value;
    __ubuf__ uint32_t* source_index = ub.candidate_index;
    __ubuf__ uint32_t* temporary_position = ub.temporary_position;
    vector_bool mask_u32 = asc_create_mask_b32(PAT_ALL);
    vector_bool mask_u16 = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t zero_u16;
    asc_duplicate_scalar(zero_u16, uint16_t(0), mask_u16);

    uint32_t source_count = warmup_source_count;
    if constexpr (!IS_WARMUP) source_count = ub.state.result_count;

    // ---------------------------------------------------------------------
    // A. Append the candidates of the newly scanned block
    // ---------------------------------------------------------------------
    // The block-local positions are compacted into the candidate index array first (it is the
    // natural scratch: the slots they occupy are overwritten by their own materialized indices
    // in the gather loop below), then each surviving position is turned into a real row index
    // and its value is pulled out of the input staging.
    if constexpr (!IS_WARMUP) {
        const uint32_t destination_count = source_count;
        __ubuf__ float* filter_source = ub.overlay.input_staging + filter_stage * BLOCK_SIZE;
        auto* filter_source_ptr = filter_source;

        vector_uint32_t threshold_bits;
        asc_loadalign_brc(threshold_bits, ub.threshold);
        vector_store_unalign index_state;
        vector_uint32_t packed_position;
        asc_squeeze_and_storeunalign_init();
        auto* index_output = source_index + destination_count;
        vector_uint32_t position;
        asc_arange(reinterpret_cast<vector_int32_t&>(position), int32_t(0));
        uint32_t remaining_in = filter_source_count;
        #pragma unroll 1
        for (uint32_t offset = 0; offset < filter_source_count; offset += NUM_FP32_PER_VEC) {
            vector_bool active = asc_update_mask_b32(remaining_in);
            vector_float value;
            asc_loadalign_postupdate(value, filter_source_ptr, static_cast<int32_t>(NUM_FP32_PER_VEC));
            vector_bool less_equal, selected;
            asc_le(less_equal, value, reinterpret_cast<vector_float&>(threshold_bits), active); // Use le + not to select all NaN
            asc_not(selected, less_equal, active);
            asc_squeeze_and_storeunalign(index_output, packed_position, index_state, position,
                                          selected);
            asc_add_scalar(position, position, uint32_t(NUM_FP32_PER_VEC), mask_u32);
        }
        asc_squeeze_and_storeunalign_finalize(index_output, index_state);
        asc_mem_bar(VST_VLD);
        // The AR counter is only reachable from `__aicore__` code, so it is spilled to UB here
        // and read back by the scalar side after the store->load barrier.
        auto* count_scratch = reinterpret_cast<__ubuf__ uint32_t*>(ub.histogram);
        asc_storealign_squeeze_status(count_scratch);
        asc_mem_bar(VST_LD);
        source_count = destination_count + count_scratch[0] / sizeof(uint32_t);

        uint32_t remaining = source_count - destination_count;
        auto* position_in_ptr = source_index + destination_count;
        auto* value_out_ptr = source_value + destination_count;
        auto* index_out_ptr = source_index + destination_count;
        while (remaining >= NUM_FP32_PER_VEC) {
            vector_uint32_t current_position;
            asc_loadalign_postupdate(current_position, position_in_ptr,
                                     static_cast<int32_t>(NUM_FP32_PER_VEC));
            vector_float value;
            asc_gather(value, filter_source, current_position, mask_u32);
            vector_uint32_t index;
            materialize_index(ub, current_position, filter_page_base, index, mask_u32);
            asc_storealign_postupdate(value_out_ptr, value, static_cast<int32_t>(NUM_FP32_PER_VEC),
                                      mask_u32);
            asc_storealign_postupdate(index_out_ptr, index, static_cast<int32_t>(NUM_FP32_PER_VEC),
                                      mask_u32);
            remaining -= NUM_FP32_PER_VEC;
        }
        if (remaining != 0) {
            uint32_t aligned_remaining = aligned(remaining, VST_VLD_ALIGN_ELEMENTS);
            vector_bool tail = asc_update_mask_b32(remaining);
            vector_bool aligned_tail = asc_update_mask_b32(aligned_remaining);
            vector_uint32_t current_position;
            asc_loadalign(current_position, position_in_ptr);
            // Lanes past the tail must not index outside the staging block.
            vector_uint32_t position_zero;
            asc_duplicate_scalar(position_zero, uint32_t(0), mask_u32);
            asc_select(current_position, current_position, position_zero, tail);
            vector_float value;
            asc_gather(value, filter_source, current_position, tail);
            vector_uint32_t index;
            materialize_index(ub, current_position, filter_page_base, index, tail);
            // Keep the value array vector sized: the lanes past the tail get the -inf bit
            // pattern, which no later pass can select below the k-th value.
            vector_uint32_t padding;
            asc_duplicate_scalar(padding, VALUE_PADDING_BITS, mask_u32);
            vector_uint32_t value_bits;
            asc_select(value_bits, reinterpret_cast<vector_uint32_t&>(value), padding, tail);
            asc_storealign(value_out_ptr, reinterpret_cast<vector_float&>(value_bits),
                           aligned_tail);
            asc_storealign(index_out_ptr, index, tail);
            source_count = aligned(source_count, VST_VLD_ALIGN_ELEMENTS);
        }
    }

    // ---------------------------------------------------------------------
    // B. Radix rebuild: reduce the buffer back to the k largest candidates
    // ---------------------------------------------------------------------
    if (IS_WARMUP || source_count > FILTER_OVERFLOW_TRIGGER || IS_FINALIZE) {
        if constexpr (!IS_WARMUP) asc_mem_bar(VST_VLD);
        vector_uint16_t hist_low, hist_high;
        asc_loadalign(hist_low, ub.high_histogram);
        asc_loadalign(hist_high, ub.high_histogram, TOPK_HISTOGRAM_BINS / 2);
        // The (clamped) top-byte histogram only ever gains entries, so the candidates discarded
        // by an earlier rebuild still count towards the global bucket of the k-th value.
        if constexpr (!IS_WARMUP) {
            const uint32_t histogram_begin = aligned(keep_count, VST_VLD_ALIGN_ELEMENTS);
            if (source_count > histogram_begin) {
                accumulate_byte_histogram<true, 3>(hist_low, hist_high, source_value + histogram_begin,
                                                  source_count - histogram_begin, 0, 0, false);
                accumulate_nan_lane_count<false>(ub, source_value + histogram_begin,
                                                 source_count - histogram_begin);
                asc_min_scalar(hist_low, hist_low, static_cast<uint16_t>(keep_count + 1), mask_u16);
                asc_min_scalar(hist_high, hist_high, static_cast<uint16_t>(keep_count + 1), mask_u16);
                asc_storealign(ub.high_histogram, hist_low, mask_u16);
                asc_storealign(ub.high_histogram, hist_high, TOPK_HISTOGRAM_BINS / 2, mask_u16);
            }
        }

        // Radix pass 1: locate the bucket of the descending top byte that holds the k-th value.
        vector_uint16_t bucket, rank, previous_count, greater_count;
        asc_duplicate_scalar(rank, static_cast<uint16_t>(keep_count), mask_u16);
        find_bucket(bucket, hist_low, hist_high, rank);
        // The threshold slot doubles as the scratch for the scalar views computed below; it is
        // written for real once the four passes have pinned the pivot down.
        __ubuf__ uint16_t* pivot_scratch = reinterpret_cast<__ubuf__ uint16_t*>(ub.threshold);
        asc_storealign_1st(pivot_scratch, bucket);
        asc_mem_bar(VST_LD);
        const uint32_t high_bucket = pivot_scratch[0] & 0xff;
        // a_high is the ascending top-byte key; `pivot_negative` follows from it, because a
        // negative value's ascending key has its sign bit clear.
        const uint32_t a_high = 0xff - high_bucket;
        const bool pivot_negative = a_high < 0x80;
        uint32_t prefix_bits = (a_high ^ (pivot_negative ? 0xffu : 0x80u)) << 24;
        // previous_count = hist[high_bucket - 1]: the candidates strictly above the bucket.
        vector_uint16_t previous_bucket;
        asc_add_scalar(previous_bucket, bucket, uint16_t(-1), mask_u16);
        histogram_lookup(previous_count, hist_low, hist_high, previous_bucket);
        vector_bool first_bucket;
        asc_eq_scalar(first_bucket, bucket, uint16_t(0), mask_u16);
        asc_select(previous_count, zero_u16, previous_count, first_bucket);
        greater_count = previous_count;
        asc_duplicate_scalar(rank, static_cast<uint16_t>(keep_count), mask_u16);
        asc_sub(rank, rank, greater_count, mask_u16);

        // Radix passes 2..4 over the candidate buffer. The three passes differ only in the byte
        // position, so they are written out; the rank bookkeeping is shared (fold_radix_pass).
        vector_uint16_t pass_low, pass_high;
        asc_duplicate_scalar(pass_low, uint16_t(0), mask_u16);
        asc_duplicate_scalar(pass_high, uint16_t(0), mask_u16);
        accumulate_byte_histogram<false, 2>(pass_low, pass_high, source_value, source_count,
                                            prefix_bits, 0xff000000u, pivot_negative);
        fold_radix_pass(pivot_scratch, pass_low, pass_high, pivot_negative,
                        static_cast<uint16_t>(keep_count), 16, prefix_bits, rank, greater_count);
        asc_duplicate_scalar(pass_low, uint16_t(0), mask_u16);
        asc_duplicate_scalar(pass_high, uint16_t(0), mask_u16);
        accumulate_byte_histogram<false, 1>(pass_low, pass_high, source_value, source_count,
                                            prefix_bits, 0xffff0000u, pivot_negative);
        fold_radix_pass(pivot_scratch, pass_low, pass_high, pivot_negative,
                        static_cast<uint16_t>(keep_count), 8, prefix_bits, rank, greater_count);
        asc_duplicate_scalar(pass_low, uint16_t(0), mask_u16);
        asc_duplicate_scalar(pass_high, uint16_t(0), mask_u16);
        accumulate_byte_histogram<false, 0>(pass_low, pass_high, source_value, source_count,
                                            prefix_bits, 0xffffff00u, pivot_negative);
        fold_radix_pass(pivot_scratch, pass_low, pass_high, pivot_negative,
                        static_cast<uint16_t>(keep_count), 0, prefix_bits, rank, greater_count);
        // `prefix_bits` is now the exact bit pattern of the k-th largest value and `rank` holds
        // the number of candidates equal to it that still have to be taken.
        asc_storealign_1st(pivot_scratch, rank);
        asc_mem_bar(VST_LD);
        const uint32_t equal_limit = pivot_scratch[0];
        vector_uint32_t threshold_bits;
        asc_duplicate_scalar(threshold_bits, prefix_bits, mask_u32);
        asc_storealign_1st(ub.threshold, threshold_bits);

        // Strict comparison in the total order of the ascending key, so that +0 counts as above
        // -0: xor both operands with the threshold's sign-dependent mask and compare as int32.
        vector_bool threshold_negative;
        asc_lt_scalar(threshold_negative, reinterpret_cast<vector_int32_t&>(threshold_bits),
                      int32_t(0), mask_u32);
        vector_uint32_t compare_mask, negative_compare_mask;
        asc_duplicate_scalar(compare_mask, uint32_t(0), mask_u32);
        asc_duplicate_scalar(negative_compare_mask, uint32_t(0x7fffffff), mask_u32);
        asc_select(compare_mask, negative_compare_mask, compare_mask, threshold_negative);
        vector_uint32_t threshold_compare;
        asc_xor(threshold_compare, threshold_bits, compare_mask, mask_u32);

        // Collect the positions of the k candidates that survive: everything above the
        // threshold, plus the first `equal_limit` candidates equal to it.
        vector_uint32_t equal_seen, equal_limit_vec;
        asc_duplicate_scalar(equal_seen, uint32_t(0), mask_u32);
        asc_duplicate_scalar(equal_limit_vec, uint32_t(equal_limit), mask_u32);
        vector_store_unalign position_state;
        vector_uint32_t packed_position;
        asc_squeeze_and_storeunalign_init();
        uint32_t remaining = source_count;
        auto* source_value_ptr = source_value;
        vector_uint32_t position;
        asc_arange(reinterpret_cast<vector_int32_t&>(position), int32_t(0));
        #pragma unroll 1
        for (uint32_t offset = 0; offset < source_count; offset += NUM_FP32_PER_VEC) {
            vector_bool active = asc_update_mask_b32(remaining);
            vector_float value;
            asc_loadalign_postupdate(value, source_value_ptr, static_cast<int32_t>(NUM_FP32_PER_VEC));
            vector_uint32_t value_compare;
            asc_xor(value_compare, reinterpret_cast<vector_uint32_t&>(value), compare_mask, active);
            vector_bool is_greater, is_equal;
            asc_gt(is_greater, reinterpret_cast<vector_int32_t&>(value_compare),
                   reinterpret_cast<vector_int32_t&>(threshold_compare), active);
            asc_eq(is_equal, reinterpret_cast<vector_int32_t&>(value_compare),
                   reinterpret_cast<vector_int32_t&>(threshold_compare), active);
            // Running count of equal candidates up to and including this lane.
            vector_uint32_t equal_prefix, equal_inclusive;
            asc_unsqueeze(equal_prefix, is_equal);
            asc_add_scalar(equal_inclusive, equal_prefix, uint32_t(1), mask_u32);
            asc_select(equal_prefix, equal_inclusive, equal_prefix, is_equal);
            asc_add(equal_prefix, equal_prefix, equal_seen, mask_u32);
            vector_bool take_equal;
            asc_ge(take_equal, equal_limit_vec, equal_prefix, is_equal);
            vector_bool selected;
            asc_or(selected, is_greater, take_equal, active);
            asc_squeeze_and_storeunalign(temporary_position, packed_position, position_state,
                                          position, selected);
            asc_duplicate_highest(equal_seen, equal_prefix, mask_u32);
            asc_add_scalar(position, position, uint32_t(NUM_FP32_PER_VEC), mask_u32);
        }
        asc_squeeze_and_storeunalign_finalize(temporary_position, position_state);
        asc_mem_bar(VST_VLD);

        // Gather the surviving values and their indices into the front of the buffer. In-place
        // gathering is safe: window i reads positions >= i, so it never touches a slot that an
        // earlier window already wrote.
        uint32_t remaining_keep = keep_count;
        auto* position_ptr = temporary_position;
        auto* keep_value_out_ptr = source_value;
        auto* keep_index_out_ptr = source_index;
        #pragma unroll 1
        for (uint32_t offset = 0; offset < keep_count; offset += NUM_FP32_PER_VEC) {
            vector_bool active = asc_update_mask_b32(remaining_keep);
            vector_uint32_t keep_position;
            asc_loadalign_postupdate(keep_position, position_ptr,
                                     static_cast<int32_t>(NUM_FP32_PER_VEC));
            // The collect step emits exactly `keep_count` positions; clamping keeps a stale slot
            // (should the invariant ever break) from turning into an out-of-bounds UB gather.
            asc_min_scalar(keep_position, keep_position, source_count - 1, mask_u32);
            vector_float keep_value;
            asc_gather(keep_value, source_value, keep_position, active);
            vector_uint32_t keep_index;
            asc_gather(keep_index, source_index, keep_position, active);
            if (offset + NUM_FP32_PER_VEC > keep_count) {
                // Lanes past the k-th candidate get the -inf bit pattern so that no later pass
                // can select them when the buffer is scanned again.
                vector_uint32_t lane;
                asc_arange(reinterpret_cast<vector_int32_t&>(lane), static_cast<int32_t>(offset));
                vector_bool out_of_bounds;
                asc_ge_scalar(out_of_bounds, lane, static_cast<int32_t>(keep_count), mask_u32);
                vector_uint32_t padding;
                asc_duplicate_scalar(padding, VALUE_PADDING_BITS, mask_u32);
                asc_select(reinterpret_cast<vector_uint32_t&>(keep_value), padding,
                           reinterpret_cast<vector_uint32_t&>(keep_value), out_of_bounds);
            }
            asc_storealign_postupdate(keep_value_out_ptr, keep_value,
                                      static_cast<int32_t>(NUM_FP32_PER_VEC), mask_u32);
            asc_storealign_postupdate(keep_index_out_ptr, keep_index,
                                      static_cast<int32_t>(NUM_FP32_PER_VEC), mask_u32);
        }
        if constexpr (!IS_WARMUP) source_count = aligned(keep_count, VST_VLD_ALIGN_ELEMENTS);
    }
    if constexpr (!IS_WARMUP) ub.state.result_count = source_count;
}

// Prepare the output directly when the number of valid elements does not exceed top-k.
template <Config CONFIG>
template <bool RETURN_VALUE>
__simd_vf__ void Kernel<CONFIG>::prepare_short_row_vf(__ubuf__ Ub& ub, uint32_t valid_count,
                                                      uint32_t topk, float value_oob_fill,
                                                      int32_t idx_oob_fill) {
    vector_bool full_mask = asc_create_mask_b32(PAT_ALL);
    accumulate_nan_lane_count<true>(ub, ub.overlay.input_staging, valid_count);
    if constexpr (RETURN_VALUE) {
        __ubuf__ float* source_ptr = ub.overlay.input_staging;
        __ubuf__ float* destination_value_out_ptr = ub.candidate_value;
        vector_int32_t lane;
        vector_float fill_value;
        asc_arange(lane, int32_t(0));
        asc_duplicate_scalar(fill_value, value_oob_fill, full_mask);
        #pragma unroll 1
        for (uint32_t offset = 0; offset < topk; offset += NUM_FP32_PER_VEC) {
            vector_float value;
            asc_loadalign_postupdate(value, source_ptr, static_cast<int32_t>(NUM_FP32_PER_VEC));
            vector_bool need_fill;
            asc_ge_scalar(need_fill, lane, static_cast<int32_t>(valid_count), full_mask);
            asc_add_scalar(lane, lane, int32_t(NUM_FP32_PER_VEC), full_mask);
            asc_select(value, fill_value, value, need_fill);
            asc_storealign_postupdate(destination_value_out_ptr, value,
                                      static_cast<int32_t>(NUM_FP32_PER_VEC), full_mask);
        }
    }

    __ubuf__ uint32_t* destination_index = ub.candidate_index;
    auto* destination_index_out_ptr = destination_index;
    vector_uint32_t index_lane, fill_index;
    asc_arange(reinterpret_cast<vector_int32_t&>(index_lane), int32_t(0));
    asc_duplicate_scalar(fill_index, static_cast<uint32_t>(idx_oob_fill), full_mask);
    #pragma unroll 1
    for (uint32_t offset = 0; offset < topk; offset += NUM_FP32_PER_VEC) {
        vector_uint32_t index;
        vector_bool need_fill;
        asc_ge_scalar(need_fill, reinterpret_cast<vector_int32_t&>(index_lane),
                      static_cast<int32_t>(valid_count), full_mask);
        asc_select(index, fill_index, index_lane, need_fill);
        asc_add_scalar(index_lane, index_lane, uint32_t(NUM_FP32_PER_VEC), full_mask);
        asc_storealign_postupdate(destination_index_out_ptr, index,
                                  static_cast<int32_t>(NUM_FP32_PER_VEC), full_mask);
    }
}

template <Config CONFIG>
__aicore__ void Kernel<CONFIG>::run(const RowArgs& row) {
    // Unpack the per-row arguments once, so the body below reads like the algorithm.
    const uint32_t valid_len = row.valid_len;
    const uint32_t topk = row.topk;
    __gm__ float* row_in = row.row_in;
    __gm__ float* row_out_value = row.row_out_value;
    const int32_t output_idx_offset = row.output_idx_offset;
    const bool return_value = row.return_value;
    const int32_t idx_oob_fill = row.idx_oob_fill;
    const float value_oob_fill = row.value_oob_fill;

    // The UB plan is a lens over raw UB: base address 0, so offsets are addresses.
    // Two steps instead of one so the compiler does not warn about the null dereference: the
    // "pointer" is never really dereferenced, it only carries the type.
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
            static_assert(MAX_TOPK * sizeof(float) <= sizeof(ub.overlay.input_staging),
                          "the short row must fit in the input staging");
            wait_flag(PIPE_V, PIPE_MTE2, stage_free_events[0]);
            copy_gm_to_ubuf_align_v2(ub.overlay.input_staging, row_in, 0, 1,
                                     valid_len * sizeof(float), 0, 0, false,
                                     static_cast<uint8_t>(
                                         LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN),
                                     valid_len * sizeof(float), valid_len * sizeof(float));
            event_t stage_full = event_t(Flags::INPUT_STAGE_FULL_BASE);
            set_flag(PIPE_MTE2, PIPE_V, stage_full);
            wait_flag(PIPE_MTE2, PIPE_V, stage_full);
        }
        if (return_value)
            prepare_short_row_vf<true>(ub, valid_len, topk, value_oob_fill, idx_oob_fill);
        else
            prepare_short_row_vf<false>(ub, valid_len, topk, value_oob_fill, idx_oob_fill);
        if (valid_len != 0) set_flag(PIPE_V, PIPE_MTE2, stage_free_events[0]);
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
                                     page_count * sizeof(float), 0, 0, false,
                                     static_cast<uint8_t>(
                                         LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN),
                                     page_count * sizeof(float), page_count * sizeof(float));
            set_flag(PIPE_MTE2, PIPE_V, warmup_events[logical_page_id]);
            wait_flag(PIPE_MTE2, PIPE_V, warmup_events[logical_page_id]);
            if (logical_page_id == 0)
                build_high_histogram_vf<true>(ub, warmup_destination, page_count, topk);
            else
                build_high_histogram_vf<false>(ub, warmup_destination, page_count, topk);
            ub.shuffle_page_ids[logical_page_id] = page_offset;
            num_warmup_elements += page_count;
        }

        // The page map is written by the scalar core and read by the vector core, and the two run concurrently, so the hand-off needs an explicit S->V event. Signal it as soon as the stores are issued and only wait right before the first VF that reads the map.
        event_t page_map_ready = event_t(Flags::PAGE_MAP_READY);
        set_flag(PIPE_S, PIPE_V, page_map_ready);

        const uint32_t stable_pages = num_shuffle_pages - num_warmup_pages;
        const uint32_t stable_blocks = ceil_div(stable_pages, PAGES_PER_BLOCK);
        wait_flag(PIPE_S, PIPE_V, page_map_ready);
        build_warmup_index_vf(ub, num_warmup_elements);
        // `IS_WARMUP` already forces the radix rebuild below, so there is no separate
        // "finalize" variant to pick here.
        select_candidates_vf<true>(ub, topk, num_warmup_elements, 0, 0, 0);
        ub.state.result_count = aligned(topk, VST_VLD_ALIGN_ELEMENTS);
        for (uint32_t stable_idx = 0; stable_idx < stable_blocks; ++stable_idx) {
            const uint32_t stage = stable_idx % NUM_STAGES;
            wait_flag(PIPE_V, PIPE_MTE2, stage_free_events[stage]);
            auto* block_destination = ub.overlay.input_staging + stage * BLOCK_SIZE;
            uint32_t count = 0;
            for (uint32_t page = 0; page < PAGES_PER_BLOCK; ++page) {
                const uint32_t logical_page_id =
                    num_warmup_pages + stable_idx * PAGES_PER_BLOCK + page;
                if (logical_page_id >= num_shuffle_pages) break;
                const ShufflePage physical_page = permutation.get_page_id(logical_page_id);
                const uint32_t page_offset = physical_page.start_offset;
                const uint32_t page_count = physical_page.len;
                copy_gm_to_ubuf_align_v2(
                    block_destination + page * SHUFFLE_PAGE_SIZE, row_in + page_offset, 0, 1,
                    page_count * sizeof(float), 0, 0, false,
                    static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN),
                    page_count * sizeof(float), page_count * sizeof(float));
                // Block-aligned page-map slot (see SHUFFLE_PAGE_CAPACITY in config.h).
                const uint32_t virtual_page_id = (stable_idx + 1) * PAGES_PER_BLOCK + page;
                ub.shuffle_page_ids[virtual_page_id] = page_offset;
                count += page_count;
            }
            set_flag(PIPE_S, PIPE_V, page_map_ready);
            event_t stage_full = event_t(Flags::INPUT_STAGE_FULL_BASE + stage);
            set_flag(PIPE_MTE2, PIPE_V, stage_full);
            wait_flag(PIPE_MTE2, PIPE_V, stage_full);
            const uint32_t page_base = (stable_idx + 1) * PAGES_PER_BLOCK;
            const bool finalize = stable_idx + 1 == stable_blocks;
            wait_flag(PIPE_S, PIPE_V, page_map_ready);
            if (finalize)
                select_candidates_vf<false, true>(ub, topk, 0, count, page_base, stage);
            else
                select_candidates_vf<false, false>(ub, topk, 0, count, page_base, stage);
            set_flag(PIPE_V, PIPE_MTE2, stage_free_events[stage]);
        }
    }

    // Raise the NaN flag as soon as the counter is final: its only writers are the
    // `accumulate_nan_lane_count` calls above, so the wait below is satisfied early and the scalar
    // core never blocks on the index preparation, the sort or the output copies that follow.
    event_t nan_lane_count_ready = event_t(Flags::NAN_LANE_COUNT_READY);
    set_flag(PIPE_V, PIPE_S, nan_lane_count_ready);

    const uint32_t valid_count = valid_len < topk ? valid_len : topk;
    prepare_index_vf(ub, topk, valid_count, output_idx_offset);

    // A short row is already in ascending index order (the warmup sample of a short row is
    // scanned sequentially), so only the sorted_value mode has to sort it.
    const bool sort_needed =
        topk > 1 && (SORTED_VALUE || (valid_len > topk &&
                                      ceil_div(valid_len, SHUFFLE_PAGE_SIZE) > NUM_WARMUP_PAGES));
    __ubuf__ float* output_value = ub.candidate_value;
    __ubuf__ uint32_t* output_index = ub.candidate_index;
    if constexpr (SORTED_INDEX) {
        if (sort_needed) {
            LocalTensor<int32_t> source_index_tensor(TPosition::VECCALC, CANDIDATE_ARRAY_BYTES,
                                                     MAX_TOPK);
            LocalTensor<int32_t> sorted_index_tensor(TPosition::VECCALC, Ub::SORT_ARRAY0_OFFSET,
                                                     MAX_TOPK);
            LocalTensor<uint32_t> permutation_tensor(TPosition::VECCALC, Ub::SORT_ARRAY1_OFFSET,
                                                     MAX_TOPK);
            LocalTensor<uint8_t> temporary_tensor(TPosition::VECCALC, Ub::SORT_TEMPORARY_OFFSET,
                                                  SORT_TEMPORARY_BYTES);
            AscendC::Sort<int32_t, true, DEFAULT_SORT_CONFIG>(
                sorted_index_tensor, permutation_tensor, source_index_tensor, temporary_tensor,
                topk);
            if (return_value) {
                get_sorted_value_vf(ub, topk);
                output_value = reinterpret_cast<__ubuf__ float*>(Ub::SORT_ARRAY1_OFFSET);
            }
            output_index = reinterpret_cast<__ubuf__ uint32_t*>(Ub::SORT_ARRAY0_OFFSET);
        }
    } else if constexpr (SORTED_VALUE) {
        if (sort_needed) {
            // The merge path takes a 32-aligned count. A short row sorts only its valid elements,
            // because the fill values that follow let nothing be compared against them (a fill can
            // be smaller than a valid value); a full row sorts topk, padded with -inf. Either way
            // only the smallest aligned window above the sorted count is touched.
            const bool short_row = valid_len <= topk;
            const uint32_t sort_input = short_row ? valid_len : topk;
            const uint32_t sort_count = aligned(sort_input, SORT_ELEMENT_ALIGN);
            pad_sort_tail_vf(ub, sort_input, sort_count);
            // The 6-argument overload takes the row indices as the sort payload, so it returns the
            // sorted values and the sorted indices together and the permutation gather that the
            // radix path needs disappears.
            LocalTensor<float> source_value_tensor(TPosition::VECCALC, 0, SORT_COUNT);
            LocalTensor<uint32_t> source_index_tensor(TPosition::VECCALC, CANDIDATE_ARRAY_BYTES,
                                                      SORT_COUNT);
            LocalTensor<float> merged_tensor(TPosition::VECCALC, Ub::SORT_ARRAY0_OFFSET,
                                             2 * SORT_COUNT);
            LocalTensor<uint32_t> index_tensor(TPosition::VECCALC, Ub::SORT_ARRAY1_OFFSET,
                                               SORT_COUNT);
            LocalTensor<uint8_t> temporary_tensor(TPosition::VECCALC, Ub::SORT_TEMPORARY_OFFSET,
                                                  SORT_TEMPORARY_BYTES);
            // A row with no valid element has nothing to sort, and the library requires a
            // positive count; the fill restoration below still covers the whole output.
            if (sort_count != 0) {
                AscendC::Sort<float, uint32_t, true, SORT_MERGE_DESCENDING>(
                    merged_tensor, index_tensor, source_value_tensor, source_index_tensor,
                    temporary_tensor, sort_count);
            }
            output_value = reinterpret_cast<__ubuf__ float*>(Ub::SORT_ARRAY0_OFFSET);
            output_index = reinterpret_cast<__ubuf__ uint32_t*>(Ub::SORT_ARRAY1_OFFSET);
            if (short_row) restore_short_row_fill_vf(ub, sort_input, topk, value_oob_fill, idx_oob_fill);
        }
    }

    if (return_value) {
        event_t value_ready = event_t(Flags::OUTPUT_READY);
        set_flag(PIPE_V, PIPE_MTE3, value_ready);
        wait_flag(PIPE_V, PIPE_MTE3, value_ready);
        copy_ubuf_to_gm_align_v2(
            row_out_value, output_value, 0, 1, topk * sizeof(float),
            static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV), topk * sizeof(float),
            topk * sizeof(float));
    }
    store_index(ub, reinterpret_cast<__ubuf__ int32_t*>(output_index), row.row_out_index, topk);

    // Already satisfied: the flag was raised before the output work, and nothing after that point
    // touches the counter.
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
            fill_nan_sentinel_vf(ub, reinterpret_cast<__ubuf__ int32_t*>(output_index), topk);
            store_index(ub, reinterpret_cast<__ubuf__ int32_t*>(output_index), row.row_out_index,
                        topk);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-row dispatch, __global__ wrapper and host-side launch
// ---------------------------------------------------------------------------
// One row = one block.
template <Config CONFIG>
__aicore__ inline void run_topk_select_row(const TopkSelectArgs& args) {
    const uint32_t row_idx = get_block_idx();
    RowArgs row;
    const uint32_t raw_valid_len =
        args.end_ptr == nullptr ? args.vocab_size : static_cast<uint32_t>(args.end_ptr[row_idx]);
    row.valid_len = raw_valid_len < args.vocab_size ? raw_valid_len : args.vocab_size;
    row.topk = args.topk;
    row.abort_when_nan_found = args.abort_when_nan_found;
    row.row_in = reinterpret_cast<__gm__ float*>(args.input) + row_idx * args.stride_input_batch;
    row.row_out_value =
        args.return_value
            ? reinterpret_cast<__gm__ float*>(args.output_value) +
                  row_idx * args.stride_output_value_batch
            : nullptr;
    row.row_out_index = reinterpret_cast<__gm__ void*>(
        reinterpret_cast<__gm__ typename Kernel<CONFIG>::OutIdxT*>(args.output_index) +
        row_idx * args.stride_output_index_batch);
    row.output_idx_offset = args.output_idx_offset == nullptr ? 0 : args.output_idx_offset[row_idx];
    row.return_value = args.return_value;
    row.idx_oob_fill = args.idx_oob_fill_value;
    row.value_oob_fill = args.value_oob_fill_value;

    Kernel<CONFIG>::run(row);
}

template <Config CONFIG>
__global__ __vector__ __aicore__ void topk_select_kernel(const TopkSelectArgs args) {
    AscendC::InitSocState();
    int64_t pc = asc_get_program_counter() & 0xFFFFFFFFFFFF;
        asc_icache_preload(reinterpret_cast<void *>(pc), 8);    // 8 = 16K (AIV's icache size) / 2K

    run_topk_select_row<CONFIG>(args);
}

// Host-side launch. Runtime invariants that would otherwise silently produce wrong output (a
// mismatched dispatch, for instance) are checked here, not in the kernel body.
template <Config CONFIG>
void run_topk_select_kernel(const TopkSelectArgs& args) {
    KU_ASSERT_NPU(args.topk <= CONFIG.MAX_TOPK,
                  "topk (%u) exceeds the compiled MAX_TOPK (%u): dispatch failure", args.topk,
                  CONFIG.MAX_TOPK);
    topk_select_kernel<CONFIG><<<args.batch_size, TOPK_UB_LIMIT_BYTES, args.stream>>>(args);
    KU_ACLRT_CHECK_NPU(aclrtGetLastError(aclrtLastErrLevel::ACL_RT_THREAD_LEVEL));
}

/*
Widen `count` int32 indices into int64, sign extended. One source vector (64 int32) turns into two
destination vectors: `asc_intlv` interleaves the low and the high word of each value, and its two
outputs are the first and the second half of the widened window. The tail window is masked per
half, because an output lane is a 32-bit word while the valid region is measured in values.
*/
template <Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::widen_index_vf(__ubuf__ Ub& ub, __ubuf__ int32_t* source,
                                                __ubuf__ int64_t* destination, uint32_t count) {
    vector_bool mask_u32 = asc_create_mask_b32(PAT_ALL);
    vector_int32_t minus_one, zero;
    asc_duplicate_scalar(minus_one, int32_t(-1), mask_u32);
    asc_duplicate_scalar(zero, int32_t(0), mask_u32);
    auto* source_ptr = source;
    auto* destination_ptr = reinterpret_cast<__ubuf__ int32_t*>(destination);
    constexpr uint32_t HALF = NUM_FP32_PER_VEC / 2;
    #pragma unroll 1
    for (uint32_t offset = 0; offset < count; offset += NUM_FP32_PER_VEC) {
        vector_int32_t value;
        asc_loadalign_postupdate(value, source_ptr, static_cast<int32_t>(NUM_FP32_PER_VEC));
        vector_bool negative;
        asc_lt_scalar(negative, value, int32_t(0), mask_u32);
        vector_int32_t high;
        asc_select(high, minus_one, zero, negative);
        vector_int32_t low_half, high_half;
        asc_intlv(low_half, high_half, value, high);
        // The last window may hold fewer than a full vector. Each value occupies two output lanes,
        // so the two halves are masked separately; the counts are locals because
        // `asc_update_mask_b32` post-decrements its argument.
        const uint32_t window = count - offset < NUM_FP32_PER_VEC ? count - offset
                                                                 : NUM_FP32_PER_VEC;
        uint32_t low_valid = window < HALF ? 2 * window : NUM_FP32_PER_VEC;
        uint32_t high_valid = window <= HALF ? 0 : 2 * (window - HALF);
        vector_bool low_mask = asc_update_mask_b32(low_valid);
        vector_bool high_mask = asc_update_mask_b32(high_valid);
        asc_storealign(destination_ptr, low_half, low_mask);
        asc_storealign(destination_ptr + NUM_FP32_PER_VEC, high_half, high_mask);
        destination_ptr += 2 * NUM_FP32_PER_VEC;
    }
}

template <Config CONFIG>
__aicore__ void Kernel<CONFIG>::store_index(__ubuf__ Ub& ub, __ubuf__ int32_t* source,
                                            __gm__ void* destination, uint32_t count) {
    auto* out = reinterpret_cast<__gm__ OutIdxT*>(destination);
    __ubuf__ int32_t* copy_source = source;
    uint32_t copy_bytes = count * sizeof(int32_t);
    if constexpr (OUT_IDX_INT64) {
        // The widened copy is twice as wide, so it needs a staging area; the Sort temporary buffer
        // is free here and sized for it (see OUT_IDX_WIDEN_BYTES in config.h).
        auto* widened = reinterpret_cast<__ubuf__ int64_t*>(ub.overlay.sort.temporary);
        widen_index_vf(ub, source, widened, count);
        copy_source = reinterpret_cast<__ubuf__ int32_t*>(widened);
        copy_bytes = count * sizeof(int64_t);
    }
    event_t output_ready = event_t(Flags::OUTPUT_READY);
    set_flag(PIPE_V, PIPE_MTE3, output_ready);
    wait_flag(PIPE_V, PIPE_MTE3, output_ready);
    copy_ubuf_to_gm_align_v2(out, copy_source, 0, 1, copy_bytes,
                             static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
                             copy_bytes, copy_bytes);
}

}  // namespace topk_select_ascend_fp32
