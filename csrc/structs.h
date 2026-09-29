#pragma once

#include <cstdint>

#if defined(DEEP_SELECT_IS_BUILD_ON_CUDA) && defined(DEEP_SELECT_IS_BUILD_ON_ASCEND)
#error "DEEP_SELECT_IS_BUILD_ON_CUDA and DEEP_SELECT_IS_BUILD_ON_ASCEND cannot both be defined"
#endif

#if defined(DEEP_SELECT_IS_BUILD_ON_CUDA)

#include <cuda_runtime_api.h>

#define TOPK_SELECT_GM

using topk_select_stream_t = cudaStream_t;

static constexpr uint32_t INPUT_STRIDE_ALIGNMENT_REQUIREMENT = 1024; // In number of bytes
static constexpr uint32_t OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT = 32; // In number of bytes

static constexpr uint32_t MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION = 1u << 23;
static constexpr uint32_t MAX_VOCAB_SIZE = 1u << 23;
static_assert(MAX_VOCAB_SIZE <= MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION);

#elif defined(DEEP_SELECT_IS_BUILD_ON_ASCEND)

#include <acl/acl_base_rt.h>

// `__gm__` qualifies the pointers that live in Ascend global memory. It only
// exists in the device pass of a `.asc` translation unit and is empty in every
// other pass (host pass of a `.asc`, plain `.cpp`).
#if defined(__NPU_DEVICE__)
#define TOPK_SELECT_GM __gm__
#else
#define TOPK_SELECT_GM
#endif

using topk_select_stream_t = aclrtStream;

static constexpr uint32_t INPUT_STRIDE_ALIGNMENT_REQUIREMENT = 32; // In number of bytes
static constexpr uint32_t OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT = 32; // In number of bytes

static constexpr uint32_t MAX_VOCAB_SIZE = 2 * 1024 * 1024;

#else
#error "Exactly one of DEEP_SELECT_IS_BUILD_ON_CUDA / DEEP_SELECT_IS_BUILD_ON_ASCEND must be defined"
#endif

struct TopkSelectArgs {
    uint32_t batch_size;
    uint32_t vocab_size;
    uint32_t topk;

    // Input / output pointers. Both backends keep them type-erased here; the
    // typed kernel casts them once on entry.
    TOPK_SELECT_GM void* input;
    TOPK_SELECT_GM void* output_value;
    TOPK_SELECT_GM void* output_index;
    TOPK_SELECT_GM int* begin_ptr;
    TOPK_SELECT_GM int* end_ptr;
    TOPK_SELECT_GM int* output_idx_offset;

    // All strides are in number of elements, not bytes
    uint64_t stride_input_batch;
    uint64_t stride_output_value_batch;
    uint64_t stride_output_index_batch;

    bool sorted_value;
    bool sorted_index;
    bool return_value;
    int idx_oob_fill_value;
    float value_oob_fill_value;
    bool abort_when_nan_found;

#if defined(DEEP_SELECT_IS_BUILD_ON_CUDA)
    uint64_t shared_memory_size_per_sm;
#else
    // Ascend writes 16-bit bf16 lanes directly, so the host pre-rounds the fill
    // value and stores its bit pattern; CUDA keeps using `value_oob_fill_value`.
    uint16_t value_oob_fill_bits;
#endif

    topk_select_stream_t stream;
};
