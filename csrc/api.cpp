// This file contains only the host-side topk() function and pybind11 module.
// Kernel template instantiations are in separate files for parallel compilation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

#include "kerutils/supplemental/torch_tensors.h"

#include "dispatch_utils.h"

#include "cuda_kernels/config.h"
#include "cuda_kernels/v3/topk_select.h"
#include "cuda_kernels/v3_fp32/topk_select.h"
#include <cstdlib>
#include "cuda_kernels/v3_cluster/topk_select.h"

void topk(
    torch::Tensor &input,
    int topk,
    c10::optional<torch::Tensor> &begin,
    c10::optional<torch::Tensor> &end,
    bool sorted_value,
    bool sorted_index,
    c10::optional<torch::Tensor> &output_value,
    torch::Tensor &output_index,
    c10::optional<torch::Tensor> &output_idx_offset,
    int idx_oob_fill_value,
    float value_oob_fill_value,
    bool return_value,
    bool abort_when_nan_found
) {
    int batch_size = input.size(0);
    int vocab_size = input.size(1);
    at::ScalarType value_t = input.scalar_type();
    at::ScalarType output_index_t = output_index.scalar_type();

    TORCH_CHECK(topk > 0, "topk must > 0");
    TORCH_CHECK(!(sorted_value && !return_value), "`return_value` must be enabled when `sorted_value` is True");
    TORCH_CHECK(!(sorted_value && sorted_index), "`sorted_value` and `sorted_index` cannot be used at the same time");
    // Contract: sorted_value is a 32-bit-value-only feature.
    TORCH_CHECK(!(sorted_value && value_t == at::kBFloat16), "`sorted_value` is only supported for float32 input");
    TORCH_CHECK(!begin.has_value(), "`begin` is not supported currently");
    if (return_value) {
        TORCH_CHECK(output_value.has_value(), "`output_value` must not be `None` when `return_value` is True");
    }
    
    KU_CHECK_DEVICE(input);
    KU_CHECK_DEVICE(begin);
    KU_CHECK_DEVICE(end);
    KU_CHECK_DEVICE(output_value);
    KU_CHECK_DEVICE(output_index);
    KU_CHECK_DEVICE(output_idx_offset);
    
    KU_CHECK_SHAPE(input, batch_size, vocab_size);
    KU_CHECK_SHAPE(begin, batch_size);
    KU_CHECK_SHAPE(end, batch_size);
    KU_CHECK_SHAPE(output_value, batch_size, topk);
    KU_CHECK_SHAPE(output_index, batch_size, topk);
    KU_CHECK_SHAPE(output_idx_offset, batch_size);
    
    KU_CHECK_DTYPE(input, value_t);
    KU_CHECK_DTYPE(begin, at::kInt);
    KU_CHECK_DTYPE(end, at::kInt);
    KU_CHECK_DTYPE(output_value, value_t);
    KU_CHECK_DTYPE(output_index, output_index_t);
    KU_CHECK_DTYPE(output_idx_offset, at::kInt);
    
    KU_CHECK_LAST_DIM_CONTIGUOUS(input);
    KU_CHECK_CONTIGUOUS(begin);
    KU_CHECK_CONTIGUOUS(end);
    KU_CHECK_LAST_DIM_CONTIGUOUS(output_value);
    KU_CHECK_LAST_DIM_CONTIGUOUS(output_index);
    KU_CHECK_CONTIGUOUS(output_idx_offset);

    auto check_dim0_stride = [&](const char tensor_name[], torch::Tensor &tensor, uint32_t alignment_requirement_bytes) {
        int64_t cur_stride = tensor.stride(0);
        uint64_t itemsize = tensor.dtype().itemsize();
        TORCH_CHECK(cur_stride * itemsize % alignment_requirement_bytes == 0,
            std::format("{}.stride(0) (currently {} numbers) must be a multiple of {} Bytes ({} numbers)",
                tensor_name, cur_stride,
                alignment_requirement_bytes, alignment_requirement_bytes / itemsize
            )
        );
    };
    check_dim0_stride("input", input, INPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    check_dim0_stride("output_index", output_index, OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    if (output_value.has_value()) {
        check_dim0_stride("value", *output_value, OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    }

    // Row strides do not guarantee alignment of a sliced tensor's first row.
    auto check_pointer_alignment = [](const char* name, const torch::Tensor& tensor) {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 32 == 0,
                    name, ".data_ptr() must be 32-byte aligned");
    };
    check_pointer_alignment("input", input);
    check_pointer_alignment("output_index", output_index);
    if (output_value.has_value())
        check_pointer_alignment("output_value", *output_value);
    TORCH_CHECK(batch_size <= 1 || output_index.stride(0) >= topk,
                "output_index rows must not overlap");
    if (output_value.has_value())
        TORCH_CHECK(batch_size <= 1 || output_value->stride(0) >= topk,
                    "output_value rows must not overlap");

    // The 3D TMA descriptor rounds the vocabulary dimension up to 128 bytes.
    // The final row needs actual backing storage for that padding as well.
    if (batch_size > 0 && vocab_size > 0) {
        const uint64_t itemsize = input.element_size();
        const uint64_t row_bytes = uint64_t(vocab_size) * itemsize;
        const uint64_t padded_row_bytes = (row_bytes + 127) / 128 * 128;
        const uint64_t last_row_offset =
            (uint64_t(input.storage_offset()) + uint64_t(batch_size - 1) * input.stride(0)) * itemsize;
        const uint64_t storage_bytes = input.storage().nbytes();
        TORCH_CHECK(last_row_offset <= storage_bytes && padded_row_bytes <= storage_bytes - last_row_offset,
                    "input backing storage must include the final row padded to a multiple of 128 bytes");
    }

    cudaDeviceProp* device_prop = at::cuda::getDeviceProperties(at::cuda::current_device());
    TORCH_CHECK(device_prop != nullptr);
    TopkSelectArgs args = {
        (uint32_t)batch_size,
        (uint32_t)vocab_size,
        (uint32_t)topk,

        input.data_ptr(),
        ku::get_optional_tensor_ptr<void>(output_value),
        output_index.data_ptr(),
        ku::get_optional_tensor_ptr<int>(begin),
        ku::get_optional_tensor_ptr<int>(end),
        ku::get_optional_tensor_ptr<int>(output_idx_offset),

        (uint64_t)input.stride(0),
        output_value.has_value() ? (uint64_t)output_value->stride(0) : 0,
        (uint64_t)output_index.stride(0),

        sorted_value,
        sorted_index,
        return_value,
        idx_oob_fill_value,
        value_oob_fill_value,
        abort_when_nan_found,

        device_prop->sharedMemPerBlockOptin,
        at::cuda::getCurrentCUDAStream().stream()
    };

    uint32_t num_sm = device_prop->multiProcessorCount;
    uint32_t num_waves = (batch_size + num_sm-1) / num_sm;

    TORCH_CHECK(value_t == at::kBFloat16 || value_t == at::kFloat, "input dtype must be bfloat16 or float32");
    if (value_t == at::kBFloat16) {
        TORCH_CHECK((uint32_t)vocab_size < MAX_VOCAB_SIZE,
                    "vocab_size must be < 2^23 for bfloat16 input");
        TORCH_CHECK(topk <= 4096, "topk must be <= 4096");
        if (batch_size <= 6 && (uint32_t)vocab_size >= 512u * 1024u && topk <= 1024) {  // TODO Tune
            INTEGER_TYPE_SWITCH(output_index_t, OutIdxT, [&]() {
                BOOL_SWITCH(sorted_index, SORTED_INDEX, [&]() {
                    BOOL_SWITCH(return_value, RETURN_VALUE, [&]() {
                        topk_select_bf16_cluster::run_topk_select_kernel<
                            TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, 1024, 256, 1, 4096, 4096, 16, 512, 16>>(args);
                    });
                });
            });
        } else {
            INTEGER_TYPE_SWITCH(output_index_t, OutIdxT, [&]() {
                BOOL_SWITCH(sorted_index, SORTED_INDEX, [&]() {
                    BOOL_SWITCH(return_value, RETURN_VALUE, [&]() {
                        //   wave == 1 -> occ1 (512t / B8192 / B2 4096 / TMA5 rounds)
                        //   otherwise -> occ2 (256t / B4096 / B2 4096 / TMA3 or 4)
                        auto dispatch = [&]<uint32_t MAX_TOPK>() {
                            if (num_waves == 1)
                                topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, 512, 1, 8192, 4096, 5>>(args);
                            else if constexpr (MAX_TOPK <= 512)
                                topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, 256, 2, 4096, 4096, 4>>(args);
                            else
                                topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, 256, 2, 4096, 4096, 3>>(args);
                        };
                        if (topk <= 512) {
                            dispatch.template operator()<512>();
                        } else if (topk <= 1024) {
                            dispatch.template operator()<1024>();
                        } else {
                            // Big-topk coverage tier, topk in (1024, 4096]: one correctness-only tuple
                            // (512t / occ1 / B8192 / B2 4096 / TMA3 / max_topk 4096), no wave split.
                            topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, 4096, 512, 1, 8192, 4096, 3>>(args);
                        }
                    });
                });
            });
        }
    } else {
        TORCH_CHECK((uint32_t)vocab_size < MAX_VOCAB_SIZE,
                    "vocab_size must be < 2^23 for float32 input");
        TORCH_CHECK(topk <= 4096, "topk must be <= 4096");

        INTEGER_TYPE_SWITCH(output_index_t, OutIdxT, [&]() {
            //   topk <= 1024        -> 512t / B8192 / B2 4096 / TMA3
            //   topk in (1024,4096] -> 256t / B4096 / B2 4096 / TMA3 (correctness-only coverage tier)
            auto dispatch = [&]<bool SORTED_VALUE, bool SORTED_INDEX, bool RETURN_VALUE>() {
                auto launch = [&]<uint32_t MAX_TOPK, uint32_t NUM_THREADS, uint32_t B>() {
                    topk_select_fp32::run_topk_select_kernel<TopkSelectConfig<float, OutIdxT, SORTED_VALUE, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, NUM_THREADS, 1, B, 4096, 3>>(args);
                };
                if (topk <= 512) {
                    launch.template operator()<512, 512, 8192>();
                } else if (topk <= 1024) {
                    launch.template operator()<1024, 512, 8192>();
                } else {
                    launch.template operator()<4096, 256, 4096>();
                }
            };
            if (sorted_value) {
                dispatch.template operator()<true, false, true>();
            } else {
                BOOL_SWITCH(sorted_index, SORTED_INDEX, [&]() {
                    BOOL_SWITCH(return_value, RETURN_VALUE, [&]() {
                        dispatch.template operator()<false, SORTED_INDEX, RETURN_VALUE>();
                    });
                });
            }
        });
    }
}

std::pair<uint32_t, uint32_t> get_alignment_requirement() {
    return {INPUT_STRIDE_ALIGNMENT_REQUIREMENT, OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT};

}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk", &topk);
    m.def("get_alignment_requirement", &get_alignment_requirement);
}
