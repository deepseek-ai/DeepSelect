#pragma once

#include <cstdint>

#include "structs.h"

namespace topk_select_ascend_fp32 {

// Compile-time knobs of the fp32 kernel: one kernel instantiation per combination.
// `SORTED_INDEX` and `SORTED_VALUE` are mutually exclusive (the API rejects both at once).
struct Config {
    uint32_t MAX_TOPK;
    bool SORTED_INDEX;
    bool SORTED_VALUE;
    bool OUT_IDX_INT64;  // Output index type: false = int32, true = int64
};

template <Config CONFIG> void run_topk_select_kernel(const TopkSelectArgs& args);

}  // namespace topk_select_ascend_fp32
