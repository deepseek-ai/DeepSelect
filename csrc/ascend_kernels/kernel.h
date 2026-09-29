#pragma once

#include <cstdint>

#include "structs.h"

namespace topk_select_ascend {

// Compile-time knobs of the Ascend kernel.
struct Config {
    uint32_t MAX_TOPK;          // Tier upper bound: 512, 1024 or 4096
    bool ABORT_WHEN_NAN_FOUND;  // true: abort the kernel on NaN; false: only write the NaN guard index
    bool SORTED_INDEX;
};

template <Config CONFIG>
void run_topk_select_kernel(const TopkSelectArgs& args);

}  // namespace topk_select_ascend
