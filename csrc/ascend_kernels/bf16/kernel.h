#pragma once

#include <cstdint>

#include "structs.h"

namespace topk_select_ascend {

// Compile-time knobs of the Ascend kernel: one kernel instantiation per combination.
struct Config {
    uint32_t MAX_TOPK;  // Tier upper bound: 512, 1024 or 4096
    bool SORTED_INDEX;
};

template <Config CONFIG> void run_topk_select_kernel(const TopkSelectArgs& args);

}  // namespace topk_select_ascend
