#include "../topk_select.cuh"

namespace topk_select_fp32 {

template
void run_topk_select_kernel<
    TopkSelectConfig<float, int64_t, false, true, true, 256, 256, 2, 4096, 4096, 2, 512, 1>
>(const TopkSelectArgs &args);

}   // topk_select_fp32
