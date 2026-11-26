#include "../topk_select.cuh"

namespace topk_select_fp32 {

template
void run_topk_select_kernel<
    TopkSelectConfig<float, int32_t, false, false, true, 256, 256, 2, 4096, 768, 4, 512, 1>
>(const TopkSelectArgs &args);

}   // topk_select_fp32
