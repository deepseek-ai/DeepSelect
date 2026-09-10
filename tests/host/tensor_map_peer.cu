#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>
#include "kerutils/host/host.h"

void encode_from_peer() {
    ku::make_tensor_map({32, 4, 2}, {128, 512}, {32, 1, 1},
                       reinterpret_cast<void*>(0x1000), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                       CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE);
}
