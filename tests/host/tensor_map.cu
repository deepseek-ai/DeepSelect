// Real CUDA types and ABI, mocked driver lookup/encoder: no GPU is required.
#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>
#include <array>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>
#include <vector>
#include "kerutils/host/host.h"

extern thread_local bool count_allocations;
extern thread_local size_t allocations;

void encode_from_peer();
static std::atomic<int> lookups{0}, encodes{0};
static int lookup_failure = 0;
static bool encode_failure = false;
struct Arguments {
    unsigned rank;
    void* pointer;
    CUtensorMapDataType dtype;
    CUtensorMapInterleave interleave;
    CUtensorMapSwizzle swizzle;
    CUtensorMapL2promotion promotion;
    CUtensorMapFloatOOBfill fill;
    std::array<uint64_t, 5> sizes{}, strides{};
    std::array<uint32_t, 5> boxes{}, elements{};
    bool operator==(const Arguments&) const = default;
};
static thread_local Arguments recorded;

static CUresult CUDAAPI encode(CUtensorMap* result, CUtensorMapDataType dtype,
    cuuint32_t rank, void* pointer, const cuuint64_t* sizes,
    const cuuint64_t* strides, const cuuint32_t* boxes, const cuuint32_t* elements,
    CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion promotion, CUtensorMapFloatOOBfill fill) {
    ++encodes;
    assert(rank >= 1 && rank <= 5);
    recorded = {};
    recorded.rank = rank; recorded.pointer = pointer; recorded.dtype = dtype;
    recorded.interleave = interleave; recorded.swizzle = swizzle;
    recorded.promotion = promotion; recorded.fill = fill;
    std::copy_n(sizes, rank, recorded.sizes.begin());
    std::copy_n(strides, rank - 1, recorded.strides.begin());
    std::copy_n(boxes, rank, recorded.boxes.begin());
    std::copy_n(elements, rank, recorded.elements.begin());
    std::memset(result, 0, sizeof(*result));
    return encode_failure ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(
    const char* name, void** pointer, unsigned version, unsigned long long flags,
    cudaDriverEntryPointQueryResult* status) {
    ++lookups;
    assert(std::strcmp(name, "cuTensorMapEncodeTiled") == 0);
    assert(version == 12000 && flags == cudaEnableDefault);
    *status = lookup_failure == 2 ? cudaDriverEntryPointSymbolNotFound : cudaDriverEntryPointSuccess;
    *pointer = lookup_failure == 3 ? nullptr : reinterpret_cast<void*>(&encode);
    return lookup_failure == 1 ? cudaErrorUnknown : cudaSuccess;
}

template<class F> void throws(F f) {
    bool caught = false;
    try { f(); } catch (const ku::KUException&) { caught = true; }
    assert(caught);
}

#ifndef TEST_BASELINE
template<size_t Rank> void compare_storage() {
    std::array<uint64_t, Rank> sizes;
    std::array<uint64_t, Rank - 1> strides;
    std::array<uint32_t, Rank> boxes, elements;
    sizes.fill(32); strides.fill(128); boxes.fill(1); elements.fill(2);
    for (bool explicit_elements : {false, true}) {
        const auto ptr = reinterpret_cast<void*>(0x2000 + Rank * 128);
        const auto dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        const auto swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
        const auto promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
        const auto interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
        const auto fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
        std::vector<uint32_t> e;
        if (explicit_elements) e.assign(elements.begin(), elements.end());
        ku::make_tensor_map({sizes.begin(), sizes.end()}, {strides.begin(), strides.end()},
            {boxes.begin(), boxes.end()}, ptr, dtype, swizzle, promotion, interleave, fill, e);
        auto expected = recorded;
        if (explicit_elements)
            ku::make_tensor_map<Rank>(sizes, strides, boxes, ptr, dtype, swizzle, promotion, interleave, fill, elements);
        else
            ku::make_tensor_map<Rank>(sizes, strides, boxes, ptr, dtype, swizzle, promotion, interleave, fill);
        assert(recorded == expected);
    }
}
#endif

// Measures actual metadata wrappers with a mock encoder, not CUDA execution.
void benchmark() {
    constexpr int count = 200000;
    auto prepare = [](int i) {
        const uint64_t vocab = 8192 + (i & 31);
#ifdef TEST_BASELINE
        ku::make_tensor_map({32, (vocab + 31) / 32, 6},
            ku::make_stride_helper<uint64_t>({32, 8448}, 4), {32, 8, 1},
#else
        ku::make_tensor_map<3>({32, (vocab + 31) / 32, 6},
            {128, 33792}, {32, 8, 1},
#endif
            reinterpret_cast<void*>(0x1000), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        assert(recorded.sizes[1] == (vocab + 31) / 32);
    };
    for (int i = 0; i < 1000; ++i) prepare(i);
    for (int round = 0; round < 7; ++round) {
        allocations = 0;
        const int before = lookups;
        count_allocations = true;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < count; ++i) prepare(i);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        count_allocations = false;
        std::printf("round=%d calls=%d allocations=%zu lookups=%d ns_per_call=%.3f\n",
            round, count, allocations, int(lookups) - before,
            std::chrono::duration<double, std::nano>(elapsed).count() / count);
#ifndef TEST_BASELINE
        assert(allocations == 0 && int(lookups) == before);
#endif
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "benchmark") == 0) {
        benchmark();
        return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "retry") == 0) {
        for (int failure : {1, 2, 3}) {
            lookup_failure = failure;
            throws(encode_from_peer);
            assert(lookups == failure && encodes == 0);
        }
        lookup_failure = 0;
    }
    const int failed_lookups = lookups;
    std::atomic<int> ready{0}; std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) threads.emplace_back([&] {
        ++ready; while (!start.load()) std::this_thread::yield(); encode_from_peer();
    });
    while (ready != 16) std::this_thread::yield();
    start = true;
    for (auto& t : threads) t.join();
    assert(encodes == 16);
    assert(lookups == failed_lookups + 1);
    ku::make_tensor_map({64, 8, 3}, {256, 2048}, {64, 2, 1},
        reinterpret_cast<void*>(0x3000), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE);
    assert(recorded.pointer == reinterpret_cast<void*>(0x3000));
    assert(recorded.sizes[0] == 64 && recorded.sizes[1] == 8 && recorded.sizes[2] == 3);
    assert(recorded.strides[0] == 256 && recorded.strides[1] == 2048);
    assert(recorded.boxes[1] == 2 && recorded.dtype == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16);
    assert(lookups == failed_lookups + 1); // Shared with the other translation unit.
    encode_failure = true;
    throws(encode_from_peer);
    encode_failure = false;
    encode_from_peer();
    assert(encodes == 19 && lookups == failed_lookups + 1);
#ifndef TEST_BASELINE
    compare_storage<1>(); compare_storage<2>(); compare_storage<3>();
    compare_storage<4>(); compare_storage<5>();
#endif
    std::puts("tensor-map host tests passed");
}
