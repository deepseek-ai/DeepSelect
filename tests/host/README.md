# Host setup regression tests

These tests use real CUDA headers and a mocked driver resolver and encoder. They require CUDA 13's compiler, but do not initialize CUDA or require a GPU. The mock records encoder arguments rather than validating device-specific tensor-map constraints.

From the repository root, build into a directory outside the checkout:

```sh
nvcc -std=c++20 -O2 --cudart shared -I csrc/3rdparty/kerutils/include -Xcompiler -pthread tests/host/tensor_map.cu tests/host/tensor_map_peer.cu tests/host/allocations.cpp -o /path/to/artifacts/tensor-map-test
/path/to/artifacts/tensor-map-test
/path/to/artifacts/tensor-map-test retry
```

The normal process tests concurrent cold initialization, one lookup shared across two translation units, current per-call tensor arguments, retry after an encoder failure, and vector/array equivalence for ranks one through five with default and explicit element strides. The separate `retry` process additionally tests runtime lookup failure, missing symbol, and a null function pointer before successful initialization. Assertion diagnostics from these intentional failures are expected.

For the baseline regression, compile the same sources with `-DTEST_BASELINE` and point the include path at baseline kerutils headers. Run without `retry`: the single-lookup assertion must fail after sixteen concurrent calls. This mode excludes the new array overload and does not attempt the baseline's unsafe null-pointer call.

Run either binary with `benchmark` to measure rank-three metadata preparation through its actual headers. It warms up 1,000 calls, then reports seven rounds of 200,000 calls, allocations, and resolver calls. The baseline uses the original vector/stride-helper path; the candidate uses fixed arrays. Both use the same recording mock encoder. This isolates host preparation and does not measure driver encoding, kernel launch, or end-to-end Top-K latency. Alternate baseline/candidate runs to check timing noise. The candidate additionally asserts zero steady-state allocations and lookups.
