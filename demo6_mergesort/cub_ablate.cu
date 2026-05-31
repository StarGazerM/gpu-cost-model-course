// Ablate CUB's REAL merge sort, one knob at a time -- the rigorous direction:
// start from exactly what CUB runs and REMOVE optimizations, so the full config is
// CUB's runtime by construction (no unknown gap) and each delta is one opt's worth.
//
// CUB's sm_89 policy (cub/device/dispatch/tuning/tuning_merge_sort.cuh, Policy600):
//   AgentMergeSortPolicy<256, 17, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE>
// We instantiate cub::DispatchMergeSort with a CUSTOM PolicyHub to dial those knobs.
//
//   nvcc -O3 -std=c++17 -arch=sm_89 -o /tmp/cuba cub_ablate.cu && /tmp/cuba 28 10

#include "../demo2_sort/common.cuh"
#include <cub/device/dispatch/dispatch_merge_sort.cuh>
#include <cub/agent/agent_merge_sort.cuh>
#include <cstdint>

struct LessOp {
  __host__ __device__ bool operator()(const int& a, const int& b) const { return a < b; }
};

// a PolicyHub whose single policy is CUB's merge agent policy with our chosen knobs.
template <int BT, int IPT, cub::BlockLoadAlgorithm LOAD,
          cub::CacheLoadModifier LMOD, cub::BlockStoreAlgorithm STORE>
struct AblHub {
  struct MaxPolicy : cub::ChainedPolicy<350, MaxPolicy, MaxPolicy> {
    using MergeSortPolicy = cub::AgentMergeSortPolicy<BT, IPT, LOAD, LMOD, STORE>;
  };
};

template <class Hub>
static float bench(const char* label, int* d, int* d_pristine, size_t n, int iters) {
  using DispatchT = cub::DispatchMergeSort<int*, cub::NullType*, int*, cub::NullType*,
                                           std::int64_t, LessOp, Hub>;
  std::size_t tmp_bytes = 0;
  DispatchT::Dispatch(nullptr, tmp_bytes, d, nullptr, d, nullptr, (std::int64_t)n, LessOp(), 0);
  void* d_tmp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
  size_t bytes = n * sizeof(int);

  // correctness once
  CUDA_CHECK(cudaMemcpy(d, d_pristine, bytes, cudaMemcpyDeviceToDevice));
  DispatchT::Dispatch(d_tmp, tmp_bytes, d, nullptr, d, nullptr, (std::int64_t)n, LessOp(), 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> out(n);
  CUDA_CHECK(cudaMemcpy(out.data(), d, bytes, cudaMemcpyDeviceToHost));
  bool ok = true;
  for (size_t i = 1; i < n; ++i) if (out[i] < out[i - 1]) { ok = false; break; }

  float ms = demo2::time_best_ms(iters, [&] {
    cudaMemcpyAsync(d, d_pristine, bytes, cudaMemcpyDeviceToDevice);
    DispatchT::Dispatch(d_tmp, tmp_bytes, d, nullptr, d, nullptr, (std::int64_t)n, LessOp(), 0);
  });
  double d2d = bytes / 810e9 * 1e3;
  double mss = ms - d2d;
  printf("  %-34s %8.2f ms  %8.1f Mkeys/s  [%s]\n", label, mss, n / 1e6 / (mss / 1e3), ok ? "PASS" : "FAIL");
  cudaFree(d_tmp);
  return (float)mss;
}

int main(int argc, char** argv) {
  int log2n = (argc > 1) ? atoi(argv[1]) : 28;
  int iters = (argc > 2) ? atoi(argv[2]) : 10;
  size_t n = (size_t)1 << log2n, bytes = n * sizeof(int);
  auto h = demo2::make_uniform_input(n);
  int *d = nullptr, *d_pristine = nullptr;
  CUDA_CHECK(cudaMalloc(&d, bytes));
  CUDA_CHECK(cudaMalloc(&d_pristine, bytes));
  CUDA_CHECK(cudaMemcpy(d_pristine, h.data(), bytes, cudaMemcpyHostToDevice));

  printf("Ablating CUB's REAL merge kernels (n=2^%d). FULL = CUB's sm_89 policy:\n", log2n);
  using cub::BLOCK_LOAD_WARP_TRANSPOSE; using cub::BLOCK_STORE_WARP_TRANSPOSE;
  using cub::BLOCK_LOAD_DIRECT;         using cub::BLOCK_STORE_DIRECT;
  using cub::LOAD_DEFAULT;

  // FULL: exactly CUB's Policy600 -> must reproduce cub::DeviceMergeSort (~47.6 ms).
  bench<AblHub<256, 17, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE>>(
      "FULL (CUB policy: 256x17, transpose)", d, d_pristine, n, iters);
  // -coalesce: keep 17 items but DIRECT (un-transposed) load/store -> uncoalesced.
  bench<AblHub<256, 17, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_STORE_DIRECT>>(
      "  - coalescing (DIRECT load/store)", d, d_pristine, n, iters);
  // -items: drop items/thread 17 -> 8 -> 4 -> 1 (smaller tiles, less ILP/occupancy).
  bench<AblHub<256, 8, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE>>(
      "  - items/thread (17 -> 8)", d, d_pristine, n, iters);
  bench<AblHub<256, 4, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE>>(
      "  - items/thread (17 -> 4)", d, d_pristine, n, iters);
  bench<AblHub<256, 1, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE>>(
      "  - items/thread (17 -> 1)", d, d_pristine, n, iters);
  // -both: DIRECT + 1 item -> the naive end of CUB's own code.
  bench<AblHub<256, 1, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_STORE_DIRECT>>(
      "  - both (DIRECT + 1 item)", d, d_pristine, n, iters);
  printf("\nFULL should match cub::DeviceMergeSort; the deltas are each opt's real worth.\n");
  return 0;
}
