// The ACTUAL cost model behind thrust's radix-vs-merge dispatch: count BYTES MOVED.
//   radix  cost ~ (key_bits/digit_bits) passes x n x key_bytes x 2   (read+write)  -> grows ~key_bits^2
//   merge  cost ~ log2(n)               passes x n x key_bytes x 2                  -> grows ~key_bits
// So radix wins big for NARROW arithmetic keys, and its lead SHRINKS as keys widen
// (more digit-passes) until it converges to merge. Measure it: 32-bit vs 64-bit keys.
//
//   nvcc -O3 -std=c++17 -arch=sm_89 -o /tmp/kw keywidth.cu && /tmp/kw 28 8

#include "../demo2_sort/common.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cstdint>
#include <random>

template <class T> struct LessOp { __device__ bool operator()(const T& a, const T& b) const { return a < b; } };

template <class T>
static void run(const char* tag, size_t n, int iters) {
  size_t bytes = n * sizeof(T);
  std::vector<T> h(n);
  std::mt19937_64 rng(7);
  for (auto& x : h) x = (T)rng();
  T *d = nullptr, *d2 = nullptr, *dp = nullptr;
  CUDA_CHECK(cudaMalloc(&d, bytes)); CUDA_CHECK(cudaMalloc(&d2, bytes)); CUDA_CHECK(cudaMalloc(&dp, bytes));
  CUDA_CHECK(cudaMemcpy(dp, h.data(), bytes, cudaMemcpyHostToDevice));

  // ---- radix (out-of-place) ----
  void* tr = nullptr; size_t trb = 0;
  cub::DeviceRadixSort::SortKeys(tr, trb, d, d2, n);
  CUDA_CHECK(cudaMalloc(&tr, trb));
  float r = demo2::time_best_ms(iters, [&]{
    cudaMemcpyAsync(d, dp, bytes, cudaMemcpyDeviceToDevice);
    cub::DeviceRadixSort::SortKeys(tr, trb, d, d2, n);
  });
  // ---- merge (in-place) ----
  void* tm = nullptr; size_t tmb = 0;
  cub::DeviceMergeSort::SortKeys(tm, tmb, d, n, LessOp<T>());
  CUDA_CHECK(cudaMalloc(&tm, tmb));
  float m = demo2::time_best_ms(iters, [&]{
    cudaMemcpyAsync(d, dp, bytes, cudaMemcpyDeviceToDevice);
    cub::DeviceMergeSort::SortKeys(tm, tmb, d, n, LessOp<T>());
  });
  double d2d = bytes / 810e9 * 1e3;
  int rp = (int)sizeof(T);                 // radix passes ~ key_bytes (8-bit digits): 4 (32b) or 8 (64b)
  printf("  %-10s (%zu-bit)  radix %6.1f ms | merge %6.1f ms | merge/radix = %.2fx   [radix ~%d passes]\n",
         tag, sizeof(T) * 8, r - d2d, m - d2d, (m - d2d) / (r - d2d), rp);
  cudaFree(d); cudaFree(d2); cudaFree(dp); cudaFree(tr); cudaFree(tm);
}

int main(int argc, char** argv) {
  int log2n = (argc > 1) ? atoi(argv[1]) : 28;
  int iters = (argc > 2) ? atoi(argv[2]) : 8;
  size_t n = (size_t)1 << log2n;
  printf("CUB radix vs merge at n=2^%d -- watch radix's lead shrink as keys widen:\n", log2n);
  run<uint32_t>("uint32", n, iters);
  run<uint64_t>("uint64", n, iters);
  printf("cost model: radix moves ~(key_bits/8)*2 passes; doubling key width ~4x's radix bytes but only ~2x's merge\n");
  printf("-> the lead halves each doubling and converges. thrust dispatches this at COMPILE time on the key type.\n");
  return 0;
}
