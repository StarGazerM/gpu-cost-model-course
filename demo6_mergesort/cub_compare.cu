// Decisive cheap measurement before building the ablation:
// CUB merge sort vs CUB radix sort, at GB scale, same input.
//
// Answers Act-2's premise: does radix actually crush a fully-optimized merge
// sort on this chip? And gives the optimized-merge baseline we'll ablate against.

#include "../demo2_sort/common.cuh"
#include <cub/cub.cuh>

struct LessOp {
  __host__ __device__ bool operator()(const int& a, const int& b) const { return a < b; }
};

int main(int argc, char** argv) {
  int log2n = (argc > 1) ? atoi(argv[1]) : 28;   // 2^28 int32 = 1 GB
  int iters = (argc > 2) ? atoi(argv[2]) : 10;
  size_t n = (size_t)1 << log2n, bytes = n * sizeof(int);
  auto h = demo2::make_uniform_input(n);

  int *d_keys = nullptr, *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, bytes));
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  // ---- CUB merge sort (in place, comparison-based) ----
  CUDA_CHECK(cudaMemcpy(d_keys, h.data(), bytes, cudaMemcpyHostToDevice));
  void* mt = nullptr; size_t mb = 0;
  cub::DeviceMergeSort::SortKeys(mt, mb, d_keys, n, LessOp());
  CUDA_CHECK(cudaMalloc(&mt, mb));
  cub::DeviceMergeSort::SortKeys(mt, mb, d_keys, n, LessOp());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> out(n);
  CUDA_CHECK(cudaMemcpy(out.data(), d_keys, bytes, cudaMemcpyDeviceToHost));
  bool okm = demo2::verify_sorted(out, h);
  // CUB merge sort is not adaptive (fixed work regardless of input order), so
  // re-running it in place is the true kernel-only time -- no PCIe re-upload,
  // fair against radix's out-of-place (which never touches the host).
  float mms = demo2::time_best_ms(iters, [&] {
    cub::DeviceMergeSort::SortKeys(mt, mb, d_keys, n, LessOp());
  });

  // ---- CUB radix sort (out of place) ----
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));
  void* rt = nullptr; size_t rb = 0;
  cub::DeviceRadixSort::SortKeys(rt, rb, d_in, d_out, (int)n);
  CUDA_CHECK(cudaMalloc(&rt, rb));
  cub::DeviceRadixSort::SortKeys(rt, rb, d_in, d_out, (int)n);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  bool okr = demo2::verify_sorted(out, h);
  float rms = demo2::time_best_ms(iters, [&] {
    cub::DeviceRadixSort::SortKeys(rt, rb, d_in, d_out, (int)n);
  });

  printf("n=2^%d (%.0fM keys, %.1f GB)\n", log2n, n / 1e6, bytes / 1e9);
  printf("  CUB merge sort : %8.3f ms  %8.1f Mkeys/s  [%s]\n", mms,
         n / 1e6 / (mms / 1e3), okm ? "PASS" : "FAIL");
  printf("  CUB radix sort : %8.3f ms  %8.1f Mkeys/s  [%s]\n", rms,
         n / 1e6 / (rms / 1e3), okr ? "PASS" : "FAIL");
  printf("  => radix is %.1fx faster than merge at this scale\n", mms / rms);
  return (okm && okr) ? 0 : 1;
}
