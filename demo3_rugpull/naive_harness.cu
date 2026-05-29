// Demo 3, harness 1: the naive query-stage loop.
//
// Mimics how a sort shows up inside a database pipeline: each iteration
// allocates fresh device buffers, copies data in, runs the kernel, copies out,
// and frees. The kernel is the SAME tuned v3 from Demo 2 -- nothing about it
// changed. Yet the per-iteration time balloons, because cudaMalloc/cudaFree are
// synchronous driver calls that serialize the whole device: no async overlap,
// a round-trip to the driver every iteration.
//
// This is thesis #3 made non-negotiable: the kernel is ~10% of the problem.
// Compare per-iter time here to the kernel-only number printed first.

#include "../demo2_sort/common.cuh"
#include "../demo2_sort/bitonic_v3.cuh"
#include <chrono>
#include <cstring>

int main(int argc, char** argv) {
  int log2n = (argc > 1) ? atoi(argv[1]) : 20;
  int iters = (argc > 2) ? atoi(argv[2]) : 200;
  size_t n = (size_t)1 << log2n, bytes = n * sizeof(int);

  auto h_src = demo2::make_uniform_input(n);
  int *h_in, *h_out;
  CUDA_CHECK(cudaMallocHost(&h_in, bytes));   // pinned: required for async copies
  CUDA_CHECK(cudaMallocHost(&h_out, bytes));
  memcpy(h_in, h_src.data(), bytes);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Kernel-only reference: persistent buffer, no per-iter allocation.
  int* d0;
  CUDA_CHECK(cudaMalloc(&d0, bytes));
  CUDA_CHECK(cudaMemcpy(d0, h_in, bytes, cudaMemcpyHostToDevice));
  float konly = demo2::time_best_ms(20, [&] { bitonic_v3::sort(d0, n); });
  CUDA_CHECK(cudaFree(d0));

  // Warmup one full iteration (and verify correctness).
  // A real query stage allocates input + output (often + scratch); we model two.
  auto one_iter = [&] {
    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
    bitonic_v3::sort(d_in, n, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_out, d_in, bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
  };
  one_iter();
  std::vector<int> got(h_out, h_out + n);
  bool ok = demo2::verify_sorted(got, h_src);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) one_iter();
  auto t1 = std::chrono::high_resolution_clock::now();
  double per_iter = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

  printf("naive (cudaMalloc/Free per iter)  n=2^%d  iters=%d  [%s]\n", log2n,
         iters, ok ? "PASS" : "FAIL");
  printf("  kernel-only : %7.3f ms\n", konly);
  printf("  per-iter    : %7.3f ms   (%.1fx the kernel)\n", per_iter,
         per_iter / konly);
  return ok ? 0 : 1;
}
