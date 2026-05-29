// Demo 3, harness 2: the one-line fix -- cudaMallocAsync.
//
// Identical loop to naive_harness, but the per-iteration allocation now comes
// from a stream-ordered memory pool (cudaMallocAsync/cudaFreeAsync). After the
// pool warms up, freed blocks are recycled instead of returned to the driver,
// so there is no driver round-trip and no implicit device sync -- the copies
// and kernel stay on the stream and overlap. Per-iter time collapses back
// toward the kernel-only number.
//
// Teaching point: the kernel never changed. The win came entirely from how
// memory was managed. That is the other 90%.

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
  CUDA_CHECK(cudaMallocHost(&h_in, bytes));
  CUDA_CHECK(cudaMallocHost(&h_out, bytes));
  memcpy(h_in, h_src.data(), bytes);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Crucial: the default pool's release threshold is 0, so cudaFreeAsync hands
  // memory back to the OS every iteration (no recycling -- it's actually slower
  // than plain cudaMalloc). Retain freed blocks in the pool so they get reused.
  cudaMemPool_t pool;
  CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, 0));
  uint64_t threshold = UINT64_MAX;
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                     &threshold));

  int* d0;
  CUDA_CHECK(cudaMalloc(&d0, bytes));
  CUDA_CHECK(cudaMemcpy(d0, h_in, bytes, cudaMemcpyHostToDevice));
  float konly = demo2::time_best_ms(20, [&] { bitonic_v3::sort(d0, n); });
  CUDA_CHECK(cudaFree(d0));

  auto one_iter = [&] {
    int *d_in, *d_out;
    CUDA_CHECK(cudaMallocAsync(&d_in, bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_out, bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
    bitonic_v3::sort(d_in, n, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_out, d_in, bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_in, stream));
    CUDA_CHECK(cudaFreeAsync(d_out, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };
  one_iter();  // warms the pool
  std::vector<int> got(h_out, h_out + n);
  bool ok = demo2::verify_sorted(got, h_src);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) one_iter();
  auto t1 = std::chrono::high_resolution_clock::now();
  double per_iter = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

  printf("pool (cudaMallocAsync per iter)  n=2^%d  iters=%d  [%s]\n", log2n,
         iters, ok ? "PASS" : "FAIL");
  printf("  kernel-only : %7.3f ms\n", konly);
  printf("  per-iter    : %7.3f ms   (%.1fx the kernel)\n", per_iter,
         per_iter / konly);
  return ok ? 0 : 1;
}
