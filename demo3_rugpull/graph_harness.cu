// Demo 3, harness 3: capture the whole stage as a CUDA graph, replay N times.
//
// The v3 sort issues ~100+ kernel launches per call. Even with pooled
// allocation, each launch and each API call pays CPU-side overhead that the GPU
// can outrun at small sizes. A CUDA graph records the entire stream-ordered
// sequence once -- alloc, H2D, every sort kernel, D2H, free -- and replays it as
// a single launch, collapsing all that per-call overhead.
//
// Teaching point: at this point you are not writing kernels at all. You are
// composing and scheduling work. Allocation, async execution, and graph capture
// are the program; the kernel is a node in it.

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

  // Retain pooled memory across replays (see pool_harness for why).
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

  // Capture one full stage (stream-ordered ops only, so cudaMallocAsync/Free).
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  int *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(cudaMallocAsync(&d_in, bytes, stream));
  CUDA_CHECK(cudaMallocAsync(&d_out, bytes, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
  bitonic_v3::sort(d_in, n, stream);
  CUDA_CHECK(cudaMemcpyAsync(d_out, d_in, bytes, cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaFreeAsync(d_in, stream));
  CUDA_CHECK(cudaFreeAsync(d_out, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  CUDA_CHECK(cudaGraphLaunch(exec, stream));  // warmup
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<int> got(h_out, h_out + n);
  bool ok = demo2::verify_sorted(got, h_src);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) CUDA_CHECK(cudaGraphLaunch(exec, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto t1 = std::chrono::high_resolution_clock::now();
  double per_iter = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

  printf("graph (capture once, replay)  n=2^%d  iters=%d  [%s]\n", log2n, iters,
         ok ? "PASS" : "FAIL");
  printf("  kernel-only : %7.3f ms\n", konly);
  printf("  per-iter    : %7.3f ms   (%.1fx the kernel)\n", per_iter,
         per_iter / konly);

  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  return ok ? 0 : 1;
}
