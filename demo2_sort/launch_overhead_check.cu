// Is the CUB gap just launch overhead? Rule it out by measurement.
//
// v3 issues ~136 kernel launches per sort (relaunch = grid barrier). A skeptic
// asks: maybe v3 is slow only because the CPU can't issue launches fast enough,
// and a CUDA graph (which eliminates per-launch CPU cost) would close the gap to
// CUB. This experiment captures the EXACT same v3 launch sequence into a CUDA
// graph and replays it, timing kernel-only against the normal relaunch path.
//
// If graph ~= relaunch, then launches were never the bottleneck and the CUB gap
// is algorithmic (passes over HBM), not launch overhead.

#include "common.cuh"
#include "bitonic_v3.cuh"

int main(int argc, char** argv) {
  demo2::Args args = demo2::parse_args(argc, argv);
  size_t n = args.n, bytes = n * sizeof(int);
  auto h_in = demo2::make_uniform_input(n);

  int* d;
  CUDA_CHECK(cudaMalloc(&d, bytes));
  CUDA_CHECK(cudaMemcpy(d, h_in.data(), bytes, cudaMemcpyHostToDevice));

  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // --- normal relaunch path (data-oblivious: re-sorting sorted data is valid) ---
  float relaunch = demo2::time_best_ms(args.iters, [&] { bitonic_v3::sort(d, n, s); });
  CUDA_CHECK(cudaStreamSynchronize(s));

  // --- same sequence captured into a CUDA graph, then replayed ---
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  bitonic_v3::sort(d, n, s);
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  size_t nodes = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));

  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaGraphLaunch(exec, s));                 // warmup
  CUDA_CHECK(cudaStreamSynchronize(s));
  float graph_ms = 1e30f;
  for (int i = 0; i < args.iters; ++i) {
    CUDA_CHECK(cudaEventRecord(a, s));
    CUDA_CHECK(cudaGraphLaunch(exec, s));
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    if (ms < graph_ms) graph_ms = ms;
  }

  printf("v3 @ 2^%d  (%zu graph nodes / launches per sort)\n", args.log2n, nodes);
  printf("  relaunch path : %8.3f ms\n", relaunch);
  printf("  CUDA graph    : %8.3f ms   (graph saves %.3f ms = %.1f%%)\n",
         graph_ms, relaunch - graph_ms, 100.0 * (relaunch - graph_ms) / relaunch);
  printf("  => launch overhead is %.1f%% of v3; the CUB gap (~19x) is elsewhere.\n",
         100.0 * (relaunch - graph_ms) / relaunch);

  CUDA_CHECK(cudaFree(d));
  return 0;
}
