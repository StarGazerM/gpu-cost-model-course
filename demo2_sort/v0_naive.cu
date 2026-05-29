// Demo 2 v0: naive global-memory bitonic sort.
//
// One thread per element. Every compare-swap stage is its own kernel launch
// (the relaunch IS the grid-wide barrier). Every read and write hits global
// memory; nothing is reused. This is the deliberately-bad baseline.
//
// Stages: for n = 2^L there are L(L+1)/2 stages (300 for n=2^24), each a full
// pass over the array. Bandwidth-naive by construction.

#include "common.cuh"

// One bitonic compare-swap stage at distance j within bitonic sequences of
// size k. Thread i pairs with i^j; only the lower index of each pair acts.
__global__ void bitonic_stage(int* __restrict__ a, size_t n, size_t j, size_t k) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  size_t ixj = i ^ j;
  if (ixj > i) {
    bool ascending = ((i & k) == 0);
    int ai = a[i], aj = a[ixj];
    if ((ai > aj) == ascending) {
      a[i] = aj;
      a[ixj] = ai;
    }
  }
}

static void bitonic_sort_v0(int* d, size_t n) {
  const int block = 256;
  const int grid = (int)((n + block - 1) / block);
  for (size_t k = 2; k <= n; k <<= 1)
    for (size_t j = k >> 1; j > 0; j >>= 1)
      bitonic_stage<<<grid, block>>>(d, n, j, k);
}

int main(int argc, char** argv) {
  demo2::Args args = demo2::parse_args(argc, argv);
  auto h_in = demo2::make_uniform_input(args.n);

  int* d = nullptr;
  size_t bytes = args.n * sizeof(int);
  CUDA_CHECK(cudaMalloc(&d, bytes));
  CUDA_CHECK(cudaMemcpy(d, h_in.data(), bytes, cudaMemcpyHostToDevice));

  // Sort once, copy back, verify.
  bitonic_sort_v0(d, args.n);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> h_out(args.n);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d, bytes, cudaMemcpyDeviceToHost));
  bool ok = demo2::verify_sorted(h_out, h_in);

  // Time kernel-only (data-oblivious network: re-sorting sorted data is valid).
  float ms = demo2::time_best_ms(args.iters, [&] { bitonic_sort_v0(d, args.n); });
  demo2::report("v0_naive", args.n, ms, ok);

  CUDA_CHECK(cudaFree(d));
  return ok ? 0 : 1;
}
