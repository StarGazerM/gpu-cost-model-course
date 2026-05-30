// Demo 5 v1b: the controlled experiment -- privatize, but in GLOBAL memory.
//
// v0 -> v1 changed two things at once: (1) reduced contention by giving each
// block its OWN histogram copy, and (2) moved the atomics from global to shared.
// This version isolates (1): each block gets a private histogram, but in GLOBAL
// memory (one copy per block), then a second kernel reduces the copies.
//
//   v0  (global, shared counters)  -> v1b: effect of PRIVATIZATION alone
//   v1b (global, private)          -> v1 : effect of SHARED LOCATION alone
//
// If v1b ~= v1, the cure was contention reduction and shared was incidental.
// If v1b << v1, shared-memory atomics mattered on their own. ncu + the clock
// decide it, not intuition.

#include "common.cuh"

__global__ void hist_global_priv(const int* __restrict__ in, size_t n,
                                 unsigned* __restrict__ partials) {
  unsigned* myhist = partials + (size_t)blockIdx.x * NBINS;  // this block's copy
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    int bin = in[i] & (NBINS - 1);
    atomicAdd(&myhist[bin], 1u);   // private copy, but GLOBAL atomics
  }
}

__global__ void reduce_partials(const unsigned* __restrict__ partials,
                                int nblocks, unsigned* __restrict__ out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= NBINS) return;
  unsigned s = 0;
  for (int k = 0; k < nblocks; ++k) s += partials[(size_t)k * NBINS + b];
  out[b] = s;
}

int main(int argc, char** argv) {
  histo::Args args = histo::parse_args(argc, argv);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int block = 256, grid = 32 * prop.multiProcessorCount;
  unsigned* d_partials = nullptr;
  CUDA_CHECK(cudaMalloc(&d_partials, (size_t)grid * NBINS * sizeof(unsigned)));
  int rc = histo::run_and_report("v1b_glob_priv", args,
      [&](const int* d_in, size_t n, unsigned* d_out) {
        CUDA_CHECK(cudaMemset(d_partials, 0, (size_t)grid * NBINS * sizeof(unsigned)));
        hist_global_priv<<<grid, block>>>(d_in, n, d_partials);
        reduce_partials<<<(NBINS + 255) / 256, 256>>>(d_partials, grid, d_out);
      });
  CUDA_CHECK(cudaFree(d_partials));
  return rc;
}
