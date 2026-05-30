// Demo 5 v1: shared-memory privatization (the "cache tiling").
//
// Each block keeps its OWN histogram in shared memory; threads atomicAdd there
// (shared atomics are ~100x cheaper than global, and contention is now per-block
// instead of device-wide). At the end, each block folds its private histogram
// into the global one with NBINS atomics. This is exactly CUB's privatized-smem
// scheme (cub/agent/agent_histogram.cuh: histograms[...][PRIVATIZED_SMEM_BINS]).
//
// Constraint the cost model exposes: the private histogram must fit in shared
// memory. NBINS*4 bytes per block -- fine for 256, impossible for millions of
// bins. That fit/no-fit boundary is a real algorithm-selection knob (CUB falls
// back to privatized GLOBAL bins when smem can't hold them).

#include "common.cuh"

__global__ void hist_shared(const int* __restrict__ in, size_t n,
                            unsigned* __restrict__ out) {
  __shared__ unsigned smem[NBINS];
  for (int b = threadIdx.x; b < NBINS; b += blockDim.x) smem[b] = 0;
  __syncthreads();

  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {           // coalesced read; contention now in smem
    int bin = histo::scale_bin(in[i]);
    if (bin >= 0) atomicAdd(&smem[bin], 1u);
  }
  __syncthreads();

  for (int b = threadIdx.x; b < NBINS; b += blockDim.x)
    if (smem[b]) atomicAdd(&out[b], smem[b]);
}

int main(int argc, char** argv) {
  histo::Args args = histo::parse_args(argc, argv);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int block = 256, grid = 32 * prop.multiProcessorCount;
  return histo::run_and_report("v1_shared", args,
      [&](const int* d_in, size_t n, unsigned* d_out) {
        hist_shared<<<grid, block>>>(d_in, n, d_out);
      });
}
