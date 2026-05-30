// Demo 5 v0: naive global-memory atomics.
//
// Every thread does atomicAdd(&global_hist[bin], 1). The cost model says this
// should be a ~1 ms job (read the input once, ~0.5 ops/byte, memory-bound). It
// isn't -- all threads serialize on NBINS global counters. The bottleneck isn't
// bandwidth, it's atomic contention, and it gets dramatically worse under skew
// (a hot bin means every thread fights for one address).
//
// Teaching point: the cost model predicts the FLOOR; the profiler tells you why
// you're not on it. Here you're contention-bound, sitting far below the line.

#include "common.cuh"

__global__ void hist_global(const int* __restrict__ in, size_t n,
                            unsigned* __restrict__ out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    int bin = histo::scale_bin(in[i]);
    if (bin >= 0) atomicAdd(&out[bin], 1u);
  }
}

int main(int argc, char** argv) {
  histo::Args args = histo::parse_args(argc, argv);
  int block = 256, grid = 0, minGrid = 0;
  cudaOccupancyMaxPotentialBlockSize(&minGrid, &block, hist_global, 0, 0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  grid = 32 * prop.multiProcessorCount;
  return histo::run_and_report("v0_global", args,
      [&](const int* d_in, size_t n, unsigned* d_out) {
        hist_global<<<grid, block>>>(d_in, n, d_out);
      });
}
