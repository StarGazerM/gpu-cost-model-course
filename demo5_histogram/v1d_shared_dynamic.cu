// Demo 5 v1d: shared privatization done properly -- DYNAMIC shared memory.
//
// v1 used a static __shared__ array, which ptxas caps at 48 KB, so it failed to
// COMPILE past ~12 K bins. That was an implementation artifact, not a real wall.
// Production code (and CUB) uses dynamic shared memory with an opt-in up to the
// hardware limit (~100 KB/block on Ada), and falls back gracefully beyond it.
//
// This version compiles for ANY bin count; the limit is a RUNTIME one (the opt-in
// fails when NBINS*4 exceeds the device max), and well before that hard wall the
// growing shared footprint quietly costs occupancy. So the honest picture is:
// privatization degrades smoothly with bin count, then hits a real (higher) wall
// -- at which point you switch algorithms (global privatization, or sort).

#include "common.cuh"

extern __shared__ unsigned smem[];  // NBINS counters, sized at launch

__global__ void hist_shared_dyn(const int* __restrict__ in, size_t n,
                                unsigned* __restrict__ out) {
  for (int b = threadIdx.x; b < NBINS; b += blockDim.x) smem[b] = 0;
  __syncthreads();
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
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
  size_t smem_bytes = (size_t)NBINS * sizeof(unsigned);

  // Opt in to >48 KB dynamic shared; this is the real (runtime) limit.
  cudaError_t e = cudaFuncSetAttribute(
      hist_shared_dyn, cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes);
  if (e != cudaSuccess) {
    printf("v1d_shared_dyn   NBINS=%d  CANNOT privatize in shared: need %zu KB "
           "> device max (%d KB). -> fall back to global/sort.\n",
           NBINS, smem_bytes / 1024, prop.sharedMemPerBlockOptin / 1024);
    return 0;
  }
  return histo::run_and_report("v1d_shared_dyn", args,
      [&](const int* d_in, size_t n, unsigned* d_out) {
        hist_shared_dyn<<<grid, block, smem_bytes>>>(d_in, n, d_out);
      });
}
