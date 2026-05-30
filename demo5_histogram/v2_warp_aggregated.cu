// Demo 5 v2: warp-aggregated atomics (branchless SIMT cycle-saving).
//
// Even in shared memory, a hot bin serializes a warp's 32 lanes on one address.
// Fix it at the warp level: lanes targeting the same bin discover each other
// with __match_any_sync, elect one leader (lowest set lane), and do a SINGLE
// atomicAdd of the group's popcount instead of 32 separate adds. No divergent
// branches -- it's ballot/popcount arithmetic. This is the canonical GPU trick,
// and it's precisely what cuts skew's penalty: the more lanes share a bin, the
// more atomics it eliminates.

#include "common.cuh"

__device__ __forceinline__ void warp_aggregated_add(unsigned* hist, int bin) {
  unsigned active = __activemask();
  unsigned peers  = __match_any_sync(active, bin);  // lanes with my bin
  int leader      = __ffs(peers) - 1;               // lowest such lane
  int count       = __popc(peers);                  // how many of us
  int lane        = threadIdx.x & 31;
  if (lane == leader) atomicAdd(&hist[bin], (unsigned)count);
}

__global__ void hist_warpagg(const int* __restrict__ in, size_t n,
                             unsigned* __restrict__ out) {
  __shared__ unsigned smem[NBINS];
  for (int b = threadIdx.x; b < NBINS; b += blockDim.x) smem[b] = 0;
  __syncthreads();

  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    int bin = in[i] & (NBINS - 1);
    warp_aggregated_add(smem, bin);
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
  return histo::run_and_report("v2_warpagg", args,
      [&](const int* d_in, size_t n, unsigned* d_out) {
        hist_warpagg<<<grid, block>>>(d_in, n, d_out);
      });
}
