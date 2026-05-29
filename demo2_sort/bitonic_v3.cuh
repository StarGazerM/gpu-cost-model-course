// Reusable v3 bitonic sort (kernels + stream-aware host launcher).
// Shared by demo2/v3_multiblock.cu and the demo3 rug-pull harnesses so they
// all drive the identical sort. See v3_multiblock.cu for the full commentary.
#pragma once
#include <cuda_runtime.h>

namespace bitonic_v3 {

static const int TILE = 8192;
static const int BD = 1024;
static const int E = TILE / BD;
static const int SWAPS = TILE / 2;
static const unsigned FULL = 0xffffffffu;

__device__ __forceinline__ void cas(int* s, int low, int high, bool ascending) {
  int a = s[low], b = s[high];
  if ((a > b) == ascending) { s[low] = b; s[high] = a; }
}
__device__ __forceinline__ int pair_low(int s, int j) {
  return ((s & ~(j - 1)) << 1) | (s & (j - 1));
}
__device__ __forceinline__ int warp_tail(int v, int lane, bool ascending) {
  for (int j = 16; j >= 1; j >>= 1) {
    int partner = __shfl_xor_sync(FULL, v, j);
    int lo = min(v, partner), hi = max(v, partner);
    v = (((lane & j) == 0) == ascending) ? lo : hi;
  }
  return v;
}

__global__ void local_sort(int* __restrict__ g) {
  __shared__ int s[TILE];
  int tid = threadIdx.x;
  size_t base = (size_t)blockIdx.x * TILE;
  for (int e = tid; e < TILE; e += BD) s[e] = g[base + e];
  __syncthreads();
  for (int k = 2; k <= TILE; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      __syncthreads();
      for (int sw = tid; sw < SWAPS; sw += BD) {
        int low = pair_low(sw, j);
        bool asc = (((base + low) & k) == 0);
        cas(s, low, low + j, asc);
      }
    }
  }
  __syncthreads();
  for (int e = tid; e < TILE; e += BD) g[base + e] = s[e];
}

__global__ void global_stage(int* __restrict__ g, size_t n, size_t j, size_t k) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  size_t ixj = i ^ j;
  if (ixj > i) {
    bool asc = ((i & k) == 0);
    int a = g[i], b = g[ixj];
    if ((a > b) == asc) { g[i] = b; g[ixj] = a; }
  }
}

__global__ void local_merge(int* __restrict__ g, size_t k) {
  __shared__ int s[TILE];
  int tid = threadIdx.x;
  size_t base = (size_t)blockIdx.x * TILE;
  for (int e = tid; e < TILE; e += BD) s[e] = g[base + e];
  __syncthreads();
  for (int j = SWAPS; j >= 32; j >>= 1) {
    __syncthreads();
    for (int sw = tid; sw < SWAPS; sw += BD) {
      int low = pair_low(sw, j);
      bool asc = (((base + low) & k) == 0);
      cas(s, low, low + j, asc);
    }
  }
  __syncthreads();
  for (int e = 0; e < E; ++e) {
    int idx = tid + e * BD;
    int lane = idx & 31;
    bool asc = (((base + idx) & k) == 0);
    g[base + idx] = warp_tail(s[idx], lane, asc);
  }
}

// Launch the full v3 sort on `stream`. n must be a power of two >= TILE.
inline void sort(int* d, size_t n, cudaStream_t stream = 0) {
  int blocks = (int)(n / TILE);
  int gstage_grid = (int)((n + 255) / 256);
  local_sort<<<blocks, BD, 0, stream>>>(d);
  for (size_t k = 2 * TILE; k <= n; k <<= 1) {
    for (size_t j = k >> 1; j >= TILE; j >>= 1)
      global_stage<<<gstage_grid, 256, 0, stream>>>(d, n, j, k);
    local_merge<<<blocks, BD, 0, stream>>>(d, k);
  }
}

}  // namespace bitonic_v3
