// Demo 2 v1: shared-memory tile bitonic (correct full sort of any pow2 array).
//
// Insight: in the bitonic network, a compare-swap at distance j pairs index i
// with i^j. When j < TILE and the array is partitioned into TILE-aligned tiles,
// BOTH elements live in the same tile -- so every small-stride stage can run in
// shared memory. Only the wide-stride stages (j >= TILE) must touch global
// memory (the paired element is in another tile). v3 attacks those.
//
// Structure:
//   local_sort  : each block bitonic-sorts its own TILE in shared memory
//                 (covers all stages with k <= TILE).
//   merge loop  : for k > TILE, do j >= TILE as global stages (like v0), then
//                 collapse the j < TILE tail into one shared-memory pass.
//
// Teaching point: the programmable scratchpad is the GPU's signature. A cache
// can't be told "keep these 2048 keys resident while I make 11 passes."

#include "common.cuh"

static const int TILE = 2048;          // keys per tile (8 KB shared)
static const int THREADS = TILE / 2;   // one thread per compare-swap (1024)

__device__ __forceinline__ void cas(int* s, int low, int high, bool ascending) {
  int a = s[low], b = s[high];
  if ((a > b) == ascending) { s[low] = b; s[high] = a; }
}

// Map a thread (0..TILE/2-1) to the unique compare-swap pair for distance j.
__device__ __forceinline__ int pair_low(int tid, int j) {
  return ((tid & ~(j - 1)) << 1) | (tid & (j - 1));
}

// Initial per-tile sort: all stages with k <= TILE, entirely in shared memory.
__global__ void local_sort(int* __restrict__ g) {
  __shared__ int s[TILE];
  int tid = threadIdx.x;
  size_t base = (size_t)blockIdx.x * TILE;
  s[tid] = g[base + tid];
  s[tid + THREADS] = g[base + tid + THREADS];
  __syncthreads();
  for (int k = 2; k <= TILE; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      int low = pair_low(tid, j);
      bool asc = (((base + low) & k) == 0);
      __syncthreads();
      cas(s, low, low + j, asc);
    }
  }
  __syncthreads();
  g[base + tid] = s[tid];
  g[base + tid + THREADS] = s[tid + THREADS];
}

// One wide-stride global stage (j >= TILE): paired element is in another tile.
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

// Collapse the j < TILE tail of one merge step (fixed k) into a shared pass.
__global__ void local_merge(int* __restrict__ g, size_t k) {
  __shared__ int s[TILE];
  int tid = threadIdx.x;
  size_t base = (size_t)blockIdx.x * TILE;
  s[tid] = g[base + tid];
  s[tid + THREADS] = g[base + tid + THREADS];
  __syncthreads();
  for (int j = THREADS; j > 0; j >>= 1) {   // j = TILE/2 .. 1
    int low = pair_low(tid, j);
    bool asc = (((base + low) & k) == 0);
    __syncthreads();
    cas(s, low, low + j, asc);
  }
  __syncthreads();
  g[base + tid] = s[tid];
  g[base + tid + THREADS] = s[tid + THREADS];
}

static void sort_v1(int* d, size_t n) {
  int blocks = (int)(n / TILE);
  int gstage_grid = (int)((n + 255) / 256);
  local_sort<<<blocks, THREADS>>>(d);
  for (size_t k = 2 * TILE; k <= n; k <<= 1) {
    for (size_t j = k >> 1; j >= TILE; j >>= 1)
      global_stage<<<gstage_grid, 256>>>(d, n, j, k);
    local_merge<<<blocks, THREADS>>>(d, k);
  }
}

int main(int argc, char** argv) {
  demo2::Args args = demo2::parse_args(argc, argv);
  if (args.n < (size_t)TILE) { fprintf(stderr, "n must be >= TILE\n"); return 1; }
  auto h_in = demo2::make_uniform_input(args.n);

  int* d = nullptr;
  size_t bytes = args.n * sizeof(int);
  CUDA_CHECK(cudaMalloc(&d, bytes));
  CUDA_CHECK(cudaMemcpy(d, h_in.data(), bytes, cudaMemcpyHostToDevice));

  sort_v1(d, args.n);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> h_out(args.n);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d, bytes, cudaMemcpyDeviceToHost));
  bool ok = demo2::verify_sorted(h_out, h_in);

  float ms = demo2::time_best_ms(args.iters, [&] { sort_v1(d, args.n); });
  demo2::report("v1_shared", args.n, ms, ok);

  CUDA_CHECK(cudaFree(d));
  return ok ? 0 : 1;
}
