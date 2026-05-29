// Demo 2 v2: warp shuffles for the innermost stages.
//
// In every merge pass, the last 5 compare-swap stages have distance j < 32, so
// both elements of each pair live in the same warp's 32-element segment. Those
// stages don't need shared memory at all: each lane holds one key in a register
// and exchanges with its partner via __shfl_xor_sync. This removes 5
// __syncthreads() and the small-stride shared-memory bank conflicts per pass.
//
// Teaching point: the register file is the GPU's fastest and largest memory
// (~36 MB across the chip here). Within a warp you can address a neighbor's
// register directly -- the SIMT equivalent of a SIMD shuffle. Shared memory is
// the scratchpad; the register file + shuffle is the layer beneath it.
//
// At HBM scale the wall-clock win is small (the wide-stride GLOBAL passes still
// dominate -- see v3). The payoff shows in the profiler: fewer MIO/shared
// stalls in WarpStateStats, more eligible warps in SchedulerStats.

#include "common.cuh"

static const int TILE = 2048;
static const int THREADS = TILE / 2;
static const unsigned FULL = 0xffffffffu;

__device__ __forceinline__ void cas(int* s, int low, int high, bool ascending) {
  int a = s[low], b = s[high];
  if ((a > b) == ascending) { s[low] = b; s[high] = a; }
}
__device__ __forceinline__ int pair_low(int tid, int j) {
  return ((tid & ~(j - 1)) << 1) | (tid & (j - 1));
}

// 5-stage intra-warp bitonic tail (j = 16,8,4,2,1) on one register-resident key.
// `ascending` is constant across the 32-segment (caller guarantees k >= 64).
__device__ __forceinline__ int warp_tail(int v, int lane, bool ascending) {
  for (int j = 16; j >= 1; j >>= 1) {
    int partner = __shfl_xor_sync(FULL, v, j);
    int lo = min(v, partner), hi = max(v, partner);
    bool i_am_lower = ((lane & j) == 0);
    v = (i_am_lower == ascending) ? lo : hi;
  }
  return v;
}

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

// Merge tail for fixed k: shared for j >= 32, warp shuffle for j < 32.
__global__ void local_merge(int* __restrict__ g, size_t k) {
  __shared__ int s[TILE];
  int tid = threadIdx.x;
  size_t base = (size_t)blockIdx.x * TILE;
  s[tid] = g[base + tid];
  s[tid + THREADS] = g[base + tid + THREADS];
  __syncthreads();
  for (int j = THREADS; j >= 32; j >>= 1) {
    int low = pair_low(tid, j);
    bool asc = (((base + low) & k) == 0);
    __syncthreads();
    cas(s, low, low + j, asc);
  }
  __syncthreads();
  // Warp-shuffle tail (j = 16..1) for both elements this thread owns.
  #pragma unroll
  for (int half = 0; half < 2; ++half) {
    int idx = tid + half * THREADS;
    int lane = idx & 31;
    bool asc = (((base + idx) & k) == 0);  // constant within the 32-segment
    int v = warp_tail(s[idx], lane, asc);
    g[base + idx] = v;
  }
}

static void sort_v2(int* d, size_t n) {
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

  sort_v2(d, args.n);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> h_out(args.n);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d, bytes, cudaMemcpyDeviceToHost));
  bool ok = demo2::verify_sorted(h_out, h_in);

  float ms = demo2::time_best_ms(args.iters, [&] { sort_v2(d, args.n); });
  demo2::report("v2_shuffle", args.n, ms, ok);

  CUDA_CHECK(cudaFree(d));
  return ok ? 0 : 1;
}
