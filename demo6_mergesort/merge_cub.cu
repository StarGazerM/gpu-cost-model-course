// CUB's merge sort, UNFOLDED -- exactly CUB's optimizations, but written as concrete
// kernels with the constants baked in and launched by a plain host loop, with NONE
// of the policy_hub / ChainedPolicy / DispatchMergeSort strategy-template machinery.
// This is the readable, fully-OPTED top of the ladder; ablations remove one opt at a
// time from HERE (the rigorous "start at parity, iterate down" direction).
//
// CUB's sm_89 policy we are matching (tuning_merge_sort.cuh, Policy600):
//   AgentMergeSortPolicy<256, 17, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE>
//
//   nvcc -O3 -std=c++17 -arch=sm_89 -o /tmp/mc merge_cub.cu && /tmp/mc 28 10

#include "../demo2_sort/common.cuh"
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <climits>
#include <algorithm>

#ifndef BT
#define BT 256          // BLOCK_THREADS
#endif
#ifndef IPT
#define IPT 17          // ITEMS_PER_THREAD  (CUB's value)
#endif
#define TILE (BT * IPT) // 4352 -- NOT a power of two, so tiles need tail handling

// coalescing ablation: -DDIRECTLOAD swaps the warp-transpose (coalesced) block load/store
// for the DIRECT (uncoalesced, blocked) one.
#ifdef DIRECTLOAD
#define LOADMODE cub::BLOCK_LOAD_DIRECT
#define STOREMODE cub::BLOCK_STORE_DIRECT
#define LOADNAME "direct"
#else
#define LOADMODE cub::BLOCK_LOAD_WARP_TRANSPOSE
#define STOREMODE cub::BLOCK_STORE_WARP_TRANSPOSE
#define LOADNAME "transpose"
#endif

struct LessOp { __device__ bool operator()(const int& a, const int& b) const { return a < b; } };

// Phase 1: sort one TILE per block, exactly as cub's DeviceMergeSortBlockSortKernel does --
// a warp-transpose coalesced load -> BlockMergeSort (per-thread odd-even net + shared
// MergePath merge) -> warp-transpose coalesced store. (BlockMergeSort IS net_sort + co-rank.)
__global__ void block_sort(int* __restrict__ g, int n) {
  using LoadT  = cub::BlockLoad<int, BT, IPT, LOADMODE>;
  using SortT  = cub::BlockMergeSort<int, BT, IPT>;
  using StoreT = cub::BlockStore<int, BT, IPT, STOREMODE>;
  __shared__ union {
    typename LoadT::TempStorage  load;
    typename SortT::TempStorage  sort;
    typename StoreT::TempStorage store;
  } smem;

  int base = blockIdx.x * TILE;
  int valid = n - base; if (valid > TILE) valid = TILE;   // last tile is partial (non-pow2 TILE)
  int keys[IPT];
  LoadT(smem.load).Load(g + base, keys, valid, INT_MAX);  // OOB lanes filled with INT_MAX
  __syncthreads();
  SortT(smem.sort).Sort(keys, LessOp(), valid, INT_MAX);
  __syncthreads();
  StoreT(smem.store).Store(g + base, keys, valid);
}

__device__ __forceinline__ int corank(const int* A, int aN, const int* B, int bN, int diag) {
  int lo = diag > bN ? diag - bN : 0, hi = diag < aN ? diag : aN;
  while (lo < hi) { int mid = (lo + hi) >> 1; if (A[mid] <= B[diag - 1 - mid]) lo = mid + 1; else hi = mid; }
  return lo;
}

// Phase 2: the TILED co-rank merge -- cub's Partition + Merge kernels, unfolded and
// generalized for the non-power-of-two tile. Each block owns one OUT_TILE of contiguous
// output (a (group, tile-in-group) pair); a tiny PARTITION kernel pre-computes each tile's
// A-split (co-rank) so the merge kernel reads it and is PURE coalesced streaming.
#ifndef MSPT
#define MSPT 16                        // merge output elements per thread (CUB-class tile)
#endif
#define OUT_TILE (BT * MSPT)           // 4096

// Ablation knob: TILED=1 (default) uses the partition+shared coalesced merge below;
// TILED=0 swaps in the naive per-thread global merge (uncoalesced) to measure tiling.
#ifndef TILED
#define TILED 1
#endif

struct TileGeom { long base; int g0, g1, aN, bN, group_out; };
__device__ __forceinline__ TileGeom geom(long b, int run, int n, int tpg) {
  TileGeom t;
  long g = b / tpg; int j = (int)(b % tpg);
  t.base = g * (2L * run);
  if (t.base >= n) { t.g0 = t.g1 = t.aN = t.bN = t.group_out = 0; return t; }
  long go = (long)2 * run; if (t.base + go > n) go = n - t.base;
  t.group_out = (int)go;
  long a = (long)j * OUT_TILE, c = a + OUT_TILE;
  t.g0 = (int)(a < go ? a : go);
  t.g1 = (int)(c < go ? c : go);
  t.aN = (int)(t.base + run <= n ? run : n - t.base);
  long bb = t.base + run;
  t.bN = (int)(bb >= n ? 0 : (bb + run <= n ? run : n - bb));
  return t;
}

__global__ void partition_kernel(const int* __restrict__ in, int n, int run, int tpg,
                                 long total, int* __restrict__ parts) {
  long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= total) return;
  TileGeom t = geom(b, run, n, tpg);
  if (t.group_out == 0) { parts[b] = 0; return; }
  parts[b] = corank(in + t.base, t.aN, in + t.base + run, t.bN, t.g0);   // the scattered binary search, isolated
}

__global__ void merge_tiled(const int* __restrict__ in, int* __restrict__ out, int n, int run,
                            int tpg, const int* __restrict__ parts) {
  __shared__ int sA[OUT_TILE], sB[OUT_TILE], sO[OUT_TILE];
  int tid = threadIdx.x;
  long b = blockIdx.x;
  TileGeom t = geom(b, run, n, tpg);
  int outlen = t.g1 - t.g0;
  if (outlen <= 0) return;
  const int* A = in + t.base; const int* B = in + t.base + run;
  int a0 = parts[b];
  int a1 = (t.g1 >= t.group_out) ? t.aN : parts[b + 1];   // precomputed -- NO global binary search
  int b0 = t.g0 - a0, b1 = t.g1 - a1;
  int aCount = a1 - a0, bCount = b1 - b0;
  for (int i = tid; i < aCount; i += BT) sA[i] = A[a0 + i];   // coalesced
  for (int i = tid; i < bCount; i += BT) sB[i] = B[b0 + i];
  __syncthreads();
  int local = tid * MSPT;
  if (local < outlen) {
    int pa = corank(sA, aCount, sB, bCount, local), pb = local - pa;
    for (int k = 0; k < MSPT && local + k < outlen; ++k) {
      if (pb >= bCount || (pa < aCount && sA[pa] <= sB[pb])) sO[local + k] = sA[pa++];
      else sO[local + k] = sB[pb++];
    }
  }
  __syncthreads();
  long out_base = t.base + t.g0;
  for (int i = tid; i < outlen; i += BT) out[out_base + i] = sO[i];   // coalesced
}

// the un-tiled device merge (ablation: TILED=0) -- each thread merges MSPT outputs
// straight from global with co-rank; writes scatter (stride MSPT) -> uncoalesced.
__global__ void merge_naive(const int* __restrict__ in, int* __restrict__ out, int n, int run) {
  long o = ((long)blockIdx.x * blockDim.x + threadIdx.x) * MSPT;
  if (o >= n) return;
  long gbase = (o / (2L * run)) * (2L * run);
  const int* A = in + gbase;
  int aN = (int)(gbase + run <= n ? run : n - gbase);
  long bb = gbase + run; const int* B = in + bb;
  int bN = (int)(bb >= n ? 0 : (bb + run <= n ? run : n - bb));
  int diag = (int)(o - gbase), a = corank(A, aN, B, bN, diag), b = diag - a;
  for (int k = 0; k < MSPT && o + k < n; ++k) {
    if (b >= bN || (a < aN && A[a] <= B[b])) out[o + k] = A[a++];
    else out[o + k] = B[b++];
  }
}

static int* merge_sort(int* d, int* d2, int n) {
  int tiles = (n + TILE - 1) / TILE;
  block_sort<<<tiles, BT>>>(d, n);
  int *in = d, *out = d2;
#if TILED
  long max_tiles = ((long)n + OUT_TILE - 1) / OUT_TILE + 4;
  int* parts = nullptr; cudaMalloc(&parts, (size_t)max_tiles * sizeof(int));
#endif
  for (long run = TILE; run < n; run *= 2) {
#if TILED
    int tpg = (int)((2 * run + OUT_TILE - 1) / OUT_TILE);
    long groups = ((long)n + 2 * run - 1) / (2 * run);
    long total = groups * tpg;
    partition_kernel<<<(int)((total + 255) / 256), 256>>>(in, n, (int)run, tpg, total, parts);
    merge_tiled<<<(int)total, BT>>>(in, out, n, (int)run, tpg, parts);
#else
    int grid = (int)(((long)n / MSPT + 1 + 255) / 256);
    merge_naive<<<grid, 256>>>(in, out, n, (int)run);
#endif
    std::swap(in, out);
  }
#if TILED
  cudaFree(parts);
#endif
  return in;
}

int main(int argc, char** argv) {
  int log2n = (argc > 1) ? atoi(argv[1]) : 28;
  int iters = (argc > 2) ? atoi(argv[2]) : 10;
  int n = 1 << log2n; size_t bytes = (size_t)n * sizeof(int);
  auto h = demo2::make_uniform_input(n);
  int *d = nullptr, *d2 = nullptr, *dp = nullptr;
  CUDA_CHECK(cudaMalloc(&d, bytes)); CUDA_CHECK(cudaMalloc(&d2, bytes)); CUDA_CHECK(cudaMalloc(&dp, bytes));
  CUDA_CHECK(cudaMemcpy(dp, h.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d, dp, bytes, cudaMemcpyDeviceToDevice));
  int* res = merge_sort(d, d2, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> out(n);
  CUDA_CHECK(cudaMemcpy(out.data(), res, bytes, cudaMemcpyDeviceToHost));
  bool ok = demo2::verify_sorted(out, h);

  float ms = demo2::time_best_ms(iters, [&] {
    cudaMemcpyAsync(d, dp, bytes, cudaMemcpyDeviceToDevice);
    merge_sort(d, d2, n);
  });
  double d2d = bytes / 810e9 * 1e3;
  printf("IPT=%-2d load=%-9s merge=%-6s  %7.2f ms  %7.1f Mkeys/s  [%s]\n",
         IPT, LOADNAME, TILED ? "tiled" : "naive", ms - d2d,
         n / 1e6 / ((ms - d2d) / 1e3), ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
