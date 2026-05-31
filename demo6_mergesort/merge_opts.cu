// Merge-sort OPTIMIZATION LADDER -- the access-mode opts that close the ~1.9x
// gap between our hand-written merge sort and cub::DeviceMergeSort, added ONE AT
// A TIME so they can be shown on demand (this is the "if someone asks" material;
// the 1-hr talk uses merge_ablation.cu's single register-blocking knob instead).
//
// The ALGORITHM is fixed (CUB-faithful: per-thread odd-even network -> shared
// MergePath block merge -> device MergePath merge). Only HOW we touch memory
// changes. Build with -DOPT=k:
//
//   OPT=0  baseline      : blocked (uncoalesced) tile load + naive global device merge
//   OPT=1  +coalesce load: striped global load -> shared -> blocked registers
//   OPT=2  +vectorize    : 128-bit int4 loads on that coalesced path
//   OPT=3  +coalesce merge: device merge stages each tile through shared memory,
//                           coalesced load + coalesced store (the dominant traffic)
//
//   nvcc -O3 -std=c++17 -arch=sm_89 -DOPT=3 -o /tmp/mo3 merge_opts.cu
//
// Cost-model note: the device merge runs log2(n/TILE) passes over ALL of memory,
// so it dominates total traffic -- expect OPT=3 (coalescing the repeated merge),
// not the one-shot load opts (OPT=1,2), to move the number. Find the big term.

#include "../demo2_sort/common.cuh"
#include <climits>
#include <algorithm>

#ifndef OPT
#define OPT 0
#endif
#define IPT 8                        // fixed (register blocking is merge_ablation.cu's knob)
#define BLOCK 256
#define TILE (BLOCK * IPT)
#define SPT 8                        // device-merge output elements per thread
#define OUT_TILE (BLOCK * SPT)       // device-merge output tile per block (OPT=3)

// number of A-elements among the first `diag` of merge(A[0:aN], B[0:bN]).
__device__ __forceinline__ int merge_path(const int* A, int aN, const int* B,
                                          int bN, int diag) {
  int lo = diag > bN ? diag - bN : 0, hi = diag < aN ? diag : aN;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (A[mid] <= B[diag - 1 - mid]) lo = mid + 1; else hi = mid;
  }
  return lo;
}

__device__ __forceinline__ void net_sort(int (&a)[IPT]) {  // odd-even network
  for (int i = 0; i < IPT; ++i)
    for (int j = (i & 1); j + 1 < IPT; j += 2) {
      int lo = min(a[j], a[j + 1]), hi = max(a[j], a[j + 1]);
      a[j] = lo; a[j + 1] = hi;
    }
}

// Sort one TILE per block. The LOAD section is the only thing OPT changes.
__global__ void block_sort(int* __restrict__ g, size_t n) {
  __shared__ __align__(16) int s[TILE];
  int tid = threadIdx.x;
  size_t base = (size_t)blockIdx.x * TILE;
  int keys[IPT];

#if OPT == 0
  // blocked load straight from global: lane L reads g[base + L*IPT + i] -> the
  // warp strides by IPT, so each 128B transaction serves few lanes (UNCOALESCED).
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    size_t idx = base + (size_t)tid * IPT + i;
    keys[i] = (idx < n) ? g[idx] : INT_MAX;
  }
#elif OPT == 1
  // striped load (consecutive lanes -> consecutive addresses = COALESCED), then
  // transpose striped->blocked through shared memory (a BlockExchange).
  for (int i = tid; i < TILE; i += BLOCK) {
    size_t idx = base + i;
    s[i] = (idx < n) ? g[idx] : INT_MAX;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < IPT; ++i) keys[i] = s[tid * IPT + i];
  __syncthreads();
#else  // OPT >= 2 : same coalesced path, but 128-bit int4 loads (assumes full tile)
  const int4* g4 = reinterpret_cast<const int4*>(g + base);
  int4* s4 = reinterpret_cast<int4*>(s);
  for (int i = tid; i < TILE / 4; i += BLOCK) s4[i] = g4[i];
  __syncthreads();
#pragma unroll
  for (int i = 0; i < IPT; ++i) keys[i] = s[tid * IPT + i];
  __syncthreads();
#endif

  net_sort(keys);  // each thread now holds a sorted run of length IPT

  for (int stride = 1; stride < BLOCK; stride *= 2) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < IPT; ++i) s[tid * IPT + i] = keys[i];
    __syncthreads();
    int group = tid / (2 * stride);
    int local = tid % (2 * stride);
    int runlen = stride * IPT;
    const int* A = s + (size_t)group * (2 * stride) * IPT;
    const int* B = A + runlen;
    int diag = local * IPT;
    int a = merge_path(A, runlen, B, runlen, diag);
    int b = diag - a;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      if (b >= runlen || (a < runlen && A[a] <= B[b])) keys[i] = A[a++];
      else keys[i] = B[b++];
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < IPT; ++i) s[tid * IPT + i] = keys[i];
  __syncthreads();
  for (int i = tid; i < TILE; i += BLOCK) {
    size_t idx = base + i;
    if (idx < n) g[idx] = s[i];
  }
}

// --- baseline device merge (OPT<=2): each thread merges SPT outputs straight
// from global. Writes are blocked (stride SPT across the warp) -> uncoalesced.
__global__ void merge_runs(const int* __restrict__ in, int* __restrict__ out,
                           size_t n, size_t run) {
  size_t o = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * SPT;
  if (o >= n) return;
  size_t base = (o / (2 * run)) * (2 * run);
  const int* A = in + base;
  const int* B = in + base + run;
  size_t diag = o - base;
  size_t lo = diag > run ? diag - run : 0, hi = diag < run ? diag : run;
  while (lo < hi) {
    size_t mid = (lo + hi) >> 1;
    if (A[mid] <= B[diag - 1 - mid]) lo = mid + 1; else hi = mid;
  }
  size_t a = lo, b = diag - lo;
  for (int k = 0; k < SPT && o + k < n; ++k) {
    if (b >= run || (a < run && A[a] <= B[b])) out[o + k] = A[a++];
    else out[o + k] = B[b++];
  }
}

// --- OPT=3 device merge: one block per OUT_TILE of contiguous output. Block-level
// MergePath finds the two source segments that feed this tile; the block loads
// them into shared COALESCED, merges in shared, and writes back COALESCED. This is
// cub::AgentMergeSort's shape. (Requires OUT_TILE | 2*run, true for powers of two
// with SPT<=2*IPT, so every output tile lies within a single merge group.)
__global__ void merge_runs_smem(const int* __restrict__ in, int* __restrict__ out,
                                size_t n, size_t run) {
  __shared__ int sA[OUT_TILE];
  __shared__ int sB[OUT_TILE];
  __shared__ int sO[OUT_TILE];
  int tid = threadIdx.x;
  size_t out_base = (size_t)blockIdx.x * OUT_TILE;
  if (out_base >= n) return;
  size_t base = (out_base / (2 * run)) * (2 * run);
  const int* A = in + base;
  const int* B = in + base + run;
  int rn = (int)run;
  int g0 = (int)(out_base - base);            // this tile's start diag in the group
  int g1 = g0 + OUT_TILE;
  int a0 = merge_path(A, rn, B, rn, g0);      // block-level co-rank at both ends
  int a1 = merge_path(A, rn, B, rn, g1);
  int b0 = g0 - a0, b1 = g1 - a1;
  int aCount = a1 - a0, bCount = b1 - b0;      // aCount + bCount == OUT_TILE
  for (int i = tid; i < aCount; i += BLOCK) sA[i] = A[a0 + i];   // coalesced loads
  for (int i = tid; i < bCount; i += BLOCK) sB[i] = B[b0 + i];
  __syncthreads();
  int local = tid * SPT;
  int pa = merge_path(sA, aCount, sB, bCount, local);            // per-thread co-rank in shared
  int pb = local - pa;
#pragma unroll
  for (int k = 0; k < SPT; ++k) {
    if (pb >= bCount || (pa < aCount && sA[pa] <= sB[pb])) sO[local + k] = sA[pa++];
    else sO[local + k] = sB[pb++];
  }
  __syncthreads();
  for (int i = tid; i < OUT_TILE; i += BLOCK) {                  // coalesced store
    size_t idx = out_base + i;
    if (idx < n) out[idx] = sO[i];
  }
}

#if OPT >= 4
// --- OPT=4: split the scattered work OUT of the merge. A tiny PARTITION kernel does
// the block-level co-rank binary searches (the only non-streaming reads) and writes
// each tile's A-split to a global array. The MERGE kernel then reads those splits --
// no binary search -- so it is PURE streaming: coalesced load -> shared merge ->
// coalesced store. This is exactly cub's DeviceMergeSortPartitionKernel + MergeKernel.
__global__ void partition_kernel(const int* __restrict__ in, size_t n, size_t run,
                                 int* __restrict__ parts) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t out_base = t * OUT_TILE;
  if (out_base >= n) return;
  size_t base = (out_base / (2 * run)) * (2 * run);
  const int* A = in + base;
  const int* B = in + base + run;
  int g0 = (int)(out_base - base);
  parts[t] = merge_path(A, (int)run, B, (int)run, g0);   // the scattered binary search, isolated
}

__global__ void merge_runs_part(const int* __restrict__ in, int* __restrict__ out,
                                size_t n, size_t run, const int* __restrict__ parts) {
  __shared__ int sA[OUT_TILE];
  __shared__ int sB[OUT_TILE];
  __shared__ int sO[OUT_TILE];
  int tid = threadIdx.x;
  size_t t = blockIdx.x;
  size_t out_base = t * OUT_TILE;
  if (out_base >= n) return;
  size_t base = (out_base / (2 * run)) * (2 * run);
  const int* A = in + base;
  const int* B = in + base + run;
  int rn = (int)run;
  int g0 = (int)(out_base - base), g1 = g0 + OUT_TILE;
  int a0 = parts[t];                                    // precomputed -- NO global binary search
  int a1 = (g1 < 2 * rn) ? parts[t + 1] : rn;           // next tile's split, or end of A at group edge
  int b0 = g0 - a0, b1 = g1 - a1;
  int aCount = a1 - a0, bCount = b1 - b0;
  for (int i = tid; i < aCount; i += BLOCK) sA[i] = A[a0 + i];   // coalesced
  for (int i = tid; i < bCount; i += BLOCK) sB[i] = B[b0 + i];
  __syncthreads();
  int local = tid * SPT;
  int pa = merge_path(sA, aCount, sB, bCount, local);           // per-thread co-rank in SHARED (cheap)
  int pb = local - pa;
#pragma unroll
  for (int k = 0; k < SPT; ++k) {
    if (pb >= bCount || (pa < aCount && sA[pa] <= sB[pb])) sO[local + k] = sA[pa++];
    else sO[local + k] = sB[pb++];
  }
  __syncthreads();
  for (int i = tid; i < OUT_TILE; i += BLOCK) {                 // coalesced store
    size_t idx = out_base + i;
    if (idx < n) out[idx] = sO[i];
  }
}
#endif

static int* merge_sort(int* d, int* d2, size_t n) {
  block_sort<<<(int)(n / TILE), BLOCK>>>(d, n);
  int *in = d, *out = d2;
  int num_tiles = (int)((n + OUT_TILE - 1) / OUT_TILE);
#if OPT >= 4
  int* parts = nullptr;
  cudaMalloc(&parts, ((size_t)num_tiles + 1) * sizeof(int));   // reused across passes
#endif
  for (size_t run = TILE; run < n; run *= 2) {
#if OPT >= 4
    partition_kernel<<<(num_tiles + BLOCK - 1) / BLOCK, BLOCK>>>(in, n, run, parts);
    merge_runs_part<<<num_tiles, BLOCK>>>(in, out, n, run, parts);
#elif OPT == 3
    merge_runs_smem<<<num_tiles, BLOCK>>>(in, out, n, run);
#else
    int mblk = 256;
    int grid = (int)(((n + SPT - 1) / SPT + mblk - 1) / mblk);
    merge_runs<<<grid, mblk>>>(in, out, n, run);
#endif
    std::swap(in, out);
  }
#if OPT >= 4
  cudaFree(parts);
#endif
  return in;
}

int main(int argc, char** argv) {
  int log2n = (argc > 1) ? atoi(argv[1]) : 28;
  int iters = (argc > 2) ? atoi(argv[2]) : 10;
  size_t n = (size_t)1 << log2n, bytes = n * sizeof(int);
  auto h = demo2::make_uniform_input(n);

  int *d = nullptr, *d2 = nullptr;
  CUDA_CHECK(cudaMalloc(&d, bytes));
  CUDA_CHECK(cudaMalloc(&d2, bytes));
  CUDA_CHECK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
  int* res = merge_sort(d, d2, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> out(n);
  CUDA_CHECK(cudaMemcpy(out.data(), res, bytes, cudaMemcpyDeviceToHost));
  bool ok = demo2::verify_sorted(out, h);

  int* d_pristine = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pristine, bytes));
  CUDA_CHECK(cudaMemcpy(d_pristine, h.data(), bytes, cudaMemcpyHostToDevice));
  float ms = demo2::time_best_ms(iters, [&] {
    cudaMemcpyAsync(d, d_pristine, bytes, cudaMemcpyDeviceToDevice);
    merge_sort(d, d2, n);
  });
  double d2d = bytes / 810e9 * 1e3;
  const char* lbl[] = {"baseline", "+coalesce-load", "+vectorize", "+coalesce-merge", "+partition-kernel"};
  printf("merge OPT=%d %-16s n=2^%d  %8.3f ms  %8.1f Mkeys/s  [%s]\n",
         OPT, lbl[OPT], log2n, ms - d2d, n / 1e6 / ((ms - d2d) / 1e3),
         ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
