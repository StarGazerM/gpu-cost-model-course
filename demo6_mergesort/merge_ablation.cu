// Merge-sort ablation: a CUB-faithful merge sort with optimizations on
// compile-time toggles, so we can remove one at a time and measure it at GB
// scale (where the chip is actually stressed). Structure mirrors CUB's
// block_merge_sort.cuh: per-thread register sort (odd-even network) -> doubling
// shared-memory merge via MergePath co-rank -> device-level MergePath merge.
//
// Knob 1 (this file): IPT = ITEMS_PER_THREAD (register blocking). Build with
//   -DIPT=1,2,4,8,16 and watch whether throughput climbs at GB scale.
//
//   nvcc -O3 -std=c++17 -arch=sm_89 -DIPT=8 -o /tmp/m8 merge_ablation.cu

#include "../demo2_sort/common.cuh"
#include <climits>
#include <algorithm>

#ifndef IPT
#define IPT 8
#endif
#ifndef BLOCK
#define BLOCK 256
#endif
#define TILE (BLOCK * IPT)
#define SPT 8                      // device-merge output elements per thread

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

__device__ __forceinline__ void net_sort(int (&a)[IPT]) {  // odd-even transposition
  for (int i = 0; i < IPT; ++i)
    for (int j = (i & 1); j + 1 < IPT; j += 2)
      if (a[j] > a[j + 1]) { int t = a[j]; a[j] = a[j + 1]; a[j + 1] = t; }
}

// Sort one TILE per block: register network per thread, then doubling merge
// through shared memory with MergePath (mirrors cub::BlockMergeSort).
__global__ void block_sort(int* __restrict__ g, size_t n) {
  __shared__ int s[TILE];
  int tid = threadIdx.x;
  size_t base = (size_t)blockIdx.x * TILE;
  int keys[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    size_t idx = base + (size_t)tid * IPT + i;
    keys[i] = (idx < n) ? g[idx] : INT_MAX;
  }
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

// Merge sorted runs of length `run` into runs of `2*run` (device-level MergePath).
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

// returns pointer to the buffer holding the sorted result (d or d2)
static int* merge_sort(int* d, int* d2, size_t n) {
  block_sort<<<(int)(n / TILE), BLOCK>>>(d, n);
  int *in = d, *out = d2;
  int mblk = 256;
  int grid = (int)(((n + SPT - 1) / SPT + mblk - 1) / mblk);
  for (size_t run = TILE; run < n; run *= 2) {
    merge_runs<<<grid, mblk>>>(in, out, n, run);
    std::swap(in, out);
  }
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

  // re-upload each iter via D2D from a pristine device copy (fair, cheap).
  int* d_pristine = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pristine, bytes));
  CUDA_CHECK(cudaMemcpy(d_pristine, h.data(), bytes, cudaMemcpyHostToDevice));
  float ms = demo2::time_best_ms(iters, [&] {
    cudaMemcpyAsync(d, d_pristine, bytes, cudaMemcpyDeviceToDevice);
    merge_sort(d, d2, n);
  });
  double d2d = bytes / 810e9 * 1e3;  // subtract the D2D refresh (~at copy BW)
  printf("merge IPT=%-2d TILE=%-5d  n=2^%d  %8.3f ms (%.1f incl D2D)  %8.1f Mkeys/s  [%s]\n",
         IPT, TILE, log2n, ms - d2d, ms, n / 1e6 / ((ms - d2d) / 1e3),
         ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
