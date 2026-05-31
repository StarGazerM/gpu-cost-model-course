// "Watch the passes" -- a TINY merge sort (same kernels as merge_ablation, shrunk
// to n=64) that DUMPS the array after every stage, so you can literally see the
// sorted runs DOUBLE as the merge walks down the memory hierarchy:
//
//   input              : 64 random values, runs of length 1
//   per-thread net_sort: runs of length IPT=4        (in REGISTERS)
//   block_sort         : runs of length TILE=16       (merged through SHARED memory)
//   device pass, run=16: runs of length 32            (merged through GLOBAL memory)
//   device pass, run=32: runs of length 64 = sorted   (one run = done)
//
// Same doubling, same MergePath merge -- only WHERE the data lives changes.
//   nvcc -O3 -std=c++17 -arch=sm_89 -o /tmp/mp merge_passes.cu && /tmp/mp

#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <climits>

#define IPT 4
#define BLOCK 4
#define TILE (BLOCK * IPT)     // 16
#define SPT 4

__device__ __forceinline__ int merge_path(const int* A, int aN, const int* B,
                                          int bN, int diag) {
  int lo = diag > bN ? diag - bN : 0, hi = diag < aN ? diag : aN;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (A[mid] <= B[diag - 1 - mid]) lo = mid + 1; else hi = mid;
  }
  return lo;
}

__device__ __forceinline__ void net_sort(int (&a)[IPT]) {
  for (int i = 0; i < IPT; ++i)
    for (int j = (i & 1); j + 1 < IPT; j += 2) {
      int lo = min(a[j], a[j + 1]), hi = max(a[j], a[j + 1]);
      a[j] = lo; a[j + 1] = hi;
    }
}

// per-thread network only: load IPT keys, sort them in registers, write back.
__global__ void net_only(int* __restrict__ g, size_t n) {
  int keys[IPT];
  size_t b = ((size_t)blockIdx.x * BLOCK + threadIdx.x) * IPT;
#pragma unroll
  for (int i = 0; i < IPT; ++i) keys[i] = (b + i < n) ? g[b + i] : INT_MAX;
  net_sort(keys);
#pragma unroll
  for (int i = 0; i < IPT; ++i) if (b + i < n) g[b + i] = keys[i];
}

// full tile sort: net_sort + doubling shared-memory MergePath merge -> sorted TILE.
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
  net_sort(keys);
  for (int stride = 1; stride < BLOCK; stride *= 2) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < IPT; ++i) s[tid * IPT + i] = keys[i];
    __syncthreads();
    int group = tid / (2 * stride), local = tid % (2 * stride), runlen = stride * IPT;
    const int* A = s + (size_t)group * (2 * stride) * IPT;
    const int* B = A + runlen;
    int diag = local * IPT;
    int a = merge_path(A, runlen, B, runlen, diag), b = diag - a;
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

__global__ void merge_runs(const int* __restrict__ in, int* __restrict__ out,
                           size_t n, size_t run) {
  size_t o = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * SPT;
  if (o >= n) return;
  size_t base = (o / (2 * run)) * (2 * run);
  const int* A = in + base, *B = in + base + run;
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

#define N 64
static void dump(const char* tag, int where, int* d, size_t runlen) {
  std::vector<int> h(N);
  cudaMemcpy(h.data(), d, N * sizeof(int), cudaMemcpyDeviceToHost);
  const char* loc[] = {"input ", "REG   ", "SHARED", "GLOBAL"};
  printf("%-22s [%s] runs of %2zu: ", tag, loc[where], runlen);
  bool all_ok = true;
  for (size_t r = 0; r < N; r += runlen) {
    printf("|");
    bool sorted = true;
    for (size_t i = r; i < r + runlen && i < N; ++i) {
      printf("%3d", h[i]);
      if (i > r && h[i] < h[i - 1]) sorted = false;
    }
    if (!sorted) all_ok = false;
  }
  printf("|  %s\n", all_ok ? "(each run sorted)" : "(UNSORTED)");
}

int main() {
  std::vector<int> h(N);
  std::mt19937 rng(7);
  for (auto& x : h) x = rng() % 100;
  int *d = nullptr, *d2 = nullptr;
  cudaMalloc(&d, N * sizeof(int));
  cudaMalloc(&d2, N * sizeof(int));
  cudaMemcpy(d, h.data(), N * sizeof(int), cudaMemcpyHostToDevice);

  printf("Parallel merge sort, n=%d, IPT=%d, BLOCK=%d, TILE=%d -- watch runs DOUBLE:\n\n", N, IPT, BLOCK, TILE);
  dump("0. input", 0, d, 1);
  net_only<<<N / TILE, BLOCK>>>(d, N);
  dump("1. per-thread net_sort", 1, d, IPT);
  cudaMemcpy(d, h.data(), N * sizeof(int), cudaMemcpyHostToDevice);  // restart for the full block sort
  block_sort<<<N / TILE, BLOCK>>>(d, N);
  dump("2. block_sort (tile)", 2, d, TILE);
  int *in = d, *out = d2;
  for (size_t run = TILE; run < N; run *= 2) {
    int grid = (int)(((N + SPT - 1) / SPT + BLOCK - 1) / BLOCK);
    merge_runs<<<grid, BLOCK>>>(in, out, N, run);
    std::swap(in, out);
    char tag[48]; snprintf(tag, sizeof tag, "3. device merge run=%zu", run);
    dump(tag, 3, in, run * 2);
  }
  printf("\n-> one run of %d = fully sorted. The algorithm never changed; the data just\n", N);
  printf("   moved registers -> shared -> global as the runs outgrew each level.\n");
  return 0;
}
