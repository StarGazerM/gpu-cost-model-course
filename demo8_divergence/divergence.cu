// The other face of a thread: within a warp, "threads" are 32 SIMD lanes in
// lockstep. Same total work -- but if the lanes do *different amounts* (a
// data-dependent branch/loop), the warp runs at its SLOWEST lane and the rest
// idle. The per-thread code never shows you this; it's a warp-level cost.
//
// uniform : every lane runs K iterations            -> no divergence
// skewed  : lane L runs L iterations (avg ~K)        -> SAME total work, warp waits for max
#include <cstdio>
#define CK(c) do{cudaError_t e=(c); if(e){printf("CUDA %s\n",cudaGetErrorString(e));return 1;}}while(0)

__global__ void work(unsigned* out, int skewed, int step) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & 31;
  int iters = (skewed ? lane : 16) * step;      // skewed avg 15.5 ~ uniform 16
  unsigned x = tid * 2654435761u;
  for (int i = 0; i < iters; ++i) x = x * 1664525u + 1013904223u;
  out[tid & 16383] = x;
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  int grid = 64 * p.multiProcessorCount, block = 256, step = 4000;
  unsigned* d; CK(cudaMalloc(&d, 16384 * sizeof(unsigned)));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  auto t = [&](int skewed) -> float {
    work<<<grid, block>>>(d, skewed, step); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < 20; ++i) work<<<grid, block>>>(d, skewed, step);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms / 20;
  };
  float u = t(0), s = t(1);
  printf("Same total work per warp (avg ~16 lanes' worth), only the *spread* differs:\n");
  printf("  uniform  (all 32 lanes do equal work):  %6.3f ms\n", u);
  printf("  skewed   (lanes do 0..31 -- a branch):  %6.3f ms   (%.1fx slower)\n", s, s / u);
  printf("\n  The 32 'threads' are one SIMD unit: the warp runs at its slowest lane.\n");
  printf("  Your scalar per-thread code never showed this -- it's a warp-level cost.\n");
  return 0;
}
