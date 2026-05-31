// Using the lanes (the right way): 32 lanes cooperatively sort 32 values with
// NOTHING but register-to-register shuffles -- no shared memory, no __syncthreads,
// and only data-OBLIVIOUS selects (no data-dependent branch -> no divergence).
//
// This is LITERALLY net_sort (demo6's per-thread odd-even network) turned sideways:
//   net_sort:        for i: for j=(i&1); j+1<N; j+=2:  compare a[j], a[j+1]   (down ONE thread's registers)
//   warp_oddeven:    for i: each lane compares with neighbor lane^... (phase i&1) (ACROSS the warp's 32 lanes)
// Same odd-even transposition network, same N rounds; only the axis differs --
// registers vs lanes, the two faces of a "thread".
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#define CK(c) do{cudaError_t e=(c); if(e){printf("CUDA %s\n",cudaGetErrorString(e));return 1;}}while(0)

__device__ int warp_oddeven_sort(int v) {            // each lane holds ONE value v
  int lane = threadIdx.x & 31;
  for (int i = 0; i < 32; ++i) {                      // N rounds, exactly like net_sort's outer loop
    int phase   = i & 1;                              // even rounds pair (0,1)(2,3)..., odd rounds (1,2)(3,4)...
    bool is_low = ((lane & 1) == phase);              // am I the LOWER element of my pair this round?
    int partner = is_low ? lane + 1 : lane - 1;       // the neighbor lane I compare-exchange with
    bool active = (unsigned)partner < 32u;            // edge lanes (0 / 31) sit out some rounds
    int pv  = __shfl_sync(0xffffffffu, v, active ? partner : lane);  // read neighbor's register
    int lo  = min(v, pv), hi = max(v, pv);            // branchless compare-exchange
    v = active ? (is_low ? lo : hi) : v;              // data-oblivious selects -> no divergence
  }
  return v;                                           // warp's 32 values now sorted across lanes
}

__global__ void warp_sort(int* g) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  g[t] = warp_oddeven_sort(g[t]);   // each warp sorts its 32 contiguous values
}

int main() {
  size_t n = 1 << 22;               // many warps' worth
  std::vector<int> h(n);
  std::mt19937 rng(1);
  for (auto& x : h) x = rng();
  int* d; CK(cudaMalloc(&d, n * sizeof(int)));
  CK(cudaMemcpy(d, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  warp_sort<<<n / 256, 256>>>(d);
  CK(cudaDeviceSynchronize());
  std::vector<int> out(n);
  CK(cudaMemcpy(out.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (size_t w = 0; w < n; w += 32) {
    for (int i = 1; i < 32; ++i) if (out[w + i] < out[w + i - 1]) ok = false;
    std::vector<int> ref(h.begin() + w, h.begin() + w + 32);
    std::sort(ref.begin(), ref.end());
    for (int i = 0; i < 32; ++i) if (out[w + i] != ref[i]) ok = false;
  }
  printf("32 lanes sorting 32 values via __shfl only -- no shared mem, no divergence: [%s]\n",
         ok ? "PASS" : "FAIL");
  printf("  this is net_sort's odd-even network (demo6), turned ACROSS lanes -- lanes cooperating.\n");
  return ok ? 0 : 1;
}
