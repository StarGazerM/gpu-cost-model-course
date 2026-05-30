// Using the lanes (the right way): 32 lanes cooperatively sort 32 values with
// NOTHING but register-to-register shuffles -- no `if` (branchless min/max), no
// shared memory. This is the sorting-NETWORK idea (same as net_sort) but ACROSS
// the warp's lanes. It's how lanes cooperate when there's no latency to hide.
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#define CK(c) do{cudaError_t e=(c); if(e){printf("CUDA %s\n",cudaGetErrorString(e));return 1;}}while(0)

__device__ int warp_bitonic_sort(int v) {
  int lane = threadIdx.x & 31;
  for (int k = 2; k <= 32; k <<= 1)
    for (int j = k >> 1; j > 0; j >>= 1) {
      int partner = __shfl_xor_sync(0xffffffffu, v, j);   // read lane^j's register
      bool keep_min = ((lane & j) == 0) == ((lane & k) == 0);
      v = keep_min ? min(v, partner) : max(v, partner);    // branchless
    }
  return v;
}

__global__ void warp_sort(int* g) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  g[t] = warp_bitonic_sort(g[t]);   // each warp sorts its 32 contiguous values
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
  printf("32 lanes sorting 32 values via __shfl_xor only -- no if, no shared mem: [%s]\n",
         ok ? "PASS" : "FAIL");
  printf("  this is net_sort's sorting network, but ACROSS lanes -- lanes cooperating.\n");
  return ok ? 0 : 1;
}
