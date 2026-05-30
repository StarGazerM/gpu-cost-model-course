// Slide: "production GPU code is library composition."
// Everything to the left of this slide — four kernels, shared memory, warp
// shuffles, a profiler — is replaced by two calls. Radix streams the array
// O(digits) ~ 8 times; your bitonic streamed it ~100-350. On a bandwidth
// machine the algorithm's pass count is destiny, and CUB ships the right one.

#include <cub/cub.cuh>

void sort_keys(int* d_in, int* d_out, int n) {
  void*  d_temp = nullptr;
  size_t temp_bytes = 0;
  // Pass 1: ask how much scratch it needs.
  cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, n);
  cudaMalloc(&d_temp, temp_bytes);
  // Pass 2: sort.
  cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, n);
  cudaFree(d_temp);
}

int main() { return 0; }
