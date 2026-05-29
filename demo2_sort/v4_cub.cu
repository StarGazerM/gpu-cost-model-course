// Demo 2 v4: CUB DeviceRadixSort -- the production baseline.
//
// Your hand-rolled bitonic, fully optimized (v3), sits at the bitonic roofline:
// it streams the array O(log^2 n) ~= 100 times through HBM. Radix sort streams
// it O(digits) times -- ~4 passes for 32-bit keys (or fewer; CUB picks the
// radix and pass count adaptively). On a bandwidth machine, passes are destiny,
// so CUB should win by several x with two lines of code.
//
// Teaching point: production GPU code is library composition, not kernel
// writing. The interesting decisions become "which library, what layout, how to
// overlap" -- not "how do I shave a stall off my inner loop." CUB is what you
// ship; bitonic was the lesson.

#include "common.cuh"
#include <cub/cub.cuh>

int main(int argc, char** argv) {
  demo2::Args args = demo2::parse_args(argc, argv);
  auto h_in = demo2::make_uniform_input(args.n);
  size_t bytes = args.n * sizeof(int);

  int *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  // Two-phase CUB idiom: query temp-storage size, then allocate and run.
  void* d_temp = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, (int)args.n);
  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

  auto run = [&] {
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, (int)args.n);
  };

  run();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> h_out(args.n);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  bool ok = demo2::verify_sorted(h_out, h_in);

  // d_in is never modified by SortKeys, so every timed iter sorts fresh input.
  float ms = demo2::time_best_ms(args.iters, run);
  demo2::report("v4_cub", args.n, ms, ok);
  printf("  (temp storage: %.1f MB)\n", temp_bytes / 1e6);

  CUDA_CHECK(cudaFree(d_temp));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
