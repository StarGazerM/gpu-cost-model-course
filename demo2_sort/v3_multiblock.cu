// Demo 2 v3: scale the scratchpad to cut global passes (near memory roofline).
//
// On a bandwidth machine the cost of bitonic sort is the number of times you
// stream the whole array through HBM. v1/v2 collapsed the j < 2048 stages into
// shared memory, leaving 120 wide-stride GLOBAL passes (which dominate runtime).
//
// v3's lever: a bigger tile. With TILE = 8192, every stage with j < 8192 runs
// in shared memory, so the global passes drop from 120 to ~91. Each tile is now
// 32 KB of shared memory and a block streams 8 keys/thread. We keep v2's warp
// shuffle tail for the j < 32 stages.
//
// Grid-level structure: the wide-stride merge across tiles is sequenced by
// kernel relaunch -- the launch boundary IS the grid-wide barrier. The modern
// alternative is one persistent kernel with cooperative_groups::grid_group::
// sync(); it removes relaunch overhead but pins occupancy to what fits
// resident. Relaunch is the portable default (spec open decision #3).
//
// Teaching point: grid-level structure -- deciding what each block owns and how
// blocks synchronize -- is where GPU code stops looking like a parallel-for and
// starts looking like an algorithm laid out across a memory hierarchy.
//
// The kernels live in bitonic_v3.cuh so the Demo 3 harnesses drive the same sort.

#include "common.cuh"
#include "bitonic_v3.cuh"

int main(int argc, char** argv) {
  demo2::Args args = demo2::parse_args(argc, argv);
  if (args.n < (size_t)bitonic_v3::TILE) {
    fprintf(stderr, "n must be >= TILE (%d)\n", bitonic_v3::TILE);
    return 1;
  }
  auto h_in = demo2::make_uniform_input(args.n);

  int* d = nullptr;
  size_t bytes = args.n * sizeof(int);
  CUDA_CHECK(cudaMalloc(&d, bytes));
  CUDA_CHECK(cudaMemcpy(d, h_in.data(), bytes, cudaMemcpyHostToDevice));

  bitonic_v3::sort(d, args.n);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> h_out(args.n);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d, bytes, cudaMemcpyDeviceToHost));
  bool ok = demo2::verify_sorted(h_out, h_in);

  float ms = demo2::time_best_ms(args.iters, [&] { bitonic_v3::sort(d, args.n); });
  demo2::report("v3_multiblock", args.n, ms, ok);

  CUDA_CHECK(cudaFree(d));
  return ok ? 0 : 1;
}
