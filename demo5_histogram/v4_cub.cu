// Demo 5 v4: cub::DeviceHistogram::HistogramEven -- the production baseline.
//
// CUB privatizes into shared memory (with bank-conflict padding), register-
// blocks the loads, and -- by default -- uses the SORT-based block strategy
// rather than atomics, precisely so throughput stays flat under skew. Compare
// its uniform vs skewed numbers to v0-v2: the hand-written atomic versions wobble
// with the input distribution; CUB barely moves. That distribution-driven choice
// (atomic vs sort) is the cost model picking the algorithm, in production.

#include "common.cuh"
#include <cub/cub.cuh>

int main(int argc, char** argv) {
  histo::Args args = histo::parse_args(argc, argv);

  // Pre-size temp storage (depends on n + level count, not the data).
  size_t temp_bytes = 0;
  cub::DeviceHistogram::HistogramEven(
      nullptr, temp_bytes, (int*)nullptr, (unsigned*)nullptr, NBINS + 1, 0, HRANGE,
      (int)args.n);
  void* d_temp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

  int rc = histo::run_and_report("v4_cub", args,
      [&](const int* d_in, size_t n, unsigned* d_out) {
        cub::DeviceHistogram::HistogramEven(d_temp, temp_bytes, d_in, d_out,
                                            NBINS + 1, 0, HRANGE, (int)n);
      });
  CUDA_CHECK(cudaFree(d_temp));
  return rc;
}
