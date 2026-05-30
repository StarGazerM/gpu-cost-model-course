// Demo 5 v5: the OTHER algorithm -- sort-based histogram.
//
// Not atomics at all: radix-sort the samples, then the histogram is the
// run-lengths of equal keys (DeviceRunLengthEncode). This is what CUB's
// BlockHistogram defaults to (BLOCK_HISTO_SORT). Its cost is dominated by the
// sort -- so it is ~constant regardless of bin count or skew, the exact opposite
// of atomics. Expensive when atomics is cheap (few bins), but it has no cliff:
// it works at a million bins where privatization is impossible.
//
// This is the second algorithm the "cost model chooses the algorithm" claim
// needs: compare v5 vs the atomic v1/v1d across bin counts to see the crossover.

#include "common.cuh"
#include <cub/cub.cuh>

__global__ void map_to_bins(const int* __restrict__ in, int* __restrict__ bins,
                            size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    int b = histo::scale_bin(in[i]);
    bins[i] = (b >= 0) ? b : NBINS;  // out-of-range -> sentinel bin, dropped later
  }
}

__global__ void scatter_counts(const int* uniq, const unsigned* counts,
                               const int* num_runs, unsigned* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < *num_runs && uniq[i] < NBINS) out[uniq[i]] = counts[i];
}

int main(int argc, char** argv) {
  histo::Args args = histo::parse_args(argc, argv);
  size_t n = args.n;

  int *d_bins = nullptr, *d_sorted = nullptr, *d_uniq = nullptr, *d_runs = nullptr;
  unsigned* d_counts = nullptr;
  CUDA_CHECK(cudaMalloc(&d_bins, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_sorted, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_uniq, (NBINS + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_counts, (NBINS + 1) * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&d_runs, sizeof(int)));

  size_t sort_bytes = 0, rle_bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, (int*)nullptr,
                                 (int*)nullptr, (int)n);
  cub::DeviceRunLengthEncode::Encode(nullptr, rle_bytes, (int*)nullptr,
                                     (int*)nullptr, (unsigned*)nullptr,
                                     (int*)nullptr, (int)n);
  void *sort_temp = nullptr, *rle_temp = nullptr;
  CUDA_CHECK(cudaMalloc(&sort_temp, sort_bytes));
  CUDA_CHECK(cudaMalloc(&rle_temp, rle_bytes));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int mblock = 256, mgrid = 32 * prop.multiProcessorCount;

  int rc = histo::run_and_report("v5_sort", args,
      [&](const int* d_in, size_t nn, unsigned* d_out) {
        map_to_bins<<<mgrid, mblock>>>(d_in, d_bins, nn);     // key -> bin id
        cub::DeviceRadixSort::SortKeys(sort_temp, sort_bytes, d_bins, d_sorted, (int)nn);
        cub::DeviceRunLengthEncode::Encode(rle_temp, rle_bytes, d_sorted, d_uniq,
                                           d_counts, d_runs, (int)nn);
        scatter_counts<<<(NBINS + 256) / 256, 256>>>(d_uniq, d_counts, d_runs, d_out);
      });

  cudaFree(d_bins); cudaFree(d_sorted); cudaFree(d_uniq); cudaFree(d_counts);
  cudaFree(d_runs); cudaFree(sort_temp); cudaFree(rle_temp);
  return rc;
}
