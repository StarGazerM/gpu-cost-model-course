// Demo 5 (histogram) shared harness: input generation, verification, timing.
//
// Histogram is the cleanest cost-model example: read N samples once, trivial
// compute, tiny output -> arithmetic intensity ~0.5 ops/byte, deeply
// memory-bound. The predicted floor is just input_bytes / bandwidth. The whole
// arc is about whether you actually REACH that floor, or stall on atomic
// contention -- which is what skew controls.
//
// Samples are int32 in [0, NBINS) (so CUB HistogramEven over [0,NBINS) bins 1:1
// and every version histograms the same thing). zipf=0 is uniform; zipf>0 is a
// Zipfian skew over bins (DB selectivity / cardinality-estimation regime).

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

#ifndef NBINS
#define NBINS 256
#endif

#ifndef HRANGE
#define HRANGE (1 << 30)   // sample value range [0, HRANGE): arbitrary keys, NOT bin ids
#endif

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(_e));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace histo {

static const double PEAK_GBPS = 960.0;  // RTX 6000 Ada GDDR6 theoretical

struct Args { size_t n; int iters; int log2n; double zipf; };

inline Args parse_args(int argc, char** argv, int default_log2n = 28) {
  Args a;
  a.log2n = (argc > 1) ? atoi(argv[1]) : default_log2n;
  a.iters = (argc > 2) ? atoi(argv[2]) : 20;
  a.zipf  = (argc > 3) ? atof(argv[3]) : 0.0;   // 0 = uniform
  a.n = (size_t)1 << a.log2n;
  return a;
}

// CUB HistogramEven ScaleTransform: map an arbitrary key in [0,HRANGE) to a bin.
// bin = (sample - 0) * NBINS / HRANGE; out-of-range samples are dropped.
__host__ __device__ inline int scale_bin(int sample) {
  if (sample < 0 || sample >= HRANGE) return -1;
  return (int)(((long long)sample * (long long)NBINS) / (long long)HRANGE);
}

// Arbitrary int keys in [0,HRANGE) -- NOT bin ids. zipf<=0: uniform keys.
// zipf>0: the BIN distribution is Zipfian (a few hot value-ranges), keys uniform
// within each bin -- realistic skew, the way a real column is distributed.
inline std::vector<int> make_input(size_t n, double zipf, unsigned seed = 7) {
  std::vector<int> v(n);
  std::mt19937 rng(seed);
  if (zipf <= 0.0) {
    std::uniform_int_distribution<int> d(0, HRANGE - 1);
    for (size_t i = 0; i < n; ++i) v[i] = d(rng);
  } else {
    std::vector<double> w(NBINS);
    for (int k = 0; k < NBINS; ++k) w[k] = 1.0 / std::pow(k + 1, zipf);
    std::discrete_distribution<int> bin_dist(w.begin(), w.end());
    int bin_width = HRANGE / NBINS;
    std::uniform_int_distribution<int> off(0, bin_width - 1);
    for (size_t i = 0; i < n; ++i) v[i] = bin_dist(rng) * bin_width + off(rng);
  }
  return v;
}

inline std::vector<unsigned> cpu_histogram(const std::vector<int>& in) {
  std::vector<unsigned> h(NBINS, 0);
  for (int x : in) { int b = scale_bin(x); if (b >= 0) h[b]++; }
  return h;
}

inline bool verify(const std::vector<unsigned>& got, const std::vector<int>& in) {
  std::vector<unsigned> ref = cpu_histogram(in);
  for (int b = 0; b < NBINS; ++b)
    if (got[b] != ref[b]) {
      fprintf(stderr, "  VERIFY FAIL bin %d: got %u want %u\n", b, got[b], ref[b]);
      return false;
    }
  return true;
}

template <typename F>
inline float time_best_ms(int iters, F&& run_once) {
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float best = 1e30f;
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(s));
    run_once();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    if (ms < best) best = ms;
  }
  return best;
}

inline void report(const char* name, size_t n, double zipf, float ms, bool ok) {
  double gb = n * sizeof(int) / 1e9;
  double gbps = gb / (ms / 1e3);
  double floor_ms = gb / PEAK_GBPS * 1e3;
  printf("%-18s n=2^%d zipf=%.1f  %8.3f ms  %7.1f GB/s (%4.1f%% peak)  "
         "floor=%.2fms  [%s]\n",
         name, (int)__builtin_ctzll(n), zipf, ms, gbps, 100.0 * gbps / PEAK_GBPS,
         floor_ms, ok ? "PASS" : "FAIL");
}

// `launch(d_in, n, d_out)` must fill d_out[0..NBINS) with the histogram. The
// runner zeroes d_out before each call and times memset+launch together.
template <typename Launch>
inline int run_and_report(const char* name, Args args, Launch&& launch) {
  auto h_in = make_input(args.n, args.zipf);
  int* d_in = nullptr;
  unsigned* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, args.n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, NBINS * sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), args.n * sizeof(int),
                        cudaMemcpyHostToDevice));
  auto once = [&] {
    CUDA_CHECK(cudaMemset(d_out, 0, NBINS * sizeof(unsigned)));
    launch(d_in, args.n, d_out);
  };
  once();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<unsigned> h_out(NBINS);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NBINS * sizeof(unsigned),
                        cudaMemcpyDeviceToHost));
  bool ok = verify(h_out, h_in);
  float ms = time_best_ms(args.iters, once);
  report(name, args.n, args.zipf, ms, ok);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}

}  // namespace histo
