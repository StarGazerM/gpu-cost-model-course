// Demo 2 shared harness: input generation, verification, timing, reporting.
//
// Every vN.cu includes this and follows the same flow:
//   gen input -> H2D -> warmup -> timed iters -> D2H -> verify vs std::sort.
//
// Bitonic sort requires a power-of-two element count. All versions sort
// int32 keys; default n = 2^26 (64M keys, 256 MB) -- deliberately past the
// 96 MB L2 so the arc measures real HBM traffic, not cache residency.

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(_e));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace demo2 {

// Parse CLI: argv[1] = log2(n) (default 24), argv[2] = timed iters (default 10).
struct Args {
  size_t n;
  int iters;
  int log2n;
};

inline Args parse_args(int argc, char** argv, int default_log2n = 26) {
  Args a;
  a.log2n = (argc > 1) ? atoi(argv[1]) : default_log2n;
  a.iters = (argc > 2) ? atoi(argv[2]) : 10;
  a.n = (size_t)1 << a.log2n;
  return a;
}

inline std::vector<int> make_uniform_input(size_t n, unsigned seed = 1234) {
  std::vector<int> v(n);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, INT32_MAX);
  for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
  return v;
}

// Verify `got` equals the std::sort of `original`. Cheap insurance per spec §8.6.
inline bool verify_sorted(const std::vector<int>& got,
                          const std::vector<int>& original) {
  std::vector<int> ref = original;
  std::sort(ref.begin(), ref.end());
  if (got.size() != ref.size()) {
    fprintf(stderr, "  VERIFY FAIL: size %zu != %zu\n", got.size(), ref.size());
    return false;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != ref[i]) {
      fprintf(stderr, "  VERIFY FAIL at [%zu]: got %d, want %d\n", i, got[i],
              ref[i]);
      return false;
    }
  }
  return true;
}

// Event-timed loop. `iters` timed runs after one warmup; returns best ms.
template <typename F>
inline float time_best_ms(int iters, F&& run_once) {
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  run_once();  // warmup
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
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return best;
}

inline void report(const char* name, size_t n, float best_ms, bool ok) {
  double mkeys = (double)n / 1e6 / (best_ms / 1e3);
  printf("%-14s n=2^%d (%.0fM keys)  %8.3f ms  %8.1f Mkeys/s  [%s]\n", name,
         (int)__builtin_ctzll(n), n / 1e6, best_ms, mkeys,
         ok ? "PASS" : "FAIL");
}

}  // namespace demo2
