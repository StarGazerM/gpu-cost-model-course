// Demo 1, CPU side: STREAM Triad with OpenMP.
//
// Triad: a[i] = b[i] + scalar * c[i]  -> 2 reads + 1 write per element.
// STREAM convention counts 24 bytes/element (3 doubles); the write also incurs
// a read-for-ownership in practice, so true DRAM traffic is ~32 B/element. We
// report the STREAM-convention number (comparable to published results) and let
// the % of theoretical peak expose the LFB/MSHR-bound gap that is the point.
//
// Run pinned to one NUMA node so the memory model is uniform:
//   numactl --cpunodebind=0 --membind=0 ./stream_cpu
// First-touch init below places pages on the node that runs the threads.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

int main(int argc, char** argv) {
  // Default 80M elements => 640 MB/array, 1.9 GB total: comfortably past the
  // 64 MB L3 so we measure DRAM, not cache.
  size_t n = (argc > 1) ? strtoull(argv[1], NULL, 0) : 80000000ull;
  int iters = (argc > 2) ? atoi(argv[2]) : 20;
  STREAM_TYPE scalar = 3.0;

  STREAM_TYPE* a = (STREAM_TYPE*)malloc(n * sizeof(STREAM_TYPE));
  STREAM_TYPE* b = (STREAM_TYPE*)malloc(n * sizeof(STREAM_TYPE));
  STREAM_TYPE* c = (STREAM_TYPE*)malloc(n * sizeof(STREAM_TYPE));
  if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }

  int nthreads = 0;
  #pragma omp parallel
  {
    #pragma omp master
    nthreads = omp_get_num_threads();
  }

  // First-touch: each thread initializes the pages it will later stream,
  // so pages land on the correct NUMA node.
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.5; }

  double bytes_per_iter = 3.0 * sizeof(STREAM_TYPE) * (double)n; // STREAM triad
  double best = 1e30;
  for (int it = 0; it < iters; ++it) {
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) a[i] = b[i] + scalar * c[i];
    double dt = omp_get_wtime() - t0;
    if (dt < best) best = dt;
    // Touch a[] so the compiler can't elide the loop.
    if (a[it % n] < 0) printf("%f", a[it % n]);
  }

  double gbps = bytes_per_iter / 1e9 / best;
  printf("CPU STREAM Triad (double)\n");
  printf("Threads: %d | array: %.2f GB x3 | best time: %.3f ms\n",
         nthreads, n * sizeof(STREAM_TYPE) / 1e9, best * 1e3);
  printf("%-22s %10.1f GB/s\n", "triad bandwidth:", gbps);
  printf("(divide by your per-node theoretical peak for %% of peak)\n");
  return 0;
}
