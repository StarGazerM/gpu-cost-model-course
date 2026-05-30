// Proving latency hiding -- not just "use more of the chip".
//
// Part A: one thread, dependent loads -> raw HBM latency. One thread is slow.
//
// Part B: PIN TO ONE SM (a single block) and add warps 1..32 *on that one SM*.
//   Throughput still climbs -- and the only thing that changed is how many warps
//   the *same* SM has to switch among. That is context-switching hiding latency,
//   proven: it can't be "more cores", there is exactly one SM running.
//
// Part C: how many warps fit on an SM = occupancy = register_file / (regs/thread
//   * 32). A register-heavy kernel fits fewer warps -> hides less latency -> goes
//   slower, doing the same memory work. So "18,176 CUDA cores" is NOT 18,176 baby
//   CPUs; it's the number of tasks you can keep in flight, and it MOVES with your
//   register usage -- which is why you need ptxas (-v) and a profiler.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <random>

#define CK(c) do{cudaError_t _e=(c); if(_e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} }while(0)

// dependent chase; K extra live registers (cheap XOR/load -> still latency-bound,
// but consumes registers, so K controls occupancy without changing the memory work.
template <int K>
__global__ void chase(const int* __restrict__ next, size_t N, int steps, int* sink) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t idx = (tid * 2654435761ull) % N;     // scattered start per thread
  int r[K];
#pragma unroll
  for (int k = 0; k < K; ++k) r[k] = (int)tid + k;
  for (int i = 0; i < steps; ++i) {
    idx = next[idx];                            // the dependent load
#pragma unroll
    for (int k = 0; k < K; ++k) r[k] ^= (int)idx;
  }
  int s = 0;
#pragma unroll
  for (int k = 0; k < K; ++k) s ^= r[k];
  sink[tid & 8191] = s;                         // keep the work live
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  int coresPerSM = 128, schedPerSM = 4;  // Ada: 128 FP32 lanes, 4 warp schedulers
  printf("Chip: %d SMs x %d FP32 lanes = %d \"CUDA cores\".\n", p.multiProcessorCount,
         coresPerSM, p.multiProcessorCount * coresPerSM);
  printf("Per SM: %d warp schedulers -> it can ISSUE only %d warps per clock "
         "(128 lanes = %d warps of SIMD). Naively, '%d things at once'.\n",
         schedPerSM, schedPerSM, schedPerSM, schedPerSM);
  printf("Yet it holds up to %d warps resident. Why so many more than it can run?\n\n",
         p.maxThreadsPerMultiProcessor / 32);

  // permutation cycle over a >L2 array
  size_t N = (size_t)1 << 28;                   // 1 GB >> 96 MB L2
  std::vector<int> perm(N); std::iota(perm.begin(), perm.end(), 0);
  std::mt19937_64 rng(1);
  for (size_t i = N - 1; i > 0; --i) std::swap(perm[i], perm[rng() % (i + 1)]);
  std::vector<int> nxt(N);
  for (size_t k = 0; k < N; ++k) nxt[perm[k]] = perm[(k + 1) % N];
  int *d_next, *d_sink;
  CK(cudaMalloc(&d_next, N * sizeof(int))); CK(cudaMalloc(&d_sink, 8192 * sizeof(int)));
  CK(cudaMemcpy(d_next, nxt.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  auto time_ms = [&](int blocks, int threads, int steps) {
    chase<1><<<blocks, threads>>>(d_next, N, steps, d_sink); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a)); chase<1><<<blocks, threads>>>(d_next, N, steps, d_sink);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); return ms;
  };

  // ---- Part A: raw latency, one thread ----
  int sA = 2'000'000;
  float ms = time_ms(1, 1, sA);
  double lat = ms * 1e6 / sA;
  printf("Part A  one thread, dependent loads: %.0f ns/access (~%.0f cycles). "
         "one thread is slow.\n\n", lat, lat * (p.clockRate / 1e6));

  // ---- Part B: ONE SM (1 block), warps 1..32, vs the 4-scheduler issue width ----
  printf("Part B  ONE SM (single block), add warps past the %d schedulers it can issue:\n",
         schedPerSM);
  printf("  warps  oversub   Maccess/s   speedup\n");
  int sB = 20000; double base = 0, t4 = 0, t32 = 0;
  for (int w = 1; w <= 32; w *= 2) {
    float t = time_ms(1, 32 * w, sB);
    double acc = (double)32 * w * sB / (t / 1e3) / 1e6;
    if (w == 1) base = acc;
    if (w == schedPerSM) t4 = acc;
    if (w == 32) t32 = acc;
    printf("  %4d   %4.0fx     %9.1f     %.1fx\n", w, (double)w / schedPerSM, acc, acc / base);
  }
  printf("  -> %d warps already fills the %d schedulers; %d warps (%dx oversubscribed) is "
         "%.1fx FASTER.\n", schedPerSM, schedPerSM, 32, 32 / schedPerSM, t32 / t4);
  printf("     Those extra warps cannot be using cores -- there are only %d -- they are\n"
         "     hiding the %.0f-cycle latency. Warps are TASKS you oversubscribe, not cores.\n\n",
         schedPerSM, lat * (p.clockRate / 1e6));

  // ---- Part C: occupancy = f(registers) ----
  printf("\nPart C  warps that fit on an SM = occupancy, set by registers (it MOVES):\n");
  int maxw = p.maxThreadsPerMultiProcessor / 32;
  auto report = [&](const char* lbl, auto kern) {
    cudaFuncAttributes fa; int blk;
    CK(cudaFuncGetAttributes(&fa, kern));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blk, kern, 256, 0));
    printf("  %-11s %3d regs/thread  ->  %2d of %d warps/SM resident\n",
           lbl, fa.numRegs, blk * 256 / 32, maxw);
  };
  report("light", chase<1>);
  report("heavy", chase<48>);
  report("very heavy", chase<128>);
  printf("\n  Same silicon. The register count -- visible only via ptxas -v / the\n");
  printf("  profiler -- decides where on the Part B curve you land. Push registers\n");
  printf("  past its knee (~16 warps) and you stop hiding latency. So '18,176 cores'\n");
  printf("  is not 18,176 CPUs and is not fixed -- it is occupancy you must measure.\n");
  return 0;
}
