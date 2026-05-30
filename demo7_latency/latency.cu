// Latency vs latency-hiding: the missing keystone of the first half.
//
// Part A -- raw memory latency (pointer chase, ONE thread): dependent loads,
//   each waits for the previous, so we measure the round-trip to HBM. The GPU's
//   per-access latency is *bad* (hundreds of ns / ~hundreds of cycles).
//
// Part B -- latency hiding (copy bandwidth vs # resident warps): the SAME kernel,
//   launched with more and more warps. Few warps -> latency exposed -> low
//   bandwidth; enough warps -> the scheduler hides every stall by switching ->
//   bandwidth saturates. That curve IS the GPU's whole trick.
//
// Little's law: achievable_bandwidth = bytes_in_flight / latency. To fill the bus
//   you need thousands of concurrent loads in flight -> thousands of warps. That
//   is *why* the chip has a huge register file and lightweight threads.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <numeric>

#define CK(c) do{cudaError_t _err=(c); if(_err){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_err));exit(1);} }while(0)

// dependent loads: idx = next[idx], one thread, no parallelism to hide latency.
__global__ void chase(const int* __restrict__ next, int steps, int* sink) {
  int idx = 0;
  for (int i = 0; i < steps; ++i) idx = next[idx];
  *sink = idx;  // keep it live
}

__global__ void copy_k(float4* __restrict__ o, const float4* __restrict__ in, size_t n4) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) o[i] = in[i];
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  double clkGHz = p.clockRate / 1e6;   // kHz -> GHz

  // ---- Part A: pointer-chase latency over a >L2 array (1 GB) ----
  size_t N = (size_t)1 << 28;          // 256M ints = 1 GB >> 96 MB L2
  int steps = 2'000'000;
  std::vector<int> perm(N);            // a single random cycle visiting all N
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937_64 rng(1);
  for (size_t i = N - 1; i > 0; --i) std::swap(perm[i], perm[rng() % (i + 1)]);
  std::vector<int> nxt(N);             // nxt[perm[k]] = perm[k+1] -> one big cycle
  for (size_t k = 0; k < N; ++k) nxt[perm[k]] = perm[(k + 1) % N];

  int *d_next, *d_sink;
  CK(cudaMalloc(&d_next, N * sizeof(int))); CK(cudaMalloc(&d_sink, sizeof(int)));
  CK(cudaMemcpy(d_next, nxt.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  chase<<<1, 1>>>(d_next, 100000, d_sink); CK(cudaDeviceSynchronize());  // warmup
  CK(cudaEventRecord(t0)); chase<<<1, 1>>>(d_next, steps, d_sink); CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
  double lat_ns = ms * 1e6 / steps;
  printf("=== Part A: raw memory latency (1 thread, dependent loads) ===\n");
  printf("  HBM access latency: %.0f ns  (~%.0f cycles @ %.2f GHz)\n\n",
         lat_ns, lat_ns * clkGHz, clkGHz);

  // ---- Part B: bandwidth vs concurrency (latency hiding) ----
  size_t cn = (size_t)1 << 28;         // 1 GB copy buffer
  size_t bytes = cn * sizeof(float), n4 = cn / 4;
  float *d_in, *d_out;
  CK(cudaMalloc(&d_in, bytes)); CK(cudaMalloc(&d_out, bytes)); CK(cudaMemset(d_in, 1, bytes));
  double moved = 2.0 * bytes / 1e9;
  int block = 256;
  printf("=== Part B: copy bandwidth vs resident warps (latency hiding) ===\n");
  printf("  warps     GB/s   %% of peak\n");
  for (int blocks : {1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 32 * p.multiProcessorCount}) {
    copy_k<<<blocks, block>>>((float4*)d_out, (const float4*)d_in, n4);  // warmup
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(t0));
    for (int it = 0; it < 20; ++it) copy_k<<<blocks, block>>>((float4*)d_out, (const float4*)d_in, n4);
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    CK(cudaEventElapsedTime(&ms, t0, t1));
    double gbps = moved * 20 / (ms / 1e3);
    int warps = blocks * block / 32;
    printf("  %6d  %7.1f   %5.1f%%\n", warps, gbps, 100.0 * gbps / 960.0);
  }
  // Little's law: in-flight bytes needed to fill the bus at the measured latency
  double need_bytes = 960e9 * (lat_ns * 1e-9);
  printf("\n  Little's law: to fill ~960 GB/s at %.0f ns latency you need ~%.0f KB\n",
         lat_ns, need_bytes / 1e3);
  printf("  in flight -> thousands of concurrent loads -> thousands of warps.\n");
  printf("  That is why the GPU has a ~36 MB register file and lightweight threads.\n");
  return 0;
}
