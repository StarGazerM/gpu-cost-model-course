// Demo 1, GPU side: streaming-copy bandwidth.
//
// Thesis: the GPU saturates its memory bus by design. A trivial copy kernel
// should reach 85%+ of theoretical GDDR6 peak. Contrast with the CPU STREAM
// number, which tops out at 60-70% of its (already much smaller) DRAM peak.
//
// We report two numbers:
//   1. a hand-rolled vectorized copy kernel (out[i] = in[i])
//   2. cudaMemcpy device->device (the library path)
// Both move 2*N*sizeof bytes (one read + one write).

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(_e));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// float4 = 16 B per thread per access -> wide, fully coalesced transactions.
__global__ void copy_kernel(float4* __restrict__ out,
                            const float4* __restrict__ in, size_t n4) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) out[i] = in[i];
}

static double theoretical_peak_gbps(int dev) {
  int mem_clk_khz = 0, bus_bits = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
  // GDDR is double-data-rate: 2 transfers per clock.
  return 2.0 * (double)mem_clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;
}

int main(int argc, char** argv) {
  size_t n = (argc > 1) ? strtoull(argv[1], nullptr, 0) : (1ull << 28); // 256M floats
  int iters = (argc > 2) ? atoi(argv[2]) : 50;

  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  double peak = theoretical_peak_gbps(dev);

  size_t bytes = n * sizeof(float);
  size_t n4 = n / 4; // float4 count
  double moved_gb = 2.0 * bytes / 1e9; // read + write

  printf("GPU: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  printf("Buffer: %.2f GB x2  | theoretical peak: %.0f GB/s | L2: %.0f MB\n",
         bytes / 1e9, peak, prop.l2CacheSize / 1e6);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemset(d_in, 1, bytes));

  int block = 256;
  int grid = 0, minGrid = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &block, copy_kernel, 0, 0));
  // Saturate the device: enough blocks to fill every SM.
  grid = 32 * prop.multiProcessorCount;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // --- hand-rolled kernel ---
  copy_kernel<<<grid, block>>>((float4*)d_out, (const float4*)d_in, n4); // warmup
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    copy_kernel<<<grid, block>>>((float4*)d_out, (const float4*)d_in, n4);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  double k_gbps = moved_gb * iters / (ms / 1e3);

  // --- cudaMemcpy D2D ---
  CUDA_CHECK(cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice)); // warmup
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    CUDA_CHECK(cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  double m_gbps = moved_gb * iters / (ms / 1e3);

  printf("\n%-22s %10.1f GB/s  (%.1f%% of peak)\n", "copy kernel:", k_gbps,
         100.0 * k_gbps / peak);
  printf("%-22s %10.1f GB/s  (%.1f%% of peak)\n", "cudaMemcpy D2D:", m_gbps,
         100.0 * m_gbps / peak);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
