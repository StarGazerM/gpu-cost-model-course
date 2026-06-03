// Demo 1, GPU side: streaming-bandwidth identity.
//
// Thesis: the GPU saturates its memory bus by design. A trivial streaming
// kernel should reach 85%+ of theoretical GDDR6 peak. Contrast with the CPU
// STREAM number, which tops out at 60-70% of its (already much smaller) DRAM
// peak.
//
// To compare apples-to-apples with the CPU STREAM binary, the headline number
// is a STREAM Triad: a[i] = b[i] + s*c[i]  -> 2 reads + 1 write per element,
// moving 3*N*sizeof bytes (the same op and byte accounting as stream_cpu).
//
// We report:
//   1. STREAM Triad kernel  (the apples-to-apples number vs the CPU)
//   2. a hand-rolled vectorized copy kernel (out[i] = in[i], 2*N bytes)
//   3. cudaMemcpy device->device (the library path, 2*N bytes)

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

// STREAM Triad: a[i] = b[i] + s*c[i]. Same op as the CPU binary; 2 reads + 1
// write per element. float4-vectorized for wide coalesced transactions.
__global__ void triad_kernel(float4* __restrict__ a, const float4* __restrict__ b,
                             const float4* __restrict__ c, float s, size_t n4) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) {
    float4 bb = b[i], cc = c[i];
    a[i] = make_float4(bb.x + s * cc.x, bb.y + s * cc.y,
                       bb.z + s * cc.z, bb.w + s * cc.w);
  }
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
  double copy_gb = 2.0 * bytes / 1e9;  // read + write
  double triad_gb = 3.0 * bytes / 1e9; // 2 reads + 1 write (STREAM convention)

  printf("GPU: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  printf("Buffer: %.2f GB | theoretical peak: %.0f GB/s | L2: %.0f MB\n",
         bytes / 1e9, peak, prop.l2CacheSize / 1e6);

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemset(d_b, 1, bytes));
  CUDA_CHECK(cudaMemset(d_c, 1, bytes));
  float *d_in = d_b, *d_out = d_a; // copy/memcpy reuse two of the buffers

  int block = 256;
  int grid = 0, minGrid = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &block, triad_kernel, 0, 0));
  // Saturate the device: enough blocks to fill every SM.
  grid = 32 * prop.multiProcessorCount;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // --- STREAM Triad kernel (apples-to-apples vs the CPU) ---
  triad_kernel<<<grid, block>>>((float4*)d_a, (const float4*)d_b,
                                (const float4*)d_c, 3.0f, n4); // warmup
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    triad_kernel<<<grid, block>>>((float4*)d_a, (const float4*)d_b,
                                  (const float4*)d_c, 3.0f, n4);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float t_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&t_ms, start, stop));
  double t_gbps = triad_gb * iters / (t_ms / 1e3);

  // --- hand-rolled copy kernel ---
  copy_kernel<<<grid, block>>>((float4*)d_out, (const float4*)d_in, n4); // warmup
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    copy_kernel<<<grid, block>>>((float4*)d_out, (const float4*)d_in, n4);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  double k_gbps = copy_gb * iters / (ms / 1e3);

  // --- cudaMemcpy D2D ---
  CUDA_CHECK(cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice)); // warmup
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    CUDA_CHECK(cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  double m_gbps = copy_gb * iters / (ms / 1e3);

  printf("\n%-22s %10.1f GB/s  (%.1f%% of peak)\n", "triad kernel:", t_gbps,
         100.0 * t_gbps / peak);
  printf("%-22s %10.1f GB/s  (%.1f%% of peak)\n", "copy kernel:", k_gbps,
         100.0 * k_gbps / peak);
  printf("%-22s %10.1f GB/s  (%.1f%% of peak)\n", "cudaMemcpy D2D:", m_gbps,
         100.0 * m_gbps / peak);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}
