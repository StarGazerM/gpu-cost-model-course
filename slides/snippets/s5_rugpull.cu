// Slide: "the kernel is ~10%."
// The SAME tuned sort, wrapped the way a query stage actually runs in a pipeline.
// The kernel never changes — only how memory is managed — yet per-iteration time
// swings widely. cudaMalloc/cudaFree are synchronous driver calls that serialize
// the device; a stream-ordered pool (cudaMallocAsync) recycles memory with no
// driver round-trip.

static void sort(int*, int, cudaStream_t) {}   // the v3 kernel sequence (elsewhere)

// --- naive: a driver round-trip + device sync every iteration ---------------
void stage_naive(int* h_in, int* h_out, int n, cudaStream_t s) {
  int *d_in, *d_out;
  cudaMalloc(&d_in,  n * sizeof(int));      // <- synchronous, serializes device
  cudaMalloc(&d_out, n * sizeof(int));
  cudaMemcpyAsync(d_in, h_in, n*sizeof(int), cudaMemcpyHostToDevice, s);
  sort(d_in, n, s);
  cudaMemcpyAsync(h_out, d_in, n*sizeof(int), cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  cudaFree(d_in);  cudaFree(d_out);         // <- synchronous too
}

// --- the one-line fix: stream-ordered pool, no driver round-trip ------------
void stage_pool(int* h_in, int* h_out, int n, cudaStream_t s) {
  int *d_in, *d_out;
  cudaMallocAsync(&d_in,  n * sizeof(int), s);   // <- from the pool, on the stream
  cudaMallocAsync(&d_out, n * sizeof(int), s);
  cudaMemcpyAsync(d_in, h_in, n*sizeof(int), cudaMemcpyHostToDevice, s);
  sort(d_in, n, s);
  cudaMemcpyAsync(h_out, d_in, n*sizeof(int), cudaMemcpyDeviceToHost, s);
  cudaFreeAsync(d_in, s);  cudaFreeAsync(d_out, s);
  cudaStreamSynchronize(s);
  // (set cudaMemPoolAttrReleaseThreshold = UINT64_MAX once, or the pool returns
  //  memory to the OS each iteration and this is no faster than cudaMalloc.)
}

int main() { return 0; }
