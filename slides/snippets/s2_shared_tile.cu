// Slide: "the programmable scratchpad" — a cache you control.
// One block loads a TILE into shared memory, runs ALL the small-stride stages
// there (one global read in, one global write out), instead of a global-memory
// kernel per stage. The win is fewer HBM passes; the mechanism is shared memory,
// which a hardware cache cannot give you ("keep these 1024 keys resident").

#define TILE 1024                        // keys per tile (pow2); blockDim = TILE/2

__global__ void sort_tile(int* g) {
  __shared__ int s[TILE];
  int t = threadIdx.x;
  s[t] = g[t];  s[t + TILE/2] = g[t + TILE/2];   // coalesced load: HBM -> shared
  __syncthreads();

  for (int k = 2; k <= TILE; k <<= 1)
    for (int j = k >> 1; j > 0; j >>= 1) {
      int low = ((t & ~(j-1)) << 1) | (t & (j-1));  // this thread's pair
      int hi  = low + j;
      bool ascending = (low & k) == 0;
      __syncthreads();
      int a = s[low], b = s[hi];
      if ((a > b) == ascending) { s[low] = b; s[hi] = a; }  // swap in shared
    }

  __syncthreads();
  g[t] = s[t];  g[t + TILE/2] = s[t + TILE/2];    // coalesced store: shared -> HBM
}

int main() { return 0; }  // launch: sort_tile<<<n/TILE, TILE/2>>>(d)
