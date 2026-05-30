// Slide: "naive bitonic" — the whole algorithm, and why it's bandwidth-naive.
// One kernel launch per stage (the relaunch is the grid barrier). Every
// compare-swap reads and writes global memory: one HBM pass per stage, and
// there are L(L+1)/2 of them. That pass count IS the runtime.

__global__ void stage(int* a, int n, int j, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int ixj = i ^ j;                       // partner index
  if (ixj > i && ixj < n) {
    bool ascending = (i & k) == 0;       // direction of this sub-sequence
    if ((a[i] > a[ixj]) == ascending) {  // out of order -> swap (in global mem)
      int t = a[i]; a[i] = a[ixj]; a[ixj] = t;
    }
  }
}

void bitonic_sort(int* a, int n, int grid, int block) {
  for (int k = 2; k <= n; k <<= 1)       // ~log2(n) outer stages
    for (int j = k >> 1; j > 0; j >>= 1) // ~log2(n) inner stages each
      stage<<<grid, block>>>(a, n, j, k);// => O(log^2 n) HBM passes
}

int main() { return 0; }
