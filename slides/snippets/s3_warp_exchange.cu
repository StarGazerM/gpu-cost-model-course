// Slide: "the register file is your fastest memory."
// For the innermost stages (distance j < 32) both elements of every pair live
// in the same warp. No shared memory, no __syncthreads(): each lane holds one
// key in a register and reads its partner's register directly via a shuffle.
// This is the SIMT analogue of a SIMD shuffle.
//
// (Teaching footnote: at HBM scale this barely moves the wall clock — the global
//  passes dominate — but the profiler shows the shared/barrier stalls vanish.
//  Optimizing the part that wasn't the bottleneck is the lesson, not the speedup.)

__device__ int warp_bitonic_tail(int v, int lane, bool ascending) {
  for (int j = 16; j >= 1; j >>= 1) {                 // j = 16,8,4,2,1
    int partner = __shfl_xor_sync(0xffffffff, v, j);  // read lane^j's register
    int lo = min(v, partner), hi = max(v, partner);
    bool i_am_lower = (lane & j) == 0;
    v = (i_am_lower == ascending) ? lo : hi;          // keep min or max
  }
  return v;
}

int main() { return 0; }
