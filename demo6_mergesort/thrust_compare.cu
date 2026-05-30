// Thrust cross-check + the cost-model-in-user-code demo.
//
// Thrust's __smart_sort dispatches at COMPILE TIME (sort.h: can_use_primitive_sort):
//   thrust::sort(int*)            -> arithmetic key + default less -> RADIX
//   thrust::sort(int*, MyLess())  -> custom comparator             -> MERGE
// Same data, same ordering -- the comparator's *type* flips the algorithm, and
// the cost model makes it ~4x. This both validates the CUB numbers (Thrust falls
// back to the same two CUB calls) and shows the decision in two lines of code.

#include "../demo2_sort/common.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

// A custom functor that is NOT thrust::less<int>, so Thrust CANNOT use radix
// and must fall back to merge sort -- even though it means exactly "a < b".
struct MyLess {
  __host__ __device__ bool operator()(int a, int b) const { return a < b; }
};

int main(int argc, char** argv) {
  int log2n = (argc > 1) ? atoi(argv[1]) : 28;
  int iters = (argc > 2) ? atoi(argv[2]) : 10;
  size_t n = (size_t)1 << log2n;
  auto h = demo2::make_uniform_input(n);
  thrust::device_vector<int> pristine = h;   // unsorted master copy on device
  thrust::device_vector<int> work(n);
  std::vector<int> out(n);

  // ---- thrust::sort(x)  -> radix path (default comparator) ----
  work = pristine; thrust::sort(work.begin(), work.end());
  thrust::copy(work.begin(), work.end(), out.begin());
  bool okr = demo2::verify_sorted(out, h);
  // both paths refresh from pristine (D2D) each iter -> fair (both in-place).
  float rms = demo2::time_best_ms(iters, [&] {
    work = pristine; thrust::sort(work.begin(), work.end());
  });

  // ---- thrust::sort(x, MyLess())  -> merge path (custom comparator) ----
  work = pristine; thrust::sort(work.begin(), work.end(), MyLess());
  thrust::copy(work.begin(), work.end(), out.begin());
  bool okm = demo2::verify_sorted(out, h);
  float mms = demo2::time_best_ms(iters, [&] {
    work = pristine; thrust::sort(work.begin(), work.end(), MyLess());
  });

  printf("n=2^%d (%.0fM keys)\n", log2n, n / 1e6);
  printf("  thrust::sort(x)           -> radix : %8.3f ms  [%s]\n", rms,
         okr ? "PASS" : "FAIL");
  printf("  thrust::sort(x, MyLess()) -> merge : %8.3f ms  [%s]\n", mms,
         okm ? "PASS" : "FAIL");
  printf("  one custom comparator cost you %.1fx -- the cost model, by type.\n",
         mms / rms);
  return (okr && okm) ? 0 : 1;
}
