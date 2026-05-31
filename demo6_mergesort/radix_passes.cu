// "Watch the passes" for RADIX -- the counterpart to merge_passes.cu. LSD radix
// makes NO comparisons: it reads a DIGIT of the key and stably buckets by it, one
// digit at a time, low to high. After k digit-passes the whole key is sorted.
//
// The cost-model point: radix's cost is the NUMBER OF PASSES = key_bits / digit_bits,
// independent of n. For 8-bit keys with 2-bit digits that's 4 passes; for the real
// 32-bit keys with 8-bit digits, just 4 passes -- vs merge's ~18 passes at 2^28.
// That pass ratio is *why* radix wins on a bandwidth machine (see §5a).
//
//   nvcc -O3 -std=c++17 -o /tmp/rp radix_passes.cu && /tmp/rp

#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

#define N 24
#define KEY_BITS 8
#define DIGIT_BITS 2
#define RADIX (1 << DIGIT_BITS)          // 4 buckets
#define PASSES (KEY_BITS / DIGIT_BITS)   // 4 passes

static void print_bits(int v, int active_pass) {
  int lo = active_pass * DIGIT_BITS, hi = lo + DIGIT_BITS;   // the digit this pass reads
  for (int b = KEY_BITS - 1; b >= 0; --b) {
    if (b == hi - 1) printf("[");
    printf("%d", (v >> b) & 1);
    if (b == lo) printf("]");
  }
}

static void dump(const char* tag, const std::vector<int>& a, int pass) {
  printf("%-26s ", tag);
  for (int v : a) { print_bits(v, pass); printf(" "); }
  // sorted by the low (pass+1)*DIGIT_BITS bits?
  int mask = (1 << ((pass + 1) * DIGIT_BITS)) - 1;
  bool ok = true;
  for (int i = 1; i < (int)a.size(); ++i)
    if ((a[i] & mask) < (a[i - 1] & mask)) ok = false;
  printf(" %s\n", pass < 0 ? "" : (ok ? "(sorted by low bits so far)" : ""));
}

int main() {
  std::vector<int> a(N);
  std::mt19937 rng(3);
  for (auto& x : a) x = rng() % (1 << KEY_BITS);

  printf("LSD radix, n=%d, %d-bit keys, %d-bit digits -> %d passes. [] marks the digit each pass reads:\n\n",
         N, KEY_BITS, DIGIT_BITS, PASSES);
  std::vector<int> ref = a;
  std::sort(ref.begin(), ref.end());

  { std::vector<int> tmp = a; printf("%-26s ", "input (unsorted)");
    for (int v : tmp) print_bits(v, 0), printf(" "); printf("\n"); }

  for (int p = 0; p < PASSES; ++p) {
    // stable counting sort on digit p:  histogram -> exclusive prefix-sum -> scatter
    int count[RADIX] = {0};
    for (int v : a) count[(v >> (p * DIGIT_BITS)) & (RADIX - 1)]++;
    int off[RADIX], sum = 0;
    for (int d = 0; d < RADIX; ++d) { off[d] = sum; sum += count[d]; }   // where each bucket starts
    std::vector<int> out(N);
    for (int v : a) out[off[(v >> (p * DIGIT_BITS)) & (RADIX - 1)]++] = v;  // stable scatter
    a = out;
    char tag[32]; snprintf(tag, sizeof tag, "after pass %d (digit %d)", p + 1, p);
    dump(tag, a, p);
  }
  bool ok = (a == ref);
  printf("\n-> %d passes, NO comparisons -- each pass just read a 2-bit digit and bucketed. [%s]\n",
         PASSES, ok ? "PASS" : "FAIL");
  printf("   cost = passes = key_bits/digit_bits, INDEPENDENT of n. That's the whole story.\n");
  return ok ? 0 : 1;
}
