// CPU contrast: ONE coherence domain. The atomic is unscoped (there is no scope to pick),
// and a plain write becomes visible to other cores automatically via MESI -- you fence only
// to order, never to publish.  g++ -O2 -S coherence_cpu.cpp  ->  one 'lock' op, no scope.
#include <atomic>
void bump(std::atomic<int>& c, int n) {           // -> lock xadd / lock add  (unscoped)
  for (int i = 0; i < n; ++i) c.fetch_add(1, std::memory_order_relaxed);
}
int data; std::atomic<int> flag{0};
void producer() { data = 42; flag.store(1, std::memory_order_release); }  // no explicit "publish":
int  consumer() { while (!flag.load(std::memory_order_acquire)) {} return data; }  // MESI already made data coherent
