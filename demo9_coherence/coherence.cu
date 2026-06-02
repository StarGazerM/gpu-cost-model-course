// The GPU memory model is EXPLICIT and SCOPED -- the opposite of a CPU's transparent
// cache coherence. Two things a CPU simply does not have:
//
//  (1) SCOPE is a literal instruction modifier. The *same* cuda::atomic::fetch_add
//      compiles to  atom.add...cta  vs  atom.add...gpu  depending on thread_scope.
//      The programmer chooses the coherence domain per-op. (x86: one 'lock xadd', no scope.)
//
//  (2) Cross-SM visibility is NOT automatic. A producer must __threadfence() to push its
//      write past its (non-coherent) L1 to L2 -- the coherence/rendezvous point -- before
//      a consumer on another SM can see it. (A CPU's MESI propagates the value for free;
//      you only fence to stop *reordering*, never to *publish*.)
//
//   nvcc -O3 -std=c++17 -arch=sm_89 -o /tmp/coh demo9_coherence/coherence.cu && /tmp/coh
#include <cstdio>
#include <cuda/atomic>
#define CK(c) do{cudaError_t e=(c); if(e){printf("CUDA %s\n",cudaGetErrorString(e));return 1;}}while(0)

// (1) scope: same source, scope is a template param -> different PTX atomic (.cta vs .gpu)
template <cuda::thread_scope S>
__global__ void bump(int* c, int iters) {
  cuda::atomic_ref<int, S> a(*c);
  for (int i = 0; i < iters; ++i) a.fetch_add(1, cuda::memory_order_relaxed);
}

// (2) message passing across two blocks (-> two SMs). Producer publishes data, sets flag;
// consumer waits on flag, reads data. WITHOUT the producer fence, the data write may still
// sit in the producer's L1 -> consumer can observe flag==1 but read STALE data.
__global__ void mp(int* data, int* flag, int use_fence, int* saw) {
  cuda::atomic_ref<int, cuda::thread_scope_device> f(*flag);
  if (blockIdx.x == 0) {                                   // producer
    *data = 42;
    if (use_fence) __threadfence();                        // <-- publish to L2 (the coherence point)
    f.store(1, cuda::memory_order_release);
  } else {                                                 // consumer (block 1, another SM)
    while (f.load(cuda::memory_order_acquire) == 0) {}
    *saw = *data;                                          // 42 only if the producer published
  }
}

int main() {
  // (1)
  int blocks = 132, threads = 256, iters = 2000; long exp = (long)blocks*threads*iters;
  int *d, h; CK(cudaMalloc(&d, 4));
  CK(cudaMemset(d,0,4)); bump<cuda::thread_scope_device><<<blocks,threads>>>(d,iters); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost)); int dev=h;
  CK(cudaMemset(d,0,4)); bump<cuda::thread_scope_block><<<blocks,threads>>>(d,iters); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost)); int blk=h;
  printf("(1) SCOPE -- same fetch_add, scope picks the instruction (see PTX: atom...gpu vs atom...cta):\n");
  printf("    device scope -> %d (%s)\n", dev, dev==exp?"correct":"WRONG");
  printf("    block  scope -> %d (%s on THIS chip -- but it emits atom.cta; relying on L2\n", blk, blk==exp?"correct":"WRONG");
  printf("                    serializing it is undefined by the model -- a latent, hardware-specific bug)\n");
  // (2)
  int *data,*flag,*saw;
  CK(cudaMalloc(&data,4)); CK(cudaMalloc(&flag,4)); CK(cudaMalloc(&saw,4));
  CK(cudaMemset(data,0,4)); CK(cudaMemset(flag,0,4)); CK(cudaMemset(saw,-1,4));
  mp<<<2,1>>>(data,flag,/*use_fence=*/1,saw); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&h,saw,4,cudaMemcpyDeviceToHost));
  printf("(2) VISIBILITY -- consumer on another SM, WITH producer __threadfence(): saw data = %d (%s)\n",
         h, h==42?"published":"stale");
  printf("    remove the fence and the consumer may see flag==1 but read stale data: visibility is\n");
  printf("    YOURS to publish (fence -> L2), not the hardware's. No MESI does this for you.\n");
  return 0;
}
