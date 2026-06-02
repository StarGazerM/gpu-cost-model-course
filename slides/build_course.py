#!/usr/bin/env python3
r"""Generate course.ipynb -- CLEAN REBUILD around the agreed structure.

Run:  python3 slides/build_course.py     (writes ../course.ipynb)
Then: jupyter nbconvert --to notebook --execute --inplace course.ipynb

Design principles (settled with the user):
  * ONE canonical "beat" per measured mechanism:
        card -> concept -> figure -> GUESS -> live %%cuda (or measure) -> PROFILE -> cost-model
    Guess is used for every MEASURED demo; concept-only beats skip it.
  * Each MECHANISM is shown NAKED once, as a 5-line live %%cuda where the chip property is taught
    (so nothing is a black box). The two sorts are the CAPSTONE where those mechanisms RECUR and
    we measure their deltas -- they reference the minis, they don't re-teach them.
  * Foundation is concept/figure/data (no code) EXCEPT the programming-model-in-one-kernel beat.
  * Heavy GB-scale + deep dives are an explicit APPENDIX, not the 1-hour spine.

SECTION MAP (1-hour spine):
  0  Foundation (no code): thesis; the bet; SIMT+terminology; latency->oversubscription;
     bandwidth+coalescing; the spec sheet (read it / buy a GPU); PROGRAMMING MODEL in one kernel.
  --- the mechanisms, each the canonical beat (concept+figure+mini %%cuda+profile+cost-model) ---
  1  Bandwidth          -- streaming copy, GB/s = the chip's speed unit.
  2  Latency->concurrency-- single-SM warp sweep: oversubscription hides latency (not "more cores").
  3  A thread is a lane -- divergence even/uneven (2x); the SIMT leak.
  4  Coalescing         -- coalesced vs strided copy + `-p` (sectors 4 vs 32): feed the bus.
  5  Registers / ILP    -- LDG 1-vs-8 in flight: register blocking = concurrency inside a thread.
  6  Cost model         -- radix vs merge: bytes/passes; the key-width crossover; thrust dispatch.
  7  Two sorts capstone -- 7a radix (passes-law, watch the digits); 7b merge (doubling + the opts
     from 1/4/5 RECUR, measure deltas; honest "scaffold" note).
  8  Kernel is ~10%     -- the rug pull (allocation/orchestration dominate).
  9  Closing            -- a library is a frozen cost-model decision tree.
  APPENDIX (deep-dive, off the clock): memory model/coherence; cub_ablate; key-width sweep;
     predication proof; cache cliff; die shot; buy-a-GPU detail.
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# ---- cell helpers ----
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))
def cuda(src, args=""):                       # live, editable real-CUDA cell (nvcc4jupyter)
    head = "%%cuda " + args if args else "%%cuda"
    cells.append(nbf.v4.new_code_cell(head + "\n" + src.strip("\n")))
def cuda_file(relpath, args=""):              # embed a real .cu as a live %%cuda cell (file = source of truth)
    with open(os.path.join(_ROOT, relpath)) as f:
        cuda(f.read(), args)

# ---- canonical-beat helpers (enforce consistent structure) ----
def H(num, title, mins):                      # section header
    md(f"## {num}. {title}  *(~{mins} min)*")
def card(demo, inp, outp, why):               # the demo "card": what/in/out/why
    md(f"> **Demo** `{demo}` &middot; **in:** {inp} &middot; **out:** {outp} &middot; **why:** {why}")
def concept(s): md(s)
def fig(name, caption): md(f"![{caption}](slides/figures/{name})")
def guess(prompt, left_label, left_code, right_label, right_code):
    md(f"**Guess first** 🎲 -- {prompt}\n\n"
       f'<table style="width:100%"><tr>'
       f'<td style="width:50%;vertical-align:top"><b>{left_label}</b><pre>{left_code}</pre></td>'
       f'<td style="width:50%;vertical-align:top"><b>{right_label}</b><pre>{right_code}</pre></td>'
       f"</tr></table>")
def costmodel(s): md("**Cost model.** " + s)

# ============================================================================
# 0. FOUNDATION  (concept / figure / data -- no code except the model-in-one-kernel beat)
# ============================================================================
md(r"""
# GPU programming is cost-model first
### A one-hour tour for computer architects meeting the GPU

> **You don't pick an algorithm and make the chip run it. You compute the cost model first** --
> bytes moved, passes over memory, latency vs concurrency, where data lives, how you touch it --
> **and *that* selects the algorithm before you write a line.** Asymptotic optimality is not hardware-neutral.

The chip: a single **RTX 6000 Ada** (sm_89). Every claim below is **measured live** on it.
""")

code(r"""
# setup: live-CUDA cells (%%cuda) + plotting. nvcc/ncu on PATH; arch set once.
import os, subprocess, re
import matplotlib.pyplot as plt
ROOT = os.getcwd()
get_ipython().run_line_magic("load_ext", "nvcc4jupyter")
from nvcc4jupyter import set_defaults
set_defaults(compiler_args="-arch=sm_89 -O3 -std=c++17")
def sh(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, cwd=cwd or ROOT, capture_output=True, text=True).stdout
def grab(t, p):
    m = re.search(p, t); return float(m.group(1)) if m else float("nan")
print("ready: %%cuda live cells enabled.")
""")

md(r"""
### 0.0 A 90-second recap (you know this): scalar -> SIMD
Anchor the vocabulary the GPU is about to twist. **(1)** Classic **von Neumann**: an instruction reads/writes
**scalar registers** (32/64-bit) -- one element per instruction, one ALU, one program counter. **(2)** Data-parallel
work (graphics, DSP, ML) runs the *same* op over arrays, so you **widen the register**: AVX-512 is a 512-bit
register = **16 float lanes**, and one instruction processes all 16 in a single issue -- *same control, more data
per cycle*. **(3)** That is **SIMD** (Single Instruction, Multiple Data), the CPU's data-parallel answer.

The GPU's answer (next) is a *different* point in this space -- **SIMT**. (And **ILP** -- independent instructions
in flight -- we deliberately defer to §5, where it becomes a GPU programming lever, not a transparent CPU feature.)
""")
fig("00_scalar_simd.png", "Scalar: one instruction -> one 32/64-bit register -> one result. SIMD: one instruction -> one wide register -> 16 results.")

md(r"""
### 0.1 The bet: throughput, not latency
A CPU core spends its area making *one* instruction stream fast (out-of-order, branch prediction, big caches).
A GPU rips that out and spends the area on **many ALUs** + **enough resident threads to switch among**.
Give up single-thread latency; buy aggregate throughput. Everything weird follows from this one bet.
""")
fig("01_area.png", "CPU spends area on control+cache for one fast thread; the GPU on a sea of ALUs + a huge register file.")

md(r"""
### 0.2 SIMT -- the GPU's *different* answer, and the words that mislead
The GPU does **not** use §0.0's SIMD (one thread, one wide register). Instead, **32 lanes share one
fetch/decode/scheduler** (a **warp**) and run the same instruction in lockstep, each on its *own* scalar
registers. That is **SIMD execution with a per-lane register file** -- the width is *across threads*, not a
wide register. (So you write plain scalar per-thread code and the hardware gangs 32.) Pin the vocabulary once,
or everything later is noise:

| you'll hear | it really is | CPU analogy |
|---|---|---|
| "CUDA core" (18,176!) | an **ALU / lane** | one SIMD lane |
| thread | one lane's scalar work | a lane of a vector op |
| warp | **32 lanes** sharing an instruction | one SIMD instruction |
| **SM** | the real processor (~142) | **the core** |
| register | per-lane; the file is partitioned by occupancy | register file |

One sentence: **SM = the core; warp = a SIMD instruction; "CUDA core" = a lane; thread = one lane's work.**
""")
fig("02_simt_vs_avx.png", "AVX: one thread, one wide register. SIMT: 32 threads, one shared instruction, per-lane scalar registers.")

md(r"""
### 0.3 Each op is slow -> hide it by oversubscription
No out-of-order engine, so a dependent global load is **hundreds of cycles** and the thread just stalls.
The GPU hides it by keeping **dozens of warps resident** and switching to a ready one the instant one stalls --
free, because every resident warp's registers live on-chip. (We prove this on *one* SM in §2.)
And memory bandwidth (~960 GB/s) only materializes if a warp's 32 lanes touch **contiguous** addresses
(**coalescing**, §4). Registers (~36 MB) and shared memory (programmer-managed scratchpad) are the fast tiers;
L2 (96 MB) and GDDR are the slow ones.
""")
fig("05_latency_hiding.png", "A stalled warp is covered by another resident warp -- zero-overhead switch; the SM is never idle.")

md(r"""
### 0.4 The programming model, in one kernel
Before the detail code, map every token to the hardware. A kernel is **scalar per-thread code**; the launch
`<<<grid, block>>>` stamps out `grid x block` threads; each finds its element by index; memory-space keywords
say **where a variable lives**. Run it:
""")
cuda(r"""
#include <cstdio>
__global__ void saxpy(int n, float a, const float* x, float* y){
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // (block, thread) -> a global element index
  if (i < n) y[i] = a*x[i] + y[i];                  // each lane: ordinary scalar code on ITS i
}                                                    // __global__ = runs on device, called from host
int main(){
  int n = 1<<20; size_t b = n*sizeof(float);
  float *x,*y; cudaMallocManaged(&x,b); cudaMallocManaged(&y,b);   // host<->device memory
  for(int i=0;i<n;i++){ x[i]=1.0f; y[i]=2.0f; }
  saxpy<<<(n+255)/256, 256>>>(n, 3.0f, x, y);       // grid of blocks x 256 threads/block (=warps of 32)
  cudaDeviceSynchronize();                          // host waits for the device
  printf("y[0]=%.1f (expect 5.0)  -- %d threads = %d blocks x 256\n", y[0], n, (n+255)/256);
}
""")
md(r"""
That's the whole model: **grid -> block -> warp(32) -> thread/lane**, `__global__`/`__device__`/`__shared__`/registers
for *where data lives*, and a host that launches + synchronizes. Everything past here is making this *fast*.
""")

# ============================================================================
# 1. BANDWIDTH  (first mechanism -- the canonical beat)
# ============================================================================
H(1, "Bandwidth is the chip's identity", 4)
card("a streaming copy", "one ~1 GB array", "achieved GB/s",
     "the simplest possible kernel, so the only thing measured is the bus.")
concept(r"""
The chip's native unit of work is **moving bytes**. A trivial `out[i]=in[i]` (vectorized to 128-bit `float4`)
saturates the GDDR6 bus; the number it hits is the yardstick every later demo is measured against.
""")
guess("a GPU streaming copy vs a CPU STREAM-Triad -- how many times the bandwidth?",
      "CPU STREAM Triad", "a[i] = b[i] + s*c[i]   // ~100s GB/s",
      "GPU copy", "out[i] = in[i]        // ~?? GB/s")
cuda(r"""
#include <cstdio>
__global__ void copy_kernel(float4* __restrict__ out, const float4* __restrict__ in, size_t n4){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x, stride = (size_t)gridDim.x*blockDim.x;
  for (; i < n4; i += stride) out[i] = in[i];        // 128-bit coalesced loads/stores
}
int main(){
  size_t n = (size_t)1<<28, bytes = n*sizeof(float);  // 1 GB
  float *in,*out; cudaMalloc(&in,bytes); cudaMalloc(&out,bytes); cudaMemset(in,1,bytes);
  size_t n4 = n/4; int blk=256, grid=(int)((n4+blk-1)/blk); if(grid>65535) grid=65535;
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  copy_kernel<<<grid,blk>>>((float4*)out,(const float4*)in,n4);   // warm up
  cudaEventRecord(a); copy_kernel<<<grid,blk>>>((float4*)out,(const float4*)in,n4); cudaEventRecord(b);
  cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b);
  printf("GPU copy: %.0f GB/s  (read+write %zu MB in %.2f ms)\n", 2.0*bytes/1e9/(ms/1e3), 2*bytes/(1<<20), ms);
}
""")
costmodel(r"""
A copy moves `2*N` bytes once; time = bytes / bandwidth. There is no algorithm here -- just the bus.
Everything later is "how close to this number can a *real* computation get?"
""")

# ============================================================================
# 2. LATENCY -> CONCURRENCY  (prove it on ONE SM)
# ============================================================================
H(2, 'Latency, and why "18,176 cores" is a lie', 5)
card("a single-SM pointer chase", "warps 1..32 on ONE block (one SM)", "accesses/s vs warp count",
     "pin to one SM so the speed-up CANNOT be 'more cores' -- it can only be latency hiding.")
concept(r"""
One thread doing dependent loads is **slow** (a global load is hundreds of cycles, nothing hides it in-thread).
Pin work to **one SM** (one block) and add warps: only the scheduler's ability to switch among them changes.
If throughput climbs, that climb *is* latency hidden by concurrency -- not silicon.
""")
fig("05_latency_hiding.png", "Stalled warp -> switch to a ready one; with enough warps the SM is never idle.")
guess("one block (one SM), going 1 -> 32 warps on it: flat (only 4 schedulers!), or faster? by how much?",
      "1 warp", "chase<<<1, 32>>>(...)", "32 warps", "chase<<<1, 32*32>>>(...)")
cuda(r"""
#include <cstdio>
#include <vector>
#include <numeric>
#include <random>
__global__ void chase(const int* __restrict__ next, int N, int steps, int* sink){
  int idx = (blockIdx.x*blockDim.x + threadIdx.x) % N;
  for (int i=0;i<steps;i++) idx = next[idx];          // THE dependent load (cannot be hidden in-thread)
  sink[(blockIdx.x*blockDim.x+threadIdx.x) & 1023] = idx;
}
int main(){
  int N = 1<<22; std::vector<int> perm(N); std::iota(perm.begin(),perm.end(),0);
  std::mt19937 rng(1); for(int i=N-1;i>0;i--) std::swap(perm[i],perm[rng()%(i+1)]);
  std::vector<int> nxt(N); for(int k=0;k<N;k++) nxt[perm[k]]=perm[(k+1)%N];
  int *d,*sink; cudaMalloc(&d,N*4); cudaMalloc(&sink,1024*4); cudaMemcpy(d,nxt.data(),N*4,cudaMemcpyHostToDevice);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  printf("ONE SM (1 block) -- only warps added, 4 schedulers:\n  warps   Maccess/s   speedup\n");
  double base=0;
  for (int w=1; w<=32; w*=2){
    int steps=20000, threads=32*w;
    chase<<<1,threads>>>(d,N,steps,sink); cudaDeviceSynchronize();
    cudaEventRecord(a); chase<<<1,threads>>>(d,N,steps,sink); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    double acc=(double)threads*steps/(ms/1e3)/1e6; if(w==1) base=acc;
    printf("  %4d   %9.1f    %.1fx\n", w, acc, acc/base);
  }
}
""")
costmodel(r"""
~18x on **one SM** with only warps added: the extra warps cannot be cores (there's one SM) -- they hide the
~500-cycle latency. **"18,176 cores" is occupancy you set with registers, not 18,176 CPUs.** Concurrency, not core count.
""")

# ============================================================================
# 3. A THREAD IS ALSO A SIMD LANE  (divergence)
# ============================================================================
H(3, "A thread is also a SIMD lane", 4)
card("demo8_divergence/divergence.cu", "same total work per warp, even vs uneven across lanes",
     "ms even vs uneven", "no per-lane branch predictor -> the warp runs at its slowest lane.")
concept(r"""
In §2 a "thread" was a *task* you oversubscribe. Inside a warp it's the opposite: 32 lanes are **one SIMD unit**.
A data-dependent branch isn't 32 independent threads -- the warp executes *every* path some lane takes
(**predication/divergence**), and a data-dependent loop runs until the **slowest** lane finishes.
Same total work; spread it *unevenly* across lanes and it's ~2x slower.
""")
fig("04_divergence.png", "At a branch the warp runs path A then path B, masking idle lanes -- cost is the sum.")
guess("both kernels do the SAME total work per warp; A spreads it evenly across lanes, B unevenly (lane L does L units). Same speed?",
      "A -- even", "int iters = 16 * step;     // every lane equal",
      "B -- uneven", "int iters = lane * step;   // lanes 0..31 differ")
cuda_file("demo8_divergence/divergence.cu")
costmodel(r"""
Same work, ~2x slower uneven -- the warp moves at its slowest lane. This is *why* the per-thread sort in §7b is a
branchless **network**: data-oblivious code never diverges. A "thread" is sometimes a task, sometimes a lane.
""")

# ============================================================================
# 4. COALESCING  (feed the bus) -- the killer profiler beat
# ============================================================================
H(4, "Coalescing: bandwidth only if lanes touch contiguous memory", 4)
card("a coalesced vs strided copy", "1 GB, two access patterns", "GB/s + sectors/request (live ncu)",
     "the single biggest memory lever: contiguous lanes fuse into one 128-byte transaction.")
concept(r"""
A warp's 32 lanes each have their *own* address (§0.2). If those 32 addresses are **contiguous**, the memory
system fuses them into **one** 128-byte transaction; if scattered, it issues many. Same bytes, but the strided
version throws the bus away. We profile it **live** -- the symptom is `sectors/request`: 4 (ideal) vs up to 32.
""")
fig("06b_coalescing.png", "Coalesced: 32 lanes -> contiguous -> 1 transaction. Strided: each its own sector -> many.")
guess("the SAME copy, contiguous vs strided addresses -- same speed, or how much slower? and what does the profiler show?",
      "coalesced", "out[i] = in[i];",
      "strided", "out[(i*16)%n] = in[(i*16)%n];")
cuda(r"""
#include <cstdio>
__global__ void coalesced(const int* in,int* out,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i]; }
__global__ void strided  (const int* in,int* out,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; int j=(i*16)%n; if(i<n) out[j]=in[j]; }
int main(){
  int n=1<<26; size_t b=(size_t)n*4; int *in,*out; cudaMalloc(&in,b); cudaMalloc(&out,b); cudaMemset(in,1,b);
  cudaEvent_t a,c; cudaEventCreate(&a); cudaEventCreate(&c); float ms;
  coalesced<<<n/256,256>>>(in,out,n); cudaEventRecord(a); coalesced<<<n/256,256>>>(in,out,n); cudaEventRecord(c);
  cudaEventSynchronize(c); cudaEventElapsedTime(&ms,a,c); printf("coalesced: %5.0f GB/s\n", 2.0*b/1e9/(ms/1e3));
  strided<<<n/256,256>>>(in,out,n); cudaEventRecord(a); strided<<<n/256,256>>>(in,out,n); cudaEventRecord(c);
  cudaEventSynchronize(c); cudaEventElapsedTime(&ms,a,c); printf("strided  : %5.0f GB/s\n", 2.0*b/1e9/(ms/1e3));
}
""")
md(r"""
Now the **same program under Nsight Compute, live** -- read `sectors/request` straight off the profiler:
""")
cuda(r"""
#include <cstdio>
__global__ void coalesced(const int* in,int* out,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i]; }
__global__ void strided  (const int* in,int* out,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; int j=(i*16)%n; if(i<n) out[j]=in[j]; }
int main(){ int n=1<<22; size_t b=(size_t)n*4; int *in,*out; cudaMalloc(&in,b); cudaMalloc(&out,b);
  coalesced<<<n/256,256>>>(in,out,n); strided<<<n/256,256>>>(in,out,n); cudaDeviceSynchronize(); }
""", args='-p --profiler-args "--metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio --launch-count 2 --kernel-name regex:c|s"')
costmodel(r"""
4 vs 32 sectors/request = ~8x the memory transactions for the *same* bytes. Coalescing is the #1 question of any
GPU kernel ("is your access contiguous?"), and -- preview -- it is the dominant opt in the §7b merge.
""")

# ============================================================================
# 5. REGISTERS / ILP  (latency hiding INSIDE a thread)
# ============================================================================
H(5, "Registers & ILP: concurrency inside one thread", 4)
card("a per-thread load loop, IPT=1 vs 8 (SASS)", "the same loads, unrolled 1 vs 8", "independent LDG.E in flight",
     "register blocking gives each thread independent work -> many loads outstanding -> latency hidden without more warps.")
concept(r"""
§2 hid latency with many **warps** (across threads). The other axis is **ILP** -- *independent instructions in
flight within one thread* (exactly superscalar/out-of-order intuition). Give a thread `IPT` independent items
and the unrolled load loop becomes `IPT` independent loads the scheduler issues **back-to-back, all outstanding**.
Read it straight off the SASS: 1 item -> 1 load (stall); 8 items -> 8 loads in flight.
""")
guess("a thread loading 1 vs 8 independent items: how many global loads (LDG.E) are in flight before any is used?",
      "IPT=1", "int a = g[i];          // load, then use",
      "IPT=8", "int a[8]; for k: a[k]=g[i*8+k];  // ?")
code(r"""
# compile a tiny per-thread load kernel at IPT=1 vs 8 and count independent LDG.E (the in-flight loads)
src = r'''
template<int IPT> __global__ void load(const int* g, int* o, int base){
  int a[IPT];
  #pragma unroll
  for(int k=0;k<IPT;k++) a[k] = g[base + threadIdx.x*IPT + k];   // IPT independent loads
  int s=0;
  #pragma unroll
  for(int k=0;k<IPT;k++) s ^= a[k];
  o[threadIdx.x]=s;
}
template __global__ void load<1>(const int*,int*,int);
template __global__ void load<8>(const int*,int*,int);
'''
open("/tmp/ld.cu","w").write(src)
sh("nvcc -arch=sm_89 -std=c++17 -cubin -o /tmp/ld.cubin /tmp/ld.cu")
for ipt,name in [(1,"_Z4loadILi1EEvPKiPii"),(8,"_Z4loadILi8EEvPKiPii")]:
    sass = sh(f"cuobjdump -sass -fun {name} /tmp/ld.cubin")
    n = sass.count("LDG.E")
    print(f"IPT={ipt}: {n} independent LDG.E in flight per thread")
print("\nIPT=1: one load, used immediately -> the thread STALLS.")
print("IPT=8: eight loads issued back-to-back -> 8 latencies overlapped by ONE thread (ILP/MLP).")
""")
costmodel(r"""
More items/thread = more independent loads outstanding = latency hidden *within* a thread -- the win survives even
past a huge L2 (it's concurrency, not caching). This is the **register-blocking** lever the §7b merge dials with `IPT`.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = os.path.join(_ROOT, "course.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", os.path.normpath(out), f"({len(cells)} cells)  [REBUILD: §0 foundation + §1 bandwidth]")
