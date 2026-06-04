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

SECTION MAP -- re-estimated after the §2/§3 rewrites (they grew a lot).
The 1-HOUR course is now §0-§3 (~64 min, tight). §4+ is the 2-hour extension / cut for the 1-hr run.
  0   Foundation (~14m; concept+figure, 1 live saxpy + a term-quiz): thesis; the bet; SIMT+terminology;
      the 4-way scalar->ILP->SIMD->SIMT; programming model in one kernel; the speeds&feeds scorecard.
  --- mechanisms, each the canonical beat: card->concept->figure->guess->live %%cuda->cost-model ---
  1   Bandwidth (~5m)        -- STREAM Triad on CPU *and* GPU, one source; GB/s = the chip's speed unit.
  2   Latency, THE PREMISE (~13m) -- per-layer pointer-chase staircase, CPU vs GPU (GPU loses at every
      level); why pointer chasing; then HIDE it on one SM (warp sweep). Fills the scorecard latency column.
  2b  Warps vs registers (~6m) -- TLP vs ILP/register-blocking; the per-SM register/occupancy cap (live OOR).
  2c  Blocks -> SMs (~6m)    -- the second launch axis; escape the per-SM cap; scale to the shared ceiling.
  2.5 Cache cliff (~5m)      -- bandwidth vs working set; fills the scorecard bandwidth column.
  3   A thread is a SIMD lane (~15m) -- sort a partition: branchy merge vs cub::StableOddEvenSort; the
      trace+figure on the same 8 numbers; divergence; SASS predication (data vs control); complexity-vs-
      cycles ("terrible" O(N^2) is perfect at small N); read-CUDA-as-a-hint takeaway.
  --- 2-HOUR EXTENSION (cut from the 1-hour run) ---
  4   Coalescing            -- coalesced vs strided + `-p` (sectors 4 vs 32): feed the bus.
  5   Registers / ILP       -- LDG 1-vs-8 in flight: register blocking = concurrency inside a thread.
  6   Cost model            -- radix vs merge: bytes/passes; the key-width crossover; thrust dispatch.
  7   Two sorts capstone    -- 7a radix (passes-law); 7b merge (the opts RECUR; honest "scaffold" note).
  8   Kernel is ~10%        -- the rug pull (allocation/orchestration dominate).
  9   Closing               -- a library is a frozen cost-model decision tree.
  APPENDIX (deep-dive): memory model/coherence; cub_ablate; key-width sweep; predication; die shot.
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

md(r"""
## Outline

Each mechanism is one **canonical beat**: a card -> the concept -> a figure -> *you guess* -> a live `%%cuda` cell ->
the measured cost model. The mechanisms recur; nothing is a black box.

**The 1-hour course -- §0-§3 (~64 min):**
- **§0 -- Foundation** *(~14m)*: von Neumann -> SIMD -> SIMT; the real silicon (CPU cores vs GPU tiles); the
  programming model in one kernel; the blank **speeds-&-feeds scorecard** the rest of the hour fills in.
- **§1 -- Bandwidth** *(~5m)*: STREAM Triad on CPU *and* GPU -- GB/s is the chip's unit.
- **§2 -- Latency, the premise** *(~13m)*: the per-layer pointer-chase **staircase** (the GPU loses at *every* level),
  then **hide** it on one SM. (+ **§2b** warps vs registers · **§2c** blocks -> SMs · **§2.5** the cache cliff)
- **§3 -- A thread is a SIMD lane** *(~15m)*: sort a partition -- branchy merge vs `cub::StableOddEvenSort`; divergence,
  SASS predication (data vs control), and why a "terrible" O(N^2) sort is *perfect* here (complexity vs cycles).

**The 2-hour extension -- §4 onward** *(off the 1-hour clock)*:
- **§4** Coalescing · **§5** Registers / ILP · **§6** Cost model (radix vs merge) · **§7** the two-sort capstone ·
  **§8** "the kernel is ~10%" (the rug pull) · **§9** closing.

> One thread running below: **guess from the cost model, then confirm in cycles / SASS.** The source is a hint; the
> machine is the truth.
""")

code(r"""
# setup: live-CUDA cells (%%cuda) + plotting. nvcc/ncu on PATH; arch set once.
import os, subprocess, re
import matplotlib.pyplot as plt
ROOT = os.getcwd()
get_ipython().run_line_magic("load_ext", "nvcc4jupyter")
from nvcc4jupyter import set_defaults
set_defaults(compiler_args="-arch=sm_89 -O3 -std=c++17 -Xcompiler -fopenmp -Xcompiler -march=native")  # host code -> gcc: -fopenmp (one source runs CPU+GPU) + -march=native (real AVX, fair CPU)
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
fig("00_von_neumann.png", "The baseline machine: one program counter fetches one instruction at a time from a single memory, into scalar 32/64-bit registers. Everything the GPU does is a twist on this.")
fig("00_scalar_simd.png", "Scalar: one instruction -> one 32/64-bit register -> one result. SIMD: one instruction -> one wide register -> 16 results.")

md(r"""
### 0.1 The bet: throughput, not latency
A CPU core spends its area making *one* instruction stream fast (out-of-order, branch prediction, big caches).
A GPU rips that out and spends the area on **many ALUs** + **enough resident threads to switch among**.
Give up single-thread latency; buy aggregate throughput. Everything weird follows from this one bet.
""")
fig("01_area.png", "CPU spends area on control+cache for one fast thread; the GPU on a sea of ALUs + a huge register file.")

md(r"""
**The same bet, in real silicon -- three dies, one story.** A single **CPU core** is mostly *control + cache* (branch
prediction, decode, scheduler, microcode) wrapped around a *tiny* ALU -- almost all the area exists to keep **one**
instruction stream fast. Scale the CPU up (an **EPYC**) and you get a *few dozen* of those fat cores -- chiplets
(**CCDs**, each a core-complex / **CCX** with its own L3) around a central I/O die: still big cores, still
cache-heavy. The **GPU** scales the *opposite* way -- thousands of *tiny* lanes as a uniform field of identical SM
tiles, almost no per-lane control. **CPU = replicate a few fat cores; GPU = tile thousands of tiny ones.**
""")
md('<table style="width:100%"><tr>'
   '<td style="width:50%;vertical-align:top"><b>1. One CPU core</b><br>'
   '<img src="slides/figures/die_cpu_annotated.png" style="width:100%"><br>'
   '<small>Branch prediction, decode, scheduler, microcode, caches -- the **control + cache** machinery dwarfs the '
   'lone **Int ALU**. Most of a core exists to make *one* stream fast. <i>(annotated die, educational)</i></small></td>'
   '<td style="width:50%;vertical-align:top"><b>2. A whole CPU -- AMD EPYC</b><br>'
   '<img src="slides/figures/epyc.png" style="width:100%"><br>'
   '<small>The same fat core, replicated a *few dozen* times: chiplets (**CCDs** -- each a core-complex/**CCX** with '
   'shared L3) ringing a central **I/O die**. Big cores + lots of cache. <i>(image: TechPowerUp)</i></small></td>'
   '</tr></table>')
md(r"""
**3. A whole GPU -- the *same* picture, at two scales.** Here is the GPU, *labelled*. Both Ada **AD102** (this
course's chip, GDDR6X) and Hopper **GH100** (the datacenter part, HBM) are the identical idea: a **uniform grid of
GPC -> TPC -> SM tiles**, a **big central L2**, and **memory controllers at the edges** -- no branch predictors, no
per-core control empire, just the SM tile stamped out and fed by memory. The CPU replicates a few fat cores; the GPU
tiles *thousands* of tiny lanes -- and it looks the same whether it's a gaming Ada or a datacenter Hopper.
""")
md('<img src="slides/figures/ad102_gh100.png" style="width:100%"><br>'
   '<small>AD102 (608 mm^2, 76B transistors, GDDR6X) vs GH100 (814 mm^2, 80B, HBM): annotated GPC / TPC / SM tiles, '
   'the L2, and the memory PHYs around the rim. Renderings from NVIDIA, compiled/annotated by Locuza (not to relative '
   'scale). Labelled block diagram: '
   '<a href="https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf">Ada whitepaper</a>.</small>')

md(r"""
### 0.2 SIMT -- the GPU's *different* answer, and the words that mislead
The GPU does **not** use §0.0's SIMD (one thread, one wide register). Instead, **32 lanes share one
fetch/decode/scheduler** (a **warp**) and run the same instruction in lockstep, each on its *own* scalar
registers. That is **SIMD execution with a per-lane register file** -- the width is *across threads*, not a
wide register. (So you write plain scalar per-thread code and the hardware gangs 32.) Pin the vocabulary once,
or everything later is noise:

| you'll hear | it really is | CPU analogy | in CUDA code |
|---|---|---|---|
| "CUDA core" (18,176!) | an **ALU / lane** | one SIMD lane | you write it as a **`thread`** (a lane *is* a thread) |
| thread | one lane's scalar work | a lane of a vector op | `threadIdx`, the kernel body |
| warp | **32 lanes** sharing an instruction | one SIMD instruction | `warpSize`, `__shfl_sync`, `__ballot_sync` |
| **SM** | the real processor (~142) | **the core** | a **`block`** runs on one (`blockIdx`); `prop.multiProcessorCount` |
| register | per-lane; partitioned by occupancy | register file | a plain local var (`int x;`) vs `__shared__` |

One sentence: **SM = the core; warp = a SIMD instruction; "CUDA core" = a lane; thread = one lane's work.**
""")
fig("02_simt_vs_avx.png", "AVX: one thread, one wide register. SIMT: 32 threads, one shared instruction, per-lane scalar registers.")

md(r"""
### 0.3 The same loop, four ways: scalar -> ILP -> SIMD -> SIMT
Take the simplest data-parallel kernel, SAXPY (`for i: y[i] = a*x[i] + y[i]`). How do you make it fast?
The classic architecture progression -- ending exactly where the GPU lives.

**1. Scalar.** One element per instruction (§0.0). Each `x[i]` load is hundreds of cycles; the next iteration waits.
""")
md(r"""
**2. ILP (CPU, `-funroll-loops`).** Unroll so several *independent* loads issue back-to-back; the pipeline overlaps
their latencies -- until one **misses cache** and the consumer stalls anyway. ILP overlaps work *within one thread*,
but a single thread runs out of independent work to cover a long miss.
""")
fig("03a_cpu_pipeline.png", "Unroll overlaps independent loads; a cache miss still bubbles the consumer -- one thread can't hide it.")
md(r"""
**3. SIMD (CPU, AVX, `-O3 -march=native`).** One wide instruction does 16 lanes at once (§0.0) -- but it reads **one
contiguous block from one base address**. Per-element / data-dependent addressing (`x[idx[i]]`, a *gather*) is not
natural for SIMD; it needs a slow gather instruction. SIMD buys data *width*, not addressing *freedom*.
""")
fig("03b_simd_gather.png", "SIMD: one base address, contiguous. Scattered per-element addresses need a slow gather.")
md(r"""
**4. SIMT (GPU).** Each "thread" is the scalar loop body with its **own index and its own address** -- so the gather
SIMD couldn't do is *native*. And you launch *thousands*: when one warp stalls on the miss, the scheduler **switches
to another ready warp** -- a **"manual pipeline"** you build by exposing parallelism, not one a compiler unrolls.
That is how the GPU hides the miss ILP couldn't -- not more in-flight work *per thread*, but **more threads**.
""")
fig("05_latency_hiding.png", "The same cache miss as beat 2 -- now HIDDEN: when one warp stalls, the scheduler runs another, so the SM is never idle. The warp pipeline ILP could not build alone.")
md(r"""
Beat 2's pipeline stalled on the miss; beat 4's *warp* pipeline hides it -- same idea, but the overlap is across
**threads**, not within one. (Two follow-ons we'll *earn* later, not assume here: those per-lane addresses are only
*fast* when contiguous -- that's **coalescing**, §4; and ILP returns as a knob you dial per thread -- **register
blocking**, §5. We prove the warp-switching itself on *one* SM in §2.)
""")

md(r"""
### 0.4 The programming model, in one kernel
**That SIMT thread (§0.3, #4) in real CUDA.** Map every token to the hardware. A kernel is **scalar per-thread code**; the launch
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
**Quick quiz -- nail the terms (each answer corrects the common wrong intuition).**
Say that kernel launched `<<<4096, 256>>>` -- 4096 blocks x 256 threads:

**Q1. Thread `i = 1000` -- which "core" (SM) runs it?**
*Unknowable.* Threads come in **blocks** (`i=1000` -> block `1000/256 = 3`); the scheduler maps whole **blocks to SMs
dynamically at runtime**, and you never know or rely on which. (And "core" isn't the SM anyway -- a "CUDA core" is a
*lane*.) **There is no fixed thread -> SM map.**

**Q2. Threads `i = 100` and `i = 110` -- same warp, just different lanes?**
*Yes.* Same block, and `100/32 == 110/32 == 3` -> **warp 3**, running in **lockstep** as lanes 4 and 14.
(But `i = 100` vs `i = 200`? warps 3 and 6 -- **not** lanes of each other; they interleave, possibly different cycles.)

**Q3. Do all 256 threads of a block run at the same instant?**
*No.* They are **resident on one SM**, but its 4 schedulers issue ~4 warps/clock and the 8 warps **interleave**.
Only threads **in the same warp** are truly simultaneous. "Block = parallel" is wrong; **"warp = lockstep"** is right.

**Q4. Can thread 5 read thread 70's register?**
*Only by scope.* Same warp -> `__shfl` (registers are otherwise private). Different warps, same block -> through
**`__shared__`** + `__syncthreads`. Different block -> only **global memory** + a fence. Sharing cost grows with distance.

**Q5. Threads `i` and `i+1` read `a[i]` and `a[i+1]` -- one memory transaction, or two?**
*One.* 32 consecutive threads of a warp -> 32 contiguous addresses -> **one coalesced 128-byte transaction** (the §4 game).

If those five feel obvious now, the vocabulary is locked. Everything past here is making this model **fast**.
""")

md(r"""
### 0.5 The scorecard: a memory system is two numbers per layer
Before any optimization, here is the **whole cost model on one slide** -- and it is almost entirely blank. A memory
level is defined by exactly two numbers: **bandwidth** (bytes/s, for streaming work) and **latency** (time per
*dependent* access, for pointer-chasing work). The rest of the hour is a measuring exercise: we fill in every `?`
**live, on this card**, and never assert a number we can measure. §1 fills DRAM **bandwidth** (CPU vs GPU), §2 fills
DRAM **latency** (CPU vs GPU) and shows how the GPU *hides* it, and a working-set **size sweep** fills the on-chip
rows (L2, shared) -- the same sweep that exposes the **cache cliff**. Keep this table in your head; everything below
is one cell of it.
""")
fig("00_speeds_feeds_blank.png", "The cost model as a scorecard: bandwidth + latency per memory layer. Every '?' is measured live -- DRAM in §1/§2, the on-chip layers by the size sweep.")

# ============================================================================
# 1. BANDWIDTH  (first mechanism -- the canonical beat)
# ============================================================================
H(1, "Bandwidth is the chip's identity", 5)
card("STREAM Triad on CPU *and* GPU", "three ~1 GB arrays", "achieved GB/s, both chips",
     "the SAME op (the saxpy of §0.3) on both, so the only variable is the bus -- apples to apples.")
concept(r"""
The chip's native unit of work is **moving bytes**. To compare the two fairly we run the *same* kernel on both:
the **STREAM Triad** `a[i] = b[i] + s*c[i]` -- 2 reads + 1 write, `3*N` bytes (the classic memory-bandwidth
benchmark, and exactly the saxpy-family op from §0.3). No copy-vs-triad mismatch: identical work, identical byte
accounting. The number each chip hits is the yardstick every later demo is measured against.
""")
guess("the SAME STREAM Triad on both chips -- how many times the bandwidth does the GPU get?",
      "CPU STREAM Triad (whole socket, AVX)", "a[i] = b[i] + s*c[i]   // ~?? GB/s",
      "GPU STREAM Triad (sea of SMs)", "a[i] = b[i] + s*c[i]   // ~?? GB/s")
cuda(r"""
#include <cstdio>
#include <cstdlib>
#include <omp.h>
// ONE source, ONE compiler: nvcc compiles the __global__ for the GPU and hands the
// OpenMP host code to gcc (-Xcompiler -fopenmp). The triad body is byte-identical.

// ---- GPU: a[i] = b[i] + s*c[i], float4-vectorized, one thread per element ----
__global__ void triad_kernel(float4* __restrict__ a, const float4* __restrict__ b,
                             const float4* __restrict__ c, float s, size_t n4){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x, stride = (size_t)gridDim.x*blockDim.x;
  for (; i < n4; i += stride){ float4 bb=b[i], cc=c[i];      // 128-bit coalesced loads
    a[i] = make_float4(bb.x+s*cc.x, bb.y+s*cc.y, bb.z+s*cc.z, bb.w+s*cc.w); }
}

// Measure the GPU triad -> GB/s (1 timed launch over the whole device).
double gpu_triad(size_t n, float s, double GB){
  float *a,*b,*c, bytes = n*sizeof(float); cudaMalloc(&a,n*4); cudaMalloc(&b,n*4); cudaMalloc(&c,n*4);
  cudaMemset(b,1,n*4); cudaMemset(c,1,n*4); size_t n4 = n/4; (void)bytes;
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  triad_kernel<<<32*142,256>>>((float4*)a,(float4*)b,(float4*)c,s,n4);                  // warm up
  cudaEventRecord(e0); triad_kernel<<<32*142,256>>>((float4*)a,(float4*)b,(float4*)c,s,n4); cudaEventRecord(e1);
  cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  cudaFree(a); cudaFree(b); cudaFree(c); return GB / (ms/1e3);
}

// Measure the SAME triad on the whole CPU socket -> GB/s (best of 10 OpenMP runs).
double cpu_triad(size_t n, float s, double GB, int* threads){
  float *a=(float*)malloc(n*4), *b=(float*)malloc(n*4), *c=(float*)malloc(n*4);
  #pragma omp parallel for schedule(static)
  for (size_t i=0;i<n;i++){ a[i]=0; b[i]=1; c[i]=1; }          // first-touch: pages land near their thread
  #pragma omp parallel
  { if (omp_get_thread_num()==0) *threads = omp_get_num_threads(); }
  double best = 1e30;
  for (int it=0; it<10; it++){ double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (size_t i=0;i<n;i++) a[i] = b[i] + s*c[i];             // gcc -O3 -march=native auto-vectorizes -> AVX2 vfmadd (ymm)
    double dt = omp_get_wtime()-t0; if (dt<best) best=dt; }
  free(a); free(b); free(c); return GB / best;
}

int main(){
  size_t n = (size_t)1<<28; float s = 3.0f;                   // 1 GB/array x3
  double GB = 3.0*(double)n*sizeof(float)/1e9;                 // 2 reads + 1 write
  int th=0;
  double cpu = cpu_triad(n, s, GB, &th);
  double gpu = gpu_triad(n, s, GB);
  printf("CPU triad (%2d threads): %6.0f GB/s\n", th, cpu);
  printf("GPU triad             : %6.0f GB/s\n", gpu);
  printf("--------------------------------------\nGPU is %.1fx the CPU's memory bandwidth\n", gpu/cpu);
}
""")
costmodel(r"""
The Triad moves `3*N` bytes once; time = bytes / bandwidth. There is no algorithm here -- just the bus, and the
GPU's is ~**10x** wider than the *whole* CPU socket (24 AVX2 threads). Note the CPU loop *is* SIMD -- gcc emits
`vfmadd...ymm` -- yet **AVX2 (32 B) vs SSE2 (16 B) barely changes the number**: a streaming triad is DRAM-bound,
so vector width is irrelevant. That is the cost model in miniature -- **bytes moved, not flops, set the time.**
Everything later is one question: **how close to this number can a *real* computation get?**
""")

# ============================================================================
# 2. LATENCY -- THE PREMISE: a single GPU access is far SLOWER than a CPU's
# ============================================================================
H(2, "Latency: the GPU's is far higher -- this is the whole premise", 13)
card("a dependent pointer-chase, swept by working-set size", "L1 -> L2 -> (L3) -> DRAM, CPU and GPU",
     "ns (and GPU cycles) per single access",
     "establish the premise: per-access latency is BRUTAL on the GPU. Everything later is how you survive it.")
concept(r"""
**Why does all the latency-hiding code below make sense?** Because a *single* memory access on a GPU is far slower
than on a CPU -- and the GPU is fast anyway, only because it **hides** that latency under thousands of threads.
We measure the cost honestly first, the canonical way (lmbench `lat_mem_rd` / GPU "P-chase"): a **dependent pointer
chase** -- `idx = next[idx]`, each address from the previous load -- over a **randomized ring** that defeats the
prefetcher, swept across working-set size so each cache level shows up as a **plateau**. On the GPU we count
**cycles** with `clock64()` (DVFS-proof), then convert with the boost clock. First the CPU staircase:
""")
concept(r"""
**Why pointer chasing specifically?** Three reasons. **(1) It isolates pure latency.** Each load's address comes from
the previous load, so the hardware *cannot* overlap them -- no prefetch, no MLP, no parallelism leaks in. You measure
the naked round-trip to each level, and the cache hierarchy draws itself as plateaus. **(2) It is not a toy.** It is
the inner loop of huge real workloads: graph traversal (BFS, PageRank), **hash-join probes**, B-tree / index lookups,
linked lists -- "follow a reference, then follow the next." For DB folks this *is* the probe side of a join.
**(3) It is the pattern GPU programmers fear most:** random + dependent = **uncoalesced** (a whole 32-byte sector
fetched to use 4 bytes) *and* **latency-bound** -- the textbook worst case for a throughput machine.
""")
cuda(r"""
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cstddef>
// lmbench lat_mem_rd-style: randomized dependent chase (no prefetch), swept by size -> latency staircase.
// (Pure host code -- nvcc hands it to gcc; it runs on the CPU.)
static double chase_ns(size_t bytes){
  size_t n = bytes/sizeof(int); if (n<2) n=2;
  std::vector<int> perm(n); for(size_t i=0;i<n;i++) perm[i]=(int)i;
  std::mt19937 rng(1); for(size_t i=n-1;i>0;i--) std::swap(perm[i],perm[rng()%(i+1)]);
  std::vector<int> next(n); for(size_t k=0;k<n;k++) next[perm[k]]=perm[(k+1)%n];
  int idx=0; for(size_t i=0;i<n;i++) idx=next[idx];                  // warm this level
  size_t steps=20000000;
  auto t0=std::chrono::high_resolution_clock::now();
  for(size_t i=0;i<steps;i++) idx=next[idx];                        // serialized dependent loads
  auto t1=std::chrono::high_resolution_clock::now();
  volatile int s=idx; (void)s;
  return std::chrono::duration<double,std::nano>(t1-t0).count()/steps;
}
int main(){
  printf("CPU: one dependent load, working set grown L1 -> DRAM (randomized, prefetch defeated):\n");
  printf("     size        ns/access\n");
  size_t KB=1024, sz[]={16*KB,64*KB,256*KB,1024*KB,4096*KB,16384*KB,65536*KB,262144*KB,1048576*KB};
  for(int i=0;i<9;i++) printf("   %7zu KB    %7.2f\n", sz[i]/KB, chase_ns(sz[i]));
  printf("\n-> plateaus: L1 ~1 ns, L2 ~3 ns, L3 ~10-15 ns, DRAM ~100 ns (this box: 32 KB L1, 512 KB L2/core, 64 MB L3).\n");
}
""")
concept("Now the **same** chase on the GPU, timed in cycles with `clock64()`:")
cuda(r"""
#include <cstdio>
#include <vector>
#include <numeric>
#include <random>
__global__ void chase_cyc(const int* __restrict__ next, int steps, int* sink, long long* cyc){
  int idx=0;
  for(int i=0;i<4096;i++) idx=next[idx];               // warm the level
  long long t0=clock64();
  for(int i=0;i<steps;i++) idx=next[idx];              // serialized dependent loads (cannot overlap)
  long long t1=clock64();
  *sink=idx; *cyc=t1-t0;
}
double lat_cyc(size_t bytes){
  size_t n=bytes/sizeof(int); if(n<2)n=2;
  std::vector<int> perm(n); std::iota(perm.begin(),perm.end(),0);
  std::mt19937 rng(1); for(size_t i=n-1;i>0;i--) std::swap(perm[i],perm[rng()%(i+1)]);
  std::vector<int> nxt(n); for(size_t k=0;k<n;k++) nxt[perm[k]]=perm[(k+1)%n];
  int *d,*sink; long long* cyc; cudaMalloc(&d,n*4); cudaMalloc(&sink,4); cudaMalloc(&cyc,8);
  cudaMemcpy(d,nxt.data(),n*4,cudaMemcpyHostToDevice);
  chase_cyc<<<1,1>>>(d,3000000,sink,cyc); cudaDeviceSynchronize();
  long long c=0; cudaMemcpy(&c,cyc,8,cudaMemcpyDeviceToHost);
  cudaFree(d); cudaFree(sink); cudaFree(cyc); return (double)c/3000000;
}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  int clk=0; cudaDeviceGetAttribute(&clk,cudaDevAttrClockRate,0); double ghz=clk/1e6;
  printf("GPU: one dependent load, clock64 cycles (DVFS-proof), boost %.2f GHz, L1=128KB/SM, L2=%.0f MB:\n",
         ghz, p.l2CacheSize/1e6);
  printf("     size       cycles      ns      level\n");
  size_t KB=1024, sz[]={32*KB,64*KB,256*KB,2048*KB,8192*KB,32768*KB,131072*KB,524288*KB,1048576*KB};
  for(int i=0;i<9;i++){ double c=lat_cyc(sz[i]);
    const char* lv = (sz[i]<=128*KB) ? "L1" : ((sz[i] < (size_t)p.l2CacheSize) ? "L2" : "DRAM");
    printf("   %7zu KB   %6.0f   %6.1f     %s\n", sz[i]/KB, c, c/ghz, lv); }
  printf("\n-> GPU L1 ~40 cyc (~16 ns), L2 ~280 cyc (~110 ns), DRAM ~630 cyc (~250 ns).\n");
}
""")
costmodel(r"""
Line the two staircases up -- the GPU loses at **every** level:

| level | CPU | GPU | GPU penalty |
|---|---|---|---|
| L1 | ~1.3 ns | ~16 ns (40 cyc) | **~12x** |
| L2 | ~3 ns | ~110 ns (280 cyc) | **~37x** |
| L3 | ~10-15 ns | *(none -- GPU has no L3)* | -- |
| DRAM | ~105 ns | ~250 ns (630 cyc) | **~2.4x** |

A single GPU thread doing dependent loads is **brutally slow** -- a GPU access never beats a CPU access. (And the
huge Ada L2 buys capacity at the cost of latency: ~110 ns is *higher* than older GPUs.) **So how is a GPU ever
fast?** Not by lowering latency -- by **hiding** it: keep so many memory requests in flight that no single one's
wait is ever on the critical path. That single idea -- `in_flight = throughput x latency` (Little's law again) --
is the premise behind every demo below. The next cell *proves* it on one SM.
""")
md(r"""
**Reality check -- is pointer chasing a death sentence on a GPU? No -- *if you have enough of it.*** A single chain is
brutal (above). But real irregular workloads have *millions* of independent chains -- every vertex in a graph, every
probe in a hash join -- and the GPU overlaps each chain's latency behind all the others (Little's law again). The
evidence: GPU graph frameworks like **Gunrock** run BFS/PageRank/SSSP **~an order of magnitude faster than CPU**
frameworks *despite* fully irregular, data-dependent access ([Wang et al., *Gunrock*, ACM TOPC 2017](https://arxiv.org/pdf/1701.01170));
and GPU **hash joins** are competitive-to-winning -- random probes are still the acknowledged bottleneck, actively
optimized (e.g. ~2.3x from cutting random accesses), yet multi-GPU joins beat parallel-CPU joins by up to ~15x
([survey, 2024](https://arxiv.org/pdf/2406.13831); [joins on GPUs, 2023](https://arxiv.org/html/2312.00720v2)).
The catch matches our staircase: it only works with **massive parallelism**, it leans on the **big Ada L2** to keep
hits at ~110 ns instead of the ~250 ns DRAM trip, and uncoalesced access still wastes bandwidth. So irregular access
on a GPU is **survivable, not free** -- the thing you design *around*, not the thing that disqualifies the GPU.
""")
md(r"""
### 2.1 The resolution: hide it (prove it on ONE SM)
The premise says one access is slow. The fix is concurrency -- and we prove it can't be "more cores" by pinning to
**one SM** (one block) and adding *only warps*. If throughput climbs, that climb **is** latency hidden.
""")
fig("05_latency_hiding.png", "Stalled warp -> switch to a ready one; with enough warps the SM is never idle.")
guess("one block (one SM), 1 -> 32 warps, only warps added. Flat (just 4 schedulers!), or does throughput climb? by how much?",
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
  int N = 1<<28; std::vector<int> perm(N); std::iota(perm.begin(),perm.end(),0);  // 1 GB ring -> DRAM (the ~250 ns access from above)
  std::mt19937 rng(1); for(int i=N-1;i>0;i--) std::swap(perm[i],perm[rng()%(i+1)]);
  std::vector<int> nxt(N); for(int k=0;k<N;k++) nxt[perm[k]]=perm[(k+1)%N];

  // ---- GPU: ONE SM, only warps added (each access is the ~250 ns DRAM latency from the staircase) ----
  int *d,*sink; cudaMalloc(&d,(size_t)N*4); cudaMalloc(&sink,1024*4); cudaMemcpy(d,nxt.data(),(size_t)N*4,cudaMemcpyHostToDevice);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  printf("GPU  ONE SM (1 block) -- only warps added, 4 schedulers:\n  warps   Maccess/s   speedup\n");
  double base=0;
  for (int w=1; w<=32; w*=2){
    int steps=20000, threads=32*w;
    chase<<<1,threads>>>(d,N,steps,sink); cudaDeviceSynchronize();
    cudaEventRecord(a); chase<<<1,threads>>>(d,N,steps,sink); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    double acc=(double)threads*steps/(ms/1e3)/1e6; if(w==1) base=acc;
    printf("  %4d   %9.1f    %.1fx\n", w, acc, acc/base);
  }
  // Why does it roll off at 16->32 and stop at 32? Measure it -- it is NOT registers.
  cudaFuncAttributes fa; cudaFuncGetAttributes(&fa,chase);
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("\nwhy 32 is the ceiling here (NOT register/occupancy capped):\n");
  printf("  kernel = %d regs/thread -> registers alone allow %d warps/SM; the HW cap is %d warps/SM.\n",
         fa.numRegs, p.regsPerMultiprocessor/(fa.numRegs*32), p.maxThreadsPerMultiProcessor/32);
  printf("  but ONE block maxes at %d threads = 32 warps -> THAT caps this single-block sweep, with regs to spare.\n",
         p.maxThreadsPerBlock);
  printf("  the per-doubling gain shrinks (2x -> ~1.5x): we are climbing the latency-hiding curve -- each warp adds\n");
  printf("  in-flight loads (Little's law), with diminishing return as the SM's memory pipe fills. To go past 32\n");
  printf("  warps you need MORE BLOCKS (more SMs) -- the full-GPU story -- not more registers.\n");
}
""")
costmodel(r"""
One chain is latency-bound everywhere -- the CPU thread sits at the ~100 ns DRAM wall (prefetch dead on a random
chase). The GPU does **not** make the chain faster; it overlaps *many* of them. Each warp doubling is ~2x while
deeply latency-bound (1->2->4->8), then the gain **shrinks (~1.5x by 16->32)**: by Little's law each warp adds
in-flight loads, and the marginal warp helps less as the SM's memory pipe fills. Crucially -- and the cell *proves*
it -- this roll-off is **not** "out of warps/registers": at 22 regs/thread the registers allow ~90 warps and the SM
holds 48; we stop at 32 only because **one block maxes at 1024 threads**. To push further you add **more blocks
(more SMs)**, not registers. So the ~18x here is pure latency-hiding on a *single* SM. **"18,176 cores" is not
18,176 CPUs;** it is lanes kept busy by resident warps. (When a *bigger* kernel's registers really do cap resident
warps -- the occupancy lever -- that's next.) Concurrency, not core count.
""")

# ============================================================================
# 2b. GOING DEEPER: TWO WAYS TO HIDE LATENCY (TLP vs ILP) -- and the register cap
# ============================================================================
H("2b", "Going deeper: warps vs registers -- two ways to hide the same latency", 6)
card("demo7_latency/twocurve.cu", "same chase, 1 vs 64 chains per thread", "throughput vs warps, two curves",
     "above, more warps hid the latency. Here: prove registers can hide it too -- and that the register file then caps occupancy.")
concept(r"""
Latency hiding needs **enough loads in flight** (Little's law): `in_flight = warps x (loads per thread)`.
*(If you size connection pools or set `effective_io_concurrency` / disk queue depth, this is the **same** law you
already use -- outstanding requests = throughput x latency. The GPU is just an async-I/O machine, and a warp is one
outstanding memory request. You are sizing its "pool".)* §2 fed that with **warps** (TLP), one load per thread.
But you can also raise the *per-thread* term -- give each thread **R independent chains** so it issues R loads before
waiting (**ILP / register blocking**). Each live chain costs **registers**, and the SM's register file is finite
(65,536 / SM), so:
```cpp
template<int R> __global__ void chase_blk(const int* next, ...){
  int idx[R];                                   // R live indices -> ~R+ registers/thread
  for (i<steps) for (r<R) idx[r] = next[idx[r]];// R loads in flight per thread (MLP = R)
}
```
- **THIN R=1** (22 regs): MLP 1 -> needs **many warps** to fill the pipe; registers no constraint (runs all 32).
- **FAT  R=64** (~80 regs): MLP 64 -> fills the pipe with a **handful of warps**, but 80 regs x 1024 threads >
  64K register file, so the launch **fails** past ~25 warps -- **register-capped occupancy.**
Same destination (the SM's memory-pipe ceiling), two routes -- and a hard register wall on one of them.
""")
guess("the fat kernel runs 64 chains/thread. At 1 warp does it beat the thin kernel? And how many warps can it reach before the register file runs out?",
      "THIN (R=1)", "needs ?? warps to saturate; max 32 warps/block",
      "FAT (R=64)", "saturates by ?? warps; launch fails at ?? warps")
cuda(r"""
#include <cstdio>
#include <vector>
#include <numeric>
#include <random>
// Same chase, two register footprints. R independent chains/thread = R loads in flight (MLP=R).
template<int R>
__global__ void chase_blk(const int* __restrict__ next, int N, int steps, int* sink){
  int t = blockIdx.x*blockDim.x + threadIdx.x;
  int idx[R];
  #pragma unroll
  for (int r=0;r<R;r++) idx[r] = (int)(((long long)t*R + r) % N);
  for (int i=0;i<steps;i++)
    #pragma unroll
    for (int r=0;r<R;r++) idx[r] = next[idx[r]];        // R independent dependent-loads per thread
  int s=0;
  #pragma unroll
  for (int r=0;r<R;r++) s ^= idx[r];
  sink[t & 1023] = s;
}
// one timed launch of <R> at `warps` warps; returns Maccess/s, or -1 if the launch is rejected.
// *err receives the exact CUDA error so we can SHOW it's the register cap, not guess.
template<int R>
double run(int warps, int* d, int N, int* sink, cudaError_t* err){
  int threads = 32*warps; long target = 300000000L;
  int steps = (int)(target/((long)threads*R)); if (steps<1) steps=1;
  chase_blk<R><<<1,threads>>>(d,N,steps,sink);
  *err = cudaGetLastError();
  if (*err != cudaSuccess) return -1;                  // launch rejected -- caller prints the reason
  cudaDeviceSynchronize();
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a); chase_blk<R><<<1,threads>>>(d,N,steps,sink); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms=0; cudaEventElapsedTime(&ms,a,b);
  return (double)threads*R*steps/(ms/1e3)/1e6;
}
int main(){
  int N = 1<<26; std::vector<int> perm(N); std::iota(perm.begin(),perm.end(),0);   // 256 MB ring -> DRAM
  std::mt19937 rng(1); for(int i=N-1;i>0;i--) std::swap(perm[i],perm[rng()%(i+1)]);
  std::vector<int> nxt(N); for(int k=0;k<N;k++) nxt[perm[k]]=perm[(k+1)%N];
  int *d,*sink; cudaMalloc(&d,(size_t)N*4); cudaMalloc(&sink,1024*4); cudaMemcpy(d,nxt.data(),(size_t)N*4,cudaMemcpyHostToDevice);

  // --- registers computed LIVE, on this card ---
  cudaFuncAttributes ta, fa;
  cudaFuncGetAttributes(&ta, chase_blk<1>);
  cudaFuncGetAttributes(&fa, chase_blk<64>);
  int rf = 65536;                                       // registers per SM (sm_89)
  printf("registers measured live (cudaFuncGetAttributes), register file = %d/SM:\n", rf);
  printf("  THIN R=1  : %3d regs/thread  -> at most %2d warps fit in one block\n", ta.numRegs, (rf/ta.numRegs)/32);
  printf("  FAT  R=64 : %3d regs/thread  -> at most %2d warps fit in one block  <-- register-capped\n",
         fa.numRegs, (rf/fa.numRegs)/32);

  printf("\n warps   THIN Macc/s   FAT Macc/s\n");
  int ws[] = {1,2,4,8,12,16,20,24,28,32};
  cudaError_t te, fe;
  for (int k=0;k<10;k++){ int w = ws[k];
    double tt = run<1>(w,d,N,sink,&te), ff = run<64>(w,d,N,sink,&fe);
    printf("  %4d   %10.0f   ", w, tt);
    if (ff < 0) printf("%10s   <-- %s: \"%s\"\n", "CAP", cudaGetErrorName(fe), cudaGetErrorString(fe));
    else        printf("%10.0f\n", ff);
  }
  printf("\nboth approach the SAME peak (the SM's memory pipe): THIN climbs there via many warps (TLP);\n");
  printf("FAT is already near it at 1 warp (64 loads/thread, ILP) but CANNOT run high warp counts --\n");
  printf("%d regs/thread x 1024 threads > %d register file. in_flight = warps x loads/thread; registers are the budget.\n",
         fa.numRegs, rf);
}
""")
costmodel(r"""
The two curves reach the **same peak** -- the SM's memory pipe -- by opposite routes. **TLP** (thin) buys in-flight
loads with *warps*; **ILP / register blocking** (fat) buys them with *registers* (R loads per thread), saturating
with far fewer warps. They are **substitutes for the same goal**: `in_flight = warps x loads_per_thread`. But the
register file is the budget: 80 regs/thread literally **caps occupancy** -- the fat launch *fails* past ~25 warps.
So occupancy isn't free virtue and registers aren't free either; each kernel sits somewhere on this trade. (This is
the lever §5 turns on a real kernel: spend registers for ILP until occupancy falls too far to hide what's left.)
""")

# ============================================================================
# 2c. BLOCKS -> SMs: the second launch axis, escaping the per-SM register cap
# ============================================================================
H("2c", "Blocks -> SMs: the second axis, and escaping the per-SM cap", 6)
card("the FAT kernel as a grid", "1 block (1 SM, capped) vs many blocks (many SMs)",
     "throughput vs block count", "the register cap was PER-SM -- add SMs and it vanishes.")
concept(r"""
The launch `<<<grid, block>>>` is **two numbers, two axes** -- and that's the whole model (no 3D `dim3` "cube" needed):
- **`block`** = threads per block = the **within-one-SM** axis. A whole block runs on **one SM** -- that's what §2.1
  pinned to study a single SM, and what §2b found capped by the register file (~25 warps for the FAT kernel).
- **`grid`** = number of blocks = the **across-SMs** axis. Distinct blocks are dispatched to **distinct SMs** -- the
  ~142 real cores.

So §2b's cap was a **per-SM** limit, not a global one. Keep each block small enough to fit registers (here 8 warps,
`80 regs x 256 thr < 64K`), then launch **many** blocks: they fan out across all 142 SMs and throughput scales with
SM count -- the cap is gone. Same FAT kernel, same per-thread registers; only the *grid* changes.
""")
guess("the FAT kernel capped at ~25 warps in ONE block (~2700 Macc/s on one SM). Launch it as a GRID of small blocks across many SMs -- still capped, or does it scale with SM count?",
      "1 block -> 1 SM", "chase<<<1, 256>>>    // ~2700 Macc/s",
      "32 blocks -> 32 SMs", "chase<<<32, 256>>>   // ?")
cuda(r"""
#include <cstdio>
#include <vector>
#include <numeric>
#include <random>
// the FAT kernel from 2b (R=64 chains/thread, register-heavy). In ONE block it capped at ~25 warps.
template<int R>
__global__ void chase_blk(const int* __restrict__ next,int N,int steps,int* sink){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  int idx[R];
  #pragma unroll
  for(int r=0;r<R;r++) idx[r]=(int)(((long long)t*R+r)%N);
  for(int i=0;i<steps;i++)
    #pragma unroll
    for(int r=0;r<R;r++) idx[r]=next[idx[r]];
  int s=0;
  #pragma unroll
  for(int r=0;r<R;r++) s^=idx[r];
  sink[t&1023]=s;
}
double run(int blocks,int* d,int N,int* sink){
  int threads=256;                                  // 8 warps/block -> fits registers, NO per-block cap
  long target=400000000L; int steps=(int)(target/((long)blocks*threads*64)); if(steps<1)steps=1;
  chase_blk<64><<<blocks,threads>>>(d,N,steps,sink); cudaDeviceSynchronize();
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a); chase_blk<64><<<blocks,threads>>>(d,N,steps,sink); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms=0; cudaEventElapsedTime(&ms,a,b);
  return (double)blocks*threads*64*steps/(ms/1e3)/1e6;
}
void sweep(const char* tag,int N){
  std::vector<int> perm(N); std::iota(perm.begin(),perm.end(),0);
  std::mt19937 rng(1); for(int i=N-1;i>0;i--) std::swap(perm[i],perm[rng()%(i+1)]);
  std::vector<int> nxt(N); for(int k=0;k<N;k++) nxt[perm[k]]=perm[(k+1)%N];
  int *d,*sink; cudaMalloc(&d,(size_t)N*4); cudaMalloc(&sink,1024*4); cudaMemcpy(d,nxt.data(),(size_t)N*4,cudaMemcpyHostToDevice);
  printf("\n%-18s blocks(->SMs)   Macc/s    vs 1 block\n", tag);
  double base=0; int bs[]={1,2,8,32,71,142,284};
  for(int k=0;k<7;k++){ double m=run(bs[k],d,N,sink); if(k==0)base=m;
    printf("   %18d   %9.0f    %5.1fx\n", bs[k], m, m/base); }
  cudaFree(d); cudaFree(sink);
}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("this card has %d SMs (the real cores). one block = one SM; the grid spreads blocks across them.\n",
         p.multiProcessorCount);
  sweep("L2-resident 32 MB:", 1<<23);
  sweep("DRAM 256 MB:", 1<<26);
  printf("\n2b's register cap was PER-SM; adding blocks uses more SMs and escapes it. But you scale only to the\n");
  printf("SHARED ceiling: L2-resident work climbs ~linearly to all %d SMs; DRAM random access saturates early (~2.5's cliff).\n",
         p.multiProcessorCount);
}
""")
costmodel(r"""
Two axes, two stories. **Within a block** (one SM) you fight the register/occupancy cap (§2b). **Across blocks** you
just add SMs and that cap vanishes: the L2-resident chase scales **~linearly to all 142 SMs** (~38x over one SM).
But you only scale until you hit the **shared** resource -- random **DRAM** access saturates at ~4x (a handful of SMs
already exhaust DRAM's random-request throughput: the §2.5 cliff, felt from the other side). So "18,176 cores" really
is **142 SMs x 128 lanes**, and you scale a kernel by **filling SMs with blocks** -- right up to whichever shared
ceiling (L2 or DRAM) the access pattern lands on. That is the entire launch model: block = within an SM, grid = across SMs.

So there are **two kinds of oversubscription, one per axis**: *more warps than schedulers* (within an SM) **hides
latency**; *more blocks than SMs* (across the grid) **keeps every SM fed** -- the scheduler refills an SM the instant
a block retires. The `284`-blocks row is the second kind: ~2 waves over 142 SMs. (When the block count *isn't* a
clean multiple of 142, the last partial wave leaves SMs idle -- **wave quantization**, a real tail-effect cliff we
measure later in §5.)
""")

# ============================================================================
# 2.5 THE CACHE CLIFF  (one sweep fills the on-chip rows of the scorecard)
# ============================================================================
H("2.5", "The cache cliff: where data lives sets the speed", 5)
card("a streaming-bandwidth sweep, 8 MB -> 1 GB", "one re-read kernel, growing array", "bandwidth vs working-set size",
     "don't assert the L2/DRAM bandwidth -- grow the working set until the cliff appears, and read it off.")
concept(r"""
§2 filled the **latency** column of the scorecard (per layer, CPU vs GPU). This fills the **bandwidth** column for
the on-chip layers. Same trick -- **keep growing the working set**: while it fits in **L2 (~100 MB here)** a re-read
streaming kernel rides L2's bandwidth; once it spills, the *same code* falls off a **cliff** to DRAM. The size at
which it falls over **is** L2's capacity -- you are watching the cache boundary, not being told it.
""")
guess("grow the array from just-inside L2 (67 MB) to spilled (1 GB). What happens to streaming bandwidth?",
      "inside L2 (67 MB)", "BW ~ ?? TB/s   (on-chip)",
      "spilled to DRAM (1 GB)", "BW ~ ?? GB/s   (off-chip)")
cuda(r"""
#include <cstdio>
#include <algorithm>
// streaming bandwidth of wherever the data lives: re-read the array `reps` times so steady-state cache behavior wins.
__global__ void bw_kernel(const float* __restrict__ x, size_t n, int reps, float* out){
  size_t i = blockIdx.x*(size_t)blockDim.x+threadIdx.x, stride=(size_t)gridDim.x*blockDim.x;
  float acc=0.f;
  for(int r=0;r<reps;r++) for(size_t j=i;j<n;j+=stride) acc+=x[j];
  if(i<256) out[i]=acc;
}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  int grid=32*p.multiProcessorCount, blk=256;
  double L2 = p.l2CacheSize/1e6;
  printf("L2 = %.0f MB on this card. grow the working set until BW falls off the cliff:\n\n", L2);
  printf("  working set      BW\n     (MB)       (TB/s)\n");
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  double l2bw=0, drbw=0;
  for(int log2n=21; log2n<=28; log2n++){              // 8 MB .. 1 GB
    size_t n=(size_t)1<<log2n, bytes=n*sizeof(float); double mb=bytes/1e6;
    float *x,*out; cudaMalloc(&x,bytes); cudaMalloc(&out,256*4); cudaMemset(x,1,bytes);
    int reps=(int)std::max<size_t>(1,(size_t)8e9/bytes);
    bw_kernel<<<grid,blk>>>(x,n,reps,out); cudaDeviceSynchronize();
    cudaEventRecord(a); bw_kernel<<<grid,blk>>>(x,n,reps,out); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b); double tbs=(double)bytes*reps/1e9/(ms/1e3)/1000.0;
    printf("  %8.0f    %8.2f%s\n", mb, tbs, mb>L2 ? "   <-- spilled to DRAM" : "");
    if(mb<L2 && tbs>l2bw) l2bw=tbs;                   // best while L2-resident
    drbw=tbs;                                          // last row = DRAM
    cudaFree(x);cudaFree(out);
  }
  printf("\nSCORECARD bandwidth column filled (measured live, not asserted):\n");
  printf("  L2   : %5.1f TB/s\n", l2bw);
  printf("  DRAM : %5.2f TB/s   (read-only; cf. S1 triad ~820 GB/s)\n", drbw);
  printf("  -> L2 is ~%.0fx DRAM's streaming bandwidth. (latency for both rows: see the staircase in S2.)\n", l2bw/drbw);
}
""")
costmodel(r"""
The cliff is the cache boundary, measured not asserted: **L2 ~13 TB/s**, **DRAM ~0.9 TB/s** -- L2 is ~**14x** DRAM's
streaming bandwidth. With §2's latency staircase, the scorecard's on-chip rows are now fully filled. The whole
optimization game falls out of these two columns: a GPU access is high-latency *and* off-chip bandwidth is scarce,
so you either **keep the working set on-chip** (tiling into L2/shared, §7b) to ride the 14x-faster bandwidth and
~3x-lower latency, or **expose enough parallelism to hide the trip to the bottom** (§2). Move the hot set up the
pyramid, or hide the latency -- there is no third option.
""")

# ============================================================================
# 3. A THREAD IS ALSO A SIMD LANE  (divergence)
# ============================================================================
H(3, "A thread is also a SIMD lane: sort a partition, branchy vs branchless", 15)
card("sort 8 numbers per lane: merge (branchy) vs odd-even network (branchless)",
     "8 ints/lane, already in registers, on one SM", "cycles per warp + the exact op count",
     "the real first stage of a GPU sort -- and why CUB sorts each partition with a branchless network.")
concept(r"""
**The setup -- one SM, data already in registers.** This is the per-thread *base sort* inside a block sort. Each thread
has already loaded its 8 keys from memory into **registers** (a local `int a[8]`), and we time **only the sort** --
`clock64` brackets it, *after* the loads. So **memory is out of the picture** here (that was §1/§2); everything runs on
**one SM**, and what we measure is **pure compute**: branches, predication, lockstep. This mirrors CUB exactly -- load
items-per-thread into registers -> sort *in registers* with the network -> the block then merges the sorted runs (§7).
""")
concept(r"""
In §2 a "thread" was a *task* you oversubscribe; inside a warp it is the opposite -- 32 lanes are **one SIMD unit**,
one instruction in lockstep. A **data-dependent branch** breaks that: the warp serially runs *every* path some lane
takes. Here is the real thing instead of a toy: the first stage of a merge sort is *"sort each small partition."*
The obvious way is a small **merge** -- but its core step, `out = (a<=b) ? a : b` with the *loser* advancing, is a
**data-dependent branch** (which side moves depends on the values). The branchless alternative CUB actually uses is a
**sorting network** -- `cub::detail::StableOddEvenSort`, a *fixed* sequence of compare-exchanges whose comparator is a
**conditional swap**, `if (a[j+1] < a[j]) swap`. **"Comparison-based" does not mean "branchy":** on a *fixed* network
the compiler **predicates** that `if` into `compare + conditional-move` -- the data decides *which value moves*, never
*which instruction runs* -- so the schedule is **data-oblivious** and never diverges.
""")
fig("07_sort_network.png", "Branchy merge: which side advances depends on the values -> the warp runs both paths (divergence). Branchless odd-even network: fixed comparator positions -> every lane runs the identical sequence in lockstep.")
md(r"""
**Exactly what each does, step by step, on the same 8 numbers.** The network's comparator **positions are fixed** by
phase parity (even `(0,1)(2,3)(4,5)(6,7)`, odd `(1,2)(3,4)(5,6)`) -- identical every run, and it keeps grinding them
*even after the data is sorted* (checking "done?" would be a branch). The merge's `L`/`R` shows **which side it took**
-- that choice *is* the data-dependent branch, and a different input takes a different path:
""")
cuda(r"""
#include <cstdio>
#define K 8
__global__ void trace(){
  // ---- BRANCHLESS: cub::detail::StableOddEvenSort (keys-only, cub::Less) -- positions FIXED ----
  { int a[K]={5,2,8,1,9,3,7,4}; int cmps=0;
    printf("BRANCHLESS odd-even network = cub::detail::StableOddEvenSort (fixed positions):\n  start          : ");
    for(int i=0;i<K;i++) printf("%2d ",a[i]); printf("\n");
    for(int i=0;i<K;i++){                                  // CUB: for i in [0,N)  -- N phases
      printf("  phase %d (%-4s):", i, (i&1)?"odd":"even");
      for(int j=(1&i); j<K-1; j+=2){ printf(" cmp(%d,%d)",j,j+1); cmps++;
        if(a[j+1] < a[j]){ int t=a[j]; a[j]=a[j+1]; a[j+1]=t; } }   // if(compare_op(keys[j+1],keys[j])) Swap(...)
      printf("  -> "); for(int k=0;k<K;k++) printf("%2d ",a[k]); printf("\n"); }
    printf("  => %d compare-exchanges = N(N-1)/2, the SAME for EVERY input.\n\n", cmps);
  }
  // ---- BRANCHY: bottom-up merge -- which side is TAKEN depends on the data (a branch) ----
  { int a[K]={5,2,8,1,9,3,7,4}, tmp[K]; int cmps=0;
    printf("BRANCHY merge (L/R = which side taken -- DEPENDS on the values):\n  start          : ");
    for(int i=0;i<K;i++) printf("%2d ",a[i]); printf("\n");
    for(int w=1;w<K;w*=2){
      printf("  merge runs of %d: ", w);
      for(int lo=0;lo<K;lo+=2*w){ int mid=lo+w<K?lo+w:K, hi=lo+2*w<K?lo+2*w:K, i=lo,j=mid,k=lo;
        while(i<mid&&j<hi){ cmps++; if(a[i]<=a[j]){ printf("L%d ",a[i]); tmp[k++]=a[i++]; }
                            else  { printf("R%d ",a[j]); tmp[k++]=a[j++]; } }
        while(i<mid) tmp[k++]=a[i++]; while(j<hi) tmp[k++]=a[j++]; printf("| "); }
      for(int t=0;t<K;t++) a[t]=tmp[t];
      printf("-> "); for(int i=0;i<K;i++) printf("%2d ",a[i]); printf("\n"); }
    printf("  => %d data-dependent branches for THIS input -- another input branches differently.\n", cmps);
  }
}
int main(){ trace<<<1,1>>>(); cudaDeviceSynchronize(); }
""")
md(r"""
**What "complexity" means here -- and why the "terrible" sort is the right one.** *Complexity* (Big-O) counts how the
number of **operations** grows with the input size N, and it silently assumes **every operation costs the same**. By
that yardstick the odd-even network is **O(N^2)** -- it literally *is* a parallel **bubble sort**, the "worst sort you
were taught" -- while merge is **O(N log N)**, supposedly "better." For N=8 the network even does *more* compares
(`28` vs merge's `17`, both printed above). And yet it runs **~9x faster**.

Why? Because Big-O's "every op costs the same" is **false on this machine**. An op's real cost depends on whether it
**branches** (divergence), whether the schedule is **fixed** (so it unrolls and exposes ILP), and whether the 32 lanes
stay in **lockstep**. The network's 28 ops are cheap **predicated selects** on a fixed, fully-unrolled schedule -- *no
best/worst/average case, exactly N(N-1)/2 every time*, so its cost is an **exact number** you can budget in cycles, not
a Big-O with a hidden input-dependent constant. Merge's *fewer* ops each drag a data-dependent branch behind them.

At **small N** the `N^2` constant is tiny and **obliviousness wins**, so the "terrible" sort is **perfect**. (As N
grows, `N^2` does eventually lose to `N log N` -- which is precisely why CUB uses the network only for the **base
case**, a handful of items per thread, then *merges*. Asymptotics matter again at scale.) The lesson is the course's
whole thesis: **asymptotic optimality is not hardware-neutral** -- the right cost model here is *cycles*, not Big-O.
""")
guess("each lane sorts 8 numbers. On RANDOM data, which wins -- the branchy merge, or CUB's network (a fixed 28 compare-exchanges)? And does either change when the input is already sorted?",
      "branchy merge", "out = (a<=b) ? a++ : b++;        // data-dependent branch",
      "cub StableOddEvenSort", "if(a[j+1]<a[j]) swap;  // fixed 28 comparators -> predicated")
cuda(r"""
#include <cstdio>
#include <vector>
#include <random>
#define K 8
// branchy: bottom-up merge -- the (a<=b)? branch picks which side advances -> data-dependent, diverges.
__device__ __noinline__ void merge_sort(int* a){
  int tmp[K];
  for(int w=1;w<K;w*=2){
    for(int lo=0;lo<K;lo+=2*w){ int mid=lo+w<K?lo+w:K, hi=lo+2*w<K?lo+2*w:K, i=lo,j=mid,k=lo;
      while(i<mid&&j<hi) tmp[k++]=(a[i]<=a[j])?a[i++]:a[j++];   // <-- data-dependent branch
      while(i<mid) tmp[k++]=a[i++]; while(j<hi) tmp[k++]=a[j++]; }
    for(int t=0;t<K;t++) a[t]=tmp[t]; }
}
// branchless: EXACT cub::detail::StableOddEvenSort (keys-only, cub::Less) -- fixed N(N-1)/2 compare-exchanges.
__device__ __noinline__ void cub_oddeven(int* a){
  #pragma unroll
  for(int i=0;i<K;i++)
    #pragma unroll
    for(int j=(1&i); j<K-1; j+=2)
      if(a[j+1] < a[j]){ int t=a[j]; a[j]=a[j+1]; a[j+1]=t; }   // if(compare_op(keys[j+1],keys[j])) Swap(...)
}
// MODE is a TEMPLATE arg, not a runtime parameter: the compiler stamps out a SEPARATE kernel for
// each choice, resolved by `if constexpr` at COMPILE time. Picking merge-vs-network is a cost-model
// decision made before the kernel runs (here: partition size) -- exactly how CUB/thrust dispatch by
// type. There is no `if (mode)` left in the running kernel to branch on.
template<int MODE>
__global__ void bench(const int* in,int* out,long long* cyc){
  int t=blockIdx.x*blockDim.x+threadIdx.x, lane=threadIdx.x&31;
  int a[K];                                           // the 8 keys live in REGISTERS
  #pragma unroll
  for(int i=0;i<K;i++) a[i]=in[t*K+i];                // stage from memory -> registers (NOT timed)
  __syncwarp(); long long t0=clock64();               // timed window = the in-register sort only
  if constexpr (MODE==0) merge_sort(a); else cub_oddeven(a);   // chosen at compile time
  __syncwarp();                                       // warp span = the SLOWEST lane (divergence shows here)
  long long t1=clock64();
  #pragma unroll
  for(int i=0;i<K;i++) out[t*K+i]=a[i];
  if(lane==0) cyc[t>>5]=t1-t0;
}
template<int MODE>
double measure(const std::vector<int>& h,int warps){
  int n=warps*32; int *in,*out; long long* cyc;
  cudaMalloc(&in,(size_t)n*K*4); cudaMalloc(&out,(size_t)n*K*4); cudaMalloc(&cyc,warps*8);
  cudaMemcpy(in,h.data(),(size_t)n*K*4,cudaMemcpyHostToDevice);
  bench<MODE><<<1,n>>>(in,out,cyc); cudaDeviceSynchronize();
  std::vector<long long> c(warps); cudaMemcpy(c.data(),cyc,warps*8,cudaMemcpyDeviceToHost);
  double avg=0; for(long long v:c) avg+=v; avg/=warps;
  cudaFree(in);cudaFree(out);cudaFree(cyc); return avg;
}
int main(){
  int warps=4, n=warps*32; std::mt19937 rng(1);
  printf("each lane sorts %d ints; cycles per warp (clock64), one SM:\n", K);
  printf("   input        merge (branchy)     cub_oddeven (28 cmp, branchless)\n");
  const char* nm[]={"sorted  ","reversed","random  "};
  for(int dist=0;dist<3;dist++){
    std::vector<int> h(n*K);
    for(int t=0;t<n;t++) for(int i=0;i<K;i++)
      h[t*K+i] = dist==0 ? i : dist==1 ? K-i : (int)(rng()%1000);
    printf("   %s   %12.0f        %12.0f\n", nm[dist], measure<0>(h,warps), measure<1>(h,warps));  // MODE chosen at compile time
  }
  printf("\ncub_oddeven = fixed 28 compare-exchanges (data-oblivious); merge varies with input AND has many data-dependent branches.\n");
}
""")
md(r"""
**Read "branchless" off the hardware -- the SASS.** The whole claim ("the comparator is free, no branch") is a
statement about the *machine*, invisible in the CUDA source -- so disassemble both and count the telling opcodes
(`BRA` = a branch, `ISETP` = a compare):
""")
code(r'''
import subprocess, re, collections
src = r"""
#define K 8
__device__ void merge_sort(int* a){ int tmp[K];
  for(int w=1;w<K;w*=2){ for(int lo=0;lo<K;lo+=2*w){ int mid=lo+w<K?lo+w:K,hi=lo+2*w<K?lo+2*w:K,i=lo,j=mid,k=lo;
    while(i<mid&&j<hi) tmp[k++]=(a[i]<=a[j])?a[i++]:a[j++]; while(i<mid)tmp[k++]=a[i++]; while(j<hi)tmp[k++]=a[j++]; }
    for(int t=0;t<K;t++)a[t]=tmp[t]; } }
__device__ void cub_oddeven(int* a){
  #pragma unroll
  for(int i=0;i<K;i++)
    #pragma unroll
    for(int j=(1&i);j<K-1;j+=2) if(a[j+1]<a[j]){int t=a[j];a[j]=a[j+1];a[j+1]=t;} }
__global__ void kmerge(int* a){ merge_sort(a); }
__global__ void knet(int* a){ cub_oddeven(a); }
"""
open("/tmp/_sortsass.cu","w").write(src)
subprocess.run("nvcc -arch=sm_89 -O3 -std=c++17 -cubin -o /tmp/_sortsass.cubin /tmp/_sortsass.cu", shell=True)
sass = subprocess.run("cuobjdump -sass /tmp/_sortsass.cubin", shell=True, capture_output=True, text=True).stdout
funcs={}; cur=None
for line in sass.splitlines():
    m=re.search(r"Function : (\S+)", line)
    if m: cur=m.group(1); funcs[cur]=[]; continue
    if cur is None: continue
    s=re.sub(r"/\*.*?\*/","",line).strip()
    if not s: continue
    op=s.split()[0]
    if op.startswith("@"): op=(s.split()[1:2] or [""])[0]
    op=op.split(".")[0]
    if re.match(r"[A-Z][A-Z0-9_]*$",op): funcs[cur].append(op)
print("                              BRA   ISETP   total instrs")
for name,label in [("kmerge","merge (branchy)         "),("knet","cub_oddeven (branchless)")]:
    fn=[k for k in funcs if name in k]; ops=collections.Counter(funcs[fn[0]]) if fn else {}
    print("  %s   %4d   %4d    %5d" % (label, ops.get("BRA",0), ops.get("ISETP",0), sum(ops.values())))
print("\n-> merge: ~112 BRA = data-dependent branches (the warp diverges). cub_oddeven: 1 BRA (just the epilogue)")
print("   + 28 ISETP = the 28 comparators, each a compare feeding a PREDICATED swap -- no branch. Same source")
print("   keyword `if`, opposite hardware: a branch in merge, a conditional-move in the network.")
''')
md(r"""
**The actual SASS, side by side** (representative lines from the disassembly above). Both compute the same comparison
with the same machinery -- `ISETP` sets a predicate `P` -- but watch what the predicate **guards**:

<table style="width:100%"><tr>
<td style="width:50%;vertical-align:top"><b>cub_oddeven (branchless)</b><pre>
LDG.E R6, [R2+0x14]        ; load a[j+1]   8 loads
LDG.E R4, [R2+0x8]         ; load a[j]     in flight
ISETP.GE.AND P2, R13, R6   ; compare  ┐
ISETP.GE.AND P1, R9,  R4   ; compare  ├ 4 compares
ISETP.GE.AND P3, R17, R8   ; compare  │ back-to-back
ISETP.GE.AND P6, R7,  R5   ; compare  ┘ = ILP
@!P2 STG.E [R2+0x14], R6   ; <b>@P guards a STORE</b>
@!P1 IMAD.MOV R11, R4      ; <b>@P guards a MOVE</b>
                          ; -> swap happens, NO branch
</pre></td>
<td style="width:50%;vertical-align:top"><b>merge (branchy)</b><pre>
ISETP.GT.AND P0, R0, R5    ; compare run-fronts
ISETP.LT.U32 P1, R12, 0x2  ; data-dependent loop bound
@P0  BRA 0x80              ; <b>@P guards a BRANCH</b>
@P1  BRA 0x8a0             ; <b>@P guards a BRANCH</b>
@!P1 BRA 0x360             ; <b>@P guards a BRANCH</b>
                          ; -> control flow forks on
                          ;    the data -> warp diverges
</pre></td>
</tr></table>

**Same predicate, opposite consequence:** the network's `@P` guards a **data move** (`STG`/`MOV`) -- every lane runs
it, the predicate just decides *what value lands*, so the warp stays in lockstep. Merge's `@P` guards a **`BRA`** --
the predicate decides *which code runs*, and when lanes disagree the warp serializes both paths. That one difference
-- predicating **data** vs predicating **control** -- is the entire `~9x`, and it is **nowhere in the C++ source**.
""")
md(r"""
**Why keep the CUDA cell, if the truth is in the SASS? Two reasons.**

- **ILP is real here -- you just can't see it in the loop.** Within one phase the comparators are *independent*; the
  compiler issues several in flight (instruction-level parallelism) before any result is consumed. You wrote a
  sequential per-thread loop, but you must *think* in independent ops -- give the scheduler parallel work or a thread
  stalls on its own dependency chain. (This is the §5 register-blocking lever, in miniature: ILP **inside** one thread,
  distinct from the **across-warp** latency-hiding of §2.)
- **The language hides the cost model -- a leaky abstraction.** The *same* `if` is free here (a predicated select) and
  expensive in the merge (a divergent branch); CUDA C++ gives you no syntactic way to tell which, nor to declare "this
  schedule is data-oblivious." You write scalar imperative C++ for a 32-wide SIMT machine and *hope* `ptxas` predicates
  and vectorizes it -- CUB even hand-rolls the network and sprinkles `#pragma unroll` because the language can't express
  "fixed schedule." Tile languages like **Triton** push back on exactly this: you write block/tile ops and the compiler
  owns the warp mapping, exposing the two kinds of parallelism as *separate* knobs -- **vectorization** (true SIMD
  packing, multiple data per instruction) and **software-pipeline depth** (`num_stages`, latency hiding) -- the two
  CUDA C++ leaves tangled in your head. The cost lives in the hardware, not the syntax: which is why we measure cycles.
""")
costmodel(r"""
On random data CUB's network is **~9x faster** at K=8 (~580 vs ~5000 cyc), and its cycles are **flat** across
sorted/reversed/random (~580 each) -- data-oblivious. The merge pays for **data-dependent control flow** -- its SASS
carries **~112 `BRA`** (data-dependent branches) against the network's **1** (just the epilogue). CUB's
`if (a[j+1] < a[j]) swap`, on a *fixed* network, has none: the compiler predicates it to **exactly 28 `ISETP`
(= N(N-1)/2 comparators) + conditional moves, zero `BSSY/BSYNC`** -- the `if` never becomes a branch. The deeper win
is *costability*: a fixed `N(N-1)/2` compare-exchanges is a number you can turn into an exact cycle budget, not an
O(...) with an input-dependent constant. **Caveat (honesty):** a network is `N^2/2` work, so it only wins for **small**
partitions -- which is exactly how CUB uses it: each thread sorts a *handful* of items with `StableOddEvenSort`, then
the block **merges** those runs (§7). Network for the base case, merge for the scaling. A "thread" is sometimes a task
(§2), sometimes a SIMD lane (here).
""")

md(r"""
**Takeaway -- read CUDA as a hint, not a description.** The rule for *this* example: an `if` is **branchless** when its
body is small and maskable -- the compiler runs both sides and writes per-lane under a predicate (`@P STG/MOV`, all 32
lanes in lockstep); it stays a **branch** when the *control flow* depends on the data (a data-dependent loop bound,
"which side advances") -- then lanes need different instruction streams and the warp serializes. Same keyword `if`,
opposite cost, and **nothing in the source tells you which**.

The honest generalization for a working CUDA programmer: ordinary C++ already has a *tension with the compiler* -- you
can't fully predict inlining, vectorization, register allocation, so serious performance work already means reading the
asm and profiling. **CUDA multiplies that tension.** On top of C++'s unknowns it layers an entire SIMT cost model --
divergence, predication, coalescing, occupancy, latency hiding -- all decided by compiler and hardware, all invisible in
the syntax. So the profiler (`ncu` / `nsys`) and the disassembler (`cuobjdump -sass`) are not occasional tools here;
they are the **primary source of truth**, consulted far more often than in normal C++. You write the source; you
*verify* on the machine. **Guess from the cost model, then confirm in cycles / SASS** -- that habit is the whole method
of this course.
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
