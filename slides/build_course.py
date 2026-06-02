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
### 0.1 The bet: throughput, not latency
A CPU core spends its area making *one* instruction stream fast (out-of-order, branch prediction, big caches).
A GPU rips that out and spends the area on **many ALUs** + **enough resident threads to switch among**.
Give up single-thread latency; buy aggregate throughput. Everything weird follows from this one bet.
""")
fig("01_area.png", "CPU spends area on control+cache for one fast thread; the GPU on a sea of ALUs + a huge register file.")

md(r"""
### 0.2 SIMT, and the words that mislead
32 lanes share one fetch/decode/scheduler (a **warp**) and run the same instruction in lockstep, each on its
*own* scalar registers. That is **SIMD execution with a per-lane register file** -- width is *across threads*,
not a wide register (unlike AVX). Pin the vocabulary once, or everything later is noise:

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

# ---- (sections 2-9 + appendix: to be built next, same canonical beat) ----

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = os.path.join(_ROOT, "course.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", os.path.normpath(out), f"({len(cells)} cells)  [REBUILD: §0 foundation + §1 bandwidth]")
