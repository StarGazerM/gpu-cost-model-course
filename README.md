# GPU programming is cost-model first

A live, measured mini-course on GPGPU for **computer architects and systems/DB people meeting the GPU** — built and
benchmarked on a single **NVIDIA RTX 6000 Ada** (sm_89, AD102).

> You don't pick an algorithm and make the chip run it. You compute the **cost model** first — bytes moved, passes
> over memory, latency vs concurrency, where data lives, how you touch it — and *that* selects the algorithm before
> you write a line. **Asymptotic optimality is not hardware-neutral.**

Every claim is **measured live** in the notebook, never asserted: real `%%cuda` cells (compiled with `nvcc`), cycle
counts via `clock64()`, and disassembly via `cuobjdump -sass`.

## The course

The whole thing is **[`course.ipynb`](course.ipynb)** — a hot-editable notebook of live CUDA cells.

**The 1-hour spine is §0–§3** (~64 min):

- **§0 Foundation** — the bet (throughput over latency); SIMT vs SIMD; the 4-way `scalar → ILP → SIMD → SIMT`; the
  programming model in one kernel; a blank *speeds & feeds* scorecard the rest of the course fills in.
- **§1 Bandwidth** — STREAM Triad on CPU *and* GPU from one source; GB/s is the chip's unit.
- **§2 Latency — the premise** — per-layer pointer-chase staircase, CPU vs GPU (the GPU loses at *every* level), then
  *hide* it on one SM (warp sweep). Plus the register/occupancy cap (§2b) and the blocks→SMs axis (§2c).
- **§2.5 Cache cliff** — bandwidth vs working-set size; where the L2→DRAM boundary actually is.
- **§3 A thread is a SIMD lane** — sort a partition: branchy merge vs `cub::StableOddEvenSort`; divergence; SASS
  predication (predicating *data* vs *control*); complexity-vs-cycles (why a "terrible" O(N²) network is *perfect* at
  small N); and the takeaway — *read CUDA as a hint over a SIMT machine, not as C++; guess from the cost model, then
  confirm in cycles/SASS.*

§4 onward (coalescing, register/ILP, the radix-vs-merge cost model, the two-sort capstone, the "kernel is ~10%" rug
pull) is the **2-hour extension**.

The slides are generated, not hand-edited: **[`slides/build_course.py`](slides/build_course.py)** writes
`course.ipynb`, and **[`slides/article_figures.py`](slides/article_figures.py)** renders the schematic figures.

## Run it

Toolkit (`nvcc` / `cuobjdump` / `ncu`) is the NVIDIA HPC SDK on `PATH`, **not** a pip dependency.

```bash
python3 -m venv ~/.venvs/gpucourse && source ~/.venvs/gpucourse/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name gpucourse --display-name "GPU course"

# regenerate the notebook from source, then execute it end-to-end
python3 slides/build_course.py
jupyter nbconvert --to notebook --execute --inplace course.ipynb
```

Built/verified on system Python 3.10, RTX 6000 Ada, HPC SDK 25.5. Numbers (latency ns, GB/s, cycles) are specific to
that card — re-run on yours and the *shape* of every result should hold even as the absolute numbers move.
