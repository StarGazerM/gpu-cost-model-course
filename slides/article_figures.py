#!/usr/bin/env python3
"""Schematic figures for slides/article.md (Part 0 -- the architecture foundation).
Run:  python3 slides/article_figures.py   ->  writes slides/figures/*.png
These are CONCEPTUAL diagrams (not measured); the measured charts live in course.ipynb.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
GPU, CPU, INK, CEIL, HOT, COOL = "#76b900", "#5b6770", "#23272b", "#b9bfc4", "#c0392b", "#0b84a5"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "savefig.bbox": "tight"})


def box(ax, x, y, w, h, c, t="", tc="white", fs=10, ec="white", lw=1.2, alpha=1):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=c, edgecolor=ec, lw=lw, alpha=alpha))
    if t:
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("wrote", name)


# 0.1 -- where the transistors go: CPU (control+cache heavy) vs GPU (ALU+regfile heavy)
def f_area():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, title in [(a, "CPU core: area buys single-thread speed"),
                      (b, "GPU SM: area buys ALUs + resident threads")]:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.set_title(title, fontsize=11)
    # CPU: control + cache big; only a few ALUs; cache band at the bottom (no overlap)
    box(a, 0.5, 6.7, 5.0, 2.8, CPU, "Control\n(OoO, branch predict,\nrename, scheduler)", fs=8.5)
    box(a, 5.8, 6.7, 3.7, 2.8, COOL, "big\ncache", fs=10)
    for i in range(4):
        box(a, 0.7 + i * 1.25, 4.3, 1.0, 1.5, GPU, "ALU", fs=8)
    a.text(5, 3.7, "a few fat ALUs", ha="center", color=INK, fontsize=9, style="italic")
    box(a, 0.5, 0.5, 9.0, 2.6, "#7a8893", "L2 / L3 cache", fs=10)
    # GPU: shared control, sea of lanes, big register file (short labels so nothing clips)
    box(b, 0.5, 8.7, 9.0, 0.9, CPU, "shared Control + scheduler (1 per 32 lanes)", fs=8.5)
    for r in range(6):
        for c in range(16):
            box(b, 0.55 + c * 0.55, 3.4 + r * 0.8, 0.48, 0.7, GPU, ec="white", lw=0.6)
    b.text(5, 2.85, "a sea of ALUs (lanes)", ha="center", color=INK, fontsize=9, style="italic")
    box(b, 0.5, 0.5, 9.0, 1.8, "#b07b16", "REGISTER FILE  ~36 MB  (bigger than L2)", fs=9.5)
    save(fig, "01_area.png")


# 0.2 -- SIMT vs AVX: width across threads (per-lane scalar regs) vs one wide register
def f_simt():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for ax in (a, b):
        ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    a.set_title("CPU SIMD (AVX-512): 1 thread, 1 WIDE register", fontsize=10.5)
    box(a, 1, 8, 10, 1.2, CPU, "1 instruction  (vaddps)", fs=10)
    a.add_patch(FancyArrow(6, 7.9, 0, -1.0, width=0.05, head_width=0.4, color=INK))
    for i in range(8):
        box(a, 1 + i * 1.25, 4.5, 1.1, 1.6, COOL, f"{i}", fs=9)
    a.text(6, 3.6, "one 512-bit register, 16 lanes of ONE thread's data\n(programmer hand-vectorizes)",
           ha="center", color=INK, fontsize=9)
    b.set_title("GPU SIMT: 32 threads, per-lane SCALAR registers", fontsize=10.5)
    box(b, 1, 8, 10, 1.2, CPU, "1 instruction, shared by the warp", fs=10)
    for i in range(8):
        x = 1 + i * 1.32
        b.add_patch(FancyArrow(x + 0.5, 7.9, 0, -1.0, width=0.03, head_width=0.18, color=INK))
        box(b, x, 4.5, 1.05, 1.6, GPU, f"L{i}", fs=9)
        box(b, x, 2.7, 1.05, 1.4, "#b07b16", "reg", fs=8)
    b.text(6, 1.8, "width is ACROSS threads (lanes); each lane has ordinary scalar regs\n"
                   "(you write scalar per-thread code; hardware gangs 32)", ha="center", color=INK, fontsize=9)
    b.text(11.6, 5.3, "...x32", color=INK, fontsize=10, fontweight="bold")
    save(fig, "02_simt_vs_avx.png")


# 0.3 -- the two hierarchies (hardware vs software) and the name mapping
def f_hierarchy():
    fig, ax = plt.subplots(figsize=(11, 4.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Same hierarchy, two vocabularies -- and the words that mislead", fontsize=11)
    hw = [("Chip (AD102)", "#3a3f44"), ("SM  x142", CPU), ("sub-partition x4\n(scheduler + 32 lanes)", COOL), ("lane = 'CUDA core'", GPU)]
    sw = [("Grid", "#3a3f44"), ("Block", CPU), ("Warp = 32 threads", COOL), ("thread", GPU)]
    for i, (h, c) in enumerate(hw):
        box(ax, 0.5, 8 - i * 2.0, 5, 1.5, c, h, fs=9)
    for i, (s, c) in enumerate(sw):
        box(ax, 6.5, 8 - i * 2.0, 5, 1.5, c, s, fs=9)
    for i in range(4):
        ax.annotate("", xy=(6.5, 8.75 - i * 2.0), xytext=(5.5, 8.75 - i * 2.0),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.text(3, 9.7, "HARDWARE", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(9, 9.7, "SOFTWARE", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(6, 0.3, "the core is the SM (~142), NOT the 'CUDA core' (a lane). a 'thread' is one lane's scalar work.",
            ha="center", fontsize=9, style="italic", color=HOT)
    save(fig, "03_hierarchy.png")


# 0.4 -- divergence: a warp runs both sides of a branch, masking the idle lanes
def f_divergence():
    fig, ax = plt.subplots(figsize=(10, 3.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Divergence: no per-lane branch predictor -> the warp runs BOTH paths", fontsize=11)
    lanes = 8
    ax.text(0.2, 6.8, "if (cond):", fontsize=10, color=INK, fontweight="bold")
    for i in range(lanes):
        active = i % 2 == 0
        box(ax, 1.5 + i * 1.2, 5.3, 1.0, 1.2, GPU if active else CEIL, "run" if active else "idle",
            tc="white" if active else "#555", fs=8)
    ax.text(11.6, 5.9, "path A", color=INK, fontsize=9)
    ax.text(0.2, 3.8, "else:", fontsize=10, color=INK, fontweight="bold")
    for i in range(lanes):
        active = i % 2 == 1
        box(ax, 1.5 + i * 1.2, 2.3, 1.0, 1.2, GPU if active else CEIL, "run" if active else "idle",
            tc="white" if active else "#555", fs=8)
    ax.text(11.6, 2.9, "path B", color=INK, fontsize=9)
    ax.text(6, 0.7, "cost = path A + path B (both executed, half the lanes masked each time).\n"
                    "data-oblivious / branchless code -> one path, no waste.", ha="center", fontsize=9, color=HOT)
    save(fig, "04_divergence.png")


# 0.5 -- latency hiding: when a warp stalls, the scheduler switches to another (free)
def f_latency():
    fig, ax = plt.subplots(figsize=(11, 3.8)); ax.set_xlim(0, 22); ax.set_ylim(-0.5, 4.5)
    ax.set_title("Latency hiding: a stalled warp is covered by another resident warp (zero-overhead switch)",
                 fontsize=10.5)
    for w in range(4):
        y = 3 - w
        # issue burst
        ax.add_patch(Rectangle((w * 1.6, y), 1.4, 0.7, facecolor=GPU, edgecolor="white"))
        ax.text(w * 1.6 + 0.7, y + 0.35, f"W{w}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        # long memory stall
        ax.add_patch(Rectangle((w * 1.6 + 1.4, y), 12, 0.7, facecolor=CEIL, edgecolor="white", alpha=0.6))
        # data returns -> resume
        ax.add_patch(Rectangle((w * 1.6 + 13.4, y), 1.4, 0.7, facecolor=GPU, edgecolor="white"))
    ax.annotate("scheduler issues W0,W1,W2,W3 back-to-back\nwhile each waits on memory -> SM never idle",
                xy=(3, 3.9), fontsize=9, color=INK, ha="left")
    ax.set_yticks([]); ax.set_xlabel("time  (green = compute issued, grey = memory latency in flight)")
    for s in ax.spines.values(): s.set_visible(False)
    save(fig, "05_latency_hiding.png")


# 0.6 -- memory hierarchy (size / bandwidth / latency), registers on top
def f_mem():
    fig, ax = plt.subplots(figsize=(9, 4.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Memory hierarchy: a bandwidth machine (sizes are RTX 6000 Ada)", fontsize=11)
    rows = [("Registers", "~36 MB chip-wide -- the LARGEST on-chip", "#b07b16", 1.0, 9.0),
            ("Shared mem / L1", "~100 KB/SM -- managed scratchpad, 32 banks (bank conflicts)", GPU, 2.0, 7.4),
            ("L2 cache", "~96 MB shared", COOL, 3.0, 5.8),
            ("GDDR6 (HBM-class)", "48 GB @ ~960 GB/s -- high bandwidth, high latency", CPU, 4.0, 4.2)]
    for name, note, c, inset, y in rows:
        box(ax, inset, y, 10 - 2 * inset, 1.3, c, "", fs=9)
        ax.text(5, y + 0.65, name, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(5, y - 0.12, note, ha="center", va="top", color=INK, fontsize=8.5)
    ax.annotate("", xy=(0.5, 4.4), xytext=(0.5, 9.6), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(0.2, 7, "faster,\nsmaller", rotation=90, va="center", fontsize=8, color=INK)
    save(fig, "06_mem_hierarchy.png")


# 0.6b -- coalescing: contiguous lanes = 1 transaction; strided = many wasted transactions
def f_coalesce():
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 4))
    for ax in (a, b):
        ax.set_xlim(0, 33); ax.set_ylim(0, 3); ax.axis("off")
    a.set_title("Coalesced: 32 lanes -> contiguous addresses -> ONE 128-byte transaction", fontsize=10, color=GPU)
    for i in range(32):
        box(a, i, 1.6, 0.9, 1.0, GPU, ec="white", lw=0.5)
    a.add_patch(Rectangle((0, 0.3), 32, 0.8, facecolor=GPU, edgecolor="white", alpha=0.5))
    a.text(16, 0.7, "1 transaction (full bus utilization)", ha="center", color=INK, fontsize=9)
    b.set_title("Strided / scattered: each lane its own sector -> up to 32 transactions (bus thrown away)",
                fontsize=10, color=HOT)
    for i in range(32):
        box(b, i, 1.6, 0.9, 1.0, HOT, ec="white", lw=0.5)
        if i % 4 == 0:
            b.add_patch(Rectangle((i, 0.3), 0.9, 0.8, facecolor=HOT, edgecolor="white", alpha=0.5))
    b.text(16, 0.0, "many partial transactions = wasted bandwidth", ha="center", color=INK, fontsize=9)
    save(fig, "06b_coalescing.png")


# 0.2b -- addressing: SIMD (1 base -> contiguous) vs SIMT (32 per-lane addresses -> coalesce)
def f_addressing():
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 4.6))
    for ax in (a, b):
        ax.set_xlim(0, 34); ax.set_ylim(0, 4); ax.axis("off")
    # SIMD: one address register -> one contiguous load
    a.set_title("SIMD: ONE base address -> ONE contiguous transaction", fontsize=10.5, color=COOL)
    box(a, 0.5, 2.7, 4.5, 0.9, COOL, "1 address reg", fs=9)
    a.add_patch(FancyArrow(5.2, 3.15, 1.4, 0, width=0.04, head_width=0.3, color=INK))
    for i in range(16):
        box(a, 7 + i * 1.0, 2.7, 0.9, 0.9, COOL, ec="white", lw=0.5)
    a.add_patch(Rectangle((7, 1.4), 16, 0.7, facecolor=COOL, edgecolor="white", alpha=0.5))
    a.text(15, 1.75, "1 transaction (must be contiguous; gather needs a special slow instr)",
           ha="center", color=INK, fontsize=8.5)
    # SIMT: 32 per-lane address registers -> hardware coalesces
    b.set_title("SIMT: 32 per-lane address registers -> hardware coalesces at runtime", fontsize=10.5, color=GPU)
    for i in range(16):
        box(b, 0.5 + i * 0.62, 2.7, 0.5, 0.9, "#b07b16", ec="white", lw=0.5)
    b.text(5.4, 1.9, "32 addresses\n(each lane its own)", ha="center", color=INK, fontsize=8)
    b.add_patch(FancyArrow(10.6, 3.15, 1.2, 0, width=0.04, head_width=0.3, color=INK))
    # contiguous -> 1 txn (green) ; scattered -> many (red)
    for i in range(8):
        box(b, 13 + i * 0.95, 2.7, 0.85, 0.9, GPU, ec="white", lw=0.5)
    b.add_patch(Rectangle((13, 1.5), 7.6, 0.6, facecolor=GPU, edgecolor="white", alpha=0.5))
    b.text(16.8, 1.05, "contiguous -> 1 txn", ha="center", color=GPU, fontsize=8.5, fontweight="bold")
    for i in range(8):
        box(b, 22 + i * 1.4, 2.7, 0.85, 0.9, HOT, ec="white", lw=0.5)
        b.add_patch(Rectangle((22 + i * 1.4, 1.5), 0.85, 0.6, facecolor=HOT, edgecolor="white", alpha=0.5))
    b.text(27, 1.05, "strided -> many txns", ha="center", color=HOT, fontsize=8.5, fontweight="bold")
    save(fig, "02b_addressing.png")


# 0.0 -- the CA recap: scalar (von Neumann) -> SIMD (wider register, more data/instruction)
def f_scalar_simd():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.7))
    for ax in (a, b):
        ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    a.set_title("Scalar (von Neumann): 1 element / instruction", fontsize=10.5, color=CPU)
    box(a, 1, 8.3, 10, 1.1, CPU, "1 instruction   add r3, r1, r2", fs=10)
    a.add_patch(FancyArrow(6, 8.2, 0, -1.0, width=0.05, head_width=0.4, color=INK))
    box(a, 4.4, 5.4, 3.2, 1.4, COOL, "32/64-bit reg", fs=9)
    a.add_patch(FancyArrow(6, 5.3, 0, -1.0, width=0.05, head_width=0.4, color=INK))
    box(a, 4.7, 2.7, 2.6, 1.4, GPU, "ALU", fs=9)
    a.text(6, 1.8, "one register, one ALU, one result", ha="center", color=INK, fontsize=9)
    b.set_title("SIMD (e.g. AVX-512): 16 elements / instruction", fontsize=10.5, color=COOL)
    box(b, 1, 8.3, 10, 1.1, CPU, "1 instruction   vaddps zmm3, zmm1, zmm2", fs=9.5)
    b.add_patch(FancyArrow(6, 8.2, 0, -1.0, width=0.05, head_width=0.4, color=INK))
    for i in range(8):
        box(b, 1 + i * 1.25, 5.4, 1.1, 1.4, COOL, f"{i}", fs=8)
    b.text(11.55, 6.1, "...x16", color=INK, fontsize=9, fontweight="bold")
    box(b, 1, 2.7, 10, 1.3, GPU, "wide ALU -- 16 lanes, one cycle", fs=9)
    b.text(6, 1.9, "ONE wide register; same control, more data per instruction", ha="center", color=INK, fontsize=9)
    save(fig, "00_scalar_simd.png")


if __name__ == "__main__":
    f_scalar_simd()
    f_area(); f_simt(); f_addressing(); f_hierarchy(); f_divergence(); f_latency(); f_mem(); f_coalesce()
    print("done ->", OUT)
