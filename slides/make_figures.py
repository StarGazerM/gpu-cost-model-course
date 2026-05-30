#!/usr/bin/env python3
"""All slide figures, one consistent theme, generated from measured numbers.

Spine of the talk is the COST MODEL: on a bandwidth machine, sort runtime is set
by how many times you stream the array (HBM passes), and algorithm choice -- not
kernel tuning -- is the lever. These figures make that visible.

Outputs -> slides/figs/:
  fig1_bandwidth_identity.png   CPU STREAM vs GPU copy (Demo 1)
  fig2_cache_cliff.png          v0 throughput across the 96 MB L2 boundary
  fig3_passes_law.png           throughput vs HBM passes -- the cost model
  fig4_arc_passes.png           the v0..v4 arc, dual-encoded runtime + passes
  fig5_fit_map.png              algorithm-hardware fit; the CPU<->GPU flip
  fig6_rugpull.png              kernel vs malloc/pool/graph per-iteration (Demo 3)

Run binaries live for the measured numbers; no profiler needed for these.
"""
import os, re, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(HERE, "figs")
D1, D2, D3 = (os.path.join(ROOT, d) for d in ("demo1_bandwidth", "demo2_sort", "demo3_rugpull"))
os.makedirs(FIGS, exist_ok=True)

# ---- theme -----------------------------------------------------------------
GPU, CPU, RADIX, CEIL, INK = "#76b900", "#5b6770", "#0b84a5", "#b9bfc4", "#23272b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "figure.facecolor": "white",
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.edgecolor": "#c9ced2", "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": "#eef1f3", "grid.linewidth": 1.0,
    "axes.axisbelow": True, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.labelcolor": INK,
})

def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def run(binpath, args=()):
    try:
        return subprocess.run([binpath, *map(str, args)], capture_output=True,
                              text=True, timeout=180).stdout
    except Exception:
        return ""

def num(text, pat):
    m = re.search(pat, text)
    return float(m.group(1)) if m else float("nan")

def save(fig, name):
    p = os.path.join(FIGS, name)
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))

# ---- analytic HBM passes per version (full streams of the array) ------------
# bitonic: L(L+1)/2 stages for v0; tiled versions collapse small-stride stages.
# v1/v2 TILE=2048 -> 136; v3 TILE=8192 -> 105 (matches its 105 graph nodes);
# CUB radix: ~8 (1 histogram + 4 digit scatters, read+write, partly fused).
PASSES = {"v0_naive": 351, "v1_shared": 136, "v2_shuffle": 136,
          "v3_multiblock": 105, "v4_cub": 8}
LABEL = {"v0_naive": "v0 naive", "v1_shared": "v1 shared", "v2_shuffle": "v2 shuffle",
         "v3_multiblock": "v3 big-tile", "v4_cub": "v4 CUB radix"}
ARC = list(PASSES)


def measure_arc(size="26"):
    ms, mk = {}, {}
    for v in ARC:
        out = run(os.path.join(D2, v), [size, "10"])
        ms[v] = num(out, r"([\d.]+)\s*ms")
        mk[v] = num(out, r"([\d.]+)\s*Mkeys/s")
    return ms, mk


# ---- Fig 1: bandwidth identity ---------------------------------------------
def fig_bandwidth():
    g = num(run(os.path.join(D1, "stream_gpu")), r"copy kernel:\s*([\d.]+)")
    numa = "numactl --cpunodebind=0 --membind=0 " if subprocess.run(
        ["which", "numactl"], capture_output=True).returncode == 0 else ""
    cpu_out = subprocess.run(
        f"OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores {numa}./stream_cpu",
        shell=True, cwd=D1, capture_output=True, text=True).stdout
    c = num(cpu_out, r"triad bandwidth:\s*([\d.]+)")
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    bars = ax.bar(["CPU STREAM Triad\nDDR4, 1 NUMA node", "GPU copy kernel\nGDDR6"],
                  [c, g], color=[CPU, GPU], width=0.6)
    ax.set_ylabel("Achieved bandwidth (GB/s)")
    ax.set_title("Bandwidth is the chip's identity")
    ax.set_ylim(0, g * 1.18)
    for b, v in zip(bars, [c, g]):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.0f}", ha="center",
                va="bottom", fontweight="bold")
    ax.annotate(f"{g/c:.0f}x", xy=(1, g), xytext=(0.5, g*1.05),
                ha="center", fontsize=18, color=GPU, fontweight="bold")
    despine(ax); save(fig, "fig1_bandwidth_identity.png")


# ---- Fig 2: the L2 cache cliff ---------------------------------------------
def fig_cliff():
    sizes = [24, 25, 26, 27]
    mk = [num(run(os.path.join(D2, "v0_naive"), [s, 5]), r"([\d.]+)\s*Mkeys/s")
          for s in sizes]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    x = range(len(sizes))
    ax.plot(x, mk, "o-", color=GPU, lw=2.5, ms=9, mfc="white", mew=2.2)
    ax.axvspan(-0.4, 0.5, color="#ffe7c2", alpha=0.7, zorder=0)
    ax.text(0, mk[0], "  fits in\n  96 MB L2", va="top", ha="left", color="#a85b00",
            fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"$2^{{{s}}}$\n{2**s*4//10**6} MB" for s in sizes])
    ax.set_ylabel("v0 throughput (Mkeys/s)")
    ax.set_title("The cache cliff: same code, 4x slower once you leave L2")
    ax.set_ylim(0, max(mk)*1.1)
    despine(ax); save(fig, "fig2_cache_cliff.png")


# ---- Fig 3: the passes law (cost model) ------------------------------------
def fig_passes_law(mk):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ref_bw = 810e9  # achieved copy-kernel bandwidth, B/s
    bytes_per_key_pass = 6.0  # ~4 B read + ~2 B conditional write
    import numpy as np
    pgrid = np.logspace(0, 2.7, 50)
    ax.plot(pgrid, ref_bw / (pgrid * bytes_per_key_pass) / 1e6, "--",
            color=CEIL, lw=2, label="bandwidth ceiling (810 GB/s)")
    for v in ARC:
        col = RADIX if v == "v4_cub" else GPU
        ax.scatter(PASSES[v], mk[v], s=130, color=col, zorder=5,
                   edgecolor="white", linewidth=1.5)
    # v1 and v2 are coincident (warp shuffle changed neither passes nor BW) --
    # label the pair once; offset the cluster labels so they don't collide.
    labels = [("v0_naive", "v0 naive", (12, -2)),
              ("v3_multiblock", "v3 big-tile", (-104, 2)),
              ("v4_cub", "v4 CUB radix", (14, 6))]
    for v, txt, off in labels:
        ax.annotate(txt, (PASSES[v], mk[v]), textcoords="offset points",
                    xytext=off, fontsize=10.5, fontweight="bold", color=RADIX if v == "v4_cub" else GPU)
    ax.annotate("v1, v2 shared", (PASSES["v1_shared"], mk["v1_shared"]),
                textcoords="offset points", xytext=(2, 16), fontsize=10.5,
                fontweight="bold", color=GPU,
                arrowprops=dict(arrowstyle="-", color=GPU, lw=1))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("HBM passes  (times the whole array is streamed)")
    ax.set_ylabel("Throughput (Mkeys/s)")
    ax.set_title("Runtime is passes: the only lever is the algorithm")
    speed = mk["v4_cub"] / mk["v0_naive"]
    pfac = PASSES["v0_naive"] / PASSES["v4_cub"]
    ax.text(0.04, 0.06,
            f"v0 -> CUB = {speed:.0f}x  =  {pfac:.0f}x fewer passes"
            f"  x  {speed/pfac:.1f}x better BW use",
            transform=ax.transAxes, fontsize=10.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f7f8", ec="#d4dadd"))
    ax.legend(loc="upper right", frameon=False)
    despine(ax); save(fig, "fig3_passes_law.png")


# ---- Fig 4: the arc, dual-encoded runtime + passes -------------------------
def fig_arc(ms):
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(ARC))
    bars = ax.bar(x, [ms[v] for v in ARC], color=GPU, width=0.6, zorder=3)
    bars[-1].set_color(RADIX)
    ax.set_yscale("log")
    ax.set_ylabel("kernel-only runtime (ms, log)")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[v].replace(" ", "\n") for v in ARC])
    ax.set_ylim(top=ms["v0_naive"] * 2.2)
    base = ms["v0_naive"]
    for xi, v in zip(x, ARC):
        ax.annotate(f"{ms[v]:.1f} ms\n{base/ms[v]:.0f}x", (xi, ms[v]),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    va="bottom", fontsize=9, fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(x, [PASSES[v] for v in ARC], "s--", color=INK, lw=1.8, ms=7,
             mfc="white", zorder=4, label="HBM passes")
    ax2.set_ylabel("HBM passes (dashed)", color=INK)
    ax2.set_yscale("log"); ax2.grid(False)
    ax.set_title("Each step removes passes; CUB changes the algorithm", pad=14)
    despine(ax)
    save(fig, "fig4_arc_passes.png")


# ---- Fig 5: algorithm-hardware fit map -------------------------------------
def fig_fit_map():
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    pts = {  # name: (SIMT/coalescing fit, bandwidth/few-passes fit, color, sublabel)
        "quicksort": (1.8, 6.6, CPU, "comparison, divergent"),
        "bitonic": (8.4, 2.6, GPU, "oblivious, many passes"),
        "radix": (7.8, 8.4, RADIX, "oblivious, few passes"),
    }
    for name, (xx, yy, col, sub) in pts.items():
        ax.scatter(xx, yy, s=300, color=col, alpha=0.95, edgecolor="white", lw=2, zorder=5)
        ax.annotate(f"{name}\n({sub})", (xx, yy), textcoords="offset points",
                    xytext=(0, -34), ha="center", va="top", color=col,
                    fontsize=10.5, fontweight="bold", zorder=6)
    ax.annotate("CPU's pick", (1.8, 6.6), xytext=(1.8, 9.4), ha="center",
                color=CPU, fontweight="bold", fontsize=12,
                arrowprops=dict(arrowstyle="->", color=CPU, lw=1.6))
    ax.annotate("GPU's pick", (7.8, 8.4), xytext=(5.4, 9.6), ha="center",
                color=RADIX, fontweight="bold", fontsize=12,
                arrowprops=dict(arrowstyle="->", color=RADIX, lw=1.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.5)
    ax.set_xlabel("SIMT fit  (branch-free, coalesced)  ->")
    ax.set_ylabel("bandwidth fit  (few HBM passes)  ->")
    ax.set_title("The same problem, a different best algorithm")
    ax.text(0.04, 0.10, "GPU demands BOTH axes;\nthe CPU tolerates poor SIMT fit",
            transform=ax.transAxes, ha="left", va="center", fontsize=10,
            color="#8a9197", style="italic")
    despine(ax); save(fig, "fig5_fit_map.png")


# ---- Fig 6: the rug pull ----------------------------------------------------
def fig_rugpull():
    pit, kon = {}, float("nan")
    for h in ("naive", "pool", "graph"):
        out = run(os.path.join(D3, f"{h}_harness"), [20, 200])
        pit[h] = num(out, r"per-iter\s*:\s*([\d.]+)")
        if h == "naive":
            kon = num(out, r"kernel-only\s*:\s*([\d.]+)")
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    labels = ["kernel\nonly", "naive\nmalloc", "pool\nmallocAsync", "graph\ncapture"]
    vals = [kon, pit["naive"], pit["pool"], pit["graph"]]
    bars = ax.bar(labels, vals, color=[CPU, "#c0392b", GPU, GPU], width=0.62)
    ax.set_ylabel("per-iteration time (ms)")
    ax.set_title("Same kernel, three harnesses: the kernel is ~10%")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
    despine(ax); save(fig, "fig6_rugpull.png")


if __name__ == "__main__":
    print("building binaries...")
    subprocess.run(["make", "-s", "all"], cwd=ROOT)
    ms, mk = measure_arc()
    fig_bandwidth()
    fig_cliff()
    fig_passes_law(mk)
    fig_arc(ms)
    fig_fit_map()
    fig_rugpull()
    print("done -> slides/figs/")
