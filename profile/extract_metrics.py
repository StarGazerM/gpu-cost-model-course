#!/usr/bin/env python3
"""Turn profiler output + benchmark runs into slide CSVs and figures.

Outputs (into ../slides/figs):
  sort_kernel_metrics.csv   per-(version,kernel) table parsed from ncu_out/*.csv
  fig_demo1_bandwidth.png   CPU STREAM vs GPU copy (run demo1 binaries)
  fig_demo2_arc.png         v0..v4 runtime + speedup (run demo2 binaries)
  fig_demo2_kernels.png     DRAM %-of-peak and achieved occupancy per kernel

Run `./run_ncu.sh` first to populate ncu_out/. The figures that come from
running binaries work even without ncu data.
"""
import os, re, subprocess, glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "slides", "figs")
NCU = os.path.join(HERE, "ncu_out")
DEMO1 = os.path.join(HERE, "..", "demo1_bandwidth")
DEMO2 = os.path.join(HERE, "..", "demo2_sort")
os.makedirs(FIGS, exist_ok=True)

ARC = ["v0_naive", "v1_shared", "v2_shuffle", "v3_multiblock", "v4_cub"]
SHORT = {"v0_naive": "v0\nnaive", "v1_shared": "v1\nshared",
         "v2_shuffle": "v2\nshuffle", "v3_multiblock": "v3\nbigtile",
         "v4_cub": "v4\nCUB"}


def num(s):
    return float(str(s).replace(",", ""))


# ---------- 1. parse ncu CSVs ----------
def parse_ncu():
    rows = []
    for path in sorted(glob.glob(os.path.join(NCU, "*.csv"))):
        version = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "Metric Name" not in df.columns:
            continue
        df["kernel"] = df["Kernel Name"].str.replace(r"\(.*", "", regex=True)
        df["val"] = df["Metric Value"].map(num)
        # average across the profiled launches of each kernel
        piv = (df.groupby(["kernel", "Metric Name"])["val"].mean()
                 .unstack("Metric Name").reset_index())
        piv.insert(0, "version", version)
        rows.append(piv)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    ren = {
        "gpu__time_duration.sum": "dur_ns",
        "dram__bytes.sum.per_second": "dram_bps",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed": "l1_pct",
        "sm__warps_active.avg.pct_of_peak_sustained_active": "occ_pct",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "stall_global",
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": "stall_shared",
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "stall_barrier",
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio": "stall_mio",
    }
    out = out.rename(columns=ren)
    if "dur_ns" in out:
        out["dur_ms"] = out["dur_ns"] / 1e6
    if "dram_bps" in out:
        out["dram_gbps"] = out["dram_bps"] / 1e9
    out["version"] = pd.Categorical(out["version"], ARC, ordered=True)
    out = out.sort_values(["version", "kernel"])
    csv = os.path.join(FIGS, "sort_kernel_metrics.csv")
    out.to_csv(csv, index=False)
    print("wrote", csv, f"({len(out)} kernel rows)")
    return out


# ---------- 2. run binaries for headline numbers ----------
def run_capture(binpath, args, pattern):
    if not os.path.exists(binpath):
        return None
    try:
        out = subprocess.run([binpath, *args], capture_output=True, text=True,
                             timeout=120).stdout
    except Exception:
        return None
    m = re.search(pattern, out)
    return float(m.group(1)) if m else None


def fig_demo1():
    gpu = run_capture(os.path.join(DEMO1, "stream_gpu"), [],
                      r"copy kernel:\s*([\d.]+)\s*GB/s")
    cpu_bin = os.path.join(DEMO1, "stream_cpu")
    cpu = None
    if os.path.exists(cpu_bin):
        try:
            env = dict(os.environ, OMP_NUM_THREADS="6", OMP_PROC_BIND="close",
                       OMP_PLACES="cores")
            cmd = (["numactl", "--cpunodebind=0", "--membind=0", cpu_bin]
                   if subprocess.run(["which", "numactl"],
                                     capture_output=True).returncode == 0
                   else [cpu_bin])
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 env=env, timeout=120).stdout
            m = re.search(r"triad bandwidth:\s*([\d.]+)\s*GB/s", out)
            cpu = float(m.group(1)) if m else None
        except Exception:
            pass
    if gpu is None and cpu is None:
        print("demo1: no binaries; skipping figure")
        return
    labels, vals = [], []
    if cpu: labels.append("CPU STREAM Triad\n(1 NUMA node, DDR4)"); vals.append(cpu)
    if gpu: labels.append("GPU copy kernel\n(GDDR6)"); vals.append(gpu)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    bars = ax.bar(labels, vals, color=["#888", "#76b900"][:len(vals)])
    ax.set_ylabel("Achieved bandwidth (GB/s)")
    ax.set_title("Bandwidth identity: GPU consumes its bus")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.0f}", ha="center",
                va="bottom")
    if cpu and gpu:
        ax.text(0.5, 0.92, f"{gpu/cpu:.0f}x", transform=ax.transAxes,
                ha="center", fontsize=14, color="#76b900", weight="bold")
    fig.tight_layout()
    p = os.path.join(FIGS, "fig_demo1_bandwidth.png")
    fig.savefig(p, dpi=140); print("wrote", p)


def fig_demo2_arc(size="26"):
    runtimes = {}
    for v in ARC:
        ms = run_capture(os.path.join(DEMO2, v), [size, "10"],
                         r"([\d.]+)\s*ms")
        if ms: runtimes[v] = ms
    if not runtimes:
        print("demo2 arc: no binaries; skipping figure")
        return
    vs = [v for v in ARC if v in runtimes]
    ms = [runtimes[v] for v in vs]
    base = runtimes.get("v0_naive", ms[0])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar([SHORT[v] for v in vs], ms, color="#76b900")
    ax.set_ylabel("Kernel-only runtime (ms, log)")
    ax.set_yscale("log")
    ax.set_title(f"Sort optimization arc (n=2^{size}, 67M int32)")
    for b, v in zip(bars, vs):
        ax.text(b.get_x()+b.get_width()/2, runtimes[v],
                f"{runtimes[v]:.1f} ms\n{base/runtimes[v]:.1f}x",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIGS, "fig_demo2_arc.png")
    fig.savefig(p, dpi=140); print("wrote", p)


def fig_demo2_kernels(df):
    if df.empty or "dram_pct" not in df:
        print("demo2 kernels: no ncu data; skipping figure")
        return
    d = df[df["version"] != "v4_cub"].copy()
    d["label"] = d["version"].astype(str).str.replace("_", "\n") + "\n" + d["kernel"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.bar(d["label"], d["dram_pct"], color="#76b900")
    a1.set_ylabel("DRAM throughput (% of peak)")
    a1.set_title("Achieved HBM/GDDR6 utilization per kernel")
    a1.tick_params(axis="x", labelrotation=90, labelsize=7)
    if "occ_pct" in d:
        a2.bar(d["label"], d["occ_pct"], color="#888")
        a2.set_ylabel("Achieved occupancy (%)")
        a2.set_title("Occupancy per kernel")
        a2.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    p = os.path.join(FIGS, "fig_demo2_kernels.png")
    fig.savefig(p, dpi=140); print("wrote", p)


if __name__ == "__main__":
    df = parse_ncu()
    fig_demo1()
    fig_demo2_arc()
    fig_demo2_kernels(df)
    print("done -> slides/figs/")
