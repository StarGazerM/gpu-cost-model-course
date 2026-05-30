#!/usr/bin/env python3
"""Parse Nsight Compute output into the per-kernel slide table + appendix figure.

Outputs (into ../slides/figs):
  sort_kernel_metrics.csv   per-(version,kernel) table parsed from ncu_out/*.csv
  fig_demo2_kernels.png     DRAM %-of-peak and achieved occupancy per kernel

Run `./run_ncu.sh` first to populate ncu_out/. The main slide figures
(bandwidth, arc, passes-law, fit map, rug pull) are in slides/make_figures.py.
"""
import os, re, subprocess, glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "slides", "figs")
NCU = os.path.join(HERE, "ncu_out")
os.makedirs(FIGS, exist_ok=True)

ARC = ["v0_naive", "v1_shared", "v2_shuffle", "v3_multiblock", "v4_cub"]


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


# Main slide figures (bandwidth, arc, passes-law, fit map, rug pull) live in
# slides/make_figures.py. This script owns only the ncu-derived CSV and the
# per-kernel appendix figure below.
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
    fig_demo2_kernels(df)
    print("done -> slides/figs/ (run make_figures.py for the main slide figures)")
