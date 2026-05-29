#!/usr/bin/env bash
# Per-kernel profiling for the sort arc (Nsight Compute).
#
# Two modes:
#   (default)  Targeted metrics -> CSV per version (fast; a few launches each).
#              These CSVs feed extract_metrics.py for the slide tables/figures.
#   FULL=1     Full section set -> .ncu-rep per version for interactive opening
#              in the Nsight Compute UI (the spec's --set full command). Slower.
#
# Profiles at 2^SIZE (default 26) so kernels see real HBM traffic, not the L2.
set -euo pipefail
cd "$(dirname "$0")"

SORT=../demo2_sort
SIZE=${SIZE:-26}
OUT=ncu_out
mkdir -p "$OUT"

# Headline metrics for the slide tables.
METRICS=$(tr -d ' \n' <<'EOF'
gpu__time_duration.sum,
dram__bytes.sum.per_second,
dram__throughput.avg.pct_of_peak_sustained_elapsed,
sm__throughput.avg.pct_of_peak_sustained_elapsed,
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,
sm__warps_active.avg.pct_of_peak_sustained_active,
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
EOF
)

# version binary -> regex of kernels to profile (a few launches of each).
declare -A KRE=(
  [v0_naive]="bitonic_stage"
  [v1_shared]="global_stage|local_merge|local_sort"
  [v2_shuffle]="global_stage|local_merge|local_sort"
  [v3_multiblock]="global_stage|local_merge|local_sort"
  [v4_cub]="Onesweep|Histogram|ExclusiveSum|SingleTile"
)

for v in v0_naive v1_shared v2_shuffle v3_multiblock v4_cub; do
  bin="$SORT/$v"
  [ -x "$bin" ] || { echo "skip $v (build it first)"; continue; }
  echo "==> ncu $v (kernels: ${KRE[$v]})"
  if [ "${FULL:-0}" = "1" ]; then
    ncu --set full --launch-count 2 --kernel-name "regex:${KRE[$v]}" \
        -f -o "$OUT/$v" "$bin" "$SIZE" 1 >/dev/null
  else
    ncu --csv --launch-count 3 --kernel-name "regex:${KRE[$v]}" \
        --metrics "$METRICS" "$bin" "$SIZE" 1 \
        2>/dev/null | grep -E '^"' > "$OUT/$v.csv"
  fi
done

echo "==> wrote $OUT/*.csv  (run: python3 extract_metrics.py)"
