#!/usr/bin/env bash
# System-level timelines for Demo 3 (Nsight Systems).
#
# The rug pull's story is NOT inside a kernel -- it's in the gaps between them.
# nsys shows cudaMalloc/cudaFree serializing on the driver (naive), then the
# stream-ordered pool (pool), then a single graph launch (graph). Open the
# .nsys-rep files in the Nsight Systems UI for the three side-by-side timelines;
# the --stats summary below already prints the CUDA API time breakdown that
# makes cudaMalloc's cost obvious without the GUI.
set -euo pipefail
cd "$(dirname "$0")"

RUG=../demo3_rugpull
SIZE=${SIZE:-20}    # log2(n); small enough that allocation/launch overhead shows
ITERS=${ITERS:-30}
OUT=nsys_out
mkdir -p "$OUT"

for h in naive pool graph; do
  bin="$RUG/${h}_harness"
  [ -x "$bin" ] || { echo "skip $h (build it first)"; continue; }
  echo "==> nsys $h  (n=2^$SIZE, $ITERS iters)"
  nsys profile --stats=true --trace=cuda,nvtx --force-overwrite=true \
       -o "$OUT/timeline_$h" "$bin" "$SIZE" "$ITERS" \
       > "$OUT/stats_$h.txt" 2>&1
  echo "   --- CUDA API summary (top by total time) ---"
  awk '/CUDA API Summary|cuda_api_sum/{f=1} f&&/cudaMalloc|cudaFree|cudaMemcpy|cudaLaunch|cudaGraph/{print "   "$0}' \
      "$OUT/stats_$h.txt" | head -8 || true
done

echo "==> .nsys-rep timelines + stats in $OUT/  (open *.nsys-rep in Nsight Systems)"
