#!/usr/bin/env bash
# Iterate DOWN from a fully-opted merge sort that MATCHES CUB -- but written as concrete
# readable kernels (merge_cub.cu), with none of CUB's policy/dispatch template machinery.
# Start at parity (FULL == cub::DeviceMergeSort), then remove one opt at a time; each
# delta is that opt's real worth, with nothing left unexplained.
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
N=${1:-28}
run() { $NVCC -O3 -std=c++17 -arch=sm_89 "$@" -o /tmp/mc_x merge_cub.cu && /tmp/mc_x "$N" 10; }

echo "Iterate down from the unfolded-CUB merge (n=2^$N)."
echo "FULL = CUB's sm_89 opts: IPT=17, BLOCK_LOAD/STORE_WARP_TRANSPOSE, tiled merge + partition kernel."
echo
printf '%-26s ' "FULL (= CUB opts)";        run
printf '%-26s ' "- block-load coalescing";  run -DDIRECTLOAD
printf '%-26s ' "- tiled device merge";     run -DTILED=0
printf '%-26s ' "- items/thread 17->8";     run -DIPT=8
printf '%-26s ' "- items/thread 17->1";     run -DIPT=1
echo
echo "reference: cub::DeviceMergeSort ~= 47.6 ms / ~5640 Mkeys/s  (FULL matches it within harness noise)."
cat <<'EOF'

reading the ladder:
 * tiled (coalesced) device merge is DOMINANT (~2.3x) -- it is the log2(n/TILE)~17 passes;
   coalescing where the traffic is is everything.
 * items/thread 17->1 (~1.4x) -- register blocking = ILP/occupancy (the §2 latency story, in a thread).
 * block-load coalescing is SMALL -- the block sort is only 1 of ~18 passes.
Each line is a one-line -D toggle in merge_cub.cu; FULL starting at CUB parity means the
list is complete (no "what else?").
EOF
