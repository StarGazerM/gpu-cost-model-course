#!/usr/bin/env bash
# Show the GPU's EXPLICIT, SCOPED memory model vs the CPU's TRANSPARENT cache coherence,
# in code -- runtime + the ISA, where the difference is deterministic.
set -e
cd "$(dirname "$0")"
nvcc -O3 -std=c++17 -arch=sm_89 -o /tmp/coh coherence.cu
echo "===== GPU runtime ====="; /tmp/coh
echo; echo "===== (1) SCOPE is a literal instruction modifier (same fetch_add) ====="
nvcc -ptx -arch=sm_89 -std=c++17 coherence.cu -o /tmp/coh.ptx
grep -oE "atom\.add\.relaxed\.(cta|gpu)\.s32" /tmp/coh.ptx | sort | uniq -c | sed 's/^/   GPU  /'
echo "   __threadfence() -> $(grep -oE 'membar\.[a-z]+' /tmp/coh.ptx | sort -u)   (publish to L2, the coherence point)"
echo; echo "===== CPU contrast: ONE coherence domain -> one unscoped locked op ====="
g++ -O2 -std=c++17 -S coherence_cpu.cpp -o /tmp/coh_cpu.s
grep -oE "lock[[:space:]]+[a-z]+" /tmp/coh_cpu.s | sort | uniq -c | sed 's/^/   x86  /'
echo "   (no scope to pick; MESI publishes the value for free -- release/acquire only ORDERS)"
