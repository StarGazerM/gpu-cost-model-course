#!/usr/bin/env bash
# Demo 4: "GPU is a category" -- the SASS flex.
#
# This box has ONE GPU (RTX 6000 Ada, sm_89), so we cannot do a live hardware
# swap and compare achieved bandwidth/occupancy on two cards. The salvageable
# (and arguably sharper) demonstration of thesis #4 -- "the standard CUDA model
# is a fiction; the real chip exposes more" -- is a COMPILE-TIME one:
#
#   The SAME source compiles to measurably different machine code (SASS) per
#   architecture, because ptxas exploits arch-specific instructions. You write
#   min()/max(); on sm_90 (Hopper) it becomes VIMNMX (a fused vector int min-max
#   that does not exist on sm_89). Register allocation and scheduling differ too.
#
# What we CAN'T show without the hardware: live HBM bandwidth, occupancy, and
# anything using TMA / DSMEM / clusters (sm_90-only -- those would appear only in
# cross-compiled SASS, never execute here).
set -euo pipefail

SORT=../demo2_sort
OUT=sass_out
ARCHES=(89 90)
KERNELS=(v3_multiblock v4_cub)
mkdir -p "$OUT"

# Extract the opcode mnemonic from each SASS instruction line, dropping the
# /*addr*/ prefix and any @!Pn predicate.
opcodes() {
  grep -oE '/\*[0-9a-f]+\*/ +(@!?P[0-9T]+ +)?[A-Z][A-Z0-9._]+' "$1" \
    | sed -E 's#/\*[0-9a-f]+\*/ +##; s#@!?P[0-9T]+ +##' \
    | awk '{print $1}'
}

echo "==> Cross-compiling for sm_${ARCHES[0]} and sm_${ARCHES[1]} (no run, disasm only)"
for v in "${KERNELS[@]}"; do
  for a in "${ARCHES[@]}"; do
    nvcc -O3 -std=c++17 -arch=sm_$a -cubin -Xptxas -v \
         "$SORT/$v.cu" -o "$OUT/${v}_sm${a}.cubin" 2> "$OUT/${v}_sm${a}.ptxas"
    cuobjdump --dump-sass "$OUT/${v}_sm${a}.cubin" > "$OUT/${v}_sm${a}.sass"
  done
done

for v in "${KERNELS[@]}"; do
  echo
  echo "############################################################"
  echo "# $v"
  echo "############################################################"

  echo "-- per-kernel registers / shared memory (static, from ptxas) --"
  for a in "${ARCHES[@]}"; do
    echo "  [sm_$a]"
    grep -E "entry function|Used .* registers" "$OUT/${v}_sm${a}.ptxas" \
      | sed -E "s/.*entry function '([^']+)'.*/    kernel \1/; s/.*Used ([0-9]+) registers.*smem.*/      regs=\1 (has smem)/; s/.*Used ([0-9]+) registers.*/      regs=\1/" \
      | sed 's/_ZN10bitonic_v3[0-9]*//; s/EP.*$//'
  done

  echo "-- SASS size --"
  for a in "${ARCHES[@]}"; do
    n=$(opcodes "$OUT/${v}_sm${a}.sass" | wc -l)
    u=$(opcodes "$OUT/${v}_sm${a}.sass" | sort -u | wc -l)
    echo "  sm_$a: $n instructions, $u distinct opcodes"
  done

  opcodes "$OUT/${v}_sm89.sass" | sort -u > "$OUT/${v}_ops89.txt"
  opcodes "$OUT/${v}_sm90.sass" | sort -u > "$OUT/${v}_ops90.txt"
  echo "-- opcodes emitted on sm_90 but NOT sm_89 (newer arch exposes more) --"
  comm -13 "$OUT/${v}_ops89.txt" "$OUT/${v}_ops90.txt" | sed 's/^/    + /' | tr '\n' ' '; echo
  echo "-- opcodes on sm_89 but NOT sm_90 --"
  comm -23 "$OUT/${v}_ops89.txt" "$OUT/${v}_ops90.txt" | sed 's/^/    - /' | tr '\n' ' '; echo
done

echo
echo "==> SASS, ptxas logs, and opcode lists written to $OUT/"
echo "    e.g. diff <(cuobjdump --dump-sass $OUT/v3_multiblock_sm89.cubin) \\"
echo "              <(cuobjdump --dump-sass $OUT/v3_multiblock_sm90.cubin)"
