#!/usr/bin/env python3
"""Generate compile_commands.json so clangd understands the CUDA/C sources.

No CMake/Bear here (the build is plain nvcc Makefiles), so we synthesize a
compilation database with clang-compatible flags. clangd then gives real
cross-file navigation and diagnostics for .cu/.cuh and the OpenMP .c file.

Re-run after adding source files:  python3 scripts/gen_compile_commands.py
"""
import json, os, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CUDA_PATH = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9"
CUDA_INC = f"{CUDA_PATH}/targets/x86_64-linux/include"   # cuda_runtime.h, cub/
OMP_INC = "/usr/lib/gcc/x86_64-linux-gnu/11/include"     # omp.h
ARCH = "sm_89"

cuda_sources = (
    glob.glob(f"{ROOT}/demo*/*.cu")
)
c_sources = glob.glob(f"{ROOT}/demo*/*.c")

def cuda_args(path):
    d = os.path.dirname(path)
    return ["clang++", "-x", "cuda",
            f"--cuda-gpu-arch={ARCH}", f"--cuda-path={CUDA_PATH}",
            "-std=c++17", "-Wno-unknown-cuda-version",
            f"-I{CUDA_INC}", f"-I{d}", f"-I{ROOT}/demo2_sort",
            "-c", path]

def c_args(path):
    return ["clang", "-x", "c", "-std=c11", "-fopenmp",
            f"-I{OMP_INC}", "-c", path]

db = []
for p in sorted(cuda_sources):
    db.append({"directory": os.path.dirname(p), "file": p, "arguments": cuda_args(p)})
for p in sorted(c_sources):
    db.append({"directory": os.path.dirname(p), "file": p, "arguments": c_args(p)})

out = os.path.join(ROOT, "compile_commands.json")
with open(out, "w") as f:
    json.dump(db, f, indent=2)
print(f"wrote {out} ({len(db)} entries)")
