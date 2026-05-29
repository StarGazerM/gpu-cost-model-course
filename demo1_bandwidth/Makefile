# Demo 1: bandwidth identity. Targets sm_89 (RTX 6000 Ada).
NVCC    ?= nvcc
CC      ?= gcc
ARCH    ?= --generate-code=arch=compute_89,code=sm_89
NVFLAGS ?= -O3 -lineinfo -std=c++17 $(ARCH)
CFLAGS  ?= -O3 -fopenmp -march=native

all: stream_gpu stream_cpu

stream_gpu: stream_gpu.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

stream_cpu: stream_cpu.c
	$(CC) $(CFLAGS) -o $@ $<

# Pin STREAM to one NUMA node (6 cores / 4 channels on this box).
run-cpu: stream_cpu
	OMP_PROC_BIND=close OMP_PLACES=cores numactl --cpunodebind=0 --membind=0 ./stream_cpu

run-gpu: stream_gpu
	./stream_gpu

clean:
	rm -f stream_gpu stream_cpu

.PHONY: all run-cpu run-gpu clean
