# Demo 3: the rug pull. Targets sm_89 (RTX 6000 Ada).
NVCC    ?= nvcc
ARCH    ?= --generate-code=arch=compute_89,code=sm_89
NVFLAGS ?= -O3 -lineinfo -std=c++17 $(ARCH)

DEPS := ../demo2_sort/common.cuh ../demo2_sort/bitonic_v3.cuh
SRCS := $(wildcard *_harness.cu)
BINS := $(SRCS:.cu=)

LOG2N ?= 20
ITERS ?= 200

all: $(BINS)

%: %.cu $(DEPS)
	$(NVCC) $(NVFLAGS) -o $@ $<

run: $(BINS)
	@for b in $(BINS); do ./$$b $(LOG2N) $(ITERS); echo; done

clean:
	rm -f $(BINS)

.PHONY: all run clean
