# Demo 5: histogram cost-model arc. Targets sm_89 (RTX 6000 Ada).
NVCC    ?= nvcc
ARCH    ?= --generate-code=arch=compute_89,code=sm_89
NVFLAGS ?= -O3 -lineinfo -std=c++17 $(ARCH)

SRCS := $(wildcard v*.cu)
BINS := $(SRCS:.cu=)

LOG2N ?= 28
ITERS ?= 20
ZIPF  ?= 0

all: $(BINS)

%: %.cu common.cuh
	$(NVCC) $(NVFLAGS) -o $@ $<

# Run the whole arc at one (size, skew). e.g. make run ZIPF=1.0
run: $(BINS)
	@for b in $(BINS); do ./$$b $(LOG2N) $(ITERS) $(ZIPF); done

clean:
	rm -f $(BINS)

.PHONY: all run clean
