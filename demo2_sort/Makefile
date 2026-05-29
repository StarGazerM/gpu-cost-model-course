# Demo 2: sort optimization arc. Targets sm_89 (RTX 6000 Ada).
NVCC    ?= nvcc
ARCH    ?= --generate-code=arch=compute_89,code=sm_89
NVFLAGS ?= -O3 -lineinfo -std=c++17 $(ARCH)

SRCS := $(wildcard v*.cu) $(wildcard *_check.cu)
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu common.cuh
	$(NVCC) $(NVFLAGS) -o $@ $<

# Run every built version on the same input for an at-a-glance comparison.
run: $(BINS)
	@for b in $(BINS); do ./$$b $(LOG2N) $(ITERS); done

clean:
	rm -f $(BINS)

.PHONY: all run clean
