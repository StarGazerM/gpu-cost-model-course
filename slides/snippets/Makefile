# Compile-check the slide snippets so the examples are verified correct.
# These are teaching excerpts (no timing/verify/error-check boilerplate), not
# the production kernels -- those live in demo2_sort/.
NVCC    ?= nvcc
NVFLAGS ?= -O2 -std=c++17 --generate-code=arch=compute_89,code=sm_89

SRCS := $(wildcard s*.cu)
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
