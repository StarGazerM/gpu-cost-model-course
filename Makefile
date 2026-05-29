# GPGPU mini-course: build everything. Targets sm_89 (RTX 6000 Ada).
# Override ARCH to retarget, e.g.: make ARCH="--generate-code=arch=compute_90,code=sm_90"
DEMOS := demo1_bandwidth demo2_sort demo3_rugpull

all: $(DEMOS)

demo1_bandwidth demo2_sort demo3_rugpull:
	$(MAKE) -C $@

clean:
	@for d in $(DEMOS); do $(MAKE) -C $$d clean; done
	rm -rf demo4_hwswap/sass_out profile/ncu_out profile/nsys_out

.PHONY: all clean $(DEMOS)
