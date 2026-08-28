// REQUIRES: linux, gpu, level_zero

// Check that invalid ZE_AFFINITY_MASK values do not crash the loader.

// RUN: not not --crash %{run-unfiltered-devices} env ZE_AFFINITY_MASK=a sycl-ls
// RUN: not not --crash %{run-unfiltered-devices} env ZE_AFFINITY_MASK= sycl-ls
