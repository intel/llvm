// REQUIRES: arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_lnl_m

// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=intel_gpu_bmg input.cpp -o %t.out
// RUN: %{run} %t.out
