// REQUIRES: arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_lnl_m

// RUN: %{build} -fsycl-targets=intel_gpu_bmg %S/../Inputs/simple.cpp -o %t.out
// RUN: %{run} %t.out
