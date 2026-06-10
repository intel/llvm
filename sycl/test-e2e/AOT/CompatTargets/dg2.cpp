// REQUIRES: arch-intel_gpu_acm_g10 || arch-intel_gpu_acm_g11 || arch-intel_gpu_acm_g12

// RUN: %{build} -fsycl-targets=intel_gpu_dg2 %S/../Inputs/simple.cpp -o %t.out
// RUN: %{run} %t.out
