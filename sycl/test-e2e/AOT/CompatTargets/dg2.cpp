// REQUIRES: arch-intel_gpu_dg2_g10 || arch-intel_gpu_dg2_g11 || arch-intel_gpu_dg2_g12

// RUN: %{build} -fsycl-targets=intel_gpu_ptl %S/../Inputs/simple.cpp -o %t.out
// RUN: %{run} %t.out
