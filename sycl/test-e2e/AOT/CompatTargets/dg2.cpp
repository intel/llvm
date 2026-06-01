// REQUIRES: arch-intel_gpu_dg2_g10 || arch-intel_gpu_dg2_g11 || arch-intel_gpu_dg2_g12

// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=intel_gpu_dg2 input.cpp -o %t.out
// RUN: %{run} %t.out
