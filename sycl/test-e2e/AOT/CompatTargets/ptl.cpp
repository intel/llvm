// REQUIRES: arch-intel_gpu_ptl_u || arch-intel_gpu_ptl_h

// RUN: %{build} -fsycl-targets=intel_gpu_ptl %S/../Inputs/simple.cpp -o %t.out
// RUN: %{run} %t.out
