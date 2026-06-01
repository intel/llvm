// REQUIRES: arch-intel_gpu_ptl_u || arch-intel_gpu_ptl_h

// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=intel_gpu_ptl input.cpp -o %t.out
// RUN: %{run} %t.out
