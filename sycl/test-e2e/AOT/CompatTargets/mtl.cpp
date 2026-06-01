// REQUIRES: arch-intel_gpu_mtl_u || arch-intel_gpu_mtl_h

// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=intel_gpu_mtl input.cpp -o %t.out
// RUN: %{run} %t.out
