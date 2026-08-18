// REQUIRES: arch-intel_gpu_mtl_u || arch-intel_gpu_mtl_h

// RUN: %{build} -fsycl-targets=intel_gpu_mtl %S/../Inputs/simple.cpp -o %t.out
// RUN: %{run} %t.out
