///
/// Check if bfloat16 example works using fallback libraries AOT compiled for
/// GPU.
///

// REQUIRES: opencl-aot, ocloc, gpu-intel-dg2, any-device-is-gpu

// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2 %s -o %t.out
// RUN: %if gpu %{%{run} %t.out %}

// RUN: %clangxx -fsycl %{gpu_aot_opts} %s -o %t.out
// RUN: %if gpu %{%{run} %t.out %}

#include "bfloat16_example.hpp"

int main() {
  return runTest();
}
