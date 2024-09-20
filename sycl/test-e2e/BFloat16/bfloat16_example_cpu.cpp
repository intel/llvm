///
/// Check if bfloat16 example works using fallback libraries AOT compiled for
/// CPU.
///

// REQUIRES: opencl-aot, ocloc, gpu-intel-gen12, cpu

// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc" %s -o %t.out
// RUN: %{run} %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc" %s -o %t.out
// RUN: %{run} %t.out

#include "bfloat16_example.hpp"

int main() {
  return runTest();
}
