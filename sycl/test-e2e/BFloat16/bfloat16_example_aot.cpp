///
/// Check if bfloat16 example works using fallback libraries AOT compiled for
/// both GPU and CPU.
///

// REQUIRES: opencl-aot, ocloc, any-device-is-cpu, (!gpu-intel-gen12)

// RUN: %clangxx -fsycl -fsycl-targets=spir64 %s -o %t.out
// RUN: %{run} %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen -Xsycl-target-backend=spir64_gen "-device dg2" %s -o %t.out
// RUN: %{run} %t.out

// CPU AOT targets host isa, so we compile on the run system instead.
// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen "-device dg2" %s -o %t.out
// RUN: %{run} %t.out

#include "bfloat16_example.hpp"

int main() {
  return runTest();
}
