///
/// Check if bfloat16 example works using fallback libraries AOT compiled for
/// both GPU and CPU.
///

// REQUIRES: opencl-aot, ocloc, any-device-is-cpu, (!gpu-intel-gen12), target-spir

// RUN: %clangxx -fsycl -fsycl-targets=spir64 %s -o %t.out
// RUN: %{run} %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64,%{gpu_aot_target} %s -o %t.out
// RUN: %{run} %t.out

// CPU AOT targets host isa, so we compile on the run system instead.
// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64,%{gpu_aot_target} %s -o %t.out
// RUN: %{run} %t.out

#include "bfloat16_example.hpp"

int main() {
  return runTest();
}
