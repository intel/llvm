///
/// Check if bfloat16 example works using fallback libraries AOT compiled for
/// GPU.
///

// REQUIRES: opencl-aot, ocloc, gpu-intel-gen12, any-device-is-gpu

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device gen12lp" %s -o %t.out
// RUN: %if gpu %{%{run} %t.out %}

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device *" %s -o %t.out
// RUN: %if gpu %{%{run} %t.out %}

#include "bfloat16_example.hpp"

int main() {
  return runTest();
}
