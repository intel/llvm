///
/// Checks a simple case of bfloat16, also employed for AOT library fallback.
///

// CUDA is not compatible with SPIR.
// UNSUPPORTED: cuda

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %{run} %t.out

#include "bfloat16_example.hpp"

int main() {
  return runTest();
}
