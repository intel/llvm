/// Checks a simple case of bfloat16

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: linux && level_zero && arch-intel_gpu_mtl_u
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/23094

#include "bfloat16_example.hpp"

int main() {
  return runTest();
}
