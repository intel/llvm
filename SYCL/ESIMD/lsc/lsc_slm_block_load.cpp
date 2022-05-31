// REQUIRES: gpu-intel-pvc
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test verifies the correctness of LSC intrinsics loading
// from SLM memory.

#include "Inputs/lsc_slm_load.hpp"

// This test verifies the correctness of LSC SLM block load intrinsics.

// Id - test id.
// NGroups - number of work groups.
// LocalSize - number work items in each work group.
// VL - number of offsets used in the gather operation.
template <int Id, int NGroups, int LocalSize, int VL> bool test_load() {
  bool Passed = true;
  Passed &= test<Id, uint32_t, NGroups, LocalSize, VL, 1, true>();
  Passed &= test<Id + 1, uint64_t, NGroups, LocalSize, VL, 1, true>();
  return Passed;
}

int main() {
  bool Passed = true;

  // test_load<Id, NGroups, LocalSize, VL>();
  Passed &= test_load<0, 1, 1, 4>();
  Passed &= test_load<2, 1, 7, 16>();
  Passed &= test_load<4, 4, 7, 16>();
  Passed &= test_load<6, 16, 8, 8>();
  Passed &= test_load<8, 2, 4, 32>();
  Passed &= test_load<10, 2, 4, 64>();

  std::cout << (Passed ? "Passed" : "FAILED") << std::endl;
  return Passed ? 0 : 1;
}
