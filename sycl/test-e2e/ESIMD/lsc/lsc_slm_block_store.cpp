// REQUIRES: gpu-intel-pvc
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test verifies the correctness of LSC intrinsics storing
// to SLM memory.

#include "Inputs/lsc_slm_store.hpp"

// This test verifies the correctness of LSC SLM block store intrinsics.

// Id - test id.
// NGroups - number of work groups.
// LocalSize - number work items in each work group.
// VL - number of offsets used in the gather operation.
template <int Id, int NGroups, int LocalSize, int VL> bool test_store() {
  bool Passed = true;
  Passed &= test<Id, uint32_t, NGroups, LocalSize, VL, 1, true>();
  Passed &= test<Id + 1, uint64_t, NGroups, LocalSize, VL, 1, true>();
  return Passed;
}

int main(void) {
  bool Passed = true;

  // test_store<Id, NGroups, LocalSize, VL>();
  Passed &= test_store<0, 1, 1, 4>();
  Passed &= test_store<2, 1, 7, 16>();
  Passed &= test_store<4, 4, 3, 2>();
  Passed &= test_store<6, 16, 8, 8>();
  Passed &= test_store<8, 2, 4, 32>();
  Passed &= test_store<10, 2, 4, 64>();

  std::cout << (Passed ? "Passed" : "FAILED") << std::endl;
  return Passed ? 0 : 1;
}
