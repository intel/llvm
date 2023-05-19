// REQUIRES: gpu-intel-pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

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
  Passed &= test<Id + 2, uint8_t, NGroups, LocalSize, VL, 1, true>();
  Passed &= test<Id + 3, uint16_t, NGroups, LocalSize, VL, 1, true>();
  Passed &= test<Id + 4, float, NGroups, LocalSize, VL, 1, true>();
  Passed &= test<Id + 5, double, NGroups, LocalSize, VL, 1, true>();
  return Passed;
}

int main(void) {
  bool Passed = true;

  // test_store<Id, NGroups, LocalSize, VL>();
  Passed &= test_store<0, 1, 1, 4>();
  Passed &= test_store<6, 1, 7, 16>();
  Passed &= test_store<12, 4, 3, 2>();
  Passed &= test_store<18, 16, 8, 8>();
  Passed &= test_store<24, 2, 4, 32>();
  Passed &= test_store<30, 2, 4, 64>();

  std::cout << (Passed ? "Passed" : "FAILED") << std::endl;
  return Passed ? 0 : 1;
}
