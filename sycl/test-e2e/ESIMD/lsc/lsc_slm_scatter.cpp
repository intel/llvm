// REQUIRES: gpu-intel-pvc || gpu-intel-dg2
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies the correctness of LSC SLM scatter intrinsics.

#include "Inputs/lsc_slm_store.hpp"

// Id - test id.
// NGroups - number of work groups.
// LocalSize - number work items in each work group.
// VL - number of offsets used in the scatter operation.
// NChannels - number of stores per each address/offset.
template <int Id, int NGroups, int LocalSize, int VL, int NChannels>
bool test_scatter(uint32_t Mask) {
  bool Passed = true;
  Passed &= test<Id, uint32_t, NGroups, LocalSize, VL, NChannels, false>(Mask);
  Passed &=
      test<Id + 1, uint64_t, NGroups, LocalSize, VL, NChannels, false>(Mask);
  return Passed;
}

int main(void) {
  bool Passed = true;

  Passed &= test_scatter<0, 1, 1, 1, 1>(1);
  Passed &= test_scatter<2, 2, 7, 4, 1>(0x5f74);
  Passed &= test_scatter<4, 2, 7, 16, 2>(0x5f7f);
  Passed &= test_scatter<6, 2, 4, 32, 2>(0x32475f7a);
  // These test_scatter<>() calls exceeds 8 register per write for uint64_t.
  // Thus call them for uint32_t only.
  //  Passed &= test_scatter<8, 2, 4, 32, 3>(0x32475f7a);
  //  Passed &= test_scatter<10, 2, 4, 32, 4>(0x32475f7a);
  Passed &= test<8, uint32_t, 2, 4, 32, 3, false>(0x32475f7a);
  Passed &= test<10, uint32_t, 2, 4, 32, 4, false>(0x32475f7a);

  Passed &=
      test<100, uint32_t, 1, 7, 4, 1, false, lsc_data_size::u16u32>(0x5f7f);
  Passed &= test<101, uint32_t, 2, 7, 32, 1, false, lsc_data_size::u16u32>(
      0x3a8b5f7f);
  Passed &=
      test<102, uint32_t, 1, 7, 4, 1, false, lsc_data_size::u8u32>(0x5f7f);
  Passed &=
      test<103, uint32_t, 2, 7, 32, 1, false, lsc_data_size::u8u32>(0x3a8b5f7f);

  std::cout << (Passed ? "Passed" : "FAILED") << std::endl;
  return Passed ? 0 : 1;
}
