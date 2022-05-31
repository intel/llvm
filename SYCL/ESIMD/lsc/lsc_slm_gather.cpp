// REQUIRES: gpu-intel-pvc
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test verifies the correctness of LSC SLM gather intrinsics.

#include "Inputs/lsc_slm_load.hpp"

// Id - test id.
// NGroups - number of work groups.
// LocalSize - number work items in each work group.
// VL - number of offsets used in the gather operation.
// NChannels - number of loads per each address/offset.
template <int TestId, int NGroups, int LocalSize, int VL, int NChannels>
bool test_gather(uint32_t Mask) {
  bool Passed = true;
  Passed &=
      test<TestId, uint32_t, NGroups, LocalSize, VL, NChannels, false>(Mask);
  Passed &=
      test<TestId + 1, uint64_t, NGroups, LocalSize, VL, NChannels, false>(
          Mask);
  return Passed;
}

int main() {
  bool Passed = true;

  Passed &= test_gather<0, 1, 1, 1, 1>(1);
  Passed &= test_gather<2, 2, 7, 4, 1>(0x5f74);
  Passed &= test_gather<4, 2, 7, 16, 2>(0x5f7f);
  Passed &= test_gather<6, 2, 4, 32, 2>(0x32475f7a);
  // These test_gather<>() calls exceeds 8 register per read for uint64_t.
  // Thus call them for uint32_t only.
  //  Passed &= test_gather<8, 2, 4, 32, 3>(0x32475f7a);
  //  Passed &= test_gather<10, 2, 4, 32, 4>(0x32475f7a);
  Passed &= test<8, uint32_t, 2, 4, 32, 3, false>(0x32475f7a);
  Passed &= test<10, uint32_t, 2, 4, 32, 4, false>(0x32475f7a);

  Passed &=
      test<100, uint32_t, 1, 2, 16, 1, false, lsc_data_size::u16u32>(0x5f7f);
  Passed &= test<101, uint32_t, 2, 7, 32, 1, false, lsc_data_size::u16u32>(
      0x3a8b5f7f);

  Passed &=
      test<200, uint32_t, 1, 2, 16, 1, false, lsc_data_size::u8u32>(0x5f7f);
  Passed &=
      test<201, uint32_t, 2, 7, 32, 1, false, lsc_data_size::u8u32>(0x3a8b5f7f);

  std::cout << (Passed ? "Passed" : "FAILED") << std::endl;
  return Passed ? 0 : 1;
}
