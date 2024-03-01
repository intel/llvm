// REQUIRES: gpu-intel-pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies the correctness of LSC SLM block load intrinsics.

#include "Inputs/lsc_slm_load.hpp"

template <typename T, bool TestMerging> bool test_load(queue Q) {
  constexpr bool Transpose = true;
  constexpr int VS = 1;

  bool Passed = true;
  // test<type, NGroups, LocalSize, VL, VS, Transpose, TestMerging>(Q);
  Passed &= test<T, 1, 1, 4, VS, Transpose, TestMerging>(Q);
  Passed &= test<T, 1, 7, 16, VS, Transpose, TestMerging>(Q);
  Passed &= test<T, 4, 7, 16, VS, Transpose, TestMerging>(Q);
  Passed &= test<T, 16, 8, 8, VS, Transpose, TestMerging>(Q);
  Passed &= test<T, 2, 4, 32, VS, Transpose, TestMerging>(Q);
  Passed &= test<T, 2, 4, 64, VS, Transpose, TestMerging>(Q);

  Passed &= test<T, 1, 1, 4, VS, !Transpose, TestMerging>(Q);
  Passed &= test<T, 1, 7, 16, VS, !Transpose, TestMerging>(Q);
  Passed &= test<T, 4, 7, 16, VS, !Transpose, TestMerging>(Q);
  Passed &= test<T, 16, 8, 8, VS, !Transpose, TestMerging>(Q);
  Passed &= test<T, 2, 4, 32, VS, !Transpose, TestMerging>(Q);
  Passed &= test<T, 2, 4, 64, VS, !Transpose, TestMerging>(Q);
  return Passed;
}

int main() {
  bool Passed = true;

  constexpr bool TestMerging = true;

  auto Q = queue{gpu_selector_v};
  std::cout << "Running lsc_slm_gather() tests on "
            << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

  Passed &= test_load<uint32_t, !TestMerging>(Q);
  Passed &= test_load<uint64_t, !TestMerging>(Q);

  Passed &= test_load<uint32_t, TestMerging>(Q);
  Passed &= test_load<uint64_t, TestMerging>(Q);

  Passed &= test_load<uint16_t, !TestMerging>(Q);
  Passed &= test_load<uint8_t, !TestMerging>(Q);

  Passed &= test_load<uint16_t, TestMerging>(Q);
  Passed &= test_load<uint8_t, TestMerging>(Q);

  Passed &= test_load<float, !TestMerging>(Q);
  Passed &= test_load<double, !TestMerging>(Q);

  Passed &= test_load<float, TestMerging>(Q);
  Passed &= test_load<double, TestMerging>(Q);

  std::cout << (Passed ? "Passed" : "FAILED") << std::endl;
  return Passed ? 0 : 1;
}
