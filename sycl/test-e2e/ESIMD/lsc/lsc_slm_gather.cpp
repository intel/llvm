// REQUIRES: arch-intel_gpu_pvc || gpu-intel-dg2
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies the correctness of LSC SLM gather intrinsics.

#include "Inputs/lsc_slm_load.hpp"

template <typename T, bool CheckMerging> bool test_gather(queue Q) {
  constexpr bool Transpose = true;

  bool Passed = true;
  Passed &= test<T, 1, 1, 1, 1, !Transpose, CheckMerging>(Q, 1);
  Passed &= test<T, 2, 7, 4, 1, !Transpose, CheckMerging>(Q, 0x5f74);
  Passed &= test<T, 2, 7, 16, 2, !Transpose, CheckMerging>(Q, 0x5f7f);
  Passed &= test<T, 2, 4, 32, 2, !Transpose, CheckMerging>(Q, 0x32475f7a);

  // Do not exceed 8 registers per read (i.e. 512 bytes).
  if constexpr (sizeof(T) * 32 * 3 <= 512)
    Passed &= test<T, 2, 4, 32, 3, !Transpose, CheckMerging>(Q, 0x32475f7a);
  if constexpr (sizeof(T) * 32 * 4 <= 512)
    Passed &= test<T, 2, 4, 32, 4, !Transpose, CheckMerging>(Q, 0x32475f7a);

  if constexpr (std::is_integral_v<T> && sizeof(T) == 4) {
    Passed &=
        test<T, 1, 2, 16, 1, !Transpose, CheckMerging, lsc_data_size::u16u32>(
            Q, 0x5f7f);
    Passed &=
        test<T, 2, 7, 32, 1, !Transpose, CheckMerging, lsc_data_size::u16u32>(
            Q, 0x3a8b5f7f);

    Passed &=
        test<T, 1, 2, 16, 1, !Transpose, CheckMerging, lsc_data_size::u8u32>(
            Q, 0x5f7f);
    Passed &=
        test<T, 2, 7, 32, 1, !Transpose, CheckMerging, lsc_data_size::u8u32>(
            Q, 0x3a8b5f7f);
  }

  return Passed;
}

int main() {
  constexpr bool CheckMerging = true;

  auto Q = queue{gpu_selector_v};
  std::cout << "Running lsc_slm_gather() tests on "
            << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

  bool Passed = true;
  Passed &= test_gather<uint32_t, !CheckMerging>(Q);
  Passed &= test_gather<uint32_t, CheckMerging>(Q);

  Passed &= test_gather<uint64_t, !CheckMerging>(Q);
  Passed &= test_gather<uint64_t, CheckMerging>(Q);

  Passed &= test_gather<float, !CheckMerging>(Q);
  Passed &= test_gather<float, CheckMerging>(Q);
  if (Q.get_device().has(sycl::aspect::fp64)) {
    Passed &= test_gather<double, !CheckMerging>(Q);
    Passed &= test_gather<double, CheckMerging>(Q);
  }

  Passed &= test_gather<sycl::ext::intel::experimental::esimd::tfloat32,
                        !CheckMerging>(Q);
  Passed &= test_gather<sycl::ext::intel::experimental::esimd::tfloat32,
                        CheckMerging>(Q);

  std::cout << (Passed ? "Passed" : "FAILED") << std::endl;
  return Passed ? 0 : 1;
}
