//==---------------- dpas_int.cpp  - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc || gpu-intel-dg2
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// This test verifies DPAS support for 2,4,8-bit integers.

#include "dpas_common.hpp"

int main(int argc, const char *argv[]) {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<info::device::name>() << std::endl;

  bool Print = argc > 1 && std::string(argv[1]) == "-debug";
  bool Passed = true;

  constexpr bool LetDeduceArgs = true;

  // Test unsigned 2-bit integers.
  Passed &= tests<8, 8, u2, u2>(Q, Print);
  Passed &= tests<8, 4, u2, u2>(Q, Print);
  Passed &= tests<8, 3, u2, u2>(Q, Print);
  Passed &= tests<8, 1, u2, u2>(Q, Print);

  // Test signed 2-bit integers.
  Passed &= tests<8, 8, s2, s2>(Q, Print);
  Passed &= tests<8, 5, s2, s2>(Q, Print);
  Passed &= tests<8, 2, s2, s2>(Q, Print);
  Passed &= tests<8, 1, s2, s2>(Q, Print);

  // Test the mix of signed and unsigned 2-bit integers.
  Passed &= tests<8, 1, u2, s2>(Q, Print);
  Passed &= tests<8, 1, s2, u2>(Q, Print);

  // Test couple combinations with 4-bit integers.
  Passed &= tests<8, 8, s4, s4>(Q, Print);
  Passed &= tests<8, 4, u4, s4>(Q, Print);

  // Test couple combinations with 8-bit integers.
  Passed &= tests<8, 8, s8, s8>(Q, Print);
  Passed &= tests<8, 2, u8, s8, LetDeduceArgs>(Q, Print);

  // Test some mixes of 2/4/8-bit integers.
  Passed &= tests<8, 8, s2, s4>(Q, Print);
  Passed &= tests<8, 1, s2, s8>(Q, Print);
  Passed &= tests<8, 4, s8, s4>(Q, Print);

  std::cout << (Passed ? "Test Passed\n" : "Test FAILED\n");
  return Passed ? 0 : 1;
}
