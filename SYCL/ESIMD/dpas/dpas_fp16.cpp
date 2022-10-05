//==---------------- dpas_fp16.cpp  - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || gpu-intel-dg2 || esimd_emulator
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_kernel %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test verifies DPAS support for float16.

#include "dpas_common.hpp"

int main(int argc, const char *argv[]) {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<info::device::name>() << std::endl;

  bool Print = argc > 1 && std::string(argv[1]) == "-debug";
  bool Passed = true;

  constexpr bool LetDeduceArgs = true;
  Passed &= tests<8, 8, fp16, fp16, LetDeduceArgs>(Q, Print);
  Passed &= tests<8, 4, fp16, fp16, LetDeduceArgs>(Q, Print);
  Passed &= tests<8, 1, fp16, fp16, LetDeduceArgs>(Q, Print);

  Passed &= tests<8, 5, fp16, fp16, LetDeduceArgs>(Q, Print);
  Passed &= tests<8, 3, fp16, fp16, LetDeduceArgs>(Q, Print);

  std::cout << (Passed ? "Test Passed\n" : "Test FAILED\n");
  return Passed ? 0 : 1;
}
