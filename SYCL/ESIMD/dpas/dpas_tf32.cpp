//==---------------- dpas_bf16.cpp  - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || esimd_emulator
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_kernel %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// TODO: implement support for tfloat32 in esimd_emulator
// XFAIL: esimd_emulator

// This test verifies DPAS support for tfloat32.

#include "dpas_common.hpp"

int main(int argc, const char *argv[]) {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<info::device::name>() << std::endl;

  bool Print = argc > 1 && std::string(argv[1]) == "-debug";
  bool Passed = true;

  constexpr bool LetDeduceArgs = true;
  constexpr bool ExecSize16Only = true;
  Passed &= tests<8, 8, tf32, tf32, LetDeduceArgs, ExecSize16Only>(Q, Print);
  Passed &= tests<8, 4, tf32, tf32, LetDeduceArgs, ExecSize16Only>(Q, Print);
  Passed &= tests<8, 1, tf32, tf32, LetDeduceArgs, ExecSize16Only>(Q, Print);

  Passed &= tests<8, 5, tf32, tf32, LetDeduceArgs, ExecSize16Only>(Q, Print);
  Passed &= tests<8, 3, tf32, tf32, LetDeduceArgs, ExecSize16Only>(Q, Print);

  std::cout << (Passed ? "Test Passed\n" : "Test FAILED\n");
  return Passed ? 0 : 1;
}
