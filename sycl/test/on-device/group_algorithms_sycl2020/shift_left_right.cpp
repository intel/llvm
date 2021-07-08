// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Missing __spirv_SubgroupId, __spirv_SubgroupMaxSize, __spirv_SubgroupShuffle* on AMD:
// XFAIL: rocm_amd
//
//==------------ shift_left_right.cpp -*- C++ -*----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shift_left_right.hpp"

int main() {
  queue Queue;
  if (Queue.get_device().is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check<short>(Queue);
  check<unsigned short>(Queue);
  check<int>(Queue);
  check<int, 2>(Queue);
  check<int, 4>(Queue);
  check<int, 8>(Queue);
  check<int, 16>(Queue);
  check<unsigned int>(Queue);
  check<unsigned int, 2>(Queue);
  check<unsigned int, 4>(Queue);
  check<unsigned int, 8>(Queue);
  check<unsigned int, 16>(Queue);
  check<long>(Queue);
  check<unsigned long>(Queue);
  check<float>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
