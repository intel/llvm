// UNSUPPORTED: hip_amd
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Missing __spirv_SubgroupShuffleINTEL, __spirv_SubgroupShuffleUpINTEL,
// __spirv_SubgroupShuffleDownINTEL, __spirv_SubgroupShuffleXorINTEL on AMD
//
//==------------ shuffle.cpp - SYCL sub_group shuffle test -----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shuffle.hpp"
#include <iostream>

int main() {
  queue Queue;
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
  check<long, 2>(Queue);
  check<long, 4>(Queue);
  check<long, 8>(Queue);
  check<long, 16>(Queue);
  check<unsigned long>(Queue);
  check<unsigned long, 2>(Queue);
  check<unsigned long, 4>(Queue);
  check<unsigned long, 8>(Queue);
  check<unsigned long, 16>(Queue);
  check<float>(Queue);
  check<float, 2>(Queue);
  check<float, 4>(Queue);
  check<float, 8>(Queue);
  check<float, 16>(Queue);

  // Check long long and unsigned long long because they differ from
  // long and unsigned long according to C++ rules even if they have the same
  // size at some system.
  check<long long>(Queue);
  check<long long, 16>(Queue);
  check<unsigned long long>(Queue);
  check<unsigned long long, 16>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
