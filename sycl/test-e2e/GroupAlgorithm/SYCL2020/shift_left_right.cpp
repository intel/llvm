// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
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
