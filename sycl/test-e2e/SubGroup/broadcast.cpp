// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==--------- broadcast.cpp - SYCL sub_group broadcast test ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "broadcast.hpp"

int main() {
  queue Queue;
  check<int>(Queue);
  check<unsigned int>(Queue);
  check<long>(Queue);
  check<unsigned long>(Queue);
  check<float>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
