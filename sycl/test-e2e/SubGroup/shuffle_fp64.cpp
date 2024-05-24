// REQUIRES: aspect-fp64
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
//==------- shuffle_fp64.cpp - SYCL sub_group shuffle test -----*- C++ -*---==//
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
  check<double>(Queue);
  check<double, 2>(Queue);
  check<double, 4>(Queue);
  check<double, 8>(Queue);
  check<double, 16>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
