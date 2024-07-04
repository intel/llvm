// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==--------------- reduce.cpp - SYCL sub_group reduce test ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "reduce.hpp"
#include <iostream>
int main() {
  queue Queue;
  check<class KernelName_AJprOaCZgUmsYFRTTGNw, int>(Queue);
  check<class KernelName_ShKFIYTqaI, unsigned int>(Queue);
  check<class KernelName_TovsKTk, long>(Queue);
  check<class KernelName_JqbvoN, unsigned long>(Queue);
  check<class KernelName_mAWqKSWTT, float>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
