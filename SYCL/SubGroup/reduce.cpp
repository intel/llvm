// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

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
  if (!core_sg_supported(Queue.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check<class KernelName_AJprOaCZgUmsYFRTTGNw, int>(Queue);
  check<class KernelName_ShKFIYTqaI, unsigned int>(Queue);
  check<class KernelName_TovsKTk, long>(Queue);
  check<class KernelName_JqbvoN, unsigned long>(Queue);
  check<class KernelName_mAWqKSWTT, float>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
