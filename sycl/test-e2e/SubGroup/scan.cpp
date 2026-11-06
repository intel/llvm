// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==--------------- scan.cpp - SYCL sub_group scan test --------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "scan.hpp"
#include <iostream>

int main() {
  queue Queue;
  check<class KernelName_QTbNYAsEmawQ, int>(Queue);
  check<class KernelName_FQFNSdcVGrCLUbn, unsigned int>(Queue);
  check<class KernelName_kWYnyHJx, long>(Queue);
  check<class KernelName_qmL, unsigned long>(Queue);
  check<class KernelName_BckYc, float>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
