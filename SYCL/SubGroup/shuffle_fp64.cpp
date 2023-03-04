// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Temporarily disabled for CUDA due to failure reported in
// https://github.com/intel/llvm/issues/8516
// UNSUPPORTED: cuda
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
  if (Queue.get_device().has(sycl::aspect::fp64)) {
    check<double>(Queue);
    check<double, 2>(Queue);
    check<double, 4>(Queue);
    check<double, 8>(Queue);
    check<double, 16>(Queue);
    std::cout << "Test passed." << std::endl;
  } else {
    std::cout << "Test skipped because device doesn't support aspect::fp64"
              << std::endl;
  }
  return 0;
}
