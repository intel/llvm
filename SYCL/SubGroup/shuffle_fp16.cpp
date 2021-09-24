// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Missing __spirv_SubgroupShuffleINTEL, __spirv_SubgroupShuffleUpINTEL,
// __spirv_SubgroupShuffleDownINTEL, __spirv_SubgroupShuffleXorINTEL on AMD
// XFAIL: hip_amd
//
//==------- shuffle_fp16.cpp - SYCL sub_group shuffle test -----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shuffle.hpp"

int main() {
  queue Queue;
  if (Queue.get_device().is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check<half>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
