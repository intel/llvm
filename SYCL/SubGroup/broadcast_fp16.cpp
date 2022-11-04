// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==---- broadcast_fp16.cpp - SYCL sub_group broadcast test ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "broadcast.hpp"

int main() {
  queue Queue;
  if (Queue.get_device().has(sycl::aspect::fp16)) {
    check<sycl::half>(Queue);
    std::cout << "Test passed." << std::endl;
  } else {
    std::cout << "Test skipped because device doesn't support aspect::fp16"
              << std::endl;
  }
  return 0;
}
