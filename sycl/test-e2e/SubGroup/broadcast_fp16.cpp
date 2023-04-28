// REQUIRES: aspect-fp16
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
  check<sycl::half>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
