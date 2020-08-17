// TODO: Enable compilation w/o -fno-sycl-early-optimizations option.
// See https://github.com/intel/llvm/issues/2264 for more details.

// UNSUPPORTED: cuda
// CUDA compilation and runtime do not yet support sub-groups.
//
// RUN: %clangxx -fsycl -fno-sycl-early-optimizations -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
//==------------ shuffle_fp16.cpp - SYCL sub_group shuffle test -----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--_____--------------------------------------------------------------------===//

#include "shuffle.hpp"

int main() {
  queue Queue;
  if (!Queue.get_device().has_extension("cl_intel_subgroups")) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check<half>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
