// UNSUPPORTED: cuda
// CUDA compilation and runtime do not yet support sub-groups.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
//==------------ shuffle.cpp - SYCL sub_group shuffle test -----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shuffle.hpp"

int main() {
  queue Queue;
  if (!Queue.get_device().has_extension("cl_intel_subgroups")) {
    std::cout << "Skipping test\n";
    return 0;
  }
  if (Queue.get_device().has_extension("cl_intel_subgroups_short")) {
    check<short>(Queue);
    check<unsigned short>(Queue);
  }
  check<int>(Queue);
  check<int, 2>(Queue);
  check<int, 4>(Queue);
  check<int, 8>(Queue);
  check<int, 16>(Queue);
  check<unsigned int>(Queue);
  check<unsigned int, 2>(Queue);
  check<unsigned int, 4>(Queue);
  check<unsigned int, 8>(Queue);
  check<unsigned int, 16>(Queue);
  check<long>(Queue);
  check<unsigned long>(Queue);
  check<float>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
