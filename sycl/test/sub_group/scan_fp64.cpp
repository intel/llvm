// UNSUPPORTED: cuda || cpu
// CUDA compilation and runtime do not yet support sub-groups.
// #2252 Disable until all variants of built-ins are available in OpenCL CPU
// runtime for every supported ISA
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==--------------- scan_fp64.cpp - SYCL sub_group scan test --------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------------===//

#include "scan.hpp"

int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check<double>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
