// REQUIRES: cpu,gpu
// RUN: %clangxx -fsycl %s -o %t1.out -lOpenCL
// RUN: %t1.out

//==----------------- multictxtcpu.cpp - Multi Context USM CPU test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  queue gpu_q(gpu_selector{});
  queue cpu_q(cpu_selector{});
  device dev = cpu_q.get_device();
  context ctx = cpu_q.get_context();
  
  void *ptr = malloc_shared(128, dev, ctx);
  
  free(ptr, ctx);
  
  return 0;
}
