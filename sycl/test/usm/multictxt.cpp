// REQUIRES: cpu,gpu
// RUN: %clangxx -fsycl %s -o %t1.out -lOpenCL
// RUN: %t1.out

//==----------------- multictxt.cpp - Multi Context USM test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl::sycl;

// The multictxt test here is a sanity check that USM selects the right
// implementation when presented with multiple contexts. The extra context
// only needs to exist for this test to do its job.

void GpuCpuCpu() {
  queue gpu_q(gpu_selector{});
  queue cpu_q(cpu_selector{});
  device dev = cpu_q.get_device();
  context ctx = cpu_q.get_context();

  void *ptr = malloc_shared(128, dev, ctx);

  free(ptr, ctx);
}

void CpuGpuGpu() {
  queue cpu_q(cpu_selector{});
  queue gpu_q(gpu_selector{});
  device dev = gpu_q.get_device();
  context ctx = gpu_q.get_context();

  void *ptr = malloc_shared(128, dev, ctx);

  free(ptr, ctx);
}

void GpuCpuGpu() {
  queue gpu_q(gpu_selector{});
  queue cpu_q(cpu_selector{});
  device dev = gpu_q.get_device();
  context ctx = gpu_q.get_context();

  void *ptr = malloc_shared(128, dev, ctx);

  free(ptr, ctx);
}

int main() {
  GpuCpuCpu();
  CpuGpuGpu();
  GpuCpuGpu();
  
  return 0;
}
