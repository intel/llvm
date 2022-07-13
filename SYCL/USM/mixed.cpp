// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

//==------------------- mixed.cpp - Mixed Memory test ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

using namespace cl::sycl;

class foo;
int main() {
  const int N = 4;
  const int MAGIC_NUM = 42;

  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  if (!(dev.get_info<info::device::usm_device_allocations>() &&
        dev.get_info<info::device::usm_shared_allocations>() &&
        dev.get_info<info::device::usm_host_allocations>()))
    return 0;

  int *darray = (int *)malloc_device(N * sizeof(int), dev, ctxt);
  if (darray == nullptr) {
    return -1;
  }
  int *sarray = (int *)malloc_shared(N * sizeof(int), dev, ctxt);

  if (sarray == nullptr) {
    return -1;
  }

  int *harray = (int *)malloc_host(N * sizeof(int), ctxt);
  if (harray == nullptr) {
    return -1;
  }
  for (int i = 0; i < N; i++) {
    sarray[i] = MAGIC_NUM - 1;
    harray[i] = 1;
  }

  auto e0 = q.memset(darray, 0, N * sizeof(int));
  e0.wait();

  auto e1 = q.submit([=](handler &cgh) {
    cgh.single_task<class foo>([=]() {
      for (int i = 0; i < N; i++) {
        sarray[i] += darray[i] + harray[i];
      }
    });
  });

  e1.wait();

  for (int i = 0; i < N; i++) {
    if (sarray[i] != MAGIC_NUM) {
      return -2;
    }
  }
  free(darray, ctxt);
  free(sarray, ctxt);
  free(harray, ctxt);

  return 0;
}
