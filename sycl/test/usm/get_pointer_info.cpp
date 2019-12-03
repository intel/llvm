// RUN: %clangxx -fsycl %s -o %t1.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out

//==----------- get_pointer_info.cpp - Pointer Query test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <tuple>

using namespace cl::sycl;

class foo;
int main() {
  int *array = nullptr;
  const int N = 4;
  const int MAGIC_NUM = 42;

  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  usm::alloc Kind;
  device D;

  array = (int *)malloc_device(N * sizeof(int), q);
  if (array == nullptr) {
    return -1;
  }
  
  std::tie(Kind, D) =  get_pointer_info(array, ctxt);
  if (ctxt.is_host()) {
    // for now, host device treats all allocations
    // as host allocations
    if (Kind != usm::alloc::host) {
      return -1;
    }
  } else {
    if (Kind != usm::alloc::device) {
      return -1;
    }
  }
  if (D != dev) {
    return -1;
  }

  free(array, ctxt);

  array = (int *)malloc_shared(N * sizeof(int), q);
  if (array == nullptr) {
    return -1;
  }

  std::tie(Kind, D) =  get_pointer_info(array, ctxt);
  if (ctxt.is_host()) {
    // for now, host device treats all allocations
    // as host allocations
    if (Kind != usm::alloc::host) {
      return -1;
    }
  } else {
    if (Kind != usm::alloc::shared) {
      return -1;
    }
  }
  if (D != dev) {
    return -1;
  }

  free(array, ctxt);

  array = (int *)malloc_host(N * sizeof(int), q);
  if (array == nullptr) {
    return -1;
  }

  std::tie(Kind, D) =  get_pointer_info(array, ctxt);
  if (Kind != usm::alloc::host) {
    return -1;
  }

  free(array, ctxt);
  
  return 0;
}
