// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

//==-------------- mixed_queue.cpp - Mixed Memory test ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

using namespace sycl;

class foo;
int main() {
  const int N = 4;
  const int MAGIC_NUM = 42;
  const int SIZE = N * sizeof(int);
  queue q;
  auto dev = q.get_device();
  if (!(dev.get_info<info::device::usm_device_allocations>() &&
        dev.get_info<info::device::usm_host_allocations>() &&
        dev.get_info<info::device::usm_shared_allocations>()))
    return 0;

  int *ptr = (int *)malloc_device(SIZE, q);
  if (ptr == nullptr) {
    return -1;
  }
  free(ptr, q);

  ptr = (int *)malloc(SIZE, q, usm::alloc::device);
  if (ptr == nullptr) {
    return -1;
  }
  free(ptr, q);

  ptr = (int *)aligned_alloc_device(alignof(int), SIZE, q);
  if (ptr == nullptr) {
    return -1;
  }
  free(ptr, q);

  ptr = (int *)aligned_alloc(alignof(int), SIZE, q, usm::alloc::device);
  if (ptr == nullptr) {
    return -1;
  }
  free(ptr, q);

  ptr = (int *)malloc_shared(SIZE, q);
  if (ptr == nullptr) {
    return -1;
  }
  free(ptr, q);

  ptr = (int *)malloc(SIZE, q, usm::alloc::shared);
  if (ptr == nullptr) {
    return -1;
  }
  free(ptr, q);

  ptr = (int *)aligned_alloc_shared(alignof(int), SIZE, q);
  if (ptr == nullptr) {
    return -1;
  }
  free(ptr, q);

  ptr = (int *)aligned_alloc(alignof(int), SIZE, q, usm::alloc::shared);
  if (ptr == nullptr) {
    return -1;
  }
  free(ptr, q);

  ptr = (int *)malloc_host(SIZE, q);
  if (ptr == nullptr) {
    return -1;
  }
  free(ptr, q);

  ptr = (int *)malloc(SIZE, q, usm::alloc::host);
  if (ptr == nullptr) {
    return -1;
  }
  free(ptr, q);

  ptr = (int *)aligned_alloc_host(alignof(int), SIZE, q);
  if (ptr == nullptr) {
    return -1;
  }
  free(ptr, q);

  ptr = (int *)aligned_alloc(alignof(int), SIZE, q, usm::alloc::host);
  if (ptr == nullptr) {
    return -1;
  }
  free(ptr, q);

  return 0;
}
