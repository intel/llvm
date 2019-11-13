// RUN: %clangxx -fsycl %s -o %t1.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

//==-------------- mixed_queue.cpp - Mixed Memory test ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl::sycl;

class foo;
int main() {
  int *ptr = nullptr;
  const int N = 4;
  const int MAGIC_NUM = 42;
  const int SIZE = N * sizeof(int);
  queue q;

  ptr = (int *)malloc_device(SIZE, q);
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
