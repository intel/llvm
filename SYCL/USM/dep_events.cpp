//==---- dep_events.cpp - USM dependency test ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Prefetch operations for cuda and windows currently does not work
// cuMemPrefetchAsync returns cudaErrorInvalidDevice for this OS
// Test is temporarily disabled until this is resolved
// UNSUPPORTED: cuda && windows
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

#include <CL/sycl.hpp>

using namespace sycl;

int main() {
  queue q;

  int *x = malloc_shared<int>(1, q);
  int *y = malloc_shared<int>(1, q);
  int *z = malloc_shared<int>(1, q);

  event eMemset1 = q.memset(x, 0, sizeof(int), event{});              // x = 0
  event eMemset2 = q.memset(y, 0, sizeof(int), std::vector<event>{}); // y = 0
  event eFill = q.fill(x, 1, 1, {eMemset1, eMemset2});                // x = 1
  event eNoOpMemset = q.memset(x, 0, 0, eFill);       // 0 count, so x remains 1
  event eNoOpMemcpy = q.memcpy(x, y, 0, eNoOpMemset); // 0 count, so x remains 1
  event eNoOpCopy = q.copy(y, x, 0, eNoOpMemcpy);     // 0 count, so x remains 1
  event eMemcpy = q.memcpy(y, x, sizeof(int), eNoOpCopy);             // y = 1
  event eCopy = q.copy(y, z, 1, eMemcpy);                             // z = 1
  event ePrefetch = q.prefetch(z, sizeof(int), eCopy);                //
  q.single_task<class kernel>(ePrefetch, [=] { *z *= 2; }).wait();    // z = 2

  int error = (*z != 2) ? 1 : 0;
  std::cout << (error ? "failed\n" : "passed\n");

  free(x, q);
  free(y, q);
  free(z, q);

  return error;
}
