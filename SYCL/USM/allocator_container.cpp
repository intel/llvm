// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

//==------ allocator_container.cpp - USM allocator in containers test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cassert>

using namespace cl::sycl;

constexpr int N = 100;

template <usm::alloc AllocMode, class KernelName>
void runTest(device dev, context ctxt, queue q) {
  usm_allocator<int, AllocMode> alloc(ctxt, dev);

  std::vector<int, decltype(alloc)> vec(alloc);
  vec.resize(N);

  for (int i = 0; i < N; i++) {
    vec[i] = i;
  }

  int *vals = &vec[0];

  q.submit([=](handler &h) {
     h.single_task<KernelName>([=]() {
       for (int i = 1; i < N; i++) {
         vals[0] += vals[i];
       }
     });
   }).wait();

  assert(vals[0] == ((N * (N - 1)) / 2));
}

int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  if (dev.get_info<info::device::usm_shared_allocations>()) {
    runTest<usm::alloc::shared, class shared_test>(dev, ctxt, q);
  }

  if (dev.get_info<info::device::usm_host_allocations>()) {
    runTest<usm::alloc::host, class host_test>(dev, ctxt, q);
  }

  // usm::alloc::device is not supported by usm_allocator

  return 0;
}
