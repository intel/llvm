// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==---------- buffer_dev_to_dev.cpp - SYCL buffer basic test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cassert>
#include <memory>

using namespace cl::sycl;

int main() {
  int Data[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  {
    buffer<int, 1> Buffer(Data, range<1>(10),
                          {property::buffer::use_host_ptr()});

    default_selector Selector;

    queue FirstQueue(Selector);
    queue SecondQueue(Selector);

    assert(FirstQueue.get_context() != SecondQueue.get_context());
    FirstQueue.submit([&](handler &Cgh) {
      auto Accessor = Buffer.get_access<access::mode::read_write>(Cgh);
      Cgh.parallel_for<class init_b>(
          range<1>{10}, [=](id<1> Index) { Accessor[Index] = 0; });
    });
    SecondQueue.submit([&](handler &Cgh) {
      auto Accessor = Buffer.get_access<access::mode::read_write>(Cgh);
      Cgh.parallel_for<class increment_b>(
          range<1>{10}, [=](id<1> Index) { Accessor[Index] += 1; });
    });
  } // Data is copied back
  for (int I = 0; I < 10; I++) {
    assert(Data[I] == 1);
  }

  return 0;
}