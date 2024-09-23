// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==---------- buffer_dev_to_dev.cpp - SYCL buffer basic test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <sycl/detail/core.hpp>

using namespace sycl;

constexpr int size = 10;

int main() {
  int Data[size] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  {
    buffer<int, 1> Buffer(Data, range<1>(size),
                          {property::buffer::use_host_ptr()});

    device Device(default_selector_v);

    context FirstContext(Device);
    context SecondContext(Device);

    queue FirstQueue(FirstContext, Device);
    queue SecondQueue(SecondContext, Device);

    assert(FirstQueue.get_context() != SecondQueue.get_context());
    FirstQueue.submit([&](handler &Cgh) {
      accessor Accessor{Buffer, Cgh};
      Cgh.parallel_for<class init_b>(range<1>{size},
                                     [=](id<1> Index) { Accessor[Index] = 0; });
    });
    SecondQueue.submit([&](handler &Cgh) {
      accessor Accessor{Buffer, Cgh};
      Cgh.parallel_for<class increment_b>(
          range<1>{size}, [=](id<1> Index) { Accessor[Index] += 1; });
    });
  } // Data is copied back
  for (int I = 0; I < size; I++) {
    assert(Data[I] == 1);
  }

  return 0;
}
