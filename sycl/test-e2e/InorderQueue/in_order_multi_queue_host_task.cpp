// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//==-------- in_order_multi_queue_host_task.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>

#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

const int dataSize = 1024 * 1024;

int main() {
  queue Queue1{property::queue::in_order()};
  queue Queue2{property::queue::in_order()};

  int *dataA = malloc_host<int>(dataSize, Queue1);
  int *dataB = malloc_host<int>(dataSize, Queue1);
  int *dataC = malloc_host<int>(dataSize, Queue1);

  auto Event1 = Queue1.submit([&](handler &cgh) {
    cgh.host_task([&] {
      for (size_t i = 0; i < dataSize; ++i) {
        dataA[i] = i;
      }
    });
  });

  Queue2.submit([&](handler &cgh) {
    cgh.depends_on(Event1);
    cgh.parallel_for(range<1>(dataSize),
                     [=](id<1> idx) { dataB[idx[0]] = dataA[idx[0]]; });
  });

  Queue2.wait();

  for (size_t i = 0; i != dataSize; ++i) {
    if (dataB[i] != i) {
      std::cout << "Result mismatches " << dataB[i] << " vs expected " << i
                << " for index " << i << std::endl;
      return 1;
    }
  }

  return 0;
}
