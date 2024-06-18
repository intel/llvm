// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// SYCL in ordered queues explicit USM test.
// Simple test checking explicit USM functionality using a Queue with the
// in_order property.
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

int main() {

  {
    const int dataSize = 32;
    const size_t numBytes = static_cast<size_t>(dataSize) * sizeof(int);

    int dataA[dataSize] = {0};
    int dataB[dataSize] = {0};

    queue Queue{property::queue::in_order()};

    auto devicePtrA = malloc_device<int>(numBytes, Queue);
    Queue.memcpy(devicePtrA, &dataA, numBytes);

    Queue.submit([&](handler &cgh) {
      auto myRange = range<1>(dataSize);
      auto myKernel = ([=](id<1> idx) { devicePtrA[idx] = idx[0]; });

      cgh.parallel_for<class ordered_writer>(myRange, myKernel);
    });

    auto devicePtrB = malloc_device<int>(numBytes, Queue);
    Queue.memcpy(devicePtrB, &dataB, numBytes);

    Queue.submit([&](handler &cgh) {
      auto myRange = range<1>(dataSize);
      auto myKernel = ([=](id<1> idx) { devicePtrB[idx] = devicePtrA[idx]; });

      cgh.parallel_for<class ordered_reader>(myRange, myKernel);
    });

    Queue.memcpy(&dataB, devicePtrB, numBytes);

    Queue.wait();

    auto ctxt = Queue.get_context();
    free(devicePtrA, ctxt);
    free(devicePtrB, ctxt);

    for (int i = 0; i != dataSize; ++i) {
      if (dataB[i] != i) {
        std::cout << "Result mismatches " << dataB[i] << " vs expected " << i
                  << " for index " << i << std::endl;
      }
    }
  }
  return 0;
}
