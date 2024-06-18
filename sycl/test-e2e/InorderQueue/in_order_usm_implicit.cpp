// REQUIRES: cuda
//
// Currently only CUDA is supported: it would be necessary to generalize
// mem_advice for other devices before adding support.
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// SYCL in ordered queues implicit USM test.
// Simple test checking implicit USM functionality using a Queue with the
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
    queue Queue{property::queue::in_order()};

    const int mem_advice = PI_MEM_ADVICE_CUDA_SET_READ_MOSTLY;

    const int dataSize = 32;
    const size_t numBytes = static_cast<size_t>(dataSize) * sizeof(int);

    auto dataA = malloc_shared<int>(numBytes, Queue);
    auto dataB = malloc_shared<int>(numBytes, Queue);

    for (int i = 0; i < dataSize; i++) {
      dataA[i] = i;
      dataB[i] = 0;
    }

    Queue.mem_advise(dataA, numBytes, mem_advice);

    Queue.submit([&](handler &cgh) {
      auto myRange = range<1>(dataSize);
      auto myKernel = ([=](id<1> idx) { dataB[idx] = dataA[idx]; });

      cgh.parallel_for<class ordered_reader>(myRange, myKernel);
    });

    Queue.wait();

    for (int i = 0; i != dataSize; ++i) {
      if (dataB[i] != i) {
        std::cout << "Result mismatches " << dataB[i] << " vs expected " << i
                  << " for index " << i << std::endl;
      }
    }

    auto ctxt = Queue.get_context();
    free(dataA, ctxt);
    free(dataB, ctxt);
  }
  return 0;
}
