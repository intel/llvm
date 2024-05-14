// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//==-------- ordered_buffs.cpp - SYCL buffers in ordered queues test--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>

#include <sycl/detail/core.hpp>

#include <sycl/properties/all_properties.hpp>

using namespace sycl;

const int dataSize = 32;

int main() {
  int dataA[dataSize] = {0};
  int dataB[dataSize] = {0};

  {
    queue Queue{property::queue::in_order()};

    // Purpose of this test is to create a dependency between two kernels
    // RAW dependency
    // which requires the use of ordered queue.
    buffer<int, 1> bufA(dataA, range<1>(dataSize));
    buffer<int, 1> bufB(dataB, range<1>(dataSize));
    Queue.submit([&](handler &cgh) {
      auto writeBuffer = bufA.get_access<access::mode::write>(cgh);

      // Create a range.
      auto myRange = range<1>(dataSize);

      // Create a kernel.
      auto myKernel = ([=](id<1> idx) { writeBuffer[idx] = idx[0]; });

      cgh.parallel_for<class ordered_writer>(myRange, myKernel);
    });

    Queue.submit([&](handler &cgh) {
      auto writeBuffer = bufB.get_access<access::mode::write>(cgh);
      auto readBuffer = bufA.get_access<access::mode::read>(cgh);

      // Create a range.
      auto myRange = range<1>(dataSize);

      // Create a kernel.
      auto myKernel = ([=](id<1> idx) { writeBuffer[idx] = readBuffer[idx]; });

      cgh.parallel_for<class ordered_reader>(myRange, myKernel);
    });

    auto readBufferB = bufB.get_host_access();
    for (size_t i = 0; i != dataSize; ++i) {
      if (readBufferB[i] != i) {
        std::cout << "Result mismatches " << readBufferB[i] << " vs expected "
                  << i << " for index " << i << std::endl;
      }
    }
  }
  return 0;
}
