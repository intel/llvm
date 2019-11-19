// RUN: %clangxx -fsycl %s -o %t.out -lOpenCL
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//==-------- ordered_buffs.cpp - SYCL buffers in ordered queues test--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

const int dataSize = 32;

bool isQueueInOrder(cl_command_queue cq) {
  cl_command_queue_properties reportedProps;
  cl_int iRet = clGetCommandQueueInfo(
      cq, CL_QUEUE_PROPERTIES, sizeof(reportedProps), &reportedProps, nullptr);
  assert(CL_SUCCESS == iRet && "Failed to obtain queue info from ocl device");
  return (!(reportedProps & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE));
}

int main() {
  int dataA[dataSize] = {0};
  int dataB[dataSize] = {0};

  {
    ordered_queue Queue;

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

    bool result = true;
    cl_command_queue cq = Queue.get();
    device dev = Queue.get_device();
    bool expected_result = dev.is_host() ? true : isQueueInOrder(cq);

    if (expected_result != result) {
      std::cout << "Resulting queue order is OOO but expected order is inorder"
                << std::endl;

      return -1;
    }

    auto readBufferB = bufB.get_access<access::mode::read>();
    for (size_t i = 0; i != dataSize; ++i) {
      if (readBufferB[i] != i) {
        std::cout << "Result mismatches " << readBufferB[i] << " vs expected "
                  << i << " for index " << i << std::endl;
      }
    }
  }
  return 0;
}
