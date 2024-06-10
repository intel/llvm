// REQUIRES: opencl_icd,opencl
// RUN: %{build} -o %t.out %opencl_lib
// RUN: %{run} %t.out
//==-------- ordered_buffs.cpp - SYCL buffers in ordered queues test--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <sycl/backend.hpp>
#include <sycl/backend/opencl.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;

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
    queue Queue{property::queue::in_order()};

    bool result = true;
    cl_command_queue cq = get_native<backend::opencl>(Queue);
    device dev = Queue.get_device();
    bool expected_result = isQueueInOrder(cq);

    if (expected_result != result) {
      std::cout << "Resulting queue order is OOO but expected order is inorder"
                << std::endl;

      return -1;
    }
  }

  return 0;
}
