// RUN: %clangxx -fsycl %s -o %t1.out -lOpenCL
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

//==----------- ordered_dmemll.cpp - Device Memory Linked List test --------==//
// It uses an ordered queue where explicit waiting is not necessary between
// kernels
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl::sycl;

constexpr int numNodes = 4;

bool getQueueOrder(cl_command_queue cq) {
  cl_command_queue_properties reportedProps;
  cl_int iRet = clGetCommandQueueInfo(
      cq, CL_QUEUE_PROPERTIES, sizeof(reportedProps), &reportedProps, nullptr);
  assert(CL_SUCCESS == iRet && "Failed to obtain queue info from ocl device");
  return (reportedProps & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? true
                                                                  : false;
}

int main() {
  queue q{property::queue::out_of_order()};
  auto dev = q.get_device();
  
  cl_command_queue cq = q.get(); 
  bool expected_result = dev.is_host() ? true : getQueueOrder(cq);
  if (!expected_result) {
    std::cout << "Resulting queue order is in order but expected order is OOO"
              << std::endl;

    return -1;
  }

  return 0;
}
