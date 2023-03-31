// REQUIRES: opencl_icd,opencl
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out %opencl_lib
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out
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

#include <iostream>
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr int numNodes = 4;

bool getQueueOrder(cl_command_queue cq) {
  cl_command_queue_properties reportedProps;
  cl_int iRet = clGetCommandQueueInfo(
      cq, CL_QUEUE_PROPERTIES, sizeof(reportedProps), &reportedProps, nullptr);
  assert(CL_SUCCESS == iRet && "Failed to obtain queue info from ocl device");
  return (reportedProps & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? false
                                                                  : true;
}

int main() {
  queue q{property::queue::in_order()};
  auto dev = q.get_device();

  bool result = true;
  cl_command_queue cq = sycl::get_native<backend::opencl>(q);
  bool expected_result = getQueueOrder(cq);
  if (expected_result != result) {
    std::cout << "Resulting queue order is OOO but expected order is inorder"
              << std::endl;

    return -1;
  }

  return 0;
}
