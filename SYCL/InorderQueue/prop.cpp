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

#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

using namespace cl::sycl;

constexpr int numNodes = 4;

bool getQueueOrder(cl_command_queue cq) {
  cl_command_queue_properties reportedProps;
  cl_int iRet = clGetCommandQueueInfo(
      cq, CL_QUEUE_PROPERTIES, sizeof(reportedProps), &reportedProps, nullptr);
  assert(CL_SUCCESS == iRet && "Failed to obtain queue info from ocl device");
  return (reportedProps & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? false
                                                                  : true;
}

int CheckQueueOrder(const queue &q) {
  auto dev = q.get_device();

  cl_command_queue cq = get_native<backend::opencl>(q);
  bool expected_result = dev.is_host() ? true : getQueueOrder(cq);
  if (!expected_result)
    return -1;

  expected_result = dev.is_host() ? true : q.is_in_order();
  if (!expected_result)
    return -2;

  return 0;
}

int main() {
  queue q1{property::queue::in_order()};
  int res = CheckQueueOrder(q1);
  if (res != 0)
    return res;

  device dev{cl::sycl::default_selector{}};
  context ctx{dev};

  auto exception_handler = [](cl::sycl::exception_list exceptions) {};

  queue q2{ctx, dev, exception_handler, {sycl::property::queue::in_order()}};

  res = CheckQueueOrder(q2);
  if (res != 0)
    return res;

  return 0;
}
