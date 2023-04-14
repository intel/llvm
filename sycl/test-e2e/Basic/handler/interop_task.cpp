// REQUIRES: opencl, opencl_icd
// RUN: %clangxx -fsycl %s -o %t.out %opencl_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==------- interop_task.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

#include "../../helpers.hpp"
#include <chrono>
#include <iostream>
#include <thread>

int main() {
  constexpr size_t BufSize = 4;

  int data1[BufSize] = {1, 1, 1, 1};

  sycl::buffer<int, 1> DstBuf(sycl::range<1>{BufSize});
  sycl::buffer<int, 1> DstBuf2(sycl::range<1>{BufSize});

  TestQueue Queue{sycl::default_selector_v};

  Queue.submit([&](sycl::handler &CGH) {
    auto DstAcc = DstBuf.get_access<sycl::access::mode::write>(CGH);
    CGH.parallel_for<class Foo>(sycl::range<1>{BufSize},
                                [=](sycl::id<1> ID) { DstAcc[ID] = 42; });
  });

  Queue.submit([&](sycl::handler &CGH) {
    auto DstAcc = DstBuf.get_access<sycl::access::mode::read>(CGH);
    auto DstAcc2 = DstBuf2.get_access<sycl::access::mode::write>(CGH);

    CGH.interop_task([=](sycl::interop_handler ih) {
      cl_command_queue clQueue = ih.get_queue();
      cl_mem src = ih.get_mem(DstAcc);
      cl_mem dst2 = ih.get_mem(DstAcc2);
      clEnqueueCopyBuffer(clQueue, src, dst2, 0, 0, sizeof(int) * BufSize, 0,
                          nullptr, nullptr);
    });
  });

  {
    auto DstAcc = DstBuf.template get_access<sycl::access::mode::read_write>();
    const int Expected = 42;
    for (int I = 0; I < DstAcc.get_count(); ++I)
      if (DstAcc[I] != Expected) {
        std::cerr << "Mismatch. Elem " << I << ". Expected: " << Expected
                  << ", Got: " << DstAcc[I] << std::endl;
        return 1;
      }
  }

  {
    auto DstAcc2 =
        DstBuf2.template get_access<sycl::access::mode::read_write>();
    const int Expected = 42;
    for (int I = 0; I < DstAcc2.get_count(); ++I)
      if (DstAcc2[I] != Expected) {
        std::cerr << "Mismatch. Elem " << I << ". Expected: " << Expected
                  << ", Got: " << DstAcc2[I] << std::endl;
        return 1;
      }
  }

  std::cout << "Success" << std::endl;

  return 0;
}
