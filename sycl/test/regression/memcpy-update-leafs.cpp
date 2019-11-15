// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_PI_TRACE=1 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER

 //==----------- memcpy-update-leafs.cpp - SYCL regression test  -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

int main() {
  try {
    cl::sycl::queue Queue;

    cl::sycl::buffer<int, 1> a_buffer(cl::sycl::range<1>(1));

    {
      auto a_data = a_buffer.get_access<cl::sycl::access::mode::read_write>();
      a_data[0] = -1;
    }

    for (int i = 0; i < 2; i++) {
      Queue.submit([&](cl::sycl::handler &CGH) {
        auto acc = a_buffer.get_access<cl::sycl::access::mode::write>(CGH);
        CGH.single_task<class dummy>([=] () {
          acc[0] = 1;
        });
      });
      Queue.wait_and_throw();
    }

  } catch (cl::sycl::exception &E) {
    std::cout << E.what();
  }
  return 0;
}

// CHECK: ---> RT::piEnqueueMemBufferWrite
// CHECK-NOT: ---> RT::piEnqueueMemBufferWrite
// CHECK-NOT: ---> RT::piEnqueueMemBufferMap
// CHECK-NOT: ---> RT::piEnqueueMemUnmap
