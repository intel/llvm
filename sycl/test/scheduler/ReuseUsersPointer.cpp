// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: env SYCL_PI_TRACE=1 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=1 %ACC_RUN_PLACEHOLDER %t.out 2>&1 %ACC_CHECK_PLACEHOLDER
//==--------------------- ReuseUsersPointer.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

#include <vector>

#include "../helpers.hpp"

using namespace cl;
using sycl_access_mode = cl::sycl::access::mode;

int main() {
  {
    TestQueue Queue1(sycl::default_selector{});
    TestQueue Queue2(sycl::default_selector{});

    std::vector<int> Data(1);
    std::cout << "User pointer = " << std::hex << (void *)Data.data()
              << std::endl;
    sycl::buffer<int, 1> Buf(Data.data(), Data.size(),
                             {sycl::property::buffer::use_host_ptr()});

    Queue1.submit([&](sycl::handler &CGH) {
      auto BufAcc = Buf.get_access<sycl_access_mode::read_write>(CGH);
      CGH.single_task<class first_kernel>([=]() { BufAcc[0] = 41; });
    });

    Queue2.submit([&](sycl::handler &CGH) {
      auto BufAcc = Buf.get_access<sycl_access_mode::read_write>(CGH);
      CGH.single_task<class second_kernel>([=]() { BufAcc[0] = 42; });
    });
  }

  return 0;
}
// CHECK: User pointer = [[USER_PTR:0x.*]]
// CHECK: ---> RT::piMemBufferCreate(
// CHECK-NEXT: {{.+}}
// CHECK-NEXT: {{.+}}
// CHECK-NEXT: {{.+}}
// CHECK-NEXT: void *: [[USER_PTR]]
//
// CHECK: ---> RT::piMemBufferCreate(
// CHECK-NEXT: {{.+}}
// CHECK-NEXT: {{.+}}
// CHECK-NEXT: {{.+}}
// CHECK-NEXT: void *: 0
