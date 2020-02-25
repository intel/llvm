// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -I %sycl_source_dir %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: env SYCL_PI_TRACE=1 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=1 %ACC_RUN_PLACEHOLDER %t.out 2>&1 %ACC_CHECK_PLACEHOLDER

// TODO: error: expected string not found in input
// TODO: PI ---> pi::piProgramCreate(Context, Data, DataLen, &Program)
// XFAIL: cuda

//==------------------- ReleaseResourcesTests.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include "../helpers.hpp"

using namespace cl;
using sycl_access_mode = cl::sycl::access::mode;

int main() {
  bool Failed = false;

  // Checks creating of the second host accessor while first one is alive.
  try {
    cl::sycl::default_selector device_selector;

    sycl::range<1> BufSize{1};
    sycl::buffer<int, 1> Buf(BufSize);

    TestQueue Queue(device_selector);

    Queue.submit([&](sycl::handler &CGH) {
      auto BufAcc = Buf.get_access<sycl_access_mode::read_write>(CGH);
      CGH.parallel_for<class init_a>(
          BufSize, [=](sycl::id<1> Id) { (void)BufAcc[Id]; });
    });

    auto BufHostAcc = Buf.get_access<sycl_access_mode::read>();

    Queue.wait_and_throw();

  } catch (...) {
    std::cerr << "ReleaseResources test failed." << std::endl;
    Failed = true;
  }

  return Failed;
}

// CHECK:---> piContextCreate
// CHECK:---> piQueueCreate
// CHECK:---> piProgramCreate
// CHECK:---> piKernelCreate
// CHECK:---> piQueueRelease
// CHECK:---> piContextRelease
// CHECK:---> piKernelRelease
// CHECK:---> piProgramRelease
