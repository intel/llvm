// RUN: %{build} -fsycl-dead-args-optimization -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s
//
// XFAIL: hip_nvidia

//==------------------- ReleaseResourcesTests.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include "../helpers.hpp"

using sycl_access_mode = sycl::access::mode;

int main() {
  bool Failed = false;

  // Checks creating of the second host accessor while first one is alive.
  try {
    sycl::range<1> BufSize{1};
    sycl::buffer<int, 1> Buf(BufSize);

    TestQueue Queue(sycl::default_selector_v);

    Queue.submit([&](sycl::handler &CGH) {
      auto BufAcc = Buf.get_access<sycl_access_mode::read_write>(CGH);
      CGH.parallel_for<class init_a>(BufSize,
                                     [=](sycl::id<1> Id) { (void)BufAcc[Id]; });
    });

    auto BufHostAcc = Buf.get_host_access();

    Queue.wait_and_throw();

  } catch (...) {
    std::cerr << "ReleaseResources test failed." << std::endl;
    Failed = true;
  }

  return Failed;
}

// CHECK:---> piContextCreate
// CHECK:---> piextQueueCreate
// CHECK:---> piProgramCreate
// CHECK:---> piKernelCreate
// CHECK:---> piQueueRelease
// CHECK:---> piContextRelease
// CHECK:---> piKernelRelease
// CHECK:---> piProgramRelease
