// RUN: %{build} -Wno-error=unused-command-line-argument -fsycl-dead-args-optimization -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s %if !windows %{--check-prefix=CHECK-RELEASE%}

//==------------------- ReleaseResourcesTests.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>

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

// CHECK: <--- urContextCreate
// CHECK: <--- urQueueCreate
// CHECK: <--- urProgramCreate
// CHECK: <--- urKernelCreate

// On Windows, dlls unloading is inconsistent and if we try to release these UR
// objects manually, inconsistent hangs happen due to a race between unloading
// the UR adapters dlls (in addition to their dependency dlls) and the releasing
// of these UR objects. So, we currently shutdown without releasing them and
// windows should handle the memory cleanup.

// CHECK-RELEASE: <--- urQueueRelease
// CHECK-RELEASE: <--- urContextRelease
// CHECK-RELEASE: <--- urKernelRelease
// CHECK-RELEASE: <--- urProgramRelease
