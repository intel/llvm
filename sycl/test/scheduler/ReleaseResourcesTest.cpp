// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: env SYCL_PI_TRACE=1 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=1 %ACC_RUN_PLACEHOLDER %t.out 2>&1 %ACC_CHECK_PLACEHOLDER
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

// CHECK: PI ---> RT::piContextCreate(0, DeviceIds.size(), DeviceIds.data(), 0, 0, &m_Context)
// CHECK: PI ---> RT::piQueueCreate(Context, Device, CreationFlags, &Queue)
// CHECK: PI ---> pi::piProgramCreate(Context, Data, DataLen, &Program)
// CHECK: PI ---> RT::piKernelCreate(Program, KernelName.c_str(), &Kernel)
// CHECK: PI ---> RT::piQueueRelease(m_CommandQueue)
// CHECK: PI ---> RT::piContextRelease(m_Context)
// CHECK: PI ---> RT::piKernelRelease(KernIt.second)
// CHECK: PI ---> RT::piProgramRelease(ToBeDeleted)
