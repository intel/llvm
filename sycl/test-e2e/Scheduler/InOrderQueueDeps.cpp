// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s
//
// XFAIL: hip_nvidia

// The tested functionality is disabled with Level Zero until it is supported by
// the plugin.
// UNSUPPORTED: level_zero
//==----------------------- InOrderQueueDeps.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>

class KernelA;

void submitKernel(sycl::queue &Queue, sycl::buffer<int, 1> &Buf) {
  Queue.submit([&](sycl::handler &Cgh) {
    auto BufAcc = Buf.get_access<sycl::access::mode::read_write>(Cgh);
    Cgh.single_task<KernelA>([=]() { (void)BufAcc[0]; });
  });
}

int main() {
  int val;
  sycl::buffer<int, 1> Buf{&val, sycl::range<1>(1)};

  sycl::device Dev(sycl::default_selector_v);
  sycl::context Ctx{Dev};

  sycl::queue InOrderQueueA{Ctx, Dev, sycl::property::queue::in_order()};
  sycl::queue InOrderQueueB{Ctx, Dev, sycl::property::queue::in_order()};

  // Sequential submissions to the same in-order queue should not result in any
  // event dependencies.
  // CHECK: piEnqueueKernelLaunch
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: : 0
  submitKernel(InOrderQueueA, Buf);
  // CHECK: piEnqueueKernelLaunch
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: : 0
  submitKernel(InOrderQueueA, Buf);
  // Submisssion to a different in-order queue should explicitly depend on the
  // previous command group.
  // CHECK: piEnqueueKernelLaunch
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: : 1
  submitKernel(InOrderQueueB, Buf);

  return 0;
}
