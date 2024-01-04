// REQUIRES: level_zero
// UNSUPPORTED: ze_debug
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=-1 UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s
//
//==--- level-zero-static-link-flow.cpp.cpp - Check L0 static link flow --==//
//
// Run a simple program that uses online linking and verify that the sequence
// of calls to the plugin and to the Level Zero driver are consistent with the
// "static linking" implementation.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

// The key thing we check here is that the call to "zeModuleCreate" does not
// happen from "piProgramCompile".  Instead, we expect it to be delayed and
// called from "piProgramLink".
//
// CHECK: ---> piProgramCreate
// CHECK: ---> piProgramCompile
// CHECK-NOT: ZE ---> zeModuleCreate
// CHECK: ---> piProgramLink
// CHECK: ZE ---> zeModuleCreate

#include <sycl/sycl.hpp>

class MyKernel;

void test() {
  sycl::queue Queue;
  sycl::context Context = Queue.get_context();
  sycl::device Device = Queue.get_device();

  auto BundleInput =
      sycl::get_kernel_bundle<MyKernel, sycl::bundle_state::input>(Context,
                                                                   {Device});
  auto BundleObject = sycl::compile(BundleInput);
  sycl::link(BundleObject);

  // We need to define the kernel function, but we don't actually need to
  // submit it to the device.
  if (false) {
    Queue.submit([&](sycl::handler &CGH) { CGH.single_task<MyKernel>([=] {}); })
        .wait();
  }
}

int main() {
  test();

  return 0;
}
