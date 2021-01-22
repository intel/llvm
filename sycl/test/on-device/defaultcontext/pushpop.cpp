// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==----------- pushpop.cpp - SYCL default context test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

int main() {
  sycl::queue q;
  auto C1 = q.get_context();
  auto Device = q.get_device();
  auto Platform = Device.get_platform();

  sycl::context NewContext(Device);
  Platform.push_default_context(NewContext);

  sycl::queue q2;
  auto C2 = q2.get_context();
  assert(C1 != C2);
  assert(C2 == NewContext);

  Platform.pop_default_context();

  sycl::queue q3;
  auto C3 = q3.get_context();
  assert(C1 == C3);
  assert(C2 != C3);

  return 0;
}
