// REQUIRES: opencl

// RUN: %clangxx -fsycl -D__SYCL_INTERNAL_API %s -o %t.run
// RUN: %GPU_RUN_PLACEHOLDER %t.run
// RUN: %CPU_RUN_PLACEHOLDER %t.run
// RUN: %ACC_RUN_PLACEHOLDER %t.run

//==------------------- device_platform_interop.cpp ------------------------==//
// The test for SYCL device and platform interop constructors
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <memory>
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>
#include <unordered_map>
using namespace cl::sycl;

int main() {
  default_selector device_selector;

  device D1(device_selector);
  cl_device_id cl_device;
  {
    device D2(device_selector);
    cl_device = D2.get_native<backend::opencl>();
  }
  device D3(cl_device);
  assert(D1 == D3 && "Device impls are different");

  platform P1(device_selector);
  cl_platform_id cl_platform;
  {
    platform P2(device_selector);
    cl_platform = P2.get_native<backend::opencl>();
  }
  platform P3(cl_platform);
  assert(P1 == P3 && "Platform impls are different");

  return 0;
}
