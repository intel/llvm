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
using namespace sycl;

int main() {

  device D1(default_selector_v);
  cl_device_id cl_device;
  {
    device D2(default_selector_v);
    cl_device = get_native<backend::opencl>(D2);
  }
  device D3(cl_device);
  assert(D1 == D3 && "Device impls are different");

  platform P1(default_selector_v);
  cl_platform_id cl_platform;
  {
    platform P2(default_selector_v);
    cl_platform = get_native<backend::opencl>(P2);
  }
  platform P3(cl_platform);
  assert(P1 == P3 && "Platform impls are different");

  return 0;
}
