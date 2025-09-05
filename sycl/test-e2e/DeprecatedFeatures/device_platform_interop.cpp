// REQUIRES: opencl

// RUN: %{build} -D__SYCL_INTERNAL_API -o %t.run
// RUN: %{run} %t.run

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
#include <sycl/backend.hpp>
#include <sycl/backend/opencl.hpp>
#include <sycl/detail/core.hpp>
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
