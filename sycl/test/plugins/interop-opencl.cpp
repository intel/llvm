// REQUIRES: opencl

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -I %sycl_source_dir %s -o %t.out
// RUN: env SYCL_BE=PI_OPENCL %CPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_BE=PI_OPENCL %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_BE=PI_OPENCL %ACC_RUN_PLACEHOLDER %t.out

//==-- interop-opencl.cpp - SYCL test for OpenCL interop API --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/opencl.h>
#include <CL/sycl.hpp>
#include <CL/sycl/backend/opencl.hpp>

using namespace cl::sycl;

int main() {
  queue Queue{};
  auto Context = Queue.get_info<info::queue::context>();
  auto Device = Queue.get_info<info::queue::device>();
  auto Platform = Device.get_info<info::device::platform>();

  // Get native OpenCL handles
  auto ocl_platform = Platform.get_native<backend::opencl>();
  auto ocl_device = Device.get_native<backend::opencl>();
  auto ocl_context = Context.get_native<backend::opencl>();
  auto ocl_queue = Queue.get_native<backend::opencl>();

  // Re-create SYCL objects from native OpenCL handles
  auto PlatformInterop = opencl::make<platform>(ocl_platform);
  auto DeviceInterop = opencl::make<device>(ocl_device);
  auto ContextInterop = opencl::make<context>(ocl_context);
  auto QueueInterop = opencl::make<queue>(ContextInterop, ocl_queue);

  // Check native handles
  assert(ocl_platform == PlatformInterop.get_native<backend::opencl>());
  assert(ocl_device == DeviceInterop.get_native<backend::opencl>());
  assert(ocl_context == ContextInterop.get_native<backend::opencl>());
  assert(ocl_queue == QueueInterop.get_native<backend::opencl>());

  return 0;
}
