// REQUIRES: opencl

// RUN: %clangxx -fsycl -DSYCL2020_CONFORMANT_APIS -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl %CPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl %ACC_RUN_PLACEHOLDER %t.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl %CPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl %ACC_RUN_PLACEHOLDER %t.out

//==-- interop-opencl.cpp - SYCL test for OpenCL interop API --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/opencl.h>
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

using namespace cl::sycl;

int main() {
  queue Queue{};
  auto Context = Queue.get_info<info::queue::context>();
  auto Device = Queue.get_info<info::queue::device>();
  auto Platform = Device.get_info<info::device::platform>();

  int Data[1] = {0};
  sycl::buffer<int, 1> Buffer(&Data[0], sycl::range<1>(1));
  {
    Queue.submit([&](sycl::handler &cgh) {
      auto Acc = Buffer.get_access<sycl::access::mode::read_write>(cgh);
      cgh.host_task([=](const sycl::interop_handle &ih) {
        (void)Acc;
        auto BufNative = ih.get_native_mem<sycl::backend::opencl>(Acc);
#ifdef SYCL2020_CONFORMANT_APIS
        assert(BufNative.size() == 1);
#endif
      });
    });
  }

  // Get native OpenCL handles
  auto ocl_platform = Platform.get_native<backend::opencl>();
  auto ocl_device = Device.get_native<backend::opencl>();
  auto ocl_context = Context.get_native<backend::opencl>();
  auto ocl_queue = Queue.get_native<backend::opencl>();
  auto ocl_buffers = get_native<backend::opencl>(Buffer);
#ifdef SYCL2020_CONFORMANT_APIS
  assert(ocl_buffers.size() == 1);
#endif

  // Re-create SYCL objects from native OpenCL handles
  auto PlatformInterop = opencl::make<platform>(ocl_platform);
  auto DeviceInterop = opencl::make<device>(ocl_device);
  auto ContextInterop = opencl::make<context>(ocl_context);
  auto QueueInterop = opencl::make<queue>(ContextInterop, ocl_queue);
#ifdef SYCL2020_CONFORMANT_APIS
  auto BufferInterop =
      sycl::make_buffer<backend::opencl, int>(ocl_buffers[0], ContextInterop);
#else
  auto BufferInterop =
      sycl::make_buffer<backend::opencl, int>(ocl_buffers, ContextInterop);
#endif

  // Check native handles
  assert(ocl_platform == PlatformInterop.get_native<backend::opencl>());
  assert(ocl_device == DeviceInterop.get_native<backend::opencl>());
  assert(ocl_context == ContextInterop.get_native<backend::opencl>());
  assert(ocl_queue == QueueInterop.get_native<backend::opencl>());
  assert(ocl_buffers == get_native<backend::opencl>(BufferInterop));

  return 0;
}
