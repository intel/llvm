// REQUIRES: opencl

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==-- interop-opencl.cpp - SYCL test for OpenCL interop API --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/opencl.h>
#include <sycl/backend.hpp>
#include <sycl/backend/opencl.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/detail/host_task_impl.hpp>

using namespace sycl;

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
        assert(BufNative.size() == 1);
      });
    });
  }

  // Get native OpenCL handles
  auto ocl_platform = get_native<backend::opencl>(Platform);
  auto ocl_device = get_native<backend::opencl>(Device);
  auto ocl_context = get_native<backend::opencl>(Context);
  auto ocl_queue = get_native<backend::opencl>(Queue);
  auto ocl_buffers = get_native<backend::opencl>(Buffer);
  assert(ocl_buffers.size() == 1);

  // Re-create SYCL objects from native OpenCL handles
  auto PlatformInterop = sycl::make_platform<backend::opencl>(ocl_platform);
  auto DeviceInterop = sycl::make_device<backend::opencl>(ocl_device);
  auto ContextInterop = sycl::make_context<backend::opencl>(ocl_context);
  auto QueueInterop =
      sycl::make_queue<backend::opencl>(ocl_queue, ContextInterop);
  auto BufferInterop =
      sycl::make_buffer<backend::opencl, int>(ocl_buffers[0], ContextInterop);

  // Check native handles
  assert(ocl_platform == get_native<backend::opencl>(PlatformInterop));
  assert(ocl_device == get_native<backend::opencl>(DeviceInterop));
  assert(ocl_context == get_native<backend::opencl>(ContextInterop));
  assert(ocl_queue == get_native<backend::opencl>(QueueInterop));
  assert(ocl_buffers == get_native<backend::opencl>(BufferInterop));

  return 0;
}
