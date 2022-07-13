// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
// No execution of FPGA because it does not support images
//
// UNSUPPORTED: cuda || hip
// CUDA cannot support OpenCL spec conform images.

//==-------------- image_access.cpp - SYCL image accessors test  -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

int main() {
  try {
    cl::sycl::range<1> Range(32);
    std::vector<cl_float> Data(Range.size() * 4, 0.0f);
    cl::sycl::image<1> Image(Data.data(), cl::sycl::image_channel_order::rgba,
                             cl::sycl::image_channel_type::fp32, Range);
    cl::sycl::queue Queue;

    Queue.submit([&](cl::sycl::handler &CGH) {
      cl::sycl::accessor<cl::sycl::cl_int4, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::image,
                         cl::sycl::access::placeholder::false_t>
          A(Image, CGH);
      CGH.single_task<class MyKernel>([=]() {});
    });
    Queue.wait_and_throw();

    cl::sycl::accessor<cl::sycl::cl_int4, 1, cl::sycl::access::mode::read,
                       cl::sycl::access::target::host_image,
                       cl::sycl::access::placeholder::false_t>
        A(Image);
  } catch (cl::sycl::exception &E) {
    std::cout << E.what();
  }
  return 0;
}

// CHECK:---> piMemImageCreate
// CHECK:---> piEnqueueMemImageRead
