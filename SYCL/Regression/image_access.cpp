// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
// No execution of FPGA because it does not support images
//
// UNSUPPORTED: hip
// CUDA doesn't fully support OpenCL spec conform images.

//==-------------- image_access.cpp - SYCL image accessors test  -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  try {
    sycl::range<1> Range(32);
    std::vector<cl_float> Data(Range.size() * 4, 0.0f);
    sycl::image<1> Image(Data.data(), sycl::image_channel_order::rgba,
                         sycl::image_channel_type::fp32, Range);
    sycl::queue Queue;

    Queue.submit([&](sycl::handler &CGH) {
      sycl::accessor<sycl::cl_int4, 1, sycl::access::mode::read,
                     sycl::access::target::image,
                     sycl::access::placeholder::false_t>
          A(Image, CGH);
      CGH.single_task<class MyKernel>([=]() {});
    });
    Queue.wait_and_throw();

    sycl::accessor<sycl::cl_int4, 1, sycl::access::mode::read,
                   sycl::access::target::host_image,
                   sycl::access::placeholder::false_t>
        A(Image);
  } catch (sycl::exception &E) {
    std::cout << E.what();
  }
  return 0;
}

// CHECK:---> piMemImageCreate
// CHECK:---> piEnqueueMemImageRead
