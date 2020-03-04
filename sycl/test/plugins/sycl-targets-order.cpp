// RUN: %clangxx -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice,nvptx64-unknown-unknown-sycldevice %s -o %t-spir64-nvptx64.out
// RUN: env SYCL_BE=PI_OPENCL %t-spir64-nvptx64.out
// RUN: env SYCL_BE=PI_CUDA   %t-spir64-nvptx64.out
// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-unknown-unknown-sycldevice,spir64-unknown-unknown-sycldevice %s -o %t-nvptx64-spir64.out
// RUN: env SYCL_BE=PI_OPENCL %t-nvptx64-spir64.out
// RUN: env SYCL_BE=PI_CUDA   %t-nvptx64-spir64.out

// REQUIRES: opencl, cuda

//==------- sycl-targets-order.cpp - SYCL -fsycl-targets order test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <iostream>

int main(int argc, char **argv) {

  // select the default SYCL device
  cl::sycl::device device{cl::sycl::default_selector{}};
  std::cout << "Running on SYCL device " << device.get_info<cl::sycl::info::device::name>()
            << ", driver version " << device.get_info<cl::sycl::info::device::driver_version>()
            << std::endl;

  // create a queue
  cl::sycl::queue queue{device};

  // create a buffer of 4 ints to be used inside the kernel code
  cl::sycl::buffer<unsigned int, 1> buffer(4);

  // size of the index space for the kernel
  cl::sycl::range<1> NumOfWorkItems{buffer.get_count()};

  // submit a command group(work) to the queue
  queue.submit([&](cl::sycl::handler &cgh) {
    // get write only access to the buffer on a device
    auto accessor = buffer.get_access<cl::sycl::access::mode::write>(cgh);
    // executing the kernel
    cgh.parallel_for<class FillBuffer>(
        NumOfWorkItems, [=](cl::sycl::id<1> WIid) {
          // fill the buffer with indexes
          accessor[WIid] = WIid.get(0);
        });
  });

  // get read-only access to the buffer on the host
  // introduce an implicit barrier waiting for queue to complete the work
  const auto host_accessor = buffer.get_access<cl::sycl::access::mode::read>();

  // check the results
  bool mismatch = false;
  for (unsigned int i = 0; i < buffer.get_count(); ++i) {
    if (host_accessor[i] != i) {
      std::cout << "The result is incorrect for element: " << i
                << " , expected: " << i << " , got: " << host_accessor[i]
                << std::endl;
      mismatch = true;
    }
  }

  if (not mismatch) {
    std::cout << "The results are correct!" << std::endl;
  }

  return mismatch;
}
