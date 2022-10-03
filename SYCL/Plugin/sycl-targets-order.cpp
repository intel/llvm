// RUN: %clangxx -fsycl -fsycl-targets=spir64,nvptx64-nvidia-cuda %s -o %t-spir64-nvptx64.out
// RUN: env SYCL_DEVICE_FILTER=opencl %t-spir64-nvptx64.out
// RUN: env SYCL_DEVICE_FILTER=cuda   %t-spir64-nvptx64.out
// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 %s -o %t-nvptx64-spir64.out
// RUN: env SYCL_DEVICE_FILTER=opencl %t-nvptx64-spir64.out
// RUN: env SYCL_DEVICE_FILTER=cuda   %t-nvptx64-spir64.out

// REQUIRES: opencl, cuda

//==------- sycl-targets-order.cpp - SYCL -fsycl-targets order test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>

int main(int argc, char **argv) {

  // select the default SYCL device
  sycl::device device{sycl::default_selector_v};
  std::cout << "Running on SYCL device "
            << device.get_info<sycl::info::device::name>()
            << ", driver version "
            << device.get_info<sycl::info::device::driver_version>()
            << std::endl;

  // create a queue
  sycl::queue queue{device};

  // create a buffer of 4 ints to be used inside the kernel code
  sycl::buffer<unsigned int, 1> buffer(4);

  // size of the index space for the kernel
  sycl::range<1> NumOfWorkItems{buffer.get_count()};

  // submit a command group(work) to the queue
  queue.submit([&](sycl::handler &cgh) {
    // get write only access to the buffer on a device
    auto accessor = buffer.get_access<sycl::access::mode::write>(cgh);
    // executing the kernel
    cgh.parallel_for<class FillBuffer>(NumOfWorkItems, [=](sycl::id<1> WIid) {
      // fill the buffer with indexes
      accessor[WIid] = WIid.get(0);
    });
  });

  // get read-only access to the buffer on the host
  // introduce an implicit barrier waiting for queue to complete the work
  const auto host_accessor = buffer.get_access<sycl::access::mode::read>();

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
