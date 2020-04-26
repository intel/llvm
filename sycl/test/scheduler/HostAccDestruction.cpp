// RUN: %clangxx -fsycl -I %sycl_source_dir %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
//==---------------------- HostAccDestruction.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

int main() {
  size_t size = 3;

  cl::sycl::buffer<int, 1> buf(size);
  {
    cl::sycl::queue q;
    auto host_acc = buf.get_access<cl::sycl::access::mode::read_write>();
    q.submit([&](cl::sycl::handler &cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<class SingleTask>(
          cl::sycl::range<1>{size},
          [=](cl::sycl::id<1> id) { (void)acc[id]; });
    });
    std::cout << "host acc destructor call" << std::endl;
  }
  std::cout << "end of scope" << std::endl;

  return 0;
}

// CHECK:host acc destructor call
// CHECK:---> piEnqueueKernelLaunch(
// CHECK:end of scope
