// RUN: %{build} -fsycl-dead-args-optimization -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

//==---------------------- HostAccDestruction.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  size_t size = 3;

  sycl::buffer<int, 1> buf(size);
  {
    sycl::queue q;
    auto host_acc = buf.get_host_access();
    q.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<class SingleTask>(
          sycl::range<1>{size}, [=](sycl::id<1> id) { (void)acc[id]; });
    });
    std::cout << "host acc destructor call" << std::endl;
  }
  std::cout << "end of scope" << std::endl;

  return 0;
}

// CHECK:host acc destructor call
// CHECK:---> piEnqueueKernelLaunch(
// CHECK:end of scope
