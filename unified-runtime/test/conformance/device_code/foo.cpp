// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sycl/sycl.hpp>

int main() {
  sycl::queue deviceQueue;
  sycl::range<1> numOfItems{1};

  deviceQueue.submit([&](sycl::handler &cgh) {
    auto kern = [=](sycl::id<1>) {};
    cgh.parallel_for<class Foo>(numOfItems, kern);
  });

  return 0;
}
