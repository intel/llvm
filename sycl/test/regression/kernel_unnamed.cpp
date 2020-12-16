// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -fsycl-unnamed-lambda
// RUN: %RUN_ON_HOST %t.out

//==-- kernel_unnamed.cpp - SYCL kernel naming variants test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <iostream>

#define GOLD 10
static int NumTestCases = 0;

template <class F>
void foo(cl::sycl::queue &deviceQueue, cl::sycl::buffer<int, 1> &buf, F f) {
  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.single_task([=]() { acc[0] = f(acc[0], GOLD); });
  });
}

namespace nm {
struct Wrapper {

  int test() {
    int arr[] = {0};
    {
      // Simple test
      cl::sycl::queue deviceQueue;
      cl::sycl::buffer<int, 1> buf(arr, 1);
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task([=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;

// Test lambdas with different ordinal because of macro expansion
#ifdef __SYCL_DEVICE_ONLY__
      [] {}();
#endif
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task([=]() { acc[0] += GOLD; });
      });
      ++NumTestCases;

      // Test lambda passed to function
      foo(deviceQueue, buf, [](int a, int b) { return a + b; });
      ++NumTestCases;
    }
    return arr[0];
  }
};
} // namespace nm

int main() {
  nm::Wrapper w;
  int res = w.test();
  assert (res == GOLD * NumTestCases && "Wrong result");
}
