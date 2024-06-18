// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %if !gpu || linux %{ | FileCheck %s %}
//
// XFAIL: hip_nvidia
//==-------------- copy.cpp - SYCL stream obect auto flushing test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include <sycl/detail/core.hpp>

#include <sycl/properties/all_properties.hpp>
#include <sycl/stream.hpp>

using namespace sycl;

// Test that data is flushed to the buffer at the end of kernel execution even
// without explicit flush

void test(queue &Queue) {
  Queue.submit([&](handler &CGH) {
    stream Out(1024, 80, CGH);
    CGH.parallel_for<class auto_flush1>(
        range<1>(2), [=](id<1> i) { Out << "Hello World!\n"; });
  });
  Queue.wait();
}
int main() {
  queue Queue;
  test(Queue);
  // CHECK: Hello World!
  // CHECK-NEXT: Hello World!

  queue InOrderQueue{{sycl::property::queue::in_order()}};
  test(InOrderQueue);
  // CHECK: Hello World!
  // CHECK-NEXT: Hello World!

  return 0;
}
