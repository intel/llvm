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
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue Queue;

  // Test that data is flushed to the buffer at the end of kernel execution even
  // without explicit flush
  Queue.submit([&](handler &CGH) {
    stream Out(1024, 80, CGH);
    CGH.parallel_for<class auto_flush1>(
        range<1>(2), [=](id<1> i) { Out << "Hello World!\n"; });
  });
  Queue.wait();
  // CHECK: Hello World!
  // CHECK-NEXT: Hello World!

  return 0;
}
