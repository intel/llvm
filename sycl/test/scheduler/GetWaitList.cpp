// RUN: %clangxx -fsycl %s -o %t.out -lOpenCL
// RUN: %t.out
//==------------------- GetWaitList.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

void foo() {
  cl::sycl::event start;
  start.wait_and_throw();
  return;
}

int main() {
  cl::sycl::queue Q;
  Q.submit([&](cl::sycl::handler &CGH) {
    foo();
  });
  return 0;
}
