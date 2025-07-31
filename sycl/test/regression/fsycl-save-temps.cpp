//==--------------- fsycl-save-temps.cpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verify that a sample compilation succeeds with -save-temps
// RUN: %clangxx -fsycl -save-temps %s -o %t.out

// XFAIL: libcxx
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19616

#include <sycl/sycl.hpp>

void foo() {}

int main() {
  sycl::queue Q;
  Q.submit([](sycl::handler &Cgh) {
    Cgh.single_task<class KernelFunction>([]() { foo(); });
  });
  return 0;
}
