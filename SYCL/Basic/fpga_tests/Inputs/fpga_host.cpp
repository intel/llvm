//==--------------- fpga_host.cpp - AOT compilation for fpga ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl.hpp"
#include <cassert>

using namespace sycl;

void foo(double &, queue q, int x);

int main(void) {
  queue q(accelerator_selector{});

  double result;
  foo(result, q, 3);
  assert(result == 5);
  return 0;
}
