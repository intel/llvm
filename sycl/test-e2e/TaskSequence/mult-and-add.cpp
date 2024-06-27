//==---------------- mult-and-add.cpp - DPC++ task_sequence ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FIXME: compfail, see https://github.com/intel/llvm/issues/14284, re-enable
// when fixed:
// UNSUPPORTED: linux, windows

// REQUIRES: aspect-ext_intel_fpga_task_sequence
// RUN: %clangxx -fsycl -fintelfpga %s -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"

int mult(int a, int b) { return a * b; }

int mult_and_add(int a1, int b1, int a2, int b2) {
  task_sequence<mult, decltype(properties{invocation_capacity<1>,
                                          response_capacity<1>})>
      task1, task2;

  task1.async(a1, b1);
  task2.async(a2, b2);
  return task1.get() + task2.get();
}

int main() {
  queue myQueue;

  int result = 0;
  buffer<int, 1> res_buf(&result, range<1>(1));

  myQueue.submit([&](handler &cgh) {
    auto res_acc = res_buf.get_access<access::mode::write>(cgh);
    cgh.single_task(
        [=](kernel_handler kh) { res_acc[0] = mult_and_add(1, 2, 3, 4); });
  });
  myQueue.wait();

  assert(result == (1 * 2 + 3 * 4));
}