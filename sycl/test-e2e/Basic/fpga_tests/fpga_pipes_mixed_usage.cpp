//===-- fpga_pipes_mixed_usage.cpp -- Using pipe and experimental::pipe ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: accelerator
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// https://github.com/intel/llvm/issues/13887
// XFAIL: *
// If users need to use host pipe feature provided by experimental::pipe, all
// pipes in their design should use the experimental::pipe (as a workround).

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/pipes.hpp>

// Test for using sycl::ext::intel::pipe and
// sycl::ext::intel::experimental::pipe in the same kernel.
using NonExpPipe = sycl::ext::intel::pipe<class PipeA, int>;
using ExpPipe = sycl::ext::intel::experimental::pipe<class PipeB, short>;

int main() {
  sycl::queue q(sycl::ext::intel::fpga_emulator_selector_v);

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class SimplePipeWrite>([=]() {
      NonExpPipe::write(42);
      ExpPipe::write(24);
    });
  });
  q.wait();

  int a = 0;
  short b = 0;
  sycl::buffer<int, 1> buf_a(&a, 1);
  sycl::buffer<short, 1> buf_b(&b, 1);
  q.submit([&](sycl::handler &cgh) {
    auto acc_a = buf_a.get_access<sycl::access::mode::write>(cgh);
    auto acc_b = buf_b.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task<class SimplePipeRead>([=]() {
      acc_a[0] = NonExpPipe::read();
      acc_b[0] = ExpPipe::read();
    });
  });
  q.wait();

  if (a != 42 || b != 24) {
    std::cout << "Failed\n";
    return 1;
  }

  std::cout << "Passed\n";
  return 0;
}
