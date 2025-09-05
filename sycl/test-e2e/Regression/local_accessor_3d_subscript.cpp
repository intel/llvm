// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==------------------ local_accessor_3d_subscript.cpp ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Tests that the subscript operator on a 3-dimensional local_accessor correctly
// compiles and runs.

#include <iostream>

#include <sycl/detail/core.hpp>

int main() {
  size_t Result = 0;
  sycl::queue Q;
  {
    sycl::buffer<size_t, 1> Buf(&Result, 1);
    Q.submit([&](sycl::handler &CGH) {
      sycl::local_accessor<size_t, 3> LocalMem(sycl::range<3>(1, 1, 1), CGH);
      auto Acc = Buf.get_access(CGH);
      CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1> It) {
        LocalMem[It.get_local_id()][It.get_local_id()][It.get_local_id()] = 42;
        Acc[It.get_local_id()] =
            LocalMem[It.get_local_id()][It.get_local_id()][It.get_local_id()];
      });
    });
  }
  assert(Result == 42);
  std::cout << "success!" << std::endl;
  return 0;
}
