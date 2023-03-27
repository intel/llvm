// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

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
#include <sycl/sycl.hpp>

int main() {
  size_t Result = 0;
  sycl::queue Q;
  {
    sycl::buffer<size_t, 1> Buf(&Result, 1);
    Q.submit([&](sycl::handler &CGH) {
      sycl::local_accessor<size_t, 3> LocalMem(sycl::range<3>(1, 1, 1), CGH);
      auto Acc = Buf.get_access(CGH);
      CGH.parallel_for(1, [=](sycl::item<1> It) {
        LocalMem[It][It][It] = 42;
        Acc[It] = LocalMem[It][It][It];
      });
    });
  }
  assert(Result == 42);
  std::cout << "success!" << std::endl;
  return 0;
}
