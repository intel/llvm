//==- hier_par_indirect.cpp --- hierarchical parallelism test for WG scope--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks correctness of hierarchical kernel execution when the work
// item code is not directly inside work group scope.

#include <iostream>
#include <sycl/detail/core.hpp>

void __attribute__((noinline)) foo(sycl::group<1> work_group) {
  work_group.parallel_for_work_item([&](sycl::h_item<1> index) {});
}

int main(int argc, char **argv) {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for_work_group(sycl::range<1>{1}, sycl::range<1>{128},
                                 ([=](sycl::group<1> wGroup) { foo(wGroup); }));
   }).wait();
  std::cout << "test passed" << std::endl;
  return 0;
}
