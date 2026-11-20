//===-----------------------assert_in_one_kernel.hpp-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests device assertion handling in an application with a single kernel.
/// Each work-item triggers an assertion.
///
//===----------------------------------------------------------------------===//
#include <cassert>
#include <iostream>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>

using namespace sycl;
using namespace sycl::access;

void kernelFunc(int *Buf, int wiID) {
  Buf[wiID] = 0;
  assert(Buf[wiID] != 0 && "from assert statement");
}

void assertTest() {
  std::array<int, 4> Vec = {1, 2, 3, 4};
  sycl::range<1> numOfItems{Vec.size()};
  sycl::buffer<int, 1> Buf(Vec.data(), numOfItems);

  queue Q;
  Q.submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<access_mode::read_write>(CGH);

    CGH.parallel_for<class TheKernel>(
        numOfItems, [=](sycl::id<1> wiID) { kernelFunc(&Acc[0], wiID[0]); });
  });
  Q.wait();
}

int main() {

  assertTest();

  std::cout << "The test ended." << std::endl;
  return 0;
}
