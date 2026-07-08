// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==--------------- event_destruction.cpp - SYCL event test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/detail/core.hpp>

const size_t ITERS = 100000;

// The test checks that event destruction does not cause a stack overflow.
// The SYCL scheduler builds a chain of dependent event_impl objects; if their
// destructors recurse into each other the host stack overflows.

int main() {
  sycl::queue Q;
  sycl::buffer<int, 1> Buf(1);
  for (size_t Idx = 0; Idx < ITERS; ++Idx) {
    auto Event = Q.submit([&](sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task([=]() { Acc[0] = 1; });
    });
    Event.wait();
  }
}
