// REQUIRES: TEMPORARY_DISABLED
// Temporarily disabled because the test is out of time
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==--------------- event_destruction.cpp - SYCL event test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>

#include <sycl/detail/core.hpp>

const size_t ITERS = 100000;

// The test checks that that event destruction does not lead to stack overflow

int main() {
  sycl::queue Q;
  sycl::buffer<int, 1> Buf(3000);
  for (size_t Idx = 0; Idx < ITERS; ++Idx) {
    auto Event = Q.submit([&](sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task([=]() {
        for (size_t I = 0; I < 2000; I++) {
          Acc[I] = I * I + 2000;
        }
      });
    });
    Event.wait();
  }
}
