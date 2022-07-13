// REQUIRES: TEMPORARY_DISABLED
// Temporarily disabled because the test is out of time
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==--------------- event_destruction.cpp - SYCL event test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <sycl/sycl.hpp>

const size_t ITERS = 100000;

// The test checks that that event destruction does not lead to stack overflow

int main() {
  cl::sycl::default_selector S;
  cl::sycl::queue Q(S);
  cl::sycl::buffer<int, 1> Buf(3000);
  for (size_t Idx = 0; Idx < ITERS; ++Idx) {
    auto Event = Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.single_task([=]() {
        for (size_t I = 0; I < 2000; I++) {
          Acc[I] = I * I + 2000;
        }
      });
    });
    Event.wait();
  }
}
