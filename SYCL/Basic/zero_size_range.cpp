// REQUIRES: cuda || hip
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//==--------------- range_zero_size.cpp - SYCL range test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  queue q;
  q.submit(
      [&](handler &cgh) { cgh.parallel_for(range<1>(0), [=](id<1> i) {}); });
}
