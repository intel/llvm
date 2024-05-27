// REQUIRES: cuda || hip
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//==--------------- range_zero_size.cpp - SYCL range test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
using namespace sycl;

int main() {
  queue q;
  q.submit(
      [&](handler &cgh) { cgh.parallel_for(range<1>(0), [=](id<1> i) {}); });
}
