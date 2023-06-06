//==------- default_platform.cpp - SYCL platform default ctor test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>

int main() {
  sycl::platform Plt;
  assert(Plt == sycl::platform{sycl::default_selector_v});
  return 0;
}
