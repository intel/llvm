//==--------------- half_ptr_builtins.cpp ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verifies that builtins with pointer arguments accept pointers to sycl::half.
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning -Wno-sycl-strict
// expected-no-diagnostics

#include <sycl/sycl.hpp>

int main() {
  sycl::half x;
  sycl::modf(sycl::half{1.0}, &x);
  sycl::sincos(sycl::half{1.0}, &x);
  return 0;
}
