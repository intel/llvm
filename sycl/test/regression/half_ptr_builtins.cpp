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

SYCL_EXTERNAL auto
foo(sycl::multi_ptr<sycl::half, sycl::access::address_space::global_space,
                    sycl::access::decorated::yes>
        ptr) {
  sycl::modf(sycl::half{1.0}, ptr);
  sycl::sincos(sycl::half{1.0}, ptr);
  return 0;
}
