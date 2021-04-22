// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics
//==--------------- ctad.cpp - SYCL vector CTAD test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main() {
  sycl::vec v1(1);
  static_assert(std::is_same<decltype(v1), sycl::vec<int, 1>>::value);
  sycl::vec v2(1, 2);
  static_assert(std::is_same<decltype(v2), sycl::vec<int, 2>>::value);
}
