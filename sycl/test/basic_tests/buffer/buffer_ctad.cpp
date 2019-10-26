// RUN: %clangxx -std=c++17 -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//==------------------- buffer_ctad.cpp - SYCL buffer CTAD test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cassert>
#include <memory>

using namespace cl::sycl;

int main() {
  std::vector<int> v(5, 1);
  const std::vector<int> cv(5, 1);
  buffer b1(v.data(), range<1>(5));
  static_assert(std::is_same_v<decltype(b1), buffer<int, 1>>);
  buffer b1a(v.data(), range<1>(5), std::allocator<int>());
  static_assert(
      std::is_same_v<decltype(b1a), buffer<int, 1, std::allocator<int>>>);
  buffer b1b(cv.data(), range<1>(5));
  static_assert(std::is_same_v<decltype(b1b), buffer<int, 1>>);
  buffer b1c(v.data(), range<2>(2, 2));
  static_assert(std::is_same_v<decltype(b1c), buffer<int, 2>>);
  buffer b2(v.begin(), v.end());
  static_assert(std::is_same_v<decltype(b2), buffer<int, 1>>);
  buffer b2a(v.cbegin(), v.cend());
  static_assert(std::is_same_v<decltype(b2a), buffer<int, 1>>);
  buffer b3(v);
  static_assert(std::is_same_v<decltype(b3), buffer<int, 1>>);
  buffer b3a(cv);
  static_assert(std::is_same_v<decltype(b3a), buffer<int, 1>>);
  shared_ptr_class<int> ptr{new int[5], [](int *p) { delete[] p; }};
  buffer b4(ptr, range<1>(5));
  static_assert(std::is_same_v<decltype(b4), buffer<int, 1>>);
}
