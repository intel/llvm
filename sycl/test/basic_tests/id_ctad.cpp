// RUN: %clangxx -std=c++17 -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//==--------------- id_ctad.cpp - SYCL id CTAD test ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

using namespace std;
int main() {
  cl::sycl::id one_dim_id(64);
  cl::sycl::id two_dim_id(64, 1);
  cl::sycl::id three_dim_id(64, 1, 2);
  static_assert(std::is_same_v<decltype(one_dim_id), cl::sycl::id<1>>);
  static_assert(std::is_same_v<decltype(two_dim_id), cl::sycl::id<2>>);
  static_assert(std::is_same_v<decltype(three_dim_id), cl::sycl::id<3>>);
}
