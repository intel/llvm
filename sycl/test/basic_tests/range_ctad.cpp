// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics
//==--------------- range_ctad.cpp - SYCL range CTAD test ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

using namespace std;
int main() {
  cl::sycl::range one_dim_range(64);
  cl::sycl::range two_dim_range(64, 1);
  cl::sycl::range three_dim_range(64, 1, 2);
  static_assert(
      std::is_same<decltype(one_dim_range), cl::sycl::range<1>>::value);
  static_assert(
      std::is_same<decltype(two_dim_range), cl::sycl::range<2>>::value);
  static_assert(
      std::is_same<decltype(three_dim_range), cl::sycl::range<3>>::value);
  cl::sycl::nd_range one_dim_ndrange(one_dim_range, one_dim_range);
  cl::sycl::nd_range two_dim_ndrange(two_dim_range, two_dim_range);
  cl::sycl::nd_range three_dim_ndrange(three_dim_range, three_dim_range);
  static_assert(
      std::is_same<decltype(one_dim_ndrange), cl::sycl::nd_range<1>>::value);
  static_assert(
      std::is_same<decltype(two_dim_ndrange), cl::sycl::nd_range<2>>::value);
  static_assert(
      std::is_same<decltype(three_dim_ndrange), cl::sycl::nd_range<3>>::value);
}
