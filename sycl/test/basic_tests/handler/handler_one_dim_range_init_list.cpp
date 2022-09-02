// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -I %sycl_include
// expected-no-diagnostics
//==---------------- handler_one_dim_range_init_list.cpp -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>

#include <utility>

int main() {
  auto l = [] {};

  decltype(std::declval<sycl::handler>().parallel_for(1, l)) a1a();
  decltype(std::declval<sycl::handler>().parallel_for({1}, l)) a1b();
  decltype(std::declval<sycl::handler>().parallel_for({1, 2}, l)) a2();
  decltype(std::declval<sycl::handler>().parallel_for({1, 2, 3}, l)) a3();
}
