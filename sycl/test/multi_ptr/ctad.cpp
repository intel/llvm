// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics
//==--------------- ctad.cpp - SYCL multi_ptr CTAD test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main() {
  using sycl::access::address_space;
  using sycl::access::mode;
  using sycl::access::target;
  using globlAcc = sycl::accessor<int, 1, mode::read, target::global_buffer>;
  using constAcc = sycl::accessor<int, 1, mode::read, target::constant_buffer>;
  using localAcc = sycl::accessor<int, 1, mode::read, target::local>;
  using globlCTAD = decltype(sycl::multi_ptr(std::declval<globlAcc>()));
  using constCTAD = decltype(sycl::multi_ptr(std::declval<constAcc>()));
  using localCTAD = decltype(sycl::multi_ptr(std::declval<localAcc>()));
  using globlMPtr = sycl::multi_ptr<int, address_space::global_space>;
  using constMPtr = sycl::multi_ptr<int, address_space::constant_space>;
  using localMPtr = sycl::multi_ptr<int, address_space::local_space>;
  static_assert(std::is_same<globlCTAD, globlMPtr>::value);
  static_assert(std::is_same<constCTAD, constMPtr>::value);
  static_assert(std::is_same<localCTAD, localMPtr>::value);
}
