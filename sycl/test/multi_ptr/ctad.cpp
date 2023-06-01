// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics
//==--------------- ctad.cpp - SYCL multi_ptr CTAD test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>

using constTypeMPtr =
    sycl::multi_ptr<const int, sycl::access::address_space::global_space,
                    sycl::access::decorated::no>;

void implicit_conversion(const constTypeMPtr &cmptr) { auto v = cmptr.get(); }

int main() {
  using sycl::access::address_space;
  using sycl::access::mode;
  using sycl::access::target;
  using deviceAcc = sycl::accessor<int, 1, mode::read, target::device>;
  using globlAcc = sycl::accessor<int, 1, mode::read, target::global_buffer>;
  using constAcc = sycl::accessor<int, 1, mode::read, target::constant_buffer>;
  using localAcc = sycl::local_accessor<int, 1>;
  using localAccDep = sycl::accessor<int, 1, mode::read, target::local>;
  using deviceCTAD = decltype(sycl::multi_ptr(std::declval<deviceAcc>()));
  using globlCTAD = decltype(sycl::multi_ptr(std::declval<globlAcc>()));
  using constCTAD = decltype(sycl::multi_ptr(std::declval<constAcc>()));
  using localCTAD = decltype(sycl::multi_ptr(std::declval<localAcc>()));
  using localCTADDep = decltype(sycl::multi_ptr(std::declval<localAccDep>()));
  using deviceMPtr = sycl::multi_ptr<int, address_space::global_space,
                                     sycl::access::decorated::no>;
  using globlMPtr = sycl::multi_ptr<int, address_space::global_space,
                                    sycl::access::decorated::no>;
  using constMPtr = sycl::multi_ptr<int, address_space::constant_space,
                                    sycl::access::decorated::legacy>;
  using localMPtr = sycl::multi_ptr<int, address_space::local_space,
                                    sycl::access::decorated::no>;
  static_assert(std::is_same<deviceCTAD, deviceMPtr>::value);
  static_assert(std::is_same<deviceCTAD, globlMPtr>::value);
  static_assert(std::is_same<globlCTAD, globlMPtr>::value);
  static_assert(std::is_same<constCTAD, constMPtr>::value);
  static_assert(std::is_same<localCTAD, localMPtr>::value);
  static_assert(std::is_same<localCTADDep, localMPtr>::value);

  globlMPtr non_const_multi_ptr;
  auto constTypeMultiPtr = constTypeMPtr(non_const_multi_ptr);
  implicit_conversion(non_const_multi_ptr);
}
