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
  using rDeviceAcc = sycl::accessor<int, 1, mode::read, target::device>;
  using rGloblAcc = sycl::accessor<int, 1, mode::read, target::global_buffer>;
  using wDeviceAcc = sycl::accessor<int, 1, mode::write, target::device>;
  using wGloblAcc = sycl::accessor<int, 1, mode::write, target::global_buffer>;
  using rwDeviceAcc = sycl::accessor<int, 1, mode::read_write, target::device>;
  using rwGloblAcc =
      sycl::accessor<int, 1, mode::read_write, target::global_buffer>;
  using constAcc = sycl::accessor<int, 1, mode::read, target::constant_buffer>;
  using localAcc = sycl::local_accessor<int, 1>;
  using localAccDep = sycl::local_accessor<int, 1>;
  using rDeviceCTAD = decltype(sycl::multi_ptr(std::declval<rDeviceAcc>()));
  using rGloblCTAD = decltype(sycl::multi_ptr(std::declval<rGloblAcc>()));
  using wDeviceCTAD = decltype(sycl::multi_ptr(std::declval<wDeviceAcc>()));
  using wGloblCTAD = decltype(sycl::multi_ptr(std::declval<wGloblAcc>()));
  using rwDeviceCTAD = decltype(sycl::multi_ptr(std::declval<rwDeviceAcc>()));
  using rwGloblCTAD = decltype(sycl::multi_ptr(std::declval<rwGloblAcc>()));
  using constCTAD = decltype(sycl::multi_ptr(std::declval<constAcc>()));
  using localCTAD = decltype(sycl::multi_ptr(std::declval<localAcc>()));
  using localCTADDep = decltype(sycl::multi_ptr(std::declval<localAccDep>()));
  using deviceMPtr = sycl::multi_ptr<int, address_space::global_space,
                                     sycl::access::decorated::no>;
  using globlMPtr = sycl::multi_ptr<int, address_space::global_space,
                                    sycl::access::decorated::no>;
  using deviceConstMPtr =
      sycl::multi_ptr<const int, address_space::global_space,
                      sycl::access::decorated::no>;
  using globlConstMPtr = sycl::multi_ptr<const int, address_space::global_space,
                                         sycl::access::decorated::no>;
  using constMPtr = sycl::multi_ptr<int, address_space::constant_space,
                                    sycl::access::decorated::legacy>;
  using constDefaultMPtr = sycl::multi_ptr<int, address_space::constant_space>;
  using localMPtr = sycl::multi_ptr<int, address_space::local_space,
                                    sycl::access::decorated::no>;
  using legacyMPtr = sycl::multi_ptr<int, address_space::global_space,
                                     sycl::access::decorated::legacy>;
  static_assert(std::is_same<rwDeviceCTAD, deviceMPtr>::value);
  static_assert(std::is_same<rwDeviceCTAD, globlMPtr>::value);
  static_assert(std::is_same<rwGloblCTAD, globlMPtr>::value);
  static_assert(std::is_same<wDeviceCTAD, deviceMPtr>::value);
  static_assert(std::is_same<wDeviceCTAD, globlMPtr>::value);
  static_assert(std::is_same<wGloblCTAD, globlMPtr>::value);
  static_assert(std::is_same<rDeviceCTAD, deviceConstMPtr>::value);
  static_assert(std::is_same<rDeviceCTAD, globlConstMPtr>::value);
  static_assert(std::is_same<rGloblCTAD, globlConstMPtr>::value);
  static_assert(std::is_same<constCTAD, constMPtr>::value);
  static_assert(std::is_same<localCTAD, localMPtr>::value);
  static_assert(std::is_same<localCTADDep, localMPtr>::value);
  static_assert(std::is_same<constMPtr, constDefaultMPtr>::value);

  legacyMPtr LegacytMultiPtr;
  static_assert(
      std::is_same_v<
          decltype(LegacytMultiPtr.get_decorated()),
          typename sycl::multi_ptr<int, address_space::global_space,
                                   sycl::access::decorated::yes>::pointer>);
  static_assert(std::is_same_v<decltype(LegacytMultiPtr.get_raw()), int *>);

  globlMPtr non_const_multi_ptr;
  auto constTypeMultiPtr = constTypeMPtr(non_const_multi_ptr);
  implicit_conversion(non_const_multi_ptr);
}
