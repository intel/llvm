//==--------------- aliases.cpp - SYCL multi_ptr aliases test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning

// expected-no-diagnostics

#include <sycl/detail/core.hpp>

#include <type_traits>

template <typename T, sycl::access::decorated IsDecorated>
void test_address_space_aliases() {
  static_assert(std::is_same_v<
                sycl::generic_ptr<T, IsDecorated>,
                sycl::multi_ptr<T, sycl::access::address_space::generic_space,
                                IsDecorated>>);
  static_assert(std::is_same_v<
                sycl::global_ptr<T, IsDecorated>,
                sycl::multi_ptr<T, sycl::access ::address_space::global_space,
                                IsDecorated>>);
  static_assert(std::is_same_v<
                sycl::local_ptr<T, IsDecorated>,
                sycl::multi_ptr<T, sycl::access::address_space::local_space,
                                IsDecorated>>);
  static_assert(std::is_same_v<
                sycl::private_ptr<T, IsDecorated>,
                sycl::multi_ptr<T, sycl::access::address_space::private_space,
                                IsDecorated>>);
}

template <typename T> void test_aliases() {
  // Template specialization aliases for different pointer address spaces
  test_address_space_aliases<T, sycl::access::decorated::yes>();
  test_address_space_aliases<T, sycl::access::decorated::no>();
  test_address_space_aliases<T, sycl::access::decorated::legacy>();

  // Template specialization aliases for different pointer address spaces.
  // The interface exposes non-decorated pointer while keeping the
  // address space information internally.
  static_assert(std::is_same_v<
                sycl::raw_generic_ptr<T>,
                sycl::multi_ptr<T, sycl::access::address_space::generic_space,
                                sycl::access::decorated::no>>);
  static_assert(std::is_same_v<
                sycl::raw_global_ptr<T>,
                sycl::multi_ptr<T, sycl::access::address_space::global_space,
                                sycl::access::decorated::no>>);
  static_assert(std::is_same_v<
                sycl::raw_local_ptr<T>,
                sycl::multi_ptr<T, sycl::access::address_space::local_space,
                                sycl::access::decorated::no>>);
  static_assert(std::is_same_v<
                sycl::raw_private_ptr<T>,
                sycl::multi_ptr<T, sycl::access::address_space::private_space,
                                sycl::access::decorated::no>>);

  // Template specialization aliases for different pointer address spaces.
  // The interface exposes decorated pointer.
  static_assert(std::is_same_v<
                sycl::decorated_generic_ptr<T>,
                sycl::multi_ptr<T, sycl::access::address_space::generic_space,
                                sycl::access::decorated::yes>>);
  static_assert(std::is_same_v<
                sycl::decorated_global_ptr<T>,
                sycl::multi_ptr<T, sycl::access::address_space::global_space,
                                sycl::access::decorated::yes>>);
  static_assert(std::is_same_v<
                sycl::decorated_local_ptr<T>,
                sycl::multi_ptr<T, sycl::access::address_space::local_space,
                                sycl::access::decorated::yes>>);
  static_assert(std::is_same_v<
                sycl::decorated_private_ptr<T>,
                sycl::multi_ptr<T, sycl::access::address_space::private_space,
                                sycl::access::decorated::yes>>);
}

// Test "minimal set of types" in the CTS. As we are just testing aliases are
// present in this test, this should work for any type.

template void test_aliases<int>();
template void test_aliases<float>();
