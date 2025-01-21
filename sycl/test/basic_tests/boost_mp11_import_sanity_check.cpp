// -*- C++ -*-
//===----------------------------------------------------------------------===//
// Modifications Copyright Intel Corporation 2022
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
// Based on boost/mp11 tests obtained from
//  https://github.com/boostorg/mp11/blob/develop/test/mp_fill.cpp
//  (git commit a231733).

//===----------------------------------------------------------------------===//
//  Copyright 2015 Peter Dimov.
//
// Distributed under the Boost Software License, Version 1.0.
//
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsyntax-only -fsycl-targets=%sycl_triple %s

// This is a sanity check test to verify that the automatic boost/mp11 import
// into SYCL is not badly broken.

#include <type_traits>

#include <sycl/detail/boost/mp11.hpp>

struct X1 {};

int main() {
  using sycl::detail::boost::mp11::mp_fill;
  using sycl::detail::boost::mp11::mp_list;

  using L1 = mp_list<int, void(), float[]>;
  static_assert(std::is_same_v<mp_fill<L1, X1>, mp_list<X1, X1, X1>>);

  //

  using L2 = std::tuple<int, char, float>;
  static_assert(std::is_same_v<mp_fill<L2, X1>, std::tuple<X1, X1, X1>>);

  //

  using L3 = std::pair<char, double>;
  static_assert(std::is_same_v<mp_fill<L3, X1>, std::pair<X1, X1>>);

  return 0;
}
