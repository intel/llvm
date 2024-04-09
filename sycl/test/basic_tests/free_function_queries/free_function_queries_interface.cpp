// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -I %sycl_include
// expected-no-diagnostics
//==- free_function_queries_interface.cpp - SYCL free queries interface -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <type_traits>

template <int Dims> struct get_nd_item_caller {
  auto operator()() const {
    return sycl::ext::oneapi::this_work_item::get_nd_item<Dims>();
  }
};

template <int Dims> struct get_work_group_caller {
  auto operator()() const {
    return sycl::ext::oneapi::this_work_item::get_work_group<Dims>();
  }
};

template <template <int> class IterationPoint, int Dims,
          template <int> class Callable>
void test(Callable<Dims> &&callable) {
  static_assert(std::is_same_v<decltype(callable()), IterationPoint<Dims>>,
                "Wrong return type of free function query");
}

SYCL_EXTERNAL void test_all() {
  test<sycl::nd_item>(get_nd_item_caller<1>{});
  test<sycl::nd_item>(get_nd_item_caller<2>{});
  test<sycl::nd_item>(get_nd_item_caller<3>{});

  test<sycl::group>(get_work_group_caller<1>{});
  test<sycl::group>(get_work_group_caller<2>{});
  test<sycl::group>(get_work_group_caller<3>{});

  static_assert(
      std::is_same_v<
          decltype(sycl::ext::oneapi::this_work_item::get_sub_group()),
          sycl::sub_group>,
      "Wrong return type of free function query for Sub Group");
}
