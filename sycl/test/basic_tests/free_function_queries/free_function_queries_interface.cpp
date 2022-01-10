// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -I %sycl_include
// expected-no-diagnostics
//==- free_function_queries_interface.cpp - SYCL free queries interface -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <type_traits>

template <int Dims> struct this_id_caller {
  auto operator()() const
      -> decltype(sycl::ext::oneapi::experimental::this_id<Dims>()) {
    return sycl::ext::oneapi::experimental::this_id<Dims>();
  }
};

template <int Dims> struct this_item_caller {
  auto operator()() const
      -> decltype(sycl::ext::oneapi::experimental::this_item<Dims>()) {
    return sycl::ext::oneapi::experimental::this_item<Dims>();
  }
};

template <int Dims> struct this_nd_item_caller {
  auto operator()() const
      -> decltype(sycl::ext::oneapi::experimental::this_nd_item<Dims>()) {
    return sycl::ext::oneapi::experimental::this_nd_item<Dims>();
  }
};

template <int Dims> struct this_group_caller {
  auto operator()() const
      -> decltype(sycl::ext::oneapi::experimental::this_group<Dims>()) {
    return sycl::ext::oneapi::experimental::this_group<Dims>();
  }
};

template <template <int> class IterationPoint, int Dims,
          template <int> class Callable>
void test(Callable<Dims> &&callable) {
  static_assert(std::is_same<decltype(callable()), IterationPoint<Dims>>::value,
                "Wrong return type of free function query");
}

template <template <int, bool = true> class Item, int Dims> void test() {
  static_assert(
      std::is_same<decltype(this_item_caller<Dims>{}()), Item<Dims>>::value,
      "Wrong return type of free function query for Item");
}

SYCL_EXTERNAL void test_all() {
  test<sycl::id>(this_id_caller<1>{});
  test<sycl::id>(this_id_caller<2>{});
  test<sycl::id>(this_id_caller<3>{});

  test<sycl::item, 1>();
  test<sycl::item, 2>();
  test<sycl::item, 3>();

  test<sycl::nd_item>(this_nd_item_caller<1>{});
  test<sycl::nd_item>(this_nd_item_caller<2>{});
  test<sycl::nd_item>(this_nd_item_caller<3>{});

  test<sycl::group>(this_group_caller<1>{});
  test<sycl::group>(this_group_caller<2>{});
  test<sycl::group>(this_group_caller<3>{});

  static_assert(
      std::is_same<decltype(sycl::ext::oneapi::experimental::this_sub_group()),
                   sycl::ext::oneapi::sub_group>::value,
      "Wrong return type of free function query for Sub Group");
}
