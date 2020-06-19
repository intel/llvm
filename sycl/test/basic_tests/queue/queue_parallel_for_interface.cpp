// UNSUPPORTED: cuda
// CUDA does not support unnamed lambdas.
//
// RUN: %clangxx -fsycl -fsyntax-only -fsycl-unnamed-lambda %s -o %t.out

//==- queue_parallel_for_generic.cpp - SYCL queue parallel_for interface test -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <iostream>
#include <type_traits>

template <std::size_t... Is>
void test_range_impl(sycl::queue q, std::index_sequence<Is...>,
                     sycl::range<sizeof...(Is)> *) {
  constexpr auto dims = sizeof...(Is);

  q.parallel_for(sycl::range<dims>{Is...}, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<dims>>::value,
                  "lambda arg type is unexpected");
  });
}

template <std::size_t... Is>
void test_range_impl(sycl::queue q, std::index_sequence<Is...>,
                     sycl::nd_range<sizeof...(Is)> *) {
  constexpr auto dims = sizeof...(Is);

  sycl::nd_range<dims> ndr{sycl::range<dims>{Is...}, sycl::range<dims>{Is...}};
  q.parallel_for(ndr, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::nd_item<dims>>::value,
                  "lambda arg type is unexpected");
  });
}

template <template <int> class Range, std::size_t Dims>
void test_range(sycl::queue q) {
  test_range_impl(q, std::make_index_sequence<Dims>{},
                  static_cast<Range<Dims> *>(nullptr));
}

void test_number_braced_init_list(sycl::queue q) {
  constexpr auto n = 1;
  q.parallel_for(n, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
  });

  q.parallel_for({n}, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
  });

  q.parallel_for({n, n}, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<2>>::value,
                  "lambda arg type is unexpected");
  });

  q.parallel_for({n, n, n}, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<3>>::value,
                  "lambda arg type is unexpected");
  });
}

int main() {
  sycl::queue q{};

  test_number_braced_init_list(q);

  test_range<sycl::range, 1>(q);
  test_range<sycl::range, 2>(q);
  test_range<sycl::range, 3>(q);
  test_range<sycl::nd_range, 1>(q);
  test_range<sycl::nd_range, 2>(q);
  test_range<sycl::nd_range, 3>(q);
}
