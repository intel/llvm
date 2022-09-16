// RUN: %clangxx -fsycl -fsyntax-only %s -o %t.out

//==- queue_parallel_for_generic.cpp - SYCL queue parallel_for interface test -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>
#include <type_traits>

template <typename KernelName, std::size_t... Is>
void test_range_impl(sycl::queue q, std::index_sequence<Is...>,
                     sycl::range<sizeof...(Is)> *) {
  constexpr auto dims = sizeof...(Is);

  q.parallel_for<KernelName>(sycl::range<dims>{Is...}, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<dims>>::value,
                  "lambda arg type is unexpected");
  });
}

template <typename KernelName, std::size_t... Is>
void test_range_impl(sycl::queue q, std::index_sequence<Is...>,
                     sycl::nd_range<sizeof...(Is)> *) {
  constexpr auto dims = sizeof...(Is);

  sycl::nd_range<dims> ndr{sycl::range<dims>{Is...}, sycl::range<dims>{Is...}};
  q.parallel_for<KernelName>(ndr, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::nd_item<dims>>::value,
                  "lambda arg type is unexpected");
  });
}

template <typename KernelName, template <int> class Range, std::size_t Dims>
void test_range(sycl::queue q) {
  test_range_impl<KernelName>(q, std::make_index_sequence<Dims>{},
                              static_cast<Range<Dims> *>(nullptr));
}

template <typename KernelName, typename Arg, std::size_t... Is>
void test_braced_init_list_event_impl(sycl::queue q, Arg &&arg,
                                      std::index_sequence<Is...>,
                                      sycl::range<sizeof...(Is)> *) {
  constexpr auto dims = sizeof...(Is);

  q.parallel_for<KernelName>({Is...}, std::forward<Arg>(arg), [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<dims>>::value,
                  "lambda arg type is unexpected");
  });
}

template <typename KernelName, template <int Dims> class Range, std::size_t Dims, typename Arg>
void test_braced_init_list_event(sycl::queue q, Arg &&arg) {
  test_braced_init_list_event_impl<KernelName>(q, std::forward<Arg>(arg),
                                               std::make_index_sequence<Dims>{},
                                               static_cast<Range<Dims> *>(nullptr));
}

void test_number_braced_init_list(sycl::queue q) {
  constexpr auto n = 1;
  q.parallel_for<class Number>(n, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
  });

  q.parallel_for<class BracedInitList1>({n}, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
  });

  q.parallel_for<class BracedInitList2>({n, n}, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<2>>::value,
                  "lambda arg type is unexpected");
  });

  q.parallel_for<class BracedInitList3>({n, n, n}, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<3>>::value,
                  "lambda arg type is unexpected");
  });
}

void test_number_event(sycl::queue q, sycl::event e) {
  q.parallel_for<class NumberEvent>(1, e, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
  });
}

void test_number_event_vector(sycl::queue q,
                              const std::vector<sycl::event> &events) {
  q.parallel_for<class NumberEventVector>(1, events, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
  });
}

int main() {
  sycl::queue q{};

  test_range<class test_range1, sycl::range, 1>(q);
  test_range<class test_range2, sycl::range, 2>(q);
  test_range<class test_range3, sycl::range, 3>(q);
  test_range<class test_nd_range1, sycl::nd_range, 1>(q);
  test_range<class test_nd_range2, sycl::nd_range, 2>(q);
  test_range<class test_nd_range3, sycl::nd_range, 3>(q);

  test_number_braced_init_list(q);

  // We need to get to event for further API testsing
  // Getting them with two same calls
  auto e1 = q.parallel_for<class Event1>(1, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
  });

  // Getting the second event for further API calling
  auto e2 = q.parallel_for<class Event2>(1, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
  });

  test_number_event(q, e1);

  test_braced_init_list_event<class BracedInitList1Event, sycl::range, 1>(q, e1);
  test_braced_init_list_event<class BracedInitList2Event, sycl::range, 2>(q, e1);
  test_braced_init_list_event<class BracedInitList3Event, sycl::range, 3>(q, e1);

  test_number_event_vector(q, {e1, e2});

  std::vector<sycl::event> events{e1, e2};
  test_braced_init_list_event<class BracedInitList1EventVector, sycl::range, 1>(q, events);
  test_braced_init_list_event<class BracedInitList2EventVector, sycl::range, 2>(q, events);
  test_braced_init_list_event<class BracedInitList3EventVector, sycl::range, 3>(q, events);
}
