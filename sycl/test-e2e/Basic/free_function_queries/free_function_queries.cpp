// FIXME: Investigate OS-agnostic failures
// REQUIRES: TEMPORARY_DISABLED
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: windows
// The failure is caused by intel/llvm#5213

//==- free_function_queries.cpp - SYCL free function queries test -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>

#include <cassert>
#include <iostream>
#include <type_traits>

int main() {
  constexpr std::size_t n = 1030;

  int data[n]{};
  int counter{0};

  {
    constexpr int checks_number{3};
    int results[checks_number]{};
    {
      sycl::buffer<int> buf(data, sycl::range<1>(n));
      sycl::buffer<int> results_buf(results, sycl::range<1>(checks_number));
      sycl::queue q;
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::device>
            acc(buf.get_access<sycl::access::mode::write>(cgh));
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::device>
            results_acc(results_buf.get_access<sycl::access::mode::write>(cgh));
        cgh.parallel_for<class IdTest>(n, [=](sycl::id<1> i) {
          auto that_id = sycl::ext::oneapi::experimental::this_id<1>();
          results_acc[0] = that_id == i;

          auto that_item = sycl::ext::oneapi::experimental::this_item<1>();
          results_acc[1] = that_item.get_id() == i;
          results_acc[2] = that_item.get_range() == sycl::range<1>(n);
          acc[i]++;
        });
      });
    }
    ++counter;
    for (int i = 0; i < n; i++) {
      assert(data[i] == counter);
    }
    for (auto val : results) {
      assert(val == 1);
    }
  }

  {
    constexpr int checks_number{2};
    int results[checks_number]{};
    {
      sycl::buffer<int> buf(data, sycl::range<1>(n));
      sycl::buffer<int> results_buf(results, sycl::range<1>(checks_number));
      sycl::queue q;
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::device>
            acc(buf.get_access<sycl::access::mode::write>(cgh));
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::device>
            results_acc(results_buf.get_access<sycl::access::mode::write>(cgh));
        cgh.parallel_for<class ItemTest>(n, [=](auto i) {
          static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                        "lambda arg type is unexpected");
          auto that_id = sycl::ext::oneapi::experimental::this_id<1>();
          results_acc[0] = i.get_id() == that_id;
          auto that_item = sycl::ext::oneapi::experimental::this_item<1>();
          results_acc[1] = i == that_item;
          acc[i]++;
        });
      });
    }
    ++counter;
    for (int i = 0; i < n; i++) {
      assert(data[i] == counter);
    }
    for (auto val : results) {
      assert(val == 1);
    }
  }

  {
    constexpr int checks_number{2};
    int results[checks_number]{};
    {
      sycl::buffer<int> buf(data, sycl::range<1>(n));
      sycl::buffer<int> results_buf(results, sycl::range<1>(checks_number));
      sycl::queue q;
      sycl::id<1> offset(1);
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::device>
            acc(buf.get_access<sycl::access::mode::write>(cgh));
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::device>
            results_acc(results_buf.get_access<sycl::access::mode::write>(cgh));
        cgh.parallel_for<class ItemOffsetTest>(
            sycl::range<1>{n}, offset, [=](sycl::item<1, true> i) {
              auto that_id = sycl::ext::oneapi::experimental::this_id<1>();
              results_acc[0] = i.get_id() == that_id;
              auto that_item = sycl::ext::oneapi::experimental::this_item<1>();
              results_acc[1] = i == that_item;
              acc[that_item.get_linear_id()]++;
            });
      });
    }
    ++counter;
    for (int i = 0; i < n; i++) {
      assert(data[i] == counter);
    }
    for (auto val : results) {
      assert(val == 1);
    }
  }

  {
    constexpr int checks_number{5};
    int results[checks_number]{};
    {
      sycl::buffer<int> buf(data, sycl::range<1>(n));
      sycl::buffer<int> results_buf(results, sycl::range<1>(checks_number));
      sycl::queue q;
      sycl::nd_range<1> NDR(sycl::range<1>{n}, sycl::range<1>{2});
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::device>
            acc(buf.get_access<sycl::access::mode::write>(cgh));
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::device>
            results_acc(results_buf.get_access<sycl::access::mode::write>(cgh));
        cgh.parallel_for<class NdItemTest>(NDR, [=](auto nd_i) {
          static_assert(std::is_same<decltype(nd_i), sycl::nd_item<1>>::value,
                        "lambda arg type is unexpected");
          auto that_nd_item =
              sycl::ext::oneapi::this_work_item::get_nd_item<1>();
          results_acc[0] = that_nd_item == nd_i;
          auto nd_item_group = that_nd_item.get_group();
          results_acc[1] =
              nd_item_group ==
              sycl::ext::oneapi::this_work_item::get_work_group<1>();
          auto nd_item_id = that_nd_item.get_global_id();
          results_acc[2] =
              nd_item_id == sycl::ext::oneapi::experimental::this_id<1>();
          auto that_item = sycl::ext::oneapi::experimental::this_item<1>();
          results_acc[3] = nd_item_id == that_item.get_id();
          results_acc[4] =
              that_nd_item.get_global_range() == that_item.get_range();

          acc[that_nd_item.get_global_id(0)]++;
        });
      });
    }
    ++counter;
    for (int i = 0; i < n; i++) {
      assert(data[i] == counter);
    }
    for (auto val : results) {
      assert(val == 1);
    }
  }
}
