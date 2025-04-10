// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==- free_function_queries_sub_group.cpp - SYCL free queries for sub group -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//

#include "../../SubGroup/helper.hpp"
#include <sycl/ext/oneapi/free_function_queries.hpp>

#include <cassert>
#include <iostream>
#include <type_traits>

int main() {
  constexpr std::size_t n = 10;

  int data[n]{};
  int counter{0};
  {
    constexpr int checks_number{4};
    int results[checks_number]{41, 42, 43, 44};
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

          auto that_sub_group =
              sycl::ext::oneapi::this_work_item::get_sub_group();
          results_acc[1] = that_sub_group.get_local_linear_id() ==
                           that_nd_item.get_sub_group().get_local_linear_id();
          results_acc[2] = that_sub_group.get_local_id() ==
                           that_nd_item.get_sub_group().get_local_id();
          results_acc[3] = that_sub_group.get_local_range() ==
                           that_nd_item.get_sub_group().get_local_range();

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
