// RUN: %{build} -o %t.out
// With awkward sizes (such as 1030) make sure that there is no range rounding,
// so `get_global_id` returns correct values.
// RUN: env SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING=1 %{run} %t.out

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
#include <vector>

int main() {
  constexpr std::size_t n = 1030;

  int data[n]{};
  int counter{0};

  {
    constexpr int checks_number{3};
    std::vector<int> results(n * checks_number, 0);
    {
      sycl::buffer<int> buf(data, sycl::range<1>(n));
      sycl::buffer<int> results_buf(results.data(),
                                    sycl::range<1>(results.size()));
      sycl::queue q;
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::device>
            acc(buf.get_access<sycl::access_mode::read_write>(cgh));
        sycl::accessor<int, 1, sycl::access_mode::write,
                       sycl::access::target::device>
            results_acc(results_buf.get_access<sycl::access_mode::write>(cgh));
        cgh.parallel_for<class IdTest>(n, [=](sycl::id<1> i) {
          const size_t ResultOffset = i[0] * checks_number;
          auto that_id = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
          results_acc[ResultOffset] = that_id.get_global_id() == i;

          auto that_item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
          results_acc[ResultOffset + 1] = that_item.get_global_id() == i;
          results_acc[ResultOffset + 2] =
              that_item.get_global_range() == sycl::range<1>(n);
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
    std::vector<int> results(n * checks_number, 0);
    {
      sycl::buffer<int> buf(data, sycl::range<1>(n));
      sycl::buffer<int> results_buf(results.data(),
                                    sycl::range<1>(results.size()));
      sycl::queue q;
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::device>
            acc(buf.get_access<sycl::access_mode::read_write>(cgh));
        sycl::accessor<int, 1, sycl::access_mode::write,
                       sycl::access::target::device>
            results_acc(results_buf.get_access<sycl::access_mode::write>(cgh));
        cgh.parallel_for<class ItemTest>(n, [=](auto i) {
          static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                        "lambda arg type is unexpected");
          const size_t ResultOffset = i.get_linear_id() * checks_number;
          auto that_id = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
          results_acc[ResultOffset] = that_id.get_global_id() == i;
          auto that_item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
          results_acc[ResultOffset + 1] = that_item.get_global_id() == i;
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
// Make sure that we ignore global offset warning.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    constexpr int checks_number{2};
    std::vector<int> results(n * checks_number, 0);
    {
      sycl::buffer<int> buf(data, sycl::range<1>(n));
      sycl::buffer<int> results_buf(results.data(),
                                    sycl::range<1>(results.size()));
      sycl::queue q;
      sycl::id<1> offset(1);
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::device>
            acc(buf.get_access<sycl::access_mode::read_write>(cgh));
        sycl::accessor<int, 1, sycl::access_mode::write,
                       sycl::access::target::device>
            results_acc(results_buf.get_access<sycl::access_mode::write>(cgh));
        cgh.parallel_for<class ItemOffsetTest>(
            sycl::range<1>{n}, offset, [=](sycl::item<1, true> i) {
              const size_t ResultOffset = i.get_linear_id() * checks_number;
              auto that_id =
                  sycl::ext::oneapi::this_work_item::get_nd_item<1>();
              results_acc[ResultOffset] = i.get_id() == that_id.get_global_id();
              auto that_item =
                  sycl::ext::oneapi::this_work_item::get_nd_item<1>();
              results_acc[ResultOffset + 1] = i == that_item.get_global_id();
              acc[that_item.get_global_linear_id()]++;
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
#pragma GCC diagnostic pop
  }
}
