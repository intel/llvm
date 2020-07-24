// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==- free_function_queries.cpp - SYCL free function queries test -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <cassert>
#include <iostream>
#include <type_traits>

int main() {
  constexpr std::size_t n = 10;

  int data[n]{};
  int counter{0};
  {
    {
      sycl::buffer<int> buf(data, sycl::range<1>(n));
      sycl::queue q;
      q.submit([&](cl::sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::global_buffer>
            acc(buf.get_access<sycl::access::mode::write>(cgh));
        cgh.parallel_for<class IdTest>(n, [=](sycl::id<1> i) {
          auto this_id = sycl::this_id<1>();
          assert(this_id == i);
          auto that_item = sycl::this_item<1>();
          assert(that_item.get_id() == i);
          acc[i]++;
        });
      });
    }
    ++counter;
    for (int i = 0; i < n; i++) {
      assert(data[i] == counter);
    }
  }

  {
    {
      sycl::buffer<int> buf(data, sycl::range<1>(n));
      sycl::queue q;
      q.submit([&](cl::sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::global_buffer>
            acc(buf.get_access<sycl::access::mode::write>(cgh));
        cgh.parallel_for<class ItemTest>(n, [=](auto i) {
          static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                        "lambda arg type is unexpected");
          auto that_id = sycl::this_id<1>();
          assert(that_id == i.get_id());
          auto that_item = sycl::this_item<1>();
          assert(that_item == i);
          acc[i]++;
        });
      });
    }
    ++counter;
    for (int i = 0; i < n; i++) {
      assert(data[i] == counter);
    }
  }

  {
    {
      sycl::buffer<int> buf(data, sycl::range<1>(n));
      sycl::queue q;
      sycl::id<1> offset(1);
      q.submit([&](cl::sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::global_buffer>
            acc(buf.get_access<sycl::access::mode::write>(cgh));
        cgh.parallel_for<class ItemOffsetTest>(
            sycl::range<1>{n}, offset, [=](sycl::item<1, true> i) {
              auto that_id = sycl::this_id<1>();
              assert(that_id == i.get_id());
              auto that_item = sycl::this_item<1>();
              assert(that_item == i);
              acc[that_item.get_linear_id()]++;
            });
      });
    }
    ++counter;
    for (int i = 0; i < n; i++) {
      assert(data[i] == counter);
    }
  }

  {
    {
      sycl::buffer<int> buf(data, sycl::range<1>(n));
      sycl::queue q;
      sycl::nd_range<1> NDR(sycl::range<1>{n}, sycl::range<1>{2});
      q.submit([&](cl::sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::global_buffer>
            acc(buf.get_access<sycl::access::mode::write>(cgh));
        cgh.parallel_for<class NdItemTest>(NDR, [=](auto nd_i) {
          static_assert(std::is_same<decltype(nd_i), sycl::nd_item<1>>::value,
                        "lambda arg type is unexpected");
          auto that_nd_item = sycl::this_nd_item<1>();
          assert(that_nd_item == nd_i);
          auto nd_item_group = that_nd_item.get_group();
          assert(nd_item_group == sycl::this_group<1>());
          auto nd_item_id = that_nd_item.get_global_id();
          assert(nd_item_id == sycl::this_id<1>());
          auto that_item = sycl::this_item<1>();
          assert(nd_item_id == that_item.get_id());
          assert(that_nd_item.get_global_range() == that_item.get_range());

          acc[that_nd_item.get_global_id(0)]++;
        });
      });
    }
    ++counter;
    for (int i = 0; i < n; i++) {
      assert(data[i] == counter);
    }
  }
}
