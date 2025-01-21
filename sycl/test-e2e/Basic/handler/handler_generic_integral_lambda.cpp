// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//==-------------- handler_generic_integral_lambda.cpp ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/detail/core.hpp>

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

int main() {
  constexpr std::size_t length = 10;

  {
    int data[length];
    {
      sycl::buffer<int> buf(data, sycl::range<1>(length));
      sycl::queue q;
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::device>
            acc(buf.get_access<sycl::access::mode::write>(cgh));
        cgh.parallel_for<class GenericLambda>(
            length, [=](auto item) { acc[item.get_id()] = item; });
      });
    }
    for (int i = 0; i < length; i++) {
      assert(data[i] == i);
    }
  }

  {
    int data[length];
    {
      sycl::buffer<int> buf(data, sycl::range<1>(length));
      sycl::queue q;
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::device>
            acc(buf.get_access<sycl::access::mode::write>(cgh));
        cgh.parallel_for<class SizeTLambda>(
            length, [=](std::size_t item) { acc[item] = item; });
      });
    }
    for (int i = 0; i < length; i++) {
      assert(data[i] == i);
    }
  }

  {
    int data[length];
    {
      sycl::buffer<int> buf(data, sycl::range<1>(length));
      sycl::queue q;
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::device>
            acc(buf.get_access<sycl::access::mode::write>(cgh));
        cgh.parallel_for<class IntLambda>(length,
                                          [=](int item) { acc[item] = item; });
      });
    }
    for (int i = 0; i < length; i++) {
      assert(data[i] == i);
    }
  }

  return 0;
}
