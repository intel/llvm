// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
//==--------------- item.cpp - SYCL item test ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

using cl::sycl::detail::Builder;

int main() {
  // one dimension item without offset
  cl::sycl::item<1, false> one_dim = Builder::createItem<1, false>({4}, {2});
  assert(one_dim.get_id() == cl::sycl::id<1>{2});
  assert(one_dim.get_id(0) == 2);
  assert(one_dim[0] == 2);
  assert(one_dim.get_range() == cl::sycl::range<1>{4});
  assert(one_dim.get_range(0) == 4);
  assert(one_dim.get_linear_id() == 2);

  // two dimension item without offset
  cl::sycl::item<2, false> two_dim =
      Builder::createItem<2, false>({4, 8}, {2, 4});
  assert((two_dim.get_id() == cl::sycl::id<2>{2, 4}));
  assert(two_dim.get_id(0) == 2);
  assert(two_dim.get_id(1) == 4);
  assert(two_dim[0] == 2);
  assert(two_dim[1] == 4);
  assert((two_dim.get_range() == cl::sycl::range<2>{4, 8}));
  assert((two_dim.get_range(0) == 4));
  assert((two_dim.get_range(1) == 8));
  assert(two_dim.get_linear_id() == 20);

  // three dimension item without offset
  cl::sycl::item<3, false> three_dim =
      Builder::createItem<3, false>({4, 8, 16}, {2, 4, 8});
  assert((three_dim.get_id() == cl::sycl::id<3>{2, 4, 8}));
  assert(three_dim.get_id(0) == 2);
  assert(three_dim.get_id(1) == 4);
  assert(three_dim.get_id(2) == 8);
  assert(three_dim[0] == 2);
  assert(three_dim[1] == 4);
  assert(three_dim[2] == 8);
  assert((three_dim.get_range() == cl::sycl::range<3>{4, 8, 16}));
  assert((three_dim.get_range(0) == 4));
  assert((three_dim.get_range(1) == 8));
  assert((three_dim.get_range(2) == 16));
  assert(three_dim.get_linear_id() == 328);


  using value_type = decltype(std::declval<cl::sycl::item<1>>()[0]);
  static_assert(!std::is_reference<value_type>::value,
                "Expected a non-reference type");
}

