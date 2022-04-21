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
  // one dimension item with offset
  cl::sycl::item<1, true> one_dim_with_offset =
      Builder::createItem<1, true>({4}, {2}, {1});
  assert(one_dim_with_offset.get_id() == cl::sycl::id<1>{2});
  assert(one_dim_with_offset.get_id(0) == 2);
  assert(one_dim_with_offset[0] == 2);
  assert(one_dim_with_offset.get_range() == cl::sycl::range<1>{4});
  assert(one_dim_with_offset.get_range(0) == 4);
  assert(one_dim_with_offset.get_offset() == cl::sycl::id<1>{1});
  assert(one_dim_with_offset.get_linear_id() == 1);

  // two dimension item with offset
  cl::sycl::item<2, true> two_dim_with_offset =
      Builder::createItem<2, true>({4, 8}, {2, 4}, {1, 1});
  assert((two_dim_with_offset.get_id() == cl::sycl::id<2>{2, 4}));
  assert(two_dim_with_offset.get_id(0) == 2);
  assert(two_dim_with_offset.get_id(1) == 4);
  assert(two_dim_with_offset[0] == 2);
  assert(two_dim_with_offset[1] == 4);
  assert((two_dim_with_offset.get_range() == cl::sycl::range<2>{4, 8}));
  assert((two_dim_with_offset.get_range(0) == 4));
  assert((two_dim_with_offset.get_range(1) == 8));
  assert((two_dim_with_offset.get_offset() == cl::sycl::id<2>{1, 1}));
  assert(two_dim_with_offset.get_linear_id() == 11);

  // three dimension item with offset
  cl::sycl::item<3, true> three_dim_with_offset =
      Builder::createItem<3, true>({4, 8, 16}, {2, 4, 8}, {1, 1, 1});
  assert((three_dim_with_offset.get_id() == cl::sycl::id<3>{2, 4, 8}));
  assert(three_dim_with_offset.get_id(0) == 2);
  assert(three_dim_with_offset.get_id(1) == 4);
  assert(three_dim_with_offset.get_id(2) == 8);
  assert(three_dim_with_offset[0] == 2);
  assert(three_dim_with_offset[1] == 4);
  assert(three_dim_with_offset[2] == 8);
  assert((three_dim_with_offset.get_range() == cl::sycl::range<3>{4, 8, 16}));
  assert((three_dim_with_offset.get_range(0) == 4));
  assert((three_dim_with_offset.get_range(1) == 8));
  assert((three_dim_with_offset.get_range(2) == 16));
  assert((three_dim_with_offset.get_offset() == cl::sycl::id<3>{1, 1, 1}));
  assert(three_dim_with_offset.get_linear_id() == 183);

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

  // A conversion to item with offset
  cl::sycl::item<1, true> one_dim_transformed = one_dim;
  cl::sycl::item<1, true> one_dim_check =
      Builder::createItem<1, true>({4}, {2}, {0});
  assert(one_dim_transformed == one_dim_check);
  cl::sycl::item<2, true> two_dim_transformed = two_dim;
  cl::sycl::item<2, true> two_dim_check =
      Builder::createItem<2, true>({4, 8}, {2, 4}, {0, 0});
  assert(two_dim_transformed == two_dim_check);
  cl::sycl::item<3, true> three_dim_transformed = three_dim;
  cl::sycl::item<3, true> three_dim_check =
      Builder::createItem<3, true>({4, 8, 16}, {2, 4, 8}, {0, 0, 0});
  assert((three_dim_transformed == three_dim_check));

  using value_type = decltype(std::declval<cl::sycl::item<1>>()[0]);
  static_assert(!std::is_reference<value_type>::value,
                "Expected a non-reference type");
}

