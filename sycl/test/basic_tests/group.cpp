// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

//==--------------- group.cpp - SYCL group test ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace std;
using sycl::detail::Builder;

int main() {
  sycl::group<1> one = Builder::createGroup<1>({8}, {4}, {1});
  // one dimension group
  sycl::group<1> one_dim = Builder::createGroup<1>({8}, {4}, {1});
  assert(one_dim.get_id() == sycl::id<1>{1});
  assert(one_dim.get_id(0) == 1);
  assert((one_dim.get_global_range() == sycl::range<1>{8}));
  assert(one_dim.get_global_range(0) == 8);
  assert((one_dim.get_local_range() == sycl::range<1>{4}));
  assert(one_dim.get_local_range(0) == 4);
  assert((one_dim.get_group_range() == sycl::range<1>{2}));
  assert(one_dim.get_group_range(0) == 2);
  assert(one_dim[0] == 1);
  assert(one_dim.get_linear_id() == 1);
  assert(one_dim.get_group_linear_id() == 1);

  try {
    one_dim.get_local_id();
    assert(one_dim.get_local_id(0) == one_dim.get_local_id()[0]);
    assert(0); // get_local_id() is not implemented on host device
  } catch (sycl::runtime_error) {
  }

  try {
    one_dim.get_local_linear_id();
    assert(0); // get_local_id() is not implemented on host device
  } catch (sycl::runtime_error) {
  }

  // two dimension group
  sycl::group<2> two_dim = Builder::createGroup<2>({8, 4}, {4, 2}, {1, 1});
  assert((two_dim.get_id() == sycl::id<2>{1, 1}));
  assert(two_dim.get_id(0) == 1);
  assert(two_dim.get_id(1) == 1);
  assert((two_dim.get_global_range() == sycl::range<2>{8, 4}));
  assert(two_dim.get_global_range(0) == 8);
  assert(two_dim.get_global_range(1) == 4);
  assert((two_dim.get_local_range() == sycl::range<2>{4, 2}));
  assert(two_dim.get_local_range(0) == 4);
  assert(two_dim.get_local_range(1) == 2);
  assert((two_dim.get_group_range() == sycl::range<2>{2, 2}));
  assert(two_dim.get_group_range(0) == 2);
  assert(two_dim.get_group_range(1) == 2);
  assert(two_dim[0] == 1);
  assert(two_dim[1] == 1);
  assert(two_dim.get_linear_id() == 3);
  assert(two_dim.get_group_linear_id() == 3);

  try {
    two_dim.get_local_id();
    assert(two_dim.get_local_id(0) == two_dim.get_local_id()[0]);
    assert(two_dim.get_local_id(1) == two_dim.get_local_id()[1]);
    assert(0); // get_local_id() is not implemented on host device
  } catch (sycl::runtime_error) {
  }

  try {
    two_dim.get_local_linear_id();
    assert(0); // get_local_id() is not implemented on host device
  } catch (sycl::runtime_error) {
  }

  // three dimension group
  sycl::group<3> three_dim =
      Builder::createGroup<3>({16, 8, 4}, {8, 4, 2}, {1, 1, 1});
  assert((three_dim.get_id() == sycl::id<3>{1, 1, 1}));
  assert(three_dim.get_id(0) == 1);
  assert(three_dim.get_id(1) == 1);
  assert(three_dim.get_id(2) == 1);
  assert((three_dim.get_global_range() == sycl::range<3>{16, 8, 4}));
  assert(three_dim.get_global_range(0) == 16);
  assert(three_dim.get_global_range(1) == 8);
  assert(three_dim.get_global_range(2) == 4);
  assert((three_dim.get_local_range() == sycl::range<3>{8, 4, 2}));
  assert(three_dim.get_local_range(0) == 8);
  assert(three_dim.get_local_range(1) == 4);
  assert(three_dim.get_local_range(2) == 2);
  assert((three_dim.get_group_range() == sycl::range<3>{2, 2, 2}));
  assert(three_dim.get_group_range(0) == 2);
  assert(three_dim.get_group_range(1) == 2);
  assert(three_dim.get_group_range(2) == 2);
  assert(three_dim[0] == 1);
  assert(three_dim[1] == 1);
  assert(three_dim[2] == 1);
  assert(three_dim.get_linear_id() == 7);
  assert(three_dim.get_group_linear_id() == 7);

  try {
    three_dim.get_local_id();
    assert(three_dim.get_local_id(0) == three_dim.get_local_id()[0]);
    assert(three_dim.get_local_id(1) == three_dim.get_local_id()[1]);
    assert(three_dim.get_local_id(2) == three_dim.get_local_id()[2]);
    assert(0); // get_local_id() is not implemented on host device
  } catch (sycl::runtime_error) {
  }

  try {
    three_dim.get_local_linear_id();
    assert(0); // get_local_id() is not implemented on host device
  } catch (sycl::runtime_error) {
  }
}
