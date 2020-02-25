// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
//==--------------- nd_range.cpp - SYCL nd_range test ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

using namespace std;
int main() {
  cl::sycl::nd_range<1> one_dim_nd_range_offset({4}, {2}, {1});
  assert(one_dim_nd_range_offset.get_global_range() == cl::sycl::range<1>(4));
  assert(one_dim_nd_range_offset.get_local_range() == cl::sycl::range<1>(2));
  assert(one_dim_nd_range_offset.get_group_range() == cl::sycl::range<1>(2));
  assert(one_dim_nd_range_offset.get_offset() == cl::sycl::id<1>(1));
  cout << "one_dim_nd_range_offset passed " << endl;

  cl::sycl::nd_range<2> two_dim_nd_range_offset({8, 16}, {4, 8}, {1, 1});
  assert(two_dim_nd_range_offset.get_global_range() == cl::sycl::range<2>(8, 16));
  assert(two_dim_nd_range_offset.get_local_range() == cl::sycl::range<2>(4, 8));
  assert(two_dim_nd_range_offset.get_group_range() == cl::sycl::range<2>(2, 2));
  assert(two_dim_nd_range_offset.get_offset() == cl::sycl::id<2>(1, 1));
  cout << "two_dim_nd_range_offset passed " << endl;

  cl::sycl::nd_range<3> three_dim_nd_range_offset({32, 64, 128}, {16, 32, 64},
                                                  {1, 1, 1});
  assert(three_dim_nd_range_offset.get_global_range() ==
         cl::sycl::range<3>(32, 64, 128));
  assert(three_dim_nd_range_offset.get_local_range() ==
         cl::sycl::range<3>(16, 32, 64));
  assert(three_dim_nd_range_offset.get_group_range() == cl::sycl::range<3>(2, 2, 2));
  assert(three_dim_nd_range_offset.get_offset() == cl::sycl::id<3>(1, 1, 1));
  cout << "three_dim_nd_range_offset passed " << endl;

  cl::sycl::nd_range<1> one_dim_nd_range({4}, {2});
  assert(one_dim_nd_range.get_global_range() == cl::sycl::range<1>(4));
  assert(one_dim_nd_range.get_local_range() == cl::sycl::range<1>(2));
  assert(one_dim_nd_range.get_group_range() == cl::sycl::range<1>(2));
  assert(one_dim_nd_range.get_offset() == cl::sycl::id<1>(0));
  cout << "one_dim_nd_range passed " << endl;

  cl::sycl::nd_range<2> two_dim_nd_range({8, 16}, {4, 8});
  assert(two_dim_nd_range.get_global_range() == cl::sycl::range<2>(8, 16));
  assert(two_dim_nd_range.get_local_range() == cl::sycl::range<2>(4, 8));
  assert(two_dim_nd_range.get_group_range() == cl::sycl::range<2>(2, 2));
  assert(two_dim_nd_range.get_offset() == cl::sycl::id<2>(0, 0));
  cout << "two_dim_nd_range passed " << endl;

  cl::sycl::nd_range<3> three_dim_nd_range({32, 64, 128}, {16, 32, 64} );
  assert(three_dim_nd_range.get_global_range() == cl::sycl::range<3>(32, 64, 128));
  assert(three_dim_nd_range.get_local_range() == cl::sycl::range<3>(16, 32, 64));
  assert(three_dim_nd_range.get_group_range() == cl::sycl::range<3>(2, 2, 2));
  assert(three_dim_nd_range.get_offset() == cl::sycl::id<3>(0, 0, 0));
  cout << "three_dim_nd_range passed " << endl;
}
