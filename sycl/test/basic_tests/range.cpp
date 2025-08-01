// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
// XFAIL: libcxx
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19616
//==--------------- range.cpp - SYCL range test ----------------------------==//
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
int main() {
  sycl::range<1> one_dim_range(64);
  sycl::range<2> two_dim_range(64, 1);
  sycl::range<3> three_dim_range(64, 1, 2);
  assert(one_dim_range.size() ==64);
  assert(one_dim_range.get(0) ==64);
  assert(one_dim_range[0] ==64);
  cout << "one_dim_range passed " << endl;
  assert(two_dim_range.size() ==64);
  assert(two_dim_range.get(0) ==64);
  assert(two_dim_range[0] ==64);
  assert(two_dim_range.get(1) ==1);
  assert(two_dim_range[1] ==1);
  cout << "two_dim_range passed " << endl;
  assert(three_dim_range.size() ==128);
  assert(three_dim_range.get(0) ==64);
  assert(three_dim_range[0] ==64);
  assert(three_dim_range.get(1) ==1);
  assert(three_dim_range[1] ==1);
  assert(three_dim_range.get(2) ==2);
  assert(three_dim_range[2] ==2);
  cout << "three_dim_range passed " << endl;

  sycl::range<1> one_dim_range_neg(-64);
  sycl::range<1> one_dim_range_copy(64);
  sycl::range<2> two_dim_range_neg(-64, -1);
  sycl::range<2> two_dim_range_copy(64, 1);
  sycl::range<3> three_dim_range_copy(64, 1, 2);
  sycl::range<3> three_dim_range_neg(-64, -1, -2);

  assert((+one_dim_range) == one_dim_range);
  assert(-one_dim_range == one_dim_range_neg);
  assert((+two_dim_range) == two_dim_range);
  assert(-two_dim_range == two_dim_range_neg);
  assert((+three_dim_range) == three_dim_range);
  assert(-three_dim_range == three_dim_range_neg);

  assert((++one_dim_range) == (one_dim_range_copy + 1));
  assert((--one_dim_range) == (one_dim_range_copy));
  assert((++two_dim_range) == (two_dim_range_copy + 1));
  assert((--two_dim_range) == (two_dim_range_copy));
  assert((++three_dim_range) == (three_dim_range_copy + 1));
  assert((--three_dim_range) == (three_dim_range_copy));

  assert((one_dim_range++) == (one_dim_range_copy));
  assert((one_dim_range--) == (one_dim_range_copy + 1));
  assert((two_dim_range++) == (two_dim_range_copy));
  assert((two_dim_range--) == (two_dim_range_copy + 1));
  assert((three_dim_range++) == (three_dim_range_copy));
  assert((three_dim_range--) == (three_dim_range_copy + 1));

  sycl::range<3> default3;
  sycl::range<2> default2;
  sycl::range<1> default1;

  assert(default3[0] == 0 && default3[1] == 0 && default3[2] == 0);
  assert(default2[0] == 0 && default2[1] == 0);
  assert(default1[0] == 0);
}
