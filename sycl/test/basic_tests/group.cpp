// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
// RUN: not --crash %t.out get_local_id_1d
// RUN: not --crash %t.out get_local_linear_id_1d
// RUN: not --crash %t.out get_local_id_2d
// RUN: not --crash %t.out get_local_linear_id_2d
// RUN: not --crash %t.out get_local_id_3d
// RUN: not --crash %t.out get_local_linear_id_3d

// XFAIL: libcxx
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19616

//==--------------- group.cpp - SYCL group test ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <cstring>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace std;
using sycl::detail::Builder;

// get_local_id()/get_local_linear_id() are noexcept and call std::terminate()
// when unimplemented on host, so each case is run as a separate death test
// rather than caught via try/catch.
static void runDeathCase(const char *mode) {
  sycl::group<1> one_dim = Builder::createGroup<1>({8}, {4}, {1});
  sycl::group<2> two_dim = Builder::createGroup<2>({8, 4}, {4, 2}, {1, 1});
  sycl::group<3> three_dim =
      Builder::createGroup<3>({16, 8, 4}, {8, 4, 2}, {1, 1, 1});

  if (strcmp(mode, "get_local_id_1d") == 0)
    one_dim.get_local_id();
  else if (strcmp(mode, "get_local_linear_id_1d") == 0)
    one_dim.get_local_linear_id();
  else if (strcmp(mode, "get_local_id_2d") == 0)
    two_dim.get_local_id();
  else if (strcmp(mode, "get_local_linear_id_2d") == 0)
    two_dim.get_local_linear_id();
  else if (strcmp(mode, "get_local_id_3d") == 0)
    three_dim.get_local_id();
  else if (strcmp(mode, "get_local_linear_id_3d") == 0)
    three_dim.get_local_linear_id();
}

int main(int argc, char **argv) {
  if (argc > 1) {
    runDeathCase(argv[1]);
    return 0; // Unreachable: runDeathCase() above must have terminated.
  }

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
}
