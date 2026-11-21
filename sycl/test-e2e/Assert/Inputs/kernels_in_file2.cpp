//===-------------------------kernels_in_file2.cpp-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provides device functions and kernels
/// shared by the multi-translation-unit assertion tests.
///
//===----------------------------------------------------------------------===//
#include "kernels_in_file2.hpp"

#ifdef DEFINE_NDEBUG_INFILE2
#define NDEBUG
#else
#undef NDEBUG
#endif

#include <cassert>

using namespace sycl;
using namespace sycl::access;

int calculus(int X) {
  assert(X && "this message from calculus");
  return X * 2;
}

void check_nil(int value) { assert(value && "this message from file2"); }

static constexpr size_t BUFFER_SIZE = 4;

void enqueueKernel_1_fromFile2(queue *Q, std::latch *Sync) {
  sycl::range<1> numOfItems{BUFFER_SIZE};
  sycl::buffer<int, 1> Buf(numOfItems);
  if (Sync)
    Sync->arrive_and_wait();

  Q->submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<access_mode::read_write>(CGH);

    CGH.parallel_for<class kernel1_from_separate_file>(numOfItems,
                                                       [=](sycl::id<1> wiID) {
                                                         Acc[wiID] = wiID % 2;
                                                         check_nil(Acc[wiID]);
                                                       });
  });
}

void enqueueKernel_2_fromFile2(queue *Q, std::latch *Sync) {
  sycl::range<1> numOfItems{BUFFER_SIZE};
  sycl::buffer<int, 1> Buf(numOfItems);
  if (Sync)
    Sync->arrive_and_wait();

  Q->submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<access_mode::read_write>(CGH);

    CGH.parallel_for<class kernel2_from_separate_file>(numOfItems,
                                                       [=](sycl::id<1> wiID) {
                                                         Acc[wiID] = wiID % 2;
                                                         check_nil(Acc[wiID]);
                                                       });
  });
}
