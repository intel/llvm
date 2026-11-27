//===----------------------assert_in_multiple_tus.hpp----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests device-side assertion failures in kernels defined in two translation
/// units.
///
//===----------------------------------------------------------------------===//
#include "Inputs/kernels_in_file2.hpp"
#include <iostream>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>

#ifdef DEFINE_NDEBUG_INFILE1
#define NDEBUG
#else
#undef NDEBUG
#endif

#include <cassert>

using namespace sycl;
using namespace sycl::access;

static constexpr size_t BUFFER_SIZE = 16;

int checkFunction() {
  int X = calculus(0);
  assert(X && "Nil in result");
  return X;
}

void enqueueKernel_1_fromFile1(queue *Q) {
  sycl::range<1> numOfItems{BUFFER_SIZE};
  sycl::buffer<int, 1> Buf(numOfItems);

  Q->submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<access_mode::read_write>(CGH);

    CGH.parallel_for<class Kernel_1>(numOfItems, [=](sycl::id<1> wiID) {
      int X = 0;
      if (wiID == BUFFER_SIZE - 1)
        X = checkFunction();
      Acc[wiID] = X;
    });
  });
}

int main() {

  queue Q({sycl::property::queue::in_order{}});
  enqueueKernel_1_fromFile1(&Q);
  // Should wait for enqueueKernel1 to trigger assert to abort the program,
  // otherwise we still see assert error msg from file2 since the kernel from
  // file2 will be launched before kernel from file1 finishes.
  Q.wait();
  enqueueKernel_2_fromFile2(&Q);
  Q.wait();

  std::cout << "The test ended." << std::endl;
  return 0;
}
