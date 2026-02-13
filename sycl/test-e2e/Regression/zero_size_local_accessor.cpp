// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==---- zero_size_local_accessor.cpp - SYCL 0-size local accessor test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
     sycl::local_accessor<uint8_t, 1> ZeroSizeLocalAcc(sycl::range<1>(0), CGH);
     CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
       if (ZeroSizeLocalAcc.get_range()[0])
         ZeroSizeLocalAcc[0] = 1;
     });
   }).wait();
  return 0;
}
