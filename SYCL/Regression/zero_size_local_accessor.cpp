// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==---- zero_size_local_accessor.cpp - SYCL 0-size local accessor test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
     sycl::local_accessor<uint8_t, 1> ZeroSizeLocalAcc(sycl::range<1>(0), CGH);
     CGH.single_task([=]() {
       if (ZeroSizeLocalAcc.get_range()[0])
         ZeroSizeLocalAcc[0] = 1;
     });
   }).wait();
  return 0;
}
