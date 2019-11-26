// RUN: %clangxx -fsycl %s -o %t.out -lOpenCL
// RUN: %CPU_RUN_PLACEHOLDER %t.out
//==------------ device_white_list.cpp -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl;

int main() {
  sycl::vector_class<sycl::platform> Platforms =
      sycl::platform::get_platforms();

  int NonHostDevs = 0;
  for(sycl::platform &Plt: Platforms) {
    NonHostDevs += !Plt.is_host();
  }
}
