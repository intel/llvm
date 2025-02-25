// REQUIRES: opencl, opencl_icd

// RUN: %{build} -o %t.out %opencl_lib
// RUN: %{run} %t.out

//==--------------- event.cpp - SYCL event test ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <sycl/detail/core.hpp>

int main() {
  try {
    std::cout << "Create default event" << std::endl;
    sycl::event e;
  } catch (const sycl::exception &e) {
    std::cout << "Failed to create device for event" << std::endl;
  }
}
