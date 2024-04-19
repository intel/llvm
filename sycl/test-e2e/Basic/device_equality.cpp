// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==------- device_equality.cpp - SYCL device equality test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <utility>

using namespace sycl;

int main() {
  std::cout << "Creating q1" << std::endl;
  queue q1;
  std::cout << "Creating q2" << std::endl;
  queue q2;

  // Default selector picks the same device every time.
  // That device should compare equal to itself.
  // Its platform should too.

  auto dev1 = q1.get_device();
  auto plat1 = dev1.get_platform();

  auto dev2 = q2.get_device();
  auto plat2 = dev2.get_platform();

  assert((dev1 == dev2) && "Device 1 == Device 2");
  assert((plat1 == plat2) && "Platform 1 == Platform 2");

  return 0;
}
