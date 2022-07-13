// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

//==------- device_equality.cpp - SYCL device equality test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>
#include <utility>

using namespace cl::sycl;

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

  device h1;
  device h2;

  assert(h1.is_host() && "Device h1 is host");
  assert(h2.is_host() && "Device h2 is host");
  assert(h1 == h2 && "Host devices equal each other");

  platform hp1 = h1.get_platform();
  platform hp2 = h2.get_platform();
  assert(hp1.is_host() && "Platform hp1 is host");
  assert(hp2.is_host() && "Platform hp2 is host");
  assert(hp1 == hp2 && "Host platforms equal each other");

  return 0;
}
