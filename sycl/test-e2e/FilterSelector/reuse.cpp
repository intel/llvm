// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

//==----------------- reuse.cpp - filter_selector reuse test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;

int main() {
  std::vector<device> Devs;

  Devs = device::get_devices();

  std::cout << "# Devices found: " << Devs.size() << std::endl;

  if (Devs.size() > 1) {
    filter_selector filter("1");

    device d1(filter);
    device d2(filter);

    assert(d1 == d2);

    filter_selector f1("0");
    filter_selector f2("1");
    device d3(f1);
    device d4(f2);

    assert(d3 != d4);
  }

  return 0;
}
