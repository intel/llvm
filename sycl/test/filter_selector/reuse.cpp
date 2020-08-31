// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %t1.out

//==----------------- reuse.cpp - filter_selector reuse test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl::sycl;
using namespace cl::sycl::ONEAPI;

int main() {
  std::vector<device> Devs;

  Devs = device::get_devices();

  std::cout << "# Devices found: " << Devs.size() << std::endl;

  if (Devs.size() > 1) {
    filter_selector filter("1");
    
    device d1(filter);
    filter.reset();
    device d2(filter);

    assert (d1 == d2);
  }
  
  return 0;
}
