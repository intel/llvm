// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
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

#include <CL/sycl.hpp>

using namespace cl::sycl;
// TODO: change to 'using namespace cl::sycl::oneapi' after PR intel/llvm:4014
// is merged
using namespace cl::sycl::ext::oneapi;

int main() {
  std::vector<device> Devs;

  Devs = device::get_devices();

  std::cout << "# Devices found: " << Devs.size() << std::endl;

  if (Devs.size() > 1) {
    // TODO: change all occurrences of filter_selector to 'filter_selector' or
    // 'oneapi::filter_selector' after PR intel/llvm:4014 is merged
    ext::oneapi::filter_selector filter("1");

    device d1(filter);
    device d2(filter);

    assert(d1 == d2);

    ext::oneapi::filter_selector f1("0");
    ext::oneapi::filter_selector f2("1");
    device d3(f1);
    device d4(f2);

    assert(d3 != d4);
  }

  return 0;
}
