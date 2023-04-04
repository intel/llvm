//==--------------- fpga_device.cpp - AOT compilation for fpga -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl.hpp"

using namespace sycl;

const double big[] = {3, 2, 1, 5, 6, 7};
void foo(double &result, queue q, int x) {
  buffer<double> buf(&result, 1);
  buffer<double, 1> big_buf(big, sizeof(big) / sizeof(double));
  q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::discard_write>(cgh);
    auto big_acc = big_buf.get_access<access::mode::read>(cgh);
    cgh.single_task<class test>([=]() { acc[0] = big_acc[x]; });
  });
}
