//==----- fpga_aocx.cpp - AOT compilation for fpga using aoc with aocx -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aoc, accelerator

/// E2E test for AOCX creation/use/run for FPGA
// Produce an archive with device (AOCX) image
// RUN: %clangxx -fsycl -fintelfpga -fsycl-link=image -DDEVICE_PART %s -o %t_image.a
// Produce a host object
// RUN: %clangxx -fsycl -fintelfpga -DHOST_PART %s -c -o %t.o

// AOCX with source
// RUN: %clangxx -fsycl -fintelfpga -DHOST_PART %s %t_image.a -o %t_aocx_src.out
// AOCX with object
// RUN: %clangxx -fsycl -fintelfpga %t.o %t_image.a -o %t_aocx_obj.out
//
// RUN: env SYCL_DEVICE_TYPE=ACC %t_aocx_src.out
// RUN: env SYCL_DEVICE_TYPE=ACC %t_aocx_obj.out

#include "CL/sycl.hpp"
#include <iostream>

using namespace cl::sycl;

#ifdef DEVICE_PART

const double big[] = {3, 2, 1, 5, 6, 7};
void foo(double &result, queue q, int x) {
  buffer<double> buf(&result, 1);
  buffer<double, 1> big_buf(big, sizeof(big) / sizeof(double));
  q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::discard_write>(cgh);
    auto big_acc = big_buf.get_access<access::mode::read>(cgh);
    cgh.single_task<class test>([=]() {
      acc[0] = big_acc[x];
    });
  });
}

#endif // DEVICE_PART

#ifdef HOST_PART

void foo(double &, queue q, int x);

int main(void) {
  queue q(accelerator_selector{});

  double result;
  foo(result, q, 3);
  std::cout << "Result: " << result << "\n";
}

#endif // HOST_PART
