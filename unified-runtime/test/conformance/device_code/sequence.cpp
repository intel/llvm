// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <sycl/sycl.hpp>

class Add;
class Mul;

int main() {
  sycl::queue deviceQueue;
  uint32_t val = 0;

  auto buff = sycl::buffer<uint32_t>(&val, 1);

  deviceQueue.submit([&](sycl::handler &cgh) {
    auto acc = buff.get_access<sycl::access::mode::read_write>(cgh);
    cgh.single_task<Add>([=]() {
      for (uint32_t i = 0; i < 1000; i++) {
        volatile uint32_t tmp = acc[0];
        acc[0] = tmp + 1;
      }
    });
  });

  deviceQueue.submit([&](sycl::handler &cgh) {
    auto acc = buff.get_access<sycl::access::mode::read_write>(cgh);
    cgh.single_task<Mul>([=]() {
      for (uint32_t i = 0; i < 2; i++) {
        volatile uint32_t tmp = acc[0];
        acc[0] = tmp * 2;
      }
    });
  });

  return 0;
}
