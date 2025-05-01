// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <sycl/sycl.hpp>

struct Struct {
  uint32_t a;
  uint32_t b;
};

int main() {
  sycl::queue deviceQueue;
  sycl::range<1> numOfItems{1};

  uint32_t output = 0;

  volatile bool test_bool = true;
  volatile uint8_t test_u8 = 2;
  volatile uint32_t test_u32 = 3;
  volatile uint64_t test_u64 = 5;
  Struct test_struct{7, 5};
  volatile float test_float = 11;

  {
    sycl::buffer output_buff(&output, sycl::range(1));
    deviceQueue.submit([&](sycl::handler &cgh) {
      sycl::accessor acc{output_buff, cgh, sycl::read_write};
      auto kern = [=](sycl::id<1> id) {
        acc[id] = 100 + (test_bool ? 1 : 0) * test_u8 * test_u32 * test_u64 *
                            test_struct.a * static_cast<uint32_t>(test_float);
      };
      cgh.parallel_for<class Foo>(numOfItems, kern);
    });
    deviceQueue.wait();
  }

  return output == 2410;
}
