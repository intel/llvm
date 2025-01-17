// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <sycl/sycl.hpp>

int main() {
  size_t array_size = 16;
  std::vector<uint32_t> X(array_size, 1);
  std::vector<uint32_t> Y(array_size, 2);
  std::vector<uint32_t> Z(array_size, 0);
  uint32_t A = 42;
  auto x_buff = sycl::buffer<uint32_t>(X.data(), sycl::range<1>(array_size));
  auto y_buff = sycl::buffer<uint32_t>(Y.data(), sycl::range<1>(array_size));
  auto z_buff = sycl::buffer<uint32_t>(Z.data(), sycl::range<1>(array_size));

  sycl::queue sycl_queue;
  sycl_queue.submit([&](sycl::handler &cgh) {
    auto x_acc = x_buff.get_access<sycl::access::mode::read>(cgh);
    auto y_acc = y_buff.get_access<sycl::access::mode::read>(cgh);
    auto z_acc = z_buff.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class saxpy>(sycl::range<1>{array_size},
                                  [=](sycl::item<1> itemId) {
                                    auto i = itemId.get_id(0);
                                    z_acc[i] = A * x_acc[i] + y_acc[i];
                                  });
  });
  return 0;
}
