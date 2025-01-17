// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <sycl/sycl.hpp>

int main() {
  size_t array_size = 16;
  std::vector<uint32_t> A(array_size, 1);
  uint32_t val = 42;
  sycl::queue sycl_queue;
  auto A_buff = sycl::buffer<uint32_t>(A.data(), sycl::range<1>(array_size));
  sycl_queue.submit([&](sycl::handler &cgh) {
    auto A_acc = A_buff.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class fill>(sycl::range<1>{array_size},
                                 [A_acc, val](sycl::item<1> itemId) {
                                   auto id = itemId.get_id(0);
                                   A_acc[id] = val;
                                 });
  });
  return 0;
}
