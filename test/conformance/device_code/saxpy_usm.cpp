// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sycl/sycl.hpp>

int main() {
  size_t array_size = 16;

  sycl::queue sycl_queue;
  uint32_t *X = sycl::malloc_shared<uint32_t>(array_size, sycl_queue);
  uint32_t *Y = sycl::malloc_shared<uint32_t>(array_size, sycl_queue);
  uint32_t *Z = sycl::malloc_shared<uint32_t>(array_size, sycl_queue);
  uint32_t A = 42;

  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class saxpy>(sycl::range<1>{array_size},
                                  [=](sycl::item<1> itemId) {
                                    auto i = itemId.get_id(0);
                                    Z[i] = A * X[i] + Y[i];
                                  });
  });
  return 0;
}
