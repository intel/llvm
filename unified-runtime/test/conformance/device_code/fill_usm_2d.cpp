// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sycl/sycl.hpp>

int main() {

  size_t nd_range_x = 8;
  size_t nd_range_y = 8;

  auto nd_range = sycl::range<2>(nd_range_x, nd_range_y);

  std::vector<uint32_t> A(nd_range_x * nd_range_y, 1);
  uint32_t val = 42;
  sycl::queue sycl_queue;

  auto work_range = sycl::nd_range<2>(nd_range, sycl::range<2>(1, 1));

  uint32_t *data =
      sycl::malloc_shared<uint32_t>(nd_range_x * nd_range_y, sycl_queue);
  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class fill_2d>(work_range,
                                    [data, val](sycl::nd_item<2> item_id) {
                                      auto id = item_id.get_global_linear_id();
                                      data[id] = val;
                                    });
  });
  return 0;
}
