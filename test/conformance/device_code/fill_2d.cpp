// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <sycl/sycl.hpp>

int main() {
  size_t nd_range_x = 8;
  size_t nd_range_y = 8;
  auto nd_range = sycl::range<2>(nd_range_x, nd_range_y);

  std::vector<uint32_t> A(nd_range_x * nd_range_y, 1);
  uint32_t val = 42;
  sycl::queue sycl_queue;

  auto work_range = sycl::nd_range<2>(nd_range, sycl::range<2>(1, 1));
  auto A_buff =
      sycl::buffer<uint32_t>(A.data(), sycl::range<1>(nd_range_x * nd_range_y));
  sycl_queue.submit([&](sycl::handler &cgh) {
    auto A_acc = A_buff.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class fill_2d>(work_range,
                                    [A_acc, val](sycl::nd_item<2> item_id) {
                                      auto id = item_id.get_global_linear_id();
                                      A_acc[id] = val;
                                    });
  });
  return 0;
}
