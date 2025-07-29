// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sycl/sycl.hpp>

int main() {
  size_t array_size = 16;
  sycl::queue sycl_queue;
  std::vector<uint32_t> src(array_size, 1);
  std::vector<uint32_t> dst(array_size, 1);
  auto src_buff =
      sycl::buffer<uint32_t>(src.data(), sycl::range<1>(array_size));
  auto dst_buff =
      sycl::buffer<uint32_t>(dst.data(), sycl::range<1>(array_size));

  sycl_queue.submit([&](sycl::handler &cgh) {
    auto src_acc = src_buff.get_access<sycl::access::mode::read>(cgh);
    auto dst_acc = dst_buff.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class cpy_and_mult>(
        sycl::range<1>{array_size}, [src_acc, dst_acc](sycl::item<1> itemId) {
          auto id = itemId.get_id(0);
          dst_acc[id] = src_acc[id] * 2;
        });
  });
  return 0;
}
