// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sycl/sycl.hpp>

class inc;

int main() {
  uint32_t *ptr;
  sycl::buffer<uint32_t> buf{ptr, 1};
  sycl::queue{}.submit([&](sycl::handler &cgh) {
    sycl::accessor acc{buf, cgh};
    auto kernel = [acc](sycl::item<1> it) { acc[it]++; };
    cgh.parallel_for<inc>(sycl::range<1>{1}, kernel);
  });
}
