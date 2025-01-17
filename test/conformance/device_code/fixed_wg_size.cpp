// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <sycl/sycl.hpp>

struct KernelFunctor {
  void operator()(sycl::nd_item<3>) const {}
  void operator()(sycl::item<3>) const {}

  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::work_group_size<8, 4, 2>};
  }
};

int main() {
  sycl::queue myQueue;
  myQueue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class FixedWgSize>(sycl::range<3>(8, 8, 8),
                                        KernelFunctor{});
  });

  myQueue.wait();
  return 0;
}
