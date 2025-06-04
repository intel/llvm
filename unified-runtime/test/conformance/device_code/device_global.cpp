// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sycl/sycl.hpp>

/* device_image_scope is needed to make sure that sycl generates device code
 * with an integer global variable instead of a pointer */
sycl::ext::oneapi::experimental::device_global<
    int, decltype(sycl::ext::oneapi::experimental::properties(
             sycl::ext::oneapi::experimental::device_image_scope))>
    dev_var;

int main() {

  sycl::queue deviceQueue;
  sycl::range<1> numOfItems{1};
  deviceQueue.submit([&](sycl::handler &cgh) {
    auto kern = [=](sycl::id<1>) {
      // just increment
      dev_var = dev_var + 1;
    };
    cgh.parallel_for<class device_global>(numOfItems, kern);
  });

  return 0;
}
