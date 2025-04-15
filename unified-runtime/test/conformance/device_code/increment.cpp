// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sycl/sycl.hpp>

int main() {

  const size_t inputSize = 1;
  sycl::queue sycl_queue;
  uint32_t *inputArray = sycl::malloc_shared<uint32_t>(inputSize, sycl_queue);

  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class IncrementBy1>(
        sycl::range<1>(inputSize),
        [=](sycl::id<1> itemID) { inputArray[itemID] += 1; });
  });
  return 0;
}
