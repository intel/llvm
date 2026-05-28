// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sycl/sycl.hpp>

int main() {

  const size_t inputSize = 1;
  sycl::queue sycl_queue;
  uint32_t *inputArray = sycl::malloc_shared<uint32_t>(inputSize, sycl_queue);

  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class MultiplyBy2>(
        sycl::range<1>(inputSize),
        [=](sycl::id<1> itemID) { inputArray[itemID] *= 2; });
  });
  return 0;
}
