// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sycl/sycl.hpp>

int main() {
  sycl::queue sycl_queue;
  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class no_args>(
        sycl::range<3>{128, 128, 128},
        [](sycl::item<3> itemId) { itemId.get_id(0); });
  });
  return 0;
}
