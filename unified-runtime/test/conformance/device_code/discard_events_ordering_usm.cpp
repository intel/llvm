// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sycl/sycl.hpp>

static constexpr uint32_t MAGIC_NUM1 = 2;

int main() {
  constexpr size_t array_size = 128;
  sycl::queue queue;
  uint32_t *values1 = sycl::malloc_shared<uint32_t>(array_size, queue);
  uint32_t *values2 = sycl::malloc_shared<uint32_t>(array_size, queue);
  uint32_t *values3 = sycl::malloc_shared<uint32_t>(array_size, queue);

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class discard_events_ordering_usm>(
        sycl::range<1>{array_size}, [=](sycl::item<1> itemID) {
          size_t i = itemID.get_id(0);
          uint32_t idx = static_cast<uint32_t>(i);

          if (values1[i] == 0)
            if (values2[i] == idx)
              if (values3[i] == idx) {
                values1[i] += idx;
                values2[i] = MAGIC_NUM1;
                values3[i] = idx;
              }

          if (values1[i] == idx)
            if (values2[i] == MAGIC_NUM1)
              if (values3[i] == idx) {
                values1[i] += 10;
              }

          if (values1[i] == idx + 10)
            if (values2[i] == idx + 10)
              if (values3[i] == idx) {
                values1[i] += 100;
                values2[i] = idx;
              }

          if (values1[i] == idx + 110)
            if (values2[i] == idx)
              if (values3[i] == idx) {
                values1[i] += 1000;
              }

          if (values1[i] == idx + 1110)
            if (values2[i] == idx)
              if (values3[i] == idx) {
                values1[i] += 10000;
              }
        });
  });

  return 0;
}
