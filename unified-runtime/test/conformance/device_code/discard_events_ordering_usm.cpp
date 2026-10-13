// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../../common/discard_events_ordering_usm_consts.h"
#include <sycl/sycl.hpp>

static constexpr uint32_t MAGIC_NUM1 = 2;

static void execute_ordering_stage(uint32_t *values1, uint32_t *values2,
                                   uint32_t *values3, size_t i,
                                   uint32_t expected_value1,
                                   uint32_t expected_value2,
                                   uint32_t expected_value3,
                                   uint32_t increment) {
  if (values1[i] == expected_value1 && values2[i] == expected_value2 &&
      values3[i] == expected_value3) {
    values1[i] += increment;
  }
}

static void execute_ordering_stage_and_update_values2(
    uint32_t *values1, uint32_t *values2, uint32_t *values3, size_t i,
    uint32_t expected_value1, uint32_t expected_value2,
    uint32_t expected_value3, uint32_t increment, uint32_t new_values2_value) {
  // Preconditions must be met for this stage to proceed
  if (values1[i] == expected_value1 && values2[i] == expected_value2 &&
      values3[i] == expected_value3) {
    execute_ordering_stage(values1, values2, values3, i, expected_value1,
                           expected_value2, expected_value3, increment);
    values2[i] = new_values2_value;
  }
}

int main() {
  constexpr size_t array_size = 128;
  sycl::queue queue;
  uint32_t *values1 = sycl::malloc_shared<uint32_t>(array_size, queue);
  uint32_t *values2 = sycl::malloc_shared<uint32_t>(array_size, queue);
  uint32_t *values3 = sycl::malloc_shared<uint32_t>(array_size, queue);

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class discard_events_ordering_usm_stage_1>(
        sycl::range<1>{array_size}, [=](sycl::item<1> itemID) {
          size_t i = itemID.get_id(0);
          uint32_t idx = static_cast<uint32_t>(i);
          if (values1[i] == 0 && values2[i] == idx && values3[i] == idx) {
            execute_ordering_stage_and_update_values2(
                values1, values2, values3, i, 0, idx, idx, idx, MAGIC_NUM1);
            values3[i] = idx;
          }
        });
  });

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class discard_events_ordering_usm_stage_2>(
        sycl::range<1>{array_size}, [=](sycl::item<1> itemID) {
          size_t i = itemID.get_id(0);
          uint32_t idx = static_cast<uint32_t>(i);
          execute_ordering_stage(values1, values2, values3, i, idx, MAGIC_NUM1,
                                 idx, DISCARD_EVENTS_STAGE_2_INCREMENT);
        });
  });

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class discard_events_ordering_usm_stage_3>(
        sycl::range<1>{array_size}, [=](sycl::item<1> itemID) {
          size_t i = itemID.get_id(0);
          uint32_t idx = static_cast<uint32_t>(i);
          execute_ordering_stage_and_update_values2(
              values1, values2, values3, i,
              idx + DISCARD_EVENTS_STAGE_2_INCREMENT,
              idx + DISCARD_EVENTS_STAGE_2_INCREMENT, idx,
              DISCARD_EVENTS_STAGE_3_INCREMENT, idx);
        });
  });

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class discard_events_ordering_usm_stage_4>(
        sycl::range<1>{array_size}, [=](sycl::item<1> itemID) {
          size_t i = itemID.get_id(0);
          uint32_t idx = static_cast<uint32_t>(i);
          execute_ordering_stage(values1, values2, values3, i,
                                 idx + DISCARD_EVENTS_STAGE_2_INCREMENT +
                                     DISCARD_EVENTS_STAGE_3_INCREMENT,
                                 idx, idx, DISCARD_EVENTS_STAGE_4_INCREMENT);
        });
  });

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class discard_events_ordering_usm_stage_5>(
        sycl::range<1>{array_size}, [=](sycl::item<1> itemID) {
          size_t i = itemID.get_id(0);
          uint32_t idx = static_cast<uint32_t>(i);
          execute_ordering_stage(values1, values2, values3, i,
                                 idx + DISCARD_EVENTS_STAGE_2_INCREMENT +
                                     DISCARD_EVENTS_STAGE_3_INCREMENT +
                                     DISCARD_EVENTS_STAGE_4_INCREMENT,
                                 idx, idx, DISCARD_EVENTS_STAGE_5_INCREMENT);
        });
  });
  sycl::free(values1, queue);
  sycl::free(values2, queue);
  sycl::free(values3, queue);

  return 0;
}
