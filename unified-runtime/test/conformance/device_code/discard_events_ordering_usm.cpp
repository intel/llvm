// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sycl/sycl.hpp>

static constexpr uint32_t MAGIC_NUM1 = 2;

// Stage increments - each stage adds an order of magnitude based on STAGE_INCREMENT
static constexpr uint32_t STAGE_INCREMENT = 10;
static constexpr uint32_t STAGE_2_INCREMENT = STAGE_INCREMENT;        // 10
static constexpr uint32_t STAGE_3_INCREMENT = STAGE_INCREMENT * 10;   // 100
static constexpr uint32_t STAGE_4_INCREMENT = STAGE_INCREMENT * 100;  // 1000
static constexpr uint32_t STAGE_5_INCREMENT = STAGE_INCREMENT * 1000; // 10000

// Execute a verification stage with data-dependent ordering constraints.
// Proceeds only if all preconditions are met, preventing reordering of stages.
//
// Parameters:
//   values1, values2, values3: shared memory arrays
//   i: array index for this work-item
//   expected_value1: precondition check for values1[i]
//   expected_value2: precondition check for values2[i]
//   expected_value3: precondition check for values3[i]
//   increment: amount to add to values1[i] if preconditions pass
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

// Execute a verification stage with data-dependent ordering constraints and update values2.
// Proceeds only if all preconditions are met, preventing reordering of stages.
//
// Parameters:
//   values1, values2, values3: shared memory arrays
//   i: array index for this work-item
//   expected_value1: precondition check for values1[i]
//   expected_value2: precondition check for values2[i]
//   expected_value3: precondition check for values3[i]
//   increment: amount to add to values1[i] if preconditions pass
//   new_values2_value: value to set for values2[i] after increment
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
    cgh.parallel_for<
        class
        discard_events_ordering_usm>(sycl::range<1>{array_size}, [=](sycl::item<
                                                                     1>
                                                                         itemID) {
      size_t i = itemID.get_id(0);
      uint32_t idx = static_cast<uint32_t>(i);

      // Execute all stages in strict order.
      // Each stage can only proceed if the previous stage's conditions are met.
      // This enforces in-order execution semantics.

      execute_ordering_stage_and_update_values2(values1, values2, values3, i, 0,
                                                idx, idx, idx, MAGIC_NUM1);

      execute_ordering_stage(values1, values2, values3, i, idx, MAGIC_NUM1, idx,
                             STAGE_2_INCREMENT);

      execute_ordering_stage_and_update_values2(
          values1, values2, values3, i, idx + STAGE_2_INCREMENT,
          idx + STAGE_2_INCREMENT, idx, STAGE_3_INCREMENT, idx);

      execute_ordering_stage(values1, values2, values3, i,
                             idx + STAGE_2_INCREMENT + STAGE_3_INCREMENT, idx,
                             idx, STAGE_4_INCREMENT);

      execute_ordering_stage(values1, values2, values3, i,
                             idx + STAGE_2_INCREMENT + STAGE_3_INCREMENT +
                                 STAGE_4_INCREMENT,
                             idx, idx, STAGE_5_INCREMENT);
    });
  });
  sycl::free(values1, queue);
  sycl::free(values2, queue);
  sycl::free(values3, queue);

  return 0;
}
