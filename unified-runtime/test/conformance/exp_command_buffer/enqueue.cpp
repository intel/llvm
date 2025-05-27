// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include <array>
#include <cstring>

// Tests that adapter implementation of urEnqueueCommandBufferExp serializes
// submissions of the same UR command-buffer object with respect to previous
// submissions.
//
// Done using a kernel that increments a value, so the ordering of the two
// submission isn't tested, only that they didn't run concurrently. See the
// enqueue_update.cpp test for a test verifying the order of submissions, as
// the input/output to the kernels can be modified between the submissions.
struct urEnqueueCommandBufferExpTest
    : uur::command_buffer::urCommandBufferExpExecutionTest {
  virtual void SetUp() override {
    program_name = "increment";
    UUR_RETURN_ON_FATAL_FAILURE(urCommandBufferExpExecutionTest::SetUp());

    // Create an in-order queue
    ur_queue_properties_t queue_properties = {
        UR_STRUCTURE_TYPE_QUEUE_PROPERTIES, nullptr, 0};
    ASSERT_SUCCESS(
        urQueueCreate(context, device, &queue_properties, &in_order_queue));

    // Create an out-of-order queue
    queue_properties.flags = UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    ASSERT_SUCCESS(
        urQueueCreate(context, device, &queue_properties, &out_of_order_queue));
    ASSERT_NE(out_of_order_queue, nullptr);

    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &device_ptr));
    ASSERT_NE(device_ptr, nullptr);

    uint32_t zero_pattern = 0;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptr, sizeof(zero_pattern),
                                    &zero_pattern, allocation_size, 0, nullptr,
                                    nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    // Create command-buffer with a single kernel that does "Ptr[i] += 1;"
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, device_ptr));
    ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
        cmd_buf_handle, kernel, n_dimensions, &global_offset, &global_size,
        nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr,
        nullptr));
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
  }

  virtual void TearDown() override {
    if (in_order_queue) {
      EXPECT_SUCCESS(urQueueRelease(in_order_queue));
    }

    if (out_of_order_queue) {
      EXPECT_SUCCESS(urQueueRelease(out_of_order_queue));
    }

    if (device_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, device_ptr));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urCommandBufferExpExecutionTest::TearDown());
  }

  ur_queue_handle_t in_order_queue = nullptr;
  ur_queue_handle_t out_of_order_queue = nullptr;

  static constexpr size_t global_size = 16;
  static constexpr size_t global_offset = 0;
  static constexpr size_t n_dimensions = 1;
  static constexpr size_t allocation_size = sizeof(uint32_t) * global_size;
  void *device_ptr = nullptr;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueCommandBufferExpTest);

// Tests that the same command-buffer submitted across different in-order
// queues has an implicit dependency on first submission
TEST_P(urEnqueueCommandBufferExpTest, SerializeAcrossQueues) {
  // Execute command-buffer to first in-order queue (created by parent
  // urQueueTest fixture)
  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, cmd_buf_handle, 0, nullptr, nullptr));

  // Execute command-buffer to second in-order queue, should have implicit
  // dependency on first submission.
  ASSERT_SUCCESS(urEnqueueCommandBufferExp(in_order_queue, cmd_buf_handle, 0,
                                           nullptr, nullptr));

  // Wait for both submissions to complete
  ASSERT_SUCCESS(urQueueFlush(queue));
  ASSERT_SUCCESS(urQueueFinish(in_order_queue));

  std::vector<uint32_t> Output(global_size);
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, Output.data(), device_ptr,
                                    allocation_size, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  // Verify
  const uint32_t reference = 2;
  for (size_t i = 0; i < global_size; i++) {
    ASSERT_EQ(reference, Output[i]);
  }
}

// Tests that submitting a command-buffer twice to an out-of-order queue
// relying on implicit serialization semantics for dependencies.
TEST_P(urEnqueueCommandBufferExpTest, SerializeOutofOrderQueue) {
  // https://github.com/intel/llvm/issues/18610
  UUR_KNOWN_FAILURE_ON(uur::LevelZeroV2{});

  ASSERT_SUCCESS(urEnqueueCommandBufferExp(out_of_order_queue, cmd_buf_handle,
                                           0, nullptr, nullptr));
  ASSERT_SUCCESS(urEnqueueCommandBufferExp(out_of_order_queue, cmd_buf_handle,
                                           0, nullptr, nullptr));

  // Wait for both submissions to complete
  ASSERT_SUCCESS(urQueueFinish(out_of_order_queue));

  std::vector<uint32_t> Output(global_size);
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(out_of_order_queue, true, Output.data(),
                                    device_ptr, allocation_size, 0, nullptr,
                                    nullptr));

  // Verify
  const uint32_t reference = 2;
  for (size_t i = 0; i < global_size; i++) {
    ASSERT_EQ(reference, Output[i]);
  }
}
