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
    : uur::command_buffer::urCommandBufferExpExecutionTestWithParam<
          ur_queue_flags_t> {
  virtual void SetUp() override {
    program_name = "increment";
    UUR_RETURN_ON_FATAL_FAILURE(
        urCommandBufferExpExecutionTestWithParam::SetUp());

    // Create an in-order or out-of-order queue, depending on the passed parameter
    ur_queue_flags_t queue_type = std::get<1>(GetParam());
    ur_queue_properties_t queue_properties = {
        UR_STRUCTURE_TYPE_QUEUE_PROPERTIES, nullptr, queue_type};
    ASSERT_SUCCESS(urQueueCreate(context, device, &queue_properties,
                                 &in_or_out_of_order_queue));

    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &device_ptr));
    ASSERT_NE(device_ptr, nullptr);

    uint32_t zero_pattern = 0;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptr, sizeof(zero_pattern),
                                    &zero_pattern, allocation_size, 0, nullptr,
                                    nullptr));

    for (int i = 0; i < num_copy_buffers; i++) {
      ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                      buffer_size * sizeof(int32_t),
                                      (void **)&(dst_buffers[i])));
      ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                      buffer_size * sizeof(int32_t),
                                      (void **)&(src_buffers[i])));

      ASSERT_SUCCESS(urEnqueueUSMFill(
          queue, src_buffers[i], sizeof(zero_pattern), &zero_pattern,
          buffer_size * sizeof(int32_t), 0, nullptr, nullptr));
    }

    ASSERT_SUCCESS(urQueueFinish(queue));

    // Create command-buffer with a single kernel that does "Ptr[i] += 1;"
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, device_ptr));
    ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
        cmd_buf_handle, kernel, n_dimensions, &global_offset, &global_size,
        nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr,
        nullptr));

    // Schedule memory copying in order to prolong the buffer execution
    for (int i = 0; i < num_copy_buffers; i++) {
      ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
          cmd_buf_handle, dst_buffers[i], src_buffers[i],
          buffer_size * sizeof(int32_t), 0, nullptr, 0, nullptr, nullptr,
          nullptr, nullptr));
    }

    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
  }

  virtual void TearDown() override {
    if (in_or_out_of_order_queue) {
      EXPECT_SUCCESS(urQueueRelease(in_or_out_of_order_queue));
    }

    if (device_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, device_ptr));
    }

    for (int i = 0; i < num_copy_buffers; i++) {
      if (dst_buffers[i]) {
        EXPECT_SUCCESS(urUSMFree(context, dst_buffers[i]));
      }

      if (src_buffers[i]) {
        EXPECT_SUCCESS(urUSMFree(context, src_buffers[i]));
      }
    }

    UUR_RETURN_ON_FATAL_FAILURE(
        urCommandBufferExpExecutionTestWithParam::TearDown());
  }

  ur_queue_handle_t in_or_out_of_order_queue = nullptr;

  static constexpr size_t global_size = 16;
  static constexpr size_t global_offset = 0;
  static constexpr size_t n_dimensions = 1;
  static constexpr size_t allocation_size = sizeof(uint32_t) * global_size;
  void *device_ptr = nullptr;

  static constexpr int num_copy_buffers = 10;
  static constexpr int buffer_size = 512;
  int32_t *dst_buffers[num_copy_buffers] = {};
  int32_t *src_buffers[num_copy_buffers] = {};
};

std::string deviceTestWithQueueTypePrinter(
    const ::testing::TestParamInfo<
        std::tuple<uur::DeviceTuple, ur_queue_flags_t>> &info) {
  auto device = std::get<0>(info.param).device;
  auto queue_type = std::get<1>(info.param);

  std::stringstream ss;

  switch (queue_type) {
  case 0:
    ss << "InOrderQueue";
    break;

  case UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE:
    ss << "OutOfOrderQueue";
    break;

  default:
    ss << "UnspecifiedQueueType" << queue_type;
  }

  return uur::GetPlatformAndDeviceName(device) + "__" +
         uur::GTestSanitizeString(ss.str());
}

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urEnqueueCommandBufferExpTest,
    testing::Values(0, UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE),
    deviceTestWithQueueTypePrinter);

// Tests that the same command-buffer submitted across different in-order
// queues has an implicit dependency on first submission
TEST_P(urEnqueueCommandBufferExpTest, SerializeAcrossQueues) {
  // Execute command-buffer to first in-order queue (created by parent
  // urQueueTestWithParam fixture)
  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, cmd_buf_handle, 0, nullptr, nullptr));

  // Execute command-buffer to second in-order queue, should have implicit
  // dependency on first submission.
  ASSERT_SUCCESS(urEnqueueCommandBufferExp(
      in_or_out_of_order_queue, cmd_buf_handle, 0, nullptr, nullptr));

  // Wait for both submissions to complete
  ASSERT_SUCCESS(urQueueFlush(queue));
  ASSERT_SUCCESS(urQueueFinish(in_or_out_of_order_queue));

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

TEST_P(urEnqueueCommandBufferExpTest, SerializeInOrOutOfOrderQueue) {
  const int iterations = 5;
  for (int i = 0; i < iterations; i++) {
    ASSERT_SUCCESS(urEnqueueCommandBufferExp(
        in_or_out_of_order_queue, cmd_buf_handle, 0, nullptr, nullptr));
  }

  // Wait for both submissions to complete
  ASSERT_SUCCESS(urQueueFinish(in_or_out_of_order_queue));

  std::vector<uint32_t> Output(global_size);
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(in_or_out_of_order_queue, true,
                                    Output.data(), device_ptr, allocation_size,
                                    0, nullptr, nullptr));

  // Verify
  const uint32_t reference = iterations;
  for (size_t i = 0; i < global_size; i++) {
    ASSERT_EQ(reference, Output[i]);
  }
}
