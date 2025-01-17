// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "fixtures.h"
#include <cstring>

// Tests kernel commands using ur events for command level synchronization work
// as expected.
struct KernelCommandEventSyncTest
    : uur::command_buffer::urCommandBufferExpExecutionTest {

  void SetUp() override {
    program_name = "saxpy_usm";
    UUR_RETURN_ON_FATAL_FAILURE(urCommandBufferExpExecutionTest::SetUp());

    ur_bool_t event_support = false;
    ASSERT_SUCCESS(
        urDeviceGetInfo(device, UR_DEVICE_INFO_COMMAND_BUFFER_EVENT_SUPPORT_EXP,
                        sizeof(ur_bool_t), &event_support, nullptr));
    if (!event_support) {
      GTEST_SKIP() << "External event sync is not supported by device.";
    }

    for (auto &device_ptr : device_ptrs) {
      ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &device_ptr));
      ASSERT_NE(device_ptr, nullptr);
    }

    // Index 0 is output
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, device_ptrs[2]));
    // Index 1 is A
    ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(A), nullptr, &A));
    // Index 2 is X
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 2, nullptr, device_ptrs[0]));
    // Index 3 is Y
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 3, nullptr, device_ptrs[1]));

    // Create second command-buffer
    ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, nullptr,
                                            &second_cmd_buf_handle));
    ASSERT_NE(second_cmd_buf_handle, nullptr);
  }

  virtual void TearDown() override {
    for (auto &device_ptr : device_ptrs) {
      if (device_ptr) {
        EXPECT_SUCCESS(urUSMFree(context, device_ptr));
      }
    }

    for (auto &event : external_events) {
      if (event) {
        EXPECT_SUCCESS(urEventRelease(event));
      }
    }

    if (second_cmd_buf_handle) {
      EXPECT_SUCCESS(urCommandBufferReleaseExp(second_cmd_buf_handle));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urCommandBufferExpExecutionTest::TearDown());
  }

  // First two device pointers are inputs to be tested, last is an output
  // from saxpy kernel.
  std::array<void *, 3> device_ptrs = {nullptr, nullptr, nullptr};
  std::array<ur_event_handle_t, 2> external_events = {nullptr, nullptr};
  std::array<ur_exp_command_buffer_sync_point_t, 2> sync_points = {0, 0};
  ur_exp_command_buffer_handle_t second_cmd_buf_handle = nullptr;
  static constexpr size_t elements = 64;
  static constexpr size_t global_offset = 0;
  static constexpr size_t allocation_size = sizeof(uint32_t) * elements;
  static constexpr size_t A = 2;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(KernelCommandEventSyncTest);

// Tests using a regular enqueue event as a dependency of a command-buffer
// command, and having the signal event of that command-buffer command being
// a dependency of another enqueue command.
TEST_P(KernelCommandEventSyncTest, Basic) {
  // Initialize data X with queue submission
  uint32_t patternX = 42;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[0], sizeof(patternX),
                                  &patternX, allocation_size, 0, nullptr,
                                  &external_events[0]));

  // Initialize data Y with command-buffer command
  uint32_t patternY = 0xA;
  ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
      cmd_buf_handle, device_ptrs[1], &patternY, sizeof(patternY),
      allocation_size, 0, nullptr, 0, nullptr, &sync_points[0], nullptr,
      nullptr));

  // Kernel command for SAXPY waiting on command and signal event
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      cmd_buf_handle, kernel, 1, &global_offset, &elements, nullptr, 0, nullptr,
      1, &sync_points[0], 1, &external_events[0], &sync_points[1],
      &external_events[1], nullptr));

  // command-buffer command that reads output to host
  std::array<uint32_t, elements> host_command_ptr{};
  ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
      cmd_buf_handle, host_command_ptr.data(), device_ptrs[2], allocation_size,
      1, &sync_points[1], 0, nullptr, nullptr, nullptr, nullptr));
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));

  ASSERT_SUCCESS(
      urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));

  // Queue command that reads output to host
  std::array<uint32_t, elements> host_enqueue_ptr{};
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, host_enqueue_ptr.data(),
                                    device_ptrs[2], allocation_size, 1,
                                    &external_events[1], nullptr));

  ASSERT_SUCCESS(urQueueFinish(queue));

  for (size_t i = 0; i < elements; i++) {
    auto ref = (patternX * A) + patternY;
    ASSERT_EQ(host_command_ptr[i], ref);
    ASSERT_EQ(host_enqueue_ptr[i], ref);
  }
}

// Tests using events to synchronize between command-buffers:
TEST_P(KernelCommandEventSyncTest, InterCommandBuffer) {
  // Initialize data X with command-buffer A command
  uint32_t patternX = 42;
  std::array<uint32_t, elements> dataX{};
  std::fill(dataX.begin(), dataX.end(), patternX);
  ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
      cmd_buf_handle, device_ptrs[0], dataX.data(), allocation_size, 0, nullptr,
      0, nullptr, &sync_points[0], nullptr, nullptr));

  // Initialize data Y with command-buffer A command
  uint32_t patternY = 0xA;
  ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
      cmd_buf_handle, device_ptrs[1], &patternY, sizeof(patternY),
      allocation_size, 1, &sync_points[0], 0, nullptr, &sync_points[1],
      &external_events[0], nullptr));

  // Run SAXPY kernel with command-buffer B command, waiting on an event.
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      second_cmd_buf_handle, kernel, 1, &global_offset, &elements, nullptr, 0,
      nullptr, 0, nullptr, 1, &external_events[0], &sync_points[1], nullptr,
      nullptr));

  // Command-buffer A command that reads output to host, waiting on an event
  std::array<uint32_t, elements> host_command_ptr{};
  ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
      second_cmd_buf_handle, host_command_ptr.data(), device_ptrs[2],
      allocation_size, 1, &sync_points[1], 0, nullptr, nullptr, nullptr,
      nullptr));

  // Finalize command-buffers
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(second_cmd_buf_handle));

  // Submit command-buffers
  ASSERT_SUCCESS(
      urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(second_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));

  // Verify execution
  ASSERT_SUCCESS(urQueueFinish(queue));
  for (size_t i = 0; i < elements; i++) {
    auto ref = (patternX * A) + patternY;
    ASSERT_EQ(host_command_ptr[i], ref) << i;
  }

  // Use new data for patternX
  patternX = 666;
  std::fill(dataX.begin(), dataX.end(), patternX);

  // Submit command-buffers again to check that dependencies still enforced.
  ASSERT_SUCCESS(
      urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(second_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));

  // Verify second execution
  ASSERT_SUCCESS(urQueueFinish(queue));
  for (size_t i = 0; i < elements; i++) {
    auto ref = (patternX * A) + patternY;
    ASSERT_EQ(host_command_ptr[i], ref) << i;
  }
}

// Tests behavior of waiting on signal event before command-buffer has executed
TEST_P(KernelCommandEventSyncTest, SignalWaitBeforeEnqueue) {
  // Initialize data X with queue submission
  uint32_t patternX = 42;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[0], sizeof(patternX),
                                  &patternX, allocation_size, 0, nullptr,
                                  &external_events[0]));

  // Initialize data Y with command-buffer command
  uint32_t patternY = 0xA;
  ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
      cmd_buf_handle, device_ptrs[1], &patternY, sizeof(patternY),
      allocation_size, 0, nullptr, 0, nullptr, &sync_points[0], nullptr,
      nullptr));

  // Kernel command for SAXPY waiting on command and signal event
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      cmd_buf_handle, kernel, 1, &global_offset, &elements, nullptr, 0, nullptr,
      1, &sync_points[0], 1, &external_events[0], &sync_points[1],
      &external_events[1], nullptr));
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));

  // Event will be considered complete before first execution
  ASSERT_SUCCESS(urEventWait(1, &external_events[1]));
}
