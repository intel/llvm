// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../fixtures.h"
#include <cstring>

struct KernelCommandEventSyncUpdateTest
    : uur::command_buffer::urUpdatableCommandBufferExpExecutionTest {
  void SetUp() override {
    program_name = "saxpy_usm";
    UUR_RETURN_ON_FATAL_FAILURE(
        urUpdatableCommandBufferExpExecutionTest::SetUp());

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::checkCommandBufferUpdateSupport(
            device, UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_EVENTS));

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

    if (command_handle) {
      EXPECT_SUCCESS(urCommandBufferReleaseCommandExp(command_handle));
    }

    UUR_RETURN_ON_FATAL_FAILURE(
        urUpdatableCommandBufferExpExecutionTest::TearDown());
  }

  // First two device pointers are inputs to be tested, last is the output
  // from the saxpy kernel command.
  std::array<void *, 3> device_ptrs = {nullptr, nullptr, nullptr};
  std::array<ur_event_handle_t, 4> external_events = {nullptr, nullptr, nullptr,
                                                      nullptr};
  std::array<ur_exp_command_buffer_sync_point_t, 2> sync_points = {0, 0};
  ur_exp_command_buffer_command_handle_t command_handle = nullptr;
  static constexpr size_t elements = 64;
  static constexpr size_t global_offset = 0;
  static constexpr size_t allocation_size = sizeof(uint32_t) * elements;
  static constexpr size_t A = 2;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(KernelCommandEventSyncUpdateTest);

// Tests updating the signal and wait event dependencies of the saxpy
// command in a command-buffer.
TEST_P(KernelCommandEventSyncUpdateTest, Basic) {
  // Initialize data X with queue submission
  uint32_t patternX = 42;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[0], sizeof(patternX),
                                  &patternX, allocation_size, 0, nullptr,
                                  &external_events[0]));

  // Initialize data Y with command-buffer command
  uint32_t patternY = 0xA;
  ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
      updatable_cmd_buf_handle, device_ptrs[1], &patternY, sizeof(patternY),
      allocation_size, 0, nullptr, 0, nullptr, &sync_points[0], nullptr,
      nullptr));

  // Kernel command for SAXPY waiting on command and signal event
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      updatable_cmd_buf_handle, kernel, 1, &global_offset, &elements, nullptr,
      0, nullptr, 1, &sync_points[0], 1, &external_events[0], &sync_points[1],
      &external_events[1], &command_handle));
  ASSERT_NE(command_handle, nullptr);

  // command-buffer command that reads output to host
  std::array<uint32_t, elements> host_command_ptr{};
  ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
      updatable_cmd_buf_handle, host_command_ptr.data(), device_ptrs[2],
      allocation_size, 1, &sync_points[1], 0, nullptr, nullptr, nullptr,
      nullptr));
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));

  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));

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

  // Reset output data
  std::memset(host_command_ptr.data(), 0, allocation_size);
  std::memset(host_enqueue_ptr.data(), 0, allocation_size);

  // Set data X to new value with queue submission
  patternX = 0xBEEF;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[0], sizeof(patternX),
                                  &patternX, allocation_size, 0, nullptr,
                                  &external_events[2]));

  // Update kernel command-wait event to wait on fill of new x value
  ASSERT_SUCCESS(urCommandBufferUpdateWaitEventsExp(command_handle, 1,
                                                    &external_events[2]));

  // Get a new signal event for command-buffer
  ASSERT_SUCCESS(
      urCommandBufferUpdateSignalEventExp(command_handle, &external_events[3]));

  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));

  // Read data back with a queue operation waiting on updated kernel command
  // signal event.
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, host_enqueue_ptr.data(),
                                    device_ptrs[2], allocation_size, 1,
                                    &external_events[3], nullptr));

  // Verify results
  ASSERT_SUCCESS(urQueueFinish(queue));
  for (size_t i = 0; i < elements; i++) {
    auto ref = (patternX * A) + patternY;
    ASSERT_EQ(host_command_ptr[i], ref);
    ASSERT_EQ(host_enqueue_ptr[i], ref);
  }
}

// Test updating wait events to a command with multiple wait events
TEST_P(KernelCommandEventSyncUpdateTest, TwoWaitEvents) {
  // Initialize data X with queue submission
  uint32_t patternX = 42;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[0], sizeof(patternX),
                                  &patternX, allocation_size, 0, nullptr,
                                  &external_events[0]));

  // Initialize data Y with command-buffer command
  uint32_t patternY = 0xA;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[1], sizeof(patternY),
                                  &patternY, allocation_size, 0, nullptr,
                                  &external_events[1]));

  // Kernel command for SAXPY waiting on command and signal event
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      updatable_cmd_buf_handle, kernel, 1, &global_offset, &elements, nullptr,
      0, nullptr, 0, nullptr, 2, &external_events[0], &sync_points[0],
      &external_events[2], &command_handle));
  ASSERT_NE(command_handle, nullptr);

  // command-buffer command that reads output to host
  std::array<uint32_t, elements> host_command_ptr{};
  ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
      updatable_cmd_buf_handle, host_command_ptr.data(), device_ptrs[2],
      allocation_size, 1, &sync_points[0], 0, nullptr, nullptr, nullptr,
      nullptr));
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));

  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));

  // Queue command that reads output to host
  std::array<uint32_t, elements> host_enqueue_ptr{};
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, host_enqueue_ptr.data(),
                                    device_ptrs[2], allocation_size, 1,
                                    &external_events[2], nullptr));

  ASSERT_SUCCESS(urQueueFinish(queue));

  for (size_t i = 0; i < elements; i++) {
    auto ref = (patternX * A) + patternY;
    ASSERT_EQ(host_command_ptr[i], ref);
    ASSERT_EQ(host_enqueue_ptr[i], ref);
  }

  // Reset output data
  std::memset(host_command_ptr.data(), 0, allocation_size);
  std::memset(host_enqueue_ptr.data(), 0, allocation_size);

  // Set data X to new value with queue submission
  patternX = 0xBEEF;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[0], sizeof(patternX),
                                  &patternX, allocation_size, 0, nullptr,
                                  &external_events[3]));

  // Set data X to new value with queue submission
  patternY = 0xBAD;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[1], sizeof(patternY),
                                  &patternY, allocation_size, 0, nullptr,
                                  &external_events[4]));

  // Update kernel command-wait event to wait on fill of new x value
  ASSERT_SUCCESS(urCommandBufferUpdateWaitEventsExp(command_handle, 2,
                                                    &external_events[3]));

  // Get a new signal event for command-buffer
  ASSERT_SUCCESS(
      urCommandBufferUpdateSignalEventExp(command_handle, &external_events[5]));

  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));

  // Read data back with a queue operation waiting on updated kernel command
  // signal event.
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, host_enqueue_ptr.data(),
                                    device_ptrs[2], allocation_size, 1,
                                    &external_events[5], nullptr));

  // Verify results
  ASSERT_SUCCESS(urQueueFinish(queue));
  for (size_t i = 0; i < elements; i++) {
    auto ref = (patternX * A) + patternY;
    ASSERT_EQ(host_command_ptr[i], ref);
    ASSERT_EQ(host_enqueue_ptr[i], ref);
  }
}

// Tests the correct error is returned when a different number
// of wait events is passed during update.
TEST_P(KernelCommandEventSyncUpdateTest, InvalidWaitUpdate) {
  // Initialize data X with queue submission
  uint32_t patternX = 42;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[0], sizeof(patternX),
                                  &patternX, allocation_size, 0, nullptr,
                                  &external_events[0]));

  // Initialize data Y with queue submission
  uint32_t patternY = 0xA;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[1], sizeof(patternY),
                                  &patternY, allocation_size, 0, nullptr,
                                  &external_events[1]));

  // Initialize data Z with queue submission
  int32_t zero_pattern = 0;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[2], sizeof(zero_pattern),
                                  &zero_pattern, allocation_size, 0, nullptr,
                                  &external_events[2]));

  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      updatable_cmd_buf_handle, kernel, 1, &global_offset, &elements, nullptr,
      0, nullptr, 0, nullptr, 1, &external_events[0], nullptr, nullptr,
      &command_handle));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));

  // Increase number of events should be an error
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                   urCommandBufferUpdateWaitEventsExp(command_handle, 2,
                                                      &external_events[1]));

  // decrease number of events should be an error
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
      urCommandBufferUpdateWaitEventsExp(command_handle, 0, nullptr));
}

// Tests the correct error is returned when trying to update the
// signal event from a command that was not created with one.
TEST_P(KernelCommandEventSyncUpdateTest, InvalidSignalUpdate) {
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      updatable_cmd_buf_handle, kernel, 1, &global_offset, &elements, nullptr,
      0, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, &command_handle));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));

  uint32_t patternX = 42;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptrs[0], sizeof(patternX),
                                  &patternX, allocation_size, 0, nullptr,
                                  &external_events[0]));

  // Increase number of events should be an error
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_OPERATION,
      urCommandBufferUpdateSignalEventExp(command_handle, &external_events[0]));
}
