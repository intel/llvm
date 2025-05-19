// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../fixtures.h"

// UR reproducer for SYCL-Graph E2E test "RecordReplay/usm_copy_in_order.cpp"
// Note that the kernel code is different, in that this test uses the
// saxpy_usm kernel, but the sequence of operations is the same.
struct urCommandBufferUSMCopyInOrderTest
    : uur::command_buffer::urCommandBufferExpExecutionTest {
  virtual void SetUp() override {
    program_name = "saxpy_usm";
    UUR_RETURN_ON_FATAL_FAILURE(urCommandBufferExpExecutionTest::SetUp());

    // See URLZA-521
    UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

    // Create in-order command-buffer
    ur_exp_command_buffer_desc_t desc{
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC, // stype
        nullptr,                                   // pNext
        false,                                     // isUpdatable
        true,                                      // isInOrder
        false                                      // enableProfiling
    };
    ASSERT_SUCCESS(
        urCommandBufferCreateExp(context, device, &desc, &in_order_cmd_buf));
    ASSERT_NE(in_order_cmd_buf, nullptr);

    // Create 4 device USM allocations and initialize elements to list index
    for (unsigned i = 0; i < device_ptrs.size(); i++) {
      auto &device_ptr = device_ptrs[i];
      ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &device_ptr));
      ASSERT_NE(device_ptr, nullptr);

      uint32_t pattern = i;
      ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptr, sizeof(pattern),
                                      &pattern, allocation_size, 0, nullptr,
                                      nullptr));
    }
    ASSERT_SUCCESS(urQueueFinish(queue));

    // Index 0 is output
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, device_ptrs[0]));
    // Index 1 is A
    ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(A), nullptr, &A));
    // Index 2 is X
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 2, nullptr, device_ptrs[1]));
    // Index 3 is Y
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 3, nullptr, device_ptrs[2]));
  }

  virtual void TearDown() override {
    for (auto &device_ptr : device_ptrs) {
      if (device_ptr) {
        EXPECT_SUCCESS(urUSMFree(context, device_ptr));
      }
    }
    if (in_order_cmd_buf) {
      EXPECT_SUCCESS(urCommandBufferReleaseExp(in_order_cmd_buf));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urCommandBufferExpExecutionTest::TearDown());
  }

  ur_exp_command_buffer_handle_t in_order_cmd_buf = nullptr;
  static constexpr size_t global_size = 10;
  static constexpr size_t global_offset = 0;
  static constexpr size_t n_dimensions = 1;
  static constexpr size_t allocation_size = sizeof(uint32_t) * global_size;
  static constexpr uint32_t A = 42;
  std::array<void *, 4> device_ptrs = {nullptr, nullptr, nullptr, nullptr};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urCommandBufferUSMCopyInOrderTest);
TEST_P(urCommandBufferUSMCopyInOrderTest, Success) {
  // Do an eager kernel enqueue without wait on completion
  // D[0] = A * D[1] + D[2]
  // D[0] = 42 * 1 + 2
  // D[0] = 44
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                       &global_offset, &global_size, nullptr, 0,
                                       nullptr, nullptr));

  // command-buffer sync point used to enforce linear dependencies when
  // appending commands to the command-buffer.
  ur_exp_command_buffer_sync_point_t sync_point;

  // Add SAXPY kernel node to command-buffer
  // D[3] = A * D[1] + D[0]
  // D[3] = 42 * 1 + 44
  // D[3] = 86
  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 3, nullptr, device_ptrs[0]));
  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, device_ptrs[3]));
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      in_order_cmd_buf, kernel, n_dimensions, &global_offset, &global_size,
      nullptr, 0, nullptr, 0, nullptr, 0, nullptr, &sync_point, nullptr,
      nullptr));

  // Add device-to-device memcpy node from output of previous command to
  // the X component of the expression.
  // D[1] = 86
  ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
      in_order_cmd_buf, device_ptrs[1], device_ptrs[3], allocation_size, 0,
      nullptr, 0, nullptr, &sync_point, nullptr, nullptr));

  // Add SAXPY kernel node
  // D[3] = A * [1] + [0]
  // D[3] = 42 * 86 + 44
  // D[3] = 3656
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      in_order_cmd_buf, kernel, n_dimensions, &global_offset, &global_size,
      nullptr, 0, nullptr, 1, &sync_point, 0, nullptr, &sync_point, nullptr,
      nullptr));

  // Add device-to-device memcpy node from output of previous command to
  // currently unused USM allocation.
  // D[2] = 3656
  ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
      in_order_cmd_buf, device_ptrs[2], device_ptrs[3], allocation_size, 1,
      &sync_point, 0, nullptr, &sync_point, nullptr, nullptr));

  // Add device-to-host memcpy node
  std::vector<uint32_t> output(global_size);
  ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
      in_order_cmd_buf, output.data(), device_ptrs[2], allocation_size, 1,
      &sync_point, 0, nullptr, &sync_point, nullptr, nullptr));
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(in_order_cmd_buf));

  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, in_order_cmd_buf, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  // Verify
  constexpr uint32_t result1 = A * 1 + 2;             // eager kernel submission
  constexpr uint32_t result2 = A * 1 + result1;       // first kernel command
  constexpr uint32_t result3 = A * result2 + result1; // second kernel command
  for (size_t i = 0; i < global_size; i++) {
    ASSERT_EQ(result3, output[i]);
  }
}
