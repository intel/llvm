// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../fixtures.h"
#include <array>
#include <cstring>

// Tests that adapter implementation of urEnqueueCommandBufferExp serializes
// submissions of the same UR command-buffer object with respect to previous
// submissions. Using update to change the input and output of the kernel
// between submissions so that the ordering can be verified.
struct urUpdatableEnqueueCommandBufferExpTest
    : uur::command_buffer::urUpdatableCommandBufferExpExecutionTest {

  virtual void SetUp() override {
    program_name = "cpy_and_mult_usm";
    UUR_RETURN_ON_FATAL_FAILURE(
        urUpdatableCommandBufferExpExecutionTest::SetUp());

    ur_device_usm_access_capability_flags_t shared_usm_support = 0;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT,
        sizeof(shared_usm_support), &shared_usm_support, nullptr));
    if (0 == shared_usm_support) {
      GTEST_SKIP() << "Shared USM is not supported by device.";
    }

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

    // Created 4 shared USM allocations
    for (auto &shared_ptr : shared_ptrs) {
      ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &shared_ptr));
      ASSERT_NE(shared_ptr, nullptr);
    }

    // Initialize allocations
    int32_t *ptrX = static_cast<int32_t *>(shared_ptrs[0]);
    int32_t *ptrY = static_cast<int32_t *>(shared_ptrs[1]);
    int32_t *ptrZ = static_cast<int32_t *>(shared_ptrs[2]);
    for (size_t i = 0; i < global_size; i++) {
      ptrX[i] = i; // Input to first command-buffer enqueue
      ptrY[i] = 0; // Output to first command-buffer enqueue
      ptrZ[i] = 0; // Output to second command-buffer enqueue
    }

    // Index 0 is input ptr
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, shared_ptrs[0]));
    // Index 1 is output ptr
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 1, nullptr, shared_ptrs[1]));

    // Create command-buffer with a single kernel that does "Out[i] = In[i] * 2"
    ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
        updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
        &global_size, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr,
        nullptr, &command_handle));
    ASSERT_NE(nullptr, command_handle);

    ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
  }

  virtual void TearDown() override {
    if (in_order_queue) {
      EXPECT_SUCCESS(urQueueRelease(in_order_queue));
    }

    if (out_of_order_queue) {
      EXPECT_SUCCESS(urQueueRelease(out_of_order_queue));
    }

    for (auto &shared_ptr : shared_ptrs) {
      if (shared_ptr) {
        EXPECT_SUCCESS(urUSMFree(context, shared_ptr));
      }
    }

    UUR_RETURN_ON_FATAL_FAILURE(
        urUpdatableCommandBufferExpExecutionTest::TearDown());
  }

  void Verify() {
    // Output of first submission
    int32_t *ptrA = static_cast<int32_t *>(shared_ptrs[1]);
    // Output of second submission
    int32_t *ptrB = static_cast<int32_t *>(shared_ptrs[2]);
    for (size_t i = 0; i < global_size; i++) {
      uint32_t resultA = i * 2;
      ASSERT_EQ(resultA, ptrA[i]);

      uint32_t resultB = resultA * 2;
      ASSERT_EQ(resultB, ptrB[i]);
    }
  }

  void Update() {
    ur_exp_command_buffer_update_pointer_arg_desc_t update_ptrs[2];
    // Set output ptr of first run as input ptr to second run
    update_ptrs[0] = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
        nullptr,                                                      // pNext
        0,               // argIndex
        nullptr,         // pProperties
        &shared_ptrs[1], // pArgValue
    };

    // Set new USM pointer as kernel output
    update_ptrs[1] = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
        nullptr,                                                      // pNext
        1,               // argIndex
        nullptr,         // pProperties
        &shared_ptrs[2], // pArgValue
    };

    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        command_handle, // hCommand
        kernel,         // hNewKernel
        0,              // numNewMemObjArgs
        2,              // numNewPointerArgs
        0,              // numNewValueArgs
        n_dimensions,   // newWorkDim
        nullptr,        // pNewMemObjArgList
        update_ptrs,    // pNewPointerArgList
        nullptr,        // pNewValueArgList
        nullptr,        // pNewGlobalWorkOffset
        nullptr,        // pNewGlobalWorkSize
        nullptr,        // pNewLocalWorkSize
    };
    ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(
        updatable_cmd_buf_handle, 1, &update_desc));
  }

  ur_queue_handle_t in_order_queue = nullptr;
  ur_queue_handle_t out_of_order_queue = nullptr;
  ur_exp_command_buffer_command_handle_t command_handle = nullptr;

  static constexpr size_t global_size = 16;
  static constexpr size_t global_offset = 0;
  static constexpr size_t n_dimensions = 1;
  static constexpr size_t allocation_size = sizeof(uint32_t) * global_size;
  std::array<void *, 3> shared_ptrs = {nullptr, nullptr};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urUpdatableEnqueueCommandBufferExpTest);

// Tests that the same command-buffer submitted across different in-order
// queues has an implicit dependency on first submission
TEST_P(urUpdatableEnqueueCommandBufferExpTest, SerializeAcrossQueues) {
  // Execute command-buffer to first in-order queue (created by parent
  // urQueueTest fixture)
  ASSERT_SUCCESS(urEnqueueCommandBufferExp(queue, updatable_cmd_buf_handle, 0,
                                           nullptr, nullptr));

  // Update command-buffer to new input and output ptrs
  Update();

  // Execute command-buffer to second in-order queue, should have implicit
  // dependency on first submission.
  ASSERT_SUCCESS(urEnqueueCommandBufferExp(
      in_order_queue, updatable_cmd_buf_handle, 0, nullptr, nullptr));

  // Wait for both submissions to complete
  ASSERT_SUCCESS(urQueueFlush(queue));
  ASSERT_SUCCESS(urQueueFinish(in_order_queue));

  Verify();
}

// Tests that submitting a command-buffer twice to an out-of-order queue
// relying on implicit serialization semantics for dependencies.
TEST_P(urUpdatableEnqueueCommandBufferExpTest, SerializeOutofOrderQueue) {
  // See https://github.com/intel/llvm/issues/18722
  UUR_KNOWN_FAILURE_ON(uur::HIP{});

  // First submission to out-of-order queue
  ASSERT_SUCCESS(urEnqueueCommandBufferExp(
      out_of_order_queue, updatable_cmd_buf_handle, 0, nullptr, nullptr));

  // Update command-buffer to new input and output ptrs
  Update();

  // Second submission to out-of-order queue, which should have implicit
  // dependency on first command-buffer submission
  ASSERT_SUCCESS(urEnqueueCommandBufferExp(
      out_of_order_queue, updatable_cmd_buf_handle, 0, nullptr, nullptr));

  // Wait for both submissions to complete
  ASSERT_SUCCESS(urQueueFinish(out_of_order_queue));

  Verify();
}
