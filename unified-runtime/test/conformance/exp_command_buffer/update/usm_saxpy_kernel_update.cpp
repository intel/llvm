// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../fixtures.h"
#include <array>
#include <cstring>

// Test that updating a command-buffer with a single kernel command
// taking USM & scalar arguments works correctly.

struct USMSaxpyKernelTestBase
    : uur::command_buffer::urUpdatableCommandBufferExpExecutionTest {
  virtual void SetUp() override {
    program_name = "saxpy_usm";
    UUR_RETURN_ON_FATAL_FAILURE(
        urUpdatableCommandBufferExpExecutionTest::SetUp());

    ur_device_usm_access_capability_flags_t shared_usm_flags;
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
    if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
      GTEST_SKIP() << "Shared USM is not supported.";
    }

    const size_t allocation_size = sizeof(uint32_t) * global_size;
    for (auto &shared_ptr : shared_ptrs) {
      ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &shared_ptr));
      ASSERT_NE(shared_ptr, nullptr);

      std::vector<uint8_t> pattern(allocation_size);
      uur::generateMemFillPattern(pattern);
      std::memcpy(shared_ptr, pattern.data(), allocation_size);
    }

    // Index 0 is output
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, shared_ptrs[0]));
    // Index 1 is A
    ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(A), nullptr, &A));
    // Index 2 is X
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 2, nullptr, shared_ptrs[1]));
    // Index 3 is Y
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 3, nullptr, shared_ptrs[2]));
  }

  void Validate(uint32_t *output, uint32_t *X, uint32_t *Y, uint32_t A,
                size_t length) {
    for (size_t i = 0; i < length; i++) {
      uint32_t result = A * X[i] + Y[i];
      ASSERT_EQ(result, output[i]);
    }
  }

  virtual void TearDown() override {
    for (auto &shared_ptr : shared_ptrs) {
      if (shared_ptr) {
        EXPECT_SUCCESS(urUSMFree(context, shared_ptr));
      }
    }

    UUR_RETURN_ON_FATAL_FAILURE(
        urUpdatableCommandBufferExpExecutionTest::TearDown());
  }

  static constexpr size_t local_size = 4;
  static constexpr size_t global_size = 32;
  static constexpr size_t global_offset = 0;
  static constexpr size_t n_dimensions = 1;
  static constexpr uint32_t A = 42;
  std::array<void *, 5> shared_ptrs = {nullptr, nullptr, nullptr, nullptr};
};

struct USMSaxpyKernelTest : USMSaxpyKernelTestBase {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(USMSaxpyKernelTestBase::SetUp());

    // Append kernel command to command-buffer and close command-buffer
    ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
        updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
        &global_size, &local_size, 0, nullptr, 0, nullptr, 0, nullptr, nullptr,
        nullptr, &command_handle));
    ASSERT_NE(command_handle, nullptr);

    ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
  }

  ur_exp_command_buffer_command_handle_t command_handle = nullptr;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(USMSaxpyKernelTest);

TEST_P(USMSaxpyKernelTest, UpdateParameters) {
  // Run command-buffer prior to update an verify output
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *output = (uint32_t *)shared_ptrs[0];
  uint32_t *X = (uint32_t *)shared_ptrs[1];
  uint32_t *Y = (uint32_t *)shared_ptrs[2];
  Validate(output, X, Y, A, global_size);

  // Update inputs
  ur_exp_command_buffer_update_pointer_arg_desc_t new_input_descs[2];

  // New X at index 2
  new_input_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      2,                                                            // argIndex
      nullptr,         // pProperties
      &shared_ptrs[3], // pArgValue
  };

  // New Y at index 3
  new_input_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      3,                                                            // argIndex
      nullptr,         // pProperties
      &shared_ptrs[4], // pArgValue
  };

  // New A at index 1
  uint32_t new_A = 33;
  ur_exp_command_buffer_update_value_arg_desc_t new_A_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1,                                                          // argIndex
      sizeof(new_A),                                              // argSize
      nullptr,                                                    // pProperties
      &new_A,                                                     // hArgValue
  };

  // Update kernel inputs
  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,  // hCommand
      kernel,          // hNewKernel
      0,               // numNewMemObjArgs
      2,               // numNewPointerArgs
      1,               // numNewValueArgs
      n_dimensions,    // newWorkDim
      nullptr,         // pNewMemObjArgList
      new_input_descs, // pNewPointerArgList
      &new_A_desc,     // pNewValueArgList
      nullptr,         // pNewGlobalWorkOffset
      nullptr,         // pNewGlobalWorkSize
      nullptr,         // pNewLocalWorkSize
  };

  // Update kernel and enqueue command-buffer again
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(updatable_cmd_buf_handle,
                                                      1, &update_desc));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  // Verify that update occurred correctly
  uint32_t *new_output = (uint32_t *)shared_ptrs[0];
  uint32_t *new_X = (uint32_t *)shared_ptrs[3];
  uint32_t *new_Y = (uint32_t *)shared_ptrs[4];
  Validate(new_output, new_X, new_Y, new_A, global_size);
}

struct USMMultiSaxpyKernelTest : USMSaxpyKernelTestBase {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(USMSaxpyKernelTestBase::SetUp());

    // Append kernel command to command-buffer and close command-buffer
    for (unsigned node = 0; node < nodes; node++) {
      ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
          updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
          &global_size, &local_size, 0, nullptr, 0, nullptr, 0, nullptr,
          nullptr, nullptr, &command_handles[node]));
      ASSERT_NE(command_handles[node], nullptr);
    }

    ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
  }

  static constexpr size_t nodes = 1024;
  static constexpr uint32_t A = 42;
  std::array<ur_exp_command_buffer_command_handle_t, nodes> command_handles{};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(USMMultiSaxpyKernelTest);

TEST_P(USMMultiSaxpyKernelTest, UpdateParameters) {
  // Run command-buffer prior to update an verify output
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *output = (uint32_t *)shared_ptrs[0];
  uint32_t *X = (uint32_t *)shared_ptrs[1];
  uint32_t *Y = (uint32_t *)shared_ptrs[2];
  Validate(output, X, Y, A, global_size);

  // Update inputs
  ur_exp_command_buffer_update_pointer_arg_desc_t new_input_descs[2];

  // New X at index 2
  new_input_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      2,                                                            // argIndex
      nullptr,         // pProperties
      &shared_ptrs[3], // pArgValue
  };

  // New Y at index 3
  new_input_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      3,                                                            // argIndex
      nullptr,         // pProperties
      &shared_ptrs[4], // pArgValue
  };

  // New A at index 1
  uint32_t new_A = 33;
  ur_exp_command_buffer_update_value_arg_desc_t new_A_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1,                                                          // argIndex
      sizeof(new_A),                                              // argSize
      nullptr,                                                    // pProperties
      &new_A,                                                     // hArgValue
  };

  std::vector<ur_exp_command_buffer_update_kernel_launch_desc_t> update_descs;
  for (auto &handle : command_handles) {
    // Update kernel inputs
    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        handle,          // hCommand
        kernel,          // hNewKernel
        0,               // numNewMemObjArgs
        2,               // numNewPointerArgs
        1,               // numNewValueArgs
        n_dimensions,    // newWorkDim
        nullptr,         // pNewMemObjArgList
        new_input_descs, // pNewPointerArgList
        &new_A_desc,     // pNewValueArgList
        nullptr,         // pNewGlobalWorkOffset
        nullptr,         // pNewGlobalWorkSize
        nullptr,         // pNewLocalWorkSize
    };

    update_descs.push_back(update_desc);
  }

  // Update kernel and enqueue command-buffer again
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, update_descs.size(), update_descs.data()));

  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  // Verify that update occurred correctly
  uint32_t *new_output = (uint32_t *)shared_ptrs[0];
  uint32_t *new_X = (uint32_t *)shared_ptrs[3];
  uint32_t *new_Y = (uint32_t *)shared_ptrs[4];
  Validate(new_output, new_X, new_Y, new_A, global_size);
}

// Checks that passing nullptr to hNewKernel even when kernel binary updates
// is not supported by the adapter.
TEST_P(USMMultiSaxpyKernelTest, UpdateNullptrKernel) {
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *output = (uint32_t *)shared_ptrs[0];
  uint32_t *X = (uint32_t *)shared_ptrs[1];
  uint32_t *Y = (uint32_t *)shared_ptrs[2];
  Validate(output, X, Y, A, global_size);

  // New A at index 1
  uint32_t new_A = 33;
  ur_exp_command_buffer_update_value_arg_desc_t new_A_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1,                                                          // argIndex
      sizeof(new_A),                                              // argSize
      nullptr,                                                    // pProperties
      &new_A,                                                     // hArgValue
  };

  std::vector<ur_exp_command_buffer_update_kernel_launch_desc_t> update_descs;
  for (auto &handle : command_handles) {
    // Update kernel inputs
    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        handle,       // hCommand
        nullptr,      // hNewKernel
        0,            // numNewMemObjArgs
        0,            // numNewPointerArgs
        1,            // numNewValueArgs
        n_dimensions, // newWorkDim
        nullptr,      // pNewMemObjArgList
        nullptr,      // pNewPointerArgList
        &new_A_desc,  // pNewValueArgList
        nullptr,      // pNewGlobalWorkOffset
        nullptr,      // pNewGlobalWorkSize
        nullptr,      // pNewLocalWorkSize
    };

    update_descs.push_back(update_desc);
  }

  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, update_descs.size(), update_descs.data()));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  // Verify that update occurred correctly
  uint32_t *new_output = (uint32_t *)shared_ptrs[0];
  Validate(new_output, X, Y, new_A, global_size);
}

TEST_P(USMMultiSaxpyKernelTest, UpdateWithoutBlocking) {
  // Prepare new inputs
  ur_exp_command_buffer_update_pointer_arg_desc_t new_input_descs[2];

  // New X at index 2
  new_input_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      2,                                                            // argIndex
      nullptr,         // pProperties
      &shared_ptrs[3], // pArgValue
  };

  // New Y at index 3
  new_input_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      3,                                                            // argIndex
      nullptr,         // pProperties
      &shared_ptrs[4], // pArgValue
  };

  // New A at index 1
  uint32_t new_A = 33;
  ur_exp_command_buffer_update_value_arg_desc_t new_A_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1,                                                          // argIndex
      sizeof(new_A),                                              // argSize
      nullptr,                                                    // pProperties
      &new_A,                                                     // hArgValue
  };

  // Run command-buffer prior to update without doing a blocking wait after
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));

  std::vector<ur_exp_command_buffer_update_kernel_launch_desc_t> update_descs;
  for (auto &handle : command_handles) {
    // Update kernel inputs
    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        handle,          // hCommand
        kernel,          // hNewKernel
        0,               // numNewMemObjArgs
        2,               // numNewPointerArgs
        1,               // numNewValueArgs
        n_dimensions,    // newWorkDim
        nullptr,         // pNewMemObjArgList
        new_input_descs, // pNewPointerArgList
        &new_A_desc,     // pNewValueArgList
        nullptr,         // pNewGlobalWorkOffset
        nullptr,         // pNewGlobalWorkSize
        nullptr,         // pNewLocalWorkSize
    };
    update_descs.push_back(update_desc);
  }

  // Update kernel and enqueue command-buffer again
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, update_descs.size(), update_descs.data()));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  // Verify that update occurred correctly
  uint32_t *new_output = (uint32_t *)shared_ptrs[0];
  uint32_t *new_X = (uint32_t *)shared_ptrs[3];
  uint32_t *new_Y = (uint32_t *)shared_ptrs[4];
  Validate(new_output, new_X, new_Y, new_A, global_size);
}
