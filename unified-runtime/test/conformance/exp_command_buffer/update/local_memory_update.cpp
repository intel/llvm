// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../fixtures.h"
#include <array>
#include <cstring>

// Test that updating a command-buffer with a single kernel command
// taking local memory arguments works correctly.
struct LocalMemoryUpdateTestBase
    : uur::command_buffer::urUpdatableCommandBufferExpExecutionTest {
  virtual void SetUp() override {
    program_name = "saxpy_usm_local_mem";
    UUR_RETURN_ON_FATAL_FAILURE(
        urUpdatableCommandBufferExpExecutionTest::SetUp());

    // HIP has extra args for local memory so we define an offset for arg
    // indices here for updating
    hip_arg_offset = backend == UR_PLATFORM_BACKEND_HIP ? 3 : 0;
    ur_device_usm_access_capability_flags_t shared_usm_flags;
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
    if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
      GTEST_SKIP() << "Shared USM is not supported.";
    }

    const size_t allocation_size = sizeof(uint32_t) * global_size * local_size;
    for (auto &shared_ptr : shared_ptrs) {
      ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &shared_ptr));
      ASSERT_NE(shared_ptr, nullptr);

      std::vector<uint8_t> pattern(allocation_size);
      uur::generateMemFillPattern(pattern);
      std::memcpy(shared_ptr, pattern.data(), allocation_size);
    }
    size_t current_index = 0;
    // Index 0 is local_mem_a arg
    ASSERT_SUCCESS(urKernelSetArgLocal(kernel, current_index++,
                                       local_mem_a_size, nullptr));

    // Hip has extra args for local mem at index 1-3
    if (backend == UR_PLATFORM_BACKEND_HIP) {
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                         sizeof(hip_local_offset), nullptr,
                                         &hip_local_offset));
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                         sizeof(hip_local_offset), nullptr,
                                         &hip_local_offset));
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                         sizeof(hip_local_offset), nullptr,
                                         &hip_local_offset));
    }

    // Index 1 is local_mem_b arg
    ASSERT_SUCCESS(urKernelSetArgLocal(kernel, current_index++,
                                       local_mem_b_size, nullptr));
    if (backend == UR_PLATFORM_BACKEND_HIP) {
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                         sizeof(hip_local_offset), nullptr,
                                         &hip_local_offset));
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                         sizeof(hip_local_offset), nullptr,
                                         &hip_local_offset));
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                         sizeof(hip_local_offset), nullptr,
                                         &hip_local_offset));
    }

    // Index 2 is output
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, current_index++, nullptr,
                                         shared_ptrs[0]));
    // Index 3 is A
    ASSERT_SUCCESS(
        urKernelSetArgValue(kernel, current_index++, sizeof(A), nullptr, &A));
    // Index 4 is X
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, current_index++, nullptr,
                                         shared_ptrs[1]));
    // Index 5 is Y
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, current_index++, nullptr,
                                         shared_ptrs[2]));
  }

  void Validate(uint32_t *output, uint32_t *X, uint32_t *Y, uint32_t A,
                size_t length, size_t local_size) {
    for (size_t i = 0; i < length; i++) {
      uint32_t result = A * X[i] + Y[i] + local_size;
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
  static constexpr size_t local_mem_a_size = local_size * sizeof(uint32_t);
  static constexpr size_t local_mem_b_size = local_mem_a_size * 2;
  static constexpr size_t global_size = 16;
  static constexpr size_t global_offset = 0;
  static constexpr size_t n_dimensions = 1;
  static constexpr uint32_t A = 42;
  std::array<void *, 5> shared_ptrs = {nullptr, nullptr, nullptr, nullptr,
                                       nullptr};

  uint32_t hip_arg_offset = 0;
  static constexpr uint64_t hip_local_offset = 0;
};

struct LocalMemoryUpdateTest : LocalMemoryUpdateTestBase {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(LocalMemoryUpdateTestBase::SetUp());

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

UUR_INSTANTIATE_DEVICE_TEST_SUITE(LocalMemoryUpdateTest);

// Test updating A,X,Y parameters to new values and local memory parameters
// to original values.
TEST_P(LocalMemoryUpdateTest, UpdateParametersSameLocalSize) {
  // Run command-buffer prior to update an verify output
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *output = (uint32_t *)shared_ptrs[0];
  uint32_t *X = (uint32_t *)shared_ptrs[1];
  uint32_t *Y = (uint32_t *)shared_ptrs[2];
  Validate(output, X, Y, A, global_size, local_size);

  // Update inputs
  std::array<ur_exp_command_buffer_update_pointer_arg_desc_t, 2>
      new_input_descs;
  std::array<ur_exp_command_buffer_update_value_arg_desc_t, 3> new_value_descs;

  // New local_mem_a at index 0
  new_value_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      0,                                                          // argIndex
      local_mem_a_size,                                           // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  };

  // New local_mem_b at index 1
  new_value_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1 + hip_arg_offset,                                         // argIndex
      local_mem_b_size,                                           // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  };

  // New A at index 3
  uint32_t new_A = 33;
  new_value_descs[2] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      3 + (2 * hip_arg_offset),                                   // argIndex
      sizeof(new_A),                                              // argSize
      nullptr,                                                    // pProperties
      &new_A,                                                     // hArgValue
  };

  // New X at index 4
  new_input_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      4 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[3], // pArgValue
  };

  // New Y at index 5
  new_input_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      5 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[4], // pArgValue
  };

  // Update kernel inputs
  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,         // hCommand
      kernel,                 // hNewKernel
      0,                      // numNewMemObjArgs
      new_input_descs.size(), // numNewPointerArgs
      new_value_descs.size(), // numNewValueArgs
      n_dimensions,           // newWorkDim
      nullptr,                // pNewMemObjArgList
      new_input_descs.data(), // pNewPointerArgList
      new_value_descs.data(), // pNewValueArgList
      nullptr,                // pNewGlobalWorkOffset
      nullptr,                // pNewGlobalWorkSize
      nullptr,                // pNewLocalWorkSize
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
  Validate(new_output, new_X, new_Y, new_A, global_size, local_size);
}

// Test only passing local memory parameters to update with the original values.
TEST_P(LocalMemoryUpdateTest, UpdateLocalOnly) {
  // Run command-buffer prior to update an verify output
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *output = (uint32_t *)shared_ptrs[0];
  uint32_t *X = (uint32_t *)shared_ptrs[1];
  uint32_t *Y = (uint32_t *)shared_ptrs[2];
  Validate(output, X, Y, A, global_size, local_size);

  // Update inputs
  std::array<ur_exp_command_buffer_update_value_arg_desc_t, 2> new_value_descs;

  // New local_mem_a at index 0
  new_value_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      0,                                                          // argIndex
      local_mem_a_size,                                           // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  };

  // New local_mem_b at index 1
  new_value_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1 + hip_arg_offset,                                         // argIndex
      local_mem_b_size,                                           // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  };

  // Update kernel inputs
  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,         // hCommand
      kernel,                 // hNewKernel
      0,                      // numNewMemObjArgs
      0,                      // numNewPointerArgs
      new_value_descs.size(), // numNewValueArgs
      n_dimensions,           // newWorkDim
      nullptr,                // pNewMemObjArgList
      nullptr,                // pNewPointerArgList
      new_value_descs.data(), // pNewValueArgList
      nullptr,                // pNewGlobalWorkOffset
      nullptr,                // pNewGlobalWorkSize
      nullptr,                // pNewLocalWorkSize
  };

  // Update kernel and enqueue command-buffer again
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(updatable_cmd_buf_handle,
                                                      1, &update_desc));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  // Verify that update occurred correctly
  Validate(output, X, Y, A, global_size, local_size);
}

// Test updating A,X,Y parameters to new values and omitting local memory
// parameters from the update.
TEST_P(LocalMemoryUpdateTest, UpdateParametersEmptyLocalSize) {
  // Run command-buffer prior to update and verify output
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *output = (uint32_t *)shared_ptrs[0];
  uint32_t *X = (uint32_t *)shared_ptrs[1];
  uint32_t *Y = (uint32_t *)shared_ptrs[2];
  Validate(output, X, Y, A, global_size, local_size);

  // Update inputs
  std::array<ur_exp_command_buffer_update_pointer_arg_desc_t, 2>
      new_input_descs;
  std::array<ur_exp_command_buffer_update_value_arg_desc_t, 1> new_value_descs;

  // New A at index 3
  uint32_t new_A = 33;
  new_value_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      3 + (2 * hip_arg_offset),                                   // argIndex
      sizeof(new_A),                                              // argSize
      nullptr,                                                    // pProperties
      &new_A,                                                     // hArgValue
  };

  // New X at index 4
  new_input_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      4 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[3], // pArgValue
  };

  // New Y at index 5
  new_input_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      5 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[4], // pArgValue
  };

  // Update kernel inputs
  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,         // hCommand
      kernel,                 // hNewKernel
      0,                      // numNewMemObjArgs
      new_input_descs.size(), // numNewPointerArgs
      new_value_descs.size(), // numNewValueArgs
      n_dimensions,           // newWorkDim
      nullptr,                // pNewMemObjArgList
      new_input_descs.data(), // pNewPointerArgList
      new_value_descs.data(), // pNewValueArgList
      nullptr,                // pNewGlobalWorkOffset
      nullptr,                // pNewGlobalWorkSize
      nullptr,                // pNewLocalWorkSize
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
  Validate(new_output, new_X, new_Y, new_A, global_size, local_size);
}

// Test updating A,X,Y parameters to new values and local memory parameters
// to new smaller values.
TEST_P(LocalMemoryUpdateTest, UpdateParametersSmallerLocalSize) {

  // Run command-buffer prior to update an verify output
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *output = (uint32_t *)shared_ptrs[0];
  uint32_t *X = (uint32_t *)shared_ptrs[1];
  uint32_t *Y = (uint32_t *)shared_ptrs[2];
  Validate(output, X, Y, A, global_size, local_size);

  // Update inputs
  ur_exp_command_buffer_update_pointer_arg_desc_t new_input_descs[2];
  std::vector<ur_exp_command_buffer_update_value_arg_desc_t> new_value_descs{};

  size_t new_local_size = 2;
  size_t new_local_mem_a_size = new_local_size * sizeof(uint32_t);
  // New local_mem_a at index 0
  new_value_descs.push_back({
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      0,                                                          // argIndex
      new_local_mem_a_size,                                       // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  });

  if (backend == UR_PLATFORM_BACKEND_HIP) {
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        1,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        2,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        3,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
  }

  // New local_mem_b at index 1
  size_t new_local_mem_b_size = new_local_size * sizeof(uint32_t) * 2;
  new_value_descs.push_back({
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1 + hip_arg_offset,                                         // argIndex
      new_local_mem_b_size,                                       // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  });

  if (backend == UR_PLATFORM_BACKEND_HIP) {
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        5,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        6,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        7,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
  }

  // New A at index 3
  uint32_t new_A = 33;
  new_value_descs.push_back({
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      3 + (2 * hip_arg_offset),                                   // argIndex
      sizeof(new_A),                                              // argSize
      nullptr,                                                    // pProperties
      &new_A,                                                     // hArgValue
  });

  // New X at index 4
  new_input_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      4 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[3], // pArgValue
  };

  // New Y at index 5
  new_input_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      5 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[4], // pArgValue
  };

  // Update kernel inputs
  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,                                // hCommand
      kernel,                                        // hNewKernel
      0,                                             // numNewMemObjArgs
      2,                                             // numNewPointerArgs
      static_cast<uint32_t>(new_value_descs.size()), // numNewValueArgs
      n_dimensions,                                  // newWorkDim
      nullptr,                                       // pNewMemObjArgList
      new_input_descs,                               // pNewPointerArgList
      new_value_descs.data(),                        // pNewValueArgList
      nullptr,                                       // pNewGlobalWorkOffset
      nullptr,                                       // pNewGlobalWorkSize
      &new_local_size,                               // pNewLocalWorkSize
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
  Validate(new_output, new_X, new_Y, new_A, global_size, new_local_size);
}

// Test updating A,X,Y parameters to new values and local memory parameters
// to new larger values.
TEST_P(LocalMemoryUpdateTest, UpdateParametersLargerLocalSize) {
  // Run command-buffer prior to update and verify output
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *output = (uint32_t *)shared_ptrs[0];
  uint32_t *X = (uint32_t *)shared_ptrs[1];
  uint32_t *Y = (uint32_t *)shared_ptrs[2];
  Validate(output, X, Y, A, global_size, local_size);

  // Update inputs
  ur_exp_command_buffer_update_pointer_arg_desc_t new_input_descs[2];
  std::vector<ur_exp_command_buffer_update_value_arg_desc_t> new_value_descs{};

  size_t new_local_size = local_size * 4;
  size_t new_local_mem_a_size = new_local_size * sizeof(uint32_t);
  // New local_mem_a at index 0
  new_value_descs.push_back({
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      0,                                                          // argIndex
      new_local_mem_a_size,                                       // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  });

  if (backend == UR_PLATFORM_BACKEND_HIP) {
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        1,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        2,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        3,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
  }

  // New local_mem_b at index 1
  size_t new_local_mem_b_size = new_local_size * sizeof(uint32_t) * 2;
  new_value_descs.push_back({
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1 + hip_arg_offset,                                         // argIndex
      new_local_mem_b_size,                                       // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  });

  if (backend == UR_PLATFORM_BACKEND_HIP) {
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        5,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        6,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        7,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
  }

  // New A at index 3
  uint32_t new_A = 33;
  new_value_descs.push_back({
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      3 + (2 * hip_arg_offset),                                   // argIndex
      sizeof(new_A),                                              // argSize
      nullptr,                                                    // pProperties
      &new_A,                                                     // hArgValue
  });

  // New X at index 4
  new_input_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      4 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[3], // pArgValue
  };

  // New Y at index 5
  new_input_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      5 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[4], // pArgValue
  };

  // Update kernel inputs
  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,                                // hCommand
      kernel,                                        // hNewKernel
      0,                                             // numNewMemObjArgs
      2,                                             // numNewPointerArgs
      static_cast<uint32_t>(new_value_descs.size()), // numNewValueArgs
      n_dimensions,                                  // newWorkDim
      nullptr,                                       // pNewMemObjArgList
      new_input_descs,                               // pNewPointerArgList
      new_value_descs.data(),                        // pNewValueArgList
      nullptr,                                       // pNewGlobalWorkOffset
      nullptr,                                       // pNewGlobalWorkSize
      &new_local_size,                               // pNewLocalWorkSize
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
  Validate(new_output, new_X, new_Y, new_A, global_size, new_local_size);
}

// Test updating A,X,Y parameters to new values and only one of the local memory
// parameters, which is set to a new value. Then a separate update call for
// the other local memory argument.
TEST_P(LocalMemoryUpdateTest, UpdateParametersPartialLocalSize) {
  // Run command-buffer prior to update and verify output
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *output = (uint32_t *)shared_ptrs[0];
  uint32_t *X = (uint32_t *)shared_ptrs[1];
  uint32_t *Y = (uint32_t *)shared_ptrs[2];
  Validate(output, X, Y, A, global_size, local_size);

  // Update inputs
  ur_exp_command_buffer_update_pointer_arg_desc_t new_input_descs[2];
  std::vector<ur_exp_command_buffer_update_value_arg_desc_t> new_value_descs{};

  size_t new_local_size = local_size * 4;
  size_t new_local_mem_a_size = new_local_size * sizeof(uint32_t);
  // New local_mem_a at index 0
  new_value_descs.push_back({
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      0,                                                          // argIndex
      new_local_mem_a_size,                                       // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  });

  if (backend == UR_PLATFORM_BACKEND_HIP) {
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        1,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        2,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
    new_value_descs.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        3,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
  }

  // New A at index 3
  uint32_t new_A = 33;
  new_value_descs.push_back({
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      3 + (2 * hip_arg_offset),                                   // argIndex
      sizeof(new_A),                                              // argSize
      nullptr,                                                    // pProperties
      &new_A,                                                     // hArgValue
  });

  // New X at index 4
  new_input_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      4 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[3], // pArgValue
  };

  // New Y at index 5
  new_input_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      5 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[4], // pArgValue
  };

  // Update kernel inputs
  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,                                // hCommand
      kernel,                                        // hNewKernel
      0,                                             // numNewMemObjArgs
      2,                                             // numNewPointerArgs
      static_cast<uint32_t>(new_value_descs.size()), // numNewValueArgs
      n_dimensions,                                  // newWorkDim
      nullptr,                                       // pNewMemObjArgList
      new_input_descs,                               // pNewPointerArgList
      new_value_descs.data(),                        // pNewValueArgList
      nullptr,                                       // pNewGlobalWorkOffset
      nullptr,                                       // pNewGlobalWorkSize
      &new_local_size,                               // pNewLocalWorkSize
  };

  // Update kernel and enqueue command-buffer again
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(updatable_cmd_buf_handle,
                                                      1, &update_desc));

  std::vector<ur_exp_command_buffer_update_value_arg_desc_t>
      second_update_value_args{};

  size_t new_local_mem_b_size = new_local_size * sizeof(uint32_t) * 2;
  // New local_mem_b at index 1
  second_update_value_args.push_back({
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1 + hip_arg_offset,                                         // argIndex
      new_local_mem_b_size,                                       // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  });

  if (backend == UR_PLATFORM_BACKEND_HIP) {
    second_update_value_args.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        5,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
    second_update_value_args.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        6,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
    second_update_value_args.push_back({
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        7,                                                          // argIndex
        sizeof(hip_local_offset),                                   // argSize
        nullptr,           // pProperties
        &hip_local_offset, // hArgValue
    });
  }

  ur_exp_command_buffer_update_kernel_launch_desc_t second_update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle, // hCommand
      kernel,         // hNewKernel
      0,              // numNewMemObjArgs
      0,              // numNewPointerArgs
      static_cast<uint32_t>(second_update_value_args.size()), // numNewValueArgs
      n_dimensions,                                           // newWorkDim
      nullptr,                         // pNewMemObjArgList
      nullptr,                         // pNewPointerArgList
      second_update_value_args.data(), // pNewValueArgList
      nullptr,                         // pNewGlobalWorkOffset
      nullptr,                         // pNewGlobalWorkSize
      nullptr,                         // pNewLocalWorkSize
  };
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(updatable_cmd_buf_handle,
                                                      1, &second_update_desc));

  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  // Verify that update occurred correctly
  uint32_t *new_output = (uint32_t *)shared_ptrs[0];
  uint32_t *new_X = (uint32_t *)shared_ptrs[3];
  uint32_t *new_Y = (uint32_t *)shared_ptrs[4];
  Validate(new_output, new_X, new_Y, new_A, global_size, new_local_size);
}

struct LocalMemoryMultiUpdateTest : LocalMemoryUpdateTestBase {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(LocalMemoryUpdateTestBase::SetUp());
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

UUR_INSTANTIATE_DEVICE_TEST_SUITE(LocalMemoryMultiUpdateTest);

// Test updating A,X,Y parameters to new values and local memory parameters
// to original values.
TEST_P(LocalMemoryMultiUpdateTest, UpdateParameters) {
  // Run command-buffer prior to update an verify output
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *output = (uint32_t *)shared_ptrs[0];
  uint32_t *X = (uint32_t *)shared_ptrs[1];
  uint32_t *Y = (uint32_t *)shared_ptrs[2];
  Validate(output, X, Y, A, global_size, local_size);

  // Update inputs
  std::array<ur_exp_command_buffer_update_pointer_arg_desc_t, 2>
      new_input_descs;
  std::array<ur_exp_command_buffer_update_value_arg_desc_t, 3> new_value_descs;

  // New local_mem_a at index 0
  new_value_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      0,                                                          // argIndex
      local_mem_a_size,                                           // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  };

  // New local_mem_b at index 1
  new_value_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1 + hip_arg_offset,                                         // argIndex
      local_mem_b_size,                                           // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  };

  // New A at index 3
  uint32_t new_A = 33;
  new_value_descs[2] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      3 + (2 * hip_arg_offset),                                   // argIndex
      sizeof(new_A),                                              // argSize
      nullptr,                                                    // pProperties
      &new_A,                                                     // hArgValue
  };

  // New X at index 4
  new_input_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      4 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[3], // pArgValue
  };

  // New Y at index 5
  new_input_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      5 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[4], // pArgValue
  };

  std::vector<ur_exp_command_buffer_update_kernel_launch_desc_t> update_descs;
  for (auto &handle : command_handles) {
    // Update kernel inputs
    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        handle,                 // hCommand
        kernel,                 // hNewKernel
        0,                      // numNewMemObjArgs
        new_input_descs.size(), // numNewPointerArgs
        new_value_descs.size(), // numNewValueArgs
        n_dimensions,           // newWorkDim
        nullptr,                // pNewMemObjArgList
        new_input_descs.data(), // pNewPointerArgList
        new_value_descs.data(), // pNewValueArgList
        nullptr,                // pNewGlobalWorkOffset
        nullptr,                // pNewGlobalWorkSize
        nullptr,                // pNewLocalWorkSize
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
  Validate(new_output, new_X, new_Y, new_A, global_size, local_size);
}

// Test updating A,X,Y parameters to new values and local memory parameters
// to original values, but without doing a blocking wait.
TEST_P(LocalMemoryMultiUpdateTest, UpdateWithoutBlocking) {
  // Update inputs
  std::array<ur_exp_command_buffer_update_pointer_arg_desc_t, 2>
      new_input_descs;
  std::array<ur_exp_command_buffer_update_value_arg_desc_t, 3> new_value_descs;

  // New local_mem_a at index 0
  new_value_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      0,                                                          // argIndex
      local_mem_a_size,                                           // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  };

  // New local_mem_a at index 1
  new_value_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1 + hip_arg_offset,                                         // argIndex
      local_mem_b_size,                                           // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  };

  // New A at index 3
  uint32_t new_A = 33;
  new_value_descs[2] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      3 + (2 * hip_arg_offset),                                   // argIndex
      sizeof(new_A),                                              // argSize
      nullptr,                                                    // pProperties
      &new_A,                                                     // hArgValue
  };

  // New X at index 4
  new_input_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      4 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[3], // pArgValue
  };

  // New Y at index 5
  new_input_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      5 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[4], // pArgValue
  };

  // Enqueue without calling urQueueFinish after
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));

  std::vector<ur_exp_command_buffer_update_kernel_launch_desc_t> update_descs;
  for (auto &handle : command_handles) {
    // Update kernel inputs
    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        handle,                 // hCommand
        kernel,                 // hNewKernel
        0,                      // numNewMemObjArgs
        new_input_descs.size(), // numNewPointerArgs
        new_value_descs.size(), // numNewValueArgs
        n_dimensions,           // newWorkDim
        nullptr,                // pNewMemObjArgList
        new_input_descs.data(), // pNewPointerArgList
        new_value_descs.data(), // pNewValueArgList
        nullptr,                // pNewGlobalWorkOffset
        nullptr,                // pNewGlobalWorkSize
        nullptr,                // pNewLocalWorkSize
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
  Validate(new_output, new_X, new_Y, new_A, global_size, local_size);
}

struct LocalMemoryUpdateTestBaseOutOfOrder : LocalMemoryUpdateTestBase {
  virtual void SetUp() override {
    program_name = "saxpy_usm_local_mem";
    UUR_RETURN_ON_FATAL_FAILURE(
        urUpdatableCommandBufferExpExecutionTest::SetUp());

    // HIP has extra args for local memory so we define an offset for arg
    // indices here for updating
    hip_arg_offset = backend == UR_PLATFORM_BACKEND_HIP ? 3 : 0;
    ur_device_usm_access_capability_flags_t shared_usm_flags;
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
    if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
      GTEST_SKIP() << "Shared USM is not supported.";
    }

    const size_t allocation_size = sizeof(uint32_t) * global_size * local_size;
    for (auto &shared_ptr : shared_ptrs) {
      ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &shared_ptr));
      ASSERT_NE(shared_ptr, nullptr);

      std::vector<uint8_t> pattern(allocation_size);
      uur::generateMemFillPattern(pattern);
      std::memcpy(shared_ptr, pattern.data(), allocation_size);
    }

    std::array<size_t, 12> index_order{};
    if (backend != UR_PLATFORM_BACKEND_HIP) {
      index_order = {3, 2, 4, 5, 1, 0};
    } else {
      index_order = {9, 8, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3};
    }
    size_t current_index = 0;

    // Index 3 is A
    ASSERT_SUCCESS(urKernelSetArgValue(kernel, index_order[current_index++],
                                       sizeof(A), nullptr, &A));
    // Index 2 is output
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, index_order[current_index++],
                                         nullptr, shared_ptrs[0]));

    // Index 4 is X
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, index_order[current_index++],
                                         nullptr, shared_ptrs[1]));
    // Index 5 is Y
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, index_order[current_index++],
                                         nullptr, shared_ptrs[2]));

    // Index 1 is local_mem_b arg
    ASSERT_SUCCESS(urKernelSetArgLocal(kernel, index_order[current_index++],
                                       local_mem_b_size, nullptr));
    if (backend == UR_PLATFORM_BACKEND_HIP) {
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, index_order[current_index++],
                                         sizeof(hip_local_offset), nullptr,
                                         &hip_local_offset));
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, index_order[current_index++],
                                         sizeof(hip_local_offset), nullptr,
                                         &hip_local_offset));
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, index_order[current_index++],
                                         sizeof(hip_local_offset), nullptr,
                                         &hip_local_offset));
    }

    // Index 0 is local_mem_a arg
    ASSERT_SUCCESS(urKernelSetArgLocal(kernel, index_order[current_index++],
                                       local_mem_a_size, nullptr));

    // Hip has extra args for local mem at index 1-3
    if (backend == UR_PLATFORM_BACKEND_HIP) {
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, index_order[current_index++],
                                         sizeof(hip_local_offset), nullptr,
                                         &hip_local_offset));
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, index_order[current_index++],
                                         sizeof(hip_local_offset), nullptr,
                                         &hip_local_offset));
      ASSERT_SUCCESS(urKernelSetArgValue(kernel, index_order[current_index++],
                                         sizeof(hip_local_offset), nullptr,
                                         &hip_local_offset));
    }
  }
};

struct LocalMemoryUpdateTestOutOfOrder : LocalMemoryUpdateTestBaseOutOfOrder {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(LocalMemoryUpdateTestBaseOutOfOrder::SetUp());

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

UUR_INSTANTIATE_DEVICE_TEST_SUITE(LocalMemoryUpdateTestOutOfOrder);

// Test updating A,X,Y parameters to new values and local memory to larger
// values when the kernel arguments were added out of order.
TEST_P(LocalMemoryUpdateTestOutOfOrder, UpdateAllParameters) {
  // Run command-buffer prior to update and verify output
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *output = (uint32_t *)shared_ptrs[0];
  uint32_t *X = (uint32_t *)shared_ptrs[1];
  uint32_t *Y = (uint32_t *)shared_ptrs[2];
  Validate(output, X, Y, A, global_size, local_size);

  // Update inputs
  std::array<ur_exp_command_buffer_update_pointer_arg_desc_t, 2>
      new_input_descs;
  std::array<ur_exp_command_buffer_update_value_arg_desc_t, 3> new_value_descs;

  size_t new_local_size = local_size * 4;
  size_t new_local_mem_a_size = new_local_size * sizeof(uint32_t);

  // New local_mem_a at index 0
  new_value_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      0,                                                          // argIndex
      new_local_mem_a_size,                                       // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  };

  // New local_mem_b at index 1
  new_value_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1 + hip_arg_offset,                                         // argIndex
      local_mem_b_size,                                           // argSize
      nullptr,                                                    // pProperties
      nullptr,                                                    // hArgValue
  };

  // New A at index 3
  uint32_t new_A = 33;
  new_value_descs[2] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      3 + (2 * hip_arg_offset),                                   // argIndex
      sizeof(new_A),                                              // argSize
      nullptr,                                                    // pProperties
      &new_A,                                                     // hArgValue
  };

  // New X at index 4
  new_input_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      4 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[3], // pArgValue
  };

  // New Y at index 5
  new_input_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
      nullptr,                                                      // pNext
      5 + (2 * hip_arg_offset),                                     // argIndex
      nullptr,         // pProperties
      &shared_ptrs[4], // pArgValue
  };

  // Update kernel inputs
  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,         // hCommand
      kernel,                 // hNewKernel
      0,                      // numNewMemObjArgs
      new_input_descs.size(), // numNewPointerArgs
      new_value_descs.size(), // numNewValueArgs
      n_dimensions,           // newWorkDim
      nullptr,                // pNewMemObjArgList
      new_input_descs.data(), // pNewPointerArgList
      new_value_descs.data(), // pNewValueArgList
      nullptr,                // pNewGlobalWorkOffset
      nullptr,                // pNewGlobalWorkSize
      nullptr,                // pNewLocalWorkSize
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
  Validate(new_output, new_X, new_Y, new_A, global_size, local_size);
}
