// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../fixtures.h"
#include <array>

// Test that updating a command-buffer with a single kernel command
// taking buffer & scalar arguments works correctly.
struct BufferSaxpyKernelTest
    : uur::command_buffer::urUpdatableCommandBufferExpExecutionTest {
  void SetUp() override {
    program_name = "saxpy";
    UUR_RETURN_ON_FATAL_FAILURE(
        urUpdatableCommandBufferExpExecutionTest::SetUp());

    const size_t allocation_size = sizeof(uint32_t) * global_size;
    for (auto &buffer : buffers) {
      ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                       allocation_size, nullptr, &buffer));
      ASSERT_NE(buffer, nullptr);

      std::vector<uint8_t> init(allocation_size);
      uur::generateMemFillPattern(init);

      ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0,
                                             allocation_size, init.data(), 0,
                                             nullptr, nullptr));
    }

    // Variable that is incremented as arguments are added to the kernel
    size_t current_arg_index = 0;
    // Index 0 is output buffer for HIP/Non-HIP
    ASSERT_SUCCESS(
        urKernelSetArgMemObj(kernel, current_arg_index++, nullptr, buffers[0]));

    // Lambda to add accessor arguments depending on backend.
    // HIP has 3 offset parameters and other backends only have 1.
    auto addAccessorArgs = [&]() {
      if (backend == UR_PLATFORM_BACKEND_HIP) {
        size_t val = 0;
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index++,
                                           sizeof(size_t), nullptr, &val));
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index++,
                                           sizeof(size_t), nullptr, &val));
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index++,
                                           sizeof(size_t), nullptr, &val));
      } else {
        struct {
          size_t offsets[1] = {0};
        } accessor;
        ASSERT_SUCCESS(urKernelSetArgValue(
            kernel, current_arg_index++, sizeof(accessor), nullptr, &accessor));
      }
    };

    // Index 3 on HIP and 1 on non-HIP are accessors
    addAccessorArgs();

    // Index 4 on HIP and 2 on non-HIP is A
    ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index++, sizeof(A),
                                       nullptr, &A));

    // Index 5 on HIP and 3 on non-HIP is X buffer
    ASSERT_SUCCESS(
        urKernelSetArgMemObj(kernel, current_arg_index++, nullptr, buffers[1]));

    // Index 8 on HIP and 4 on non-HIP is X buffer accessor
    addAccessorArgs();

    // Index 9 on HIP and 5 on non-HIP is Y buffer
    ASSERT_SUCCESS(
        urKernelSetArgMemObj(kernel, current_arg_index++, nullptr, buffers[2]));

    // Index 12 on HIP and 6 on non-HIP is Y buffer accessor
    addAccessorArgs();

    // Append kernel command to command-buffer and close command-buffer
    ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
        updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
        &global_size, &local_size, 0, nullptr, 0, nullptr, 0, nullptr, nullptr,
        nullptr, &command_handle));
    ASSERT_NE(command_handle, nullptr);

    ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
  }

  void Validate(ur_mem_handle_t output, ur_mem_handle_t X, ur_mem_handle_t Y,
                uint32_t A, size_t length) {

    std::vector<uint32_t> output_data(length, 0);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, output, true, 0, length,
                                          output_data.data(), 0, nullptr,
                                          nullptr));

    std::vector<uint32_t> X_data(length, 0);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, X, true, 0, length,
                                          X_data.data(), 0, nullptr, nullptr));

    std::vector<uint32_t> Y_data(length, 0);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, Y, true, 0, length,
                                          Y_data.data(), 0, nullptr, nullptr));

    for (size_t i = 0; i < length; i++) {
      uint32_t result = A * X_data[i] + Y_data[i];
      ASSERT_EQ(result, output_data[i]);
    }
  }

  void TearDown() override {
    for (auto &buffer : buffers) {
      if (buffer) {
        EXPECT_SUCCESS(urMemRelease(buffer));
      }
    }

    UUR_RETURN_ON_FATAL_FAILURE(
        urUpdatableCommandBufferExpExecutionTest::TearDown());
  }

  static constexpr size_t local_size = 4;
  static constexpr size_t global_size = 32;
  static constexpr size_t global_offset = 0;
  static constexpr uint32_t n_dimensions = 1;
  static constexpr uint32_t A = 42;
  std::array<ur_mem_handle_t, 5> buffers = {nullptr, nullptr, nullptr, nullptr};
  ur_exp_command_buffer_command_handle_t command_handle = nullptr;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(BufferSaxpyKernelTest);

TEST_P(BufferSaxpyKernelTest, UpdateParameters) {
  // Run command-buffer prior to update an verify output
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));

  ASSERT_SUCCESS(urQueueFinish(queue));
  Validate(buffers[0], buffers[1], buffers[2], A, global_size);

  ur_exp_command_buffer_update_memobj_arg_desc_t new_input_descs[2];

  // Index 5 on HIP and 3 on non-HIP is X buffer
  const uint32_t x_arg_index = (backend == UR_PLATFORM_BACKEND_HIP) ? 5 : 3;
  new_input_descs[0] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_MEMOBJ_ARG_DESC, // stype
      nullptr,                                                     // pNext
      x_arg_index,                                                 // argIndex
      nullptr,    // pProperties
      buffers[3], // hArgValue
  };

  // Index 9 on HIP and 5 on non-HIP is Y buffer
  const uint32_t y_arg_index = backend == (UR_PLATFORM_BACKEND_HIP) ? 9 : 5;
  new_input_descs[1] = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_MEMOBJ_ARG_DESC, // stype
      nullptr,                                                     // pNext
      y_arg_index,                                                 // argIndex
      nullptr,    // pProperties
      buffers[4], // hArgValue
  };

  // Index 4 on HIP and 2 on non-HIP is A
  const uint32_t a_arg_index = (backend == UR_PLATFORM_BACKEND_HIP) ? 4 : 2;
  uint32_t new_A = 33;
  ur_exp_command_buffer_update_value_arg_desc_t new_A_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext,
      a_arg_index,                                                // argIndex
      sizeof(new_A),                                              // argSize
      nullptr,                                                    // pProperties
      &new_A,                                                     // hArgValue
  };

  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,  // hCommand
      kernel,          // hNewKernel
      2,               // numNewMemObjArgs
      0,               // numNewPointerArgs
      1,               // numNewValueArgs
      n_dimensions,    // newWorkDim
      new_input_descs, // pNewMemObjArgList
      nullptr,         // pNewPointerArgList
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
  Validate(buffers[0], buffers[3], buffers[4], new_A, global_size);
}
