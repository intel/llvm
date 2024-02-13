// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

// Test that updating a command-buffer with a single kernel command
// taking USM arguments works correctly.
struct BufferFillCommandTest
    : uur::command_buffer::urUpdatableCommandBufferExpExecutionTest {
    void SetUp() override {
        program_name = "fill";
        UUR_RETURN_ON_FATAL_FAILURE(
            urUpdatableCommandBufferExpExecutionTest::SetUp());

        // First argument is buffer to fill (will also be hidden accessor arg)
        AddBuffer1DArg(sizeof(val) * global_size, &buffer);
        // Second argument is scalar to fill with.
        AddPodArg(val);

        // Append kernel command to command-buffer and close command-buffer
        ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
            updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
            &global_size, &local_size, 0, nullptr, nullptr, &command_handle));
        ASSERT_NE(command_handle, nullptr);

        ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
    }

    void TearDown() override {
        if (new_buffer) {
            EXPECT_SUCCESS(urMemRelease(new_buffer));
        }

        if (command_handle) {
            EXPECT_SUCCESS(urCommandBufferReleaseCommandExp(command_handle));
        }

        UUR_RETURN_ON_FATAL_FAILURE(
            urUpdatableCommandBufferExpExecutionTest::TearDown());
    }

    static constexpr uint32_t val = 42;
    static constexpr size_t local_size = 4;
    static constexpr size_t global_size = 32;
    static constexpr size_t global_offset = 0;
    static constexpr size_t n_dimensions = 1;
    static constexpr size_t buffer_size = sizeof(val) * global_size;
    ur_mem_handle_t buffer = nullptr;
    ur_mem_handle_t new_buffer = nullptr;
    ur_exp_command_buffer_command_handle_t command_handle = nullptr;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(BufferFillCommandTest);

// Update kernel arguments to fill with a new scalar value to a new output
// buffer.
TEST_P(BufferFillCommandTest, UpdateParameters) {
    // Run command-buffer prior to update an verify output
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    ValidateBuffer(buffer, buffer_size, val);

    // Create a new buffer to update kernel output parameter to
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                     buffer_size, nullptr, &new_buffer));
    char zero = 0;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, new_buffer, &zero,
                                          sizeof(zero), 0, buffer_size, 0,
                                          nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    // Set argument index zero as new buffer
    ur_exp_command_buffer_update_memobj_arg_desc_t new_output_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_MEMOBJ_ARG_DESC, // stype
        nullptr,                                                     // pNext
        0,                                                           // argIndex
        nullptr,    // pProperties
        new_buffer, // hArgValue
    };

    // Set argument index 2 as new value to fill (index 1 is buffer accessor)
    uint32_t new_val = 33;
    ur_exp_command_buffer_update_value_arg_desc_t new_input_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        2,                                                          // argIndex
        sizeof(new_val),                                            // argSize
        nullptr,  // pProperties
        &new_val, // hArgValue
    };

    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        1,                // numNewMemObjArgs
        0,                // numNewPointerArgs
        1,                // numNewValueArgs
        0,                // numNewExecInfos
        0,                // newWorkDim
        &new_output_desc, // pNewMemObjArgList
        nullptr,          // pNewPointerArgList
        &new_input_desc,  // pNewValueArgList
        nullptr,          // pNewExecInfoList
        nullptr,          // pNewGlobalWorkOffset
        nullptr,          // pNewGlobalWorkSize
        nullptr,          // pNewLocalWorkSize
    };

    // Update kernel and enqueue command-buffer again
    ASSERT_SUCCESS(
        urCommandBufferUpdateKernelLaunchExp(command_handle, &update_desc));
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    // Verify that update occurred correctly
    ValidateBuffer(new_buffer, buffer_size, new_val);
}

// Test updating the global size so that the fill outputs to a larger buffer
TEST_P(BufferFillCommandTest, UpdateGlobalSize) {
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    ValidateBuffer(buffer, sizeof(val) * global_size, val);

    size_t new_global_size = 64;
    const size_t new_buffer_size = sizeof(val) * new_global_size;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                     new_buffer_size, nullptr, &new_buffer));
    char zero = 0;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, new_buffer, &zero,
                                          sizeof(zero), 0, new_buffer_size, 0,
                                          nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    ur_exp_command_buffer_update_memobj_arg_desc_t new_output_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_MEMOBJ_ARG_DESC, // stype
        nullptr,                                                     // pNext
        0,                                                           // argIndex
        nullptr,    // pProperties
        new_buffer, // hArgValue
    };

    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        1,                // numNewMemObjArgs
        0,                // numNewPointerArgs
        0,                // numNewValueArgs
        0,                // numNewExecInfos
        0,                // newWorkDim
        &new_output_desc, // pNewMemObjArgList
        nullptr,          // pNewPointerArgList
        nullptr,          // pNewValueArgList
        nullptr,          // pNewExecInfoList
        nullptr,          // pNewGlobalWorkOffset
        &new_global_size, // pNewGlobalWorkSize
        nullptr,          // pNewLocalWorkSize
    };

    ASSERT_SUCCESS(
        urCommandBufferUpdateKernelLaunchExp(command_handle, &update_desc));
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    ValidateBuffer(new_buffer, new_buffer_size, val);
}

// Test updating the input & output kernel arguments and global
// size, by calling update individually for each of these configurations.
TEST_P(BufferFillCommandTest, SeparateUpdateCalls) {
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    ValidateBuffer(buffer, sizeof(val) * global_size, val);

    size_t new_global_size = 64;
    const size_t new_buffer_size = sizeof(val) * new_global_size;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                     new_buffer_size, nullptr, &new_buffer));
    char zero = 0;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, new_buffer, &zero,
                                          sizeof(zero), 0, new_buffer_size, 0,
                                          nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    ur_exp_command_buffer_update_memobj_arg_desc_t new_output_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_MEMOBJ_ARG_DESC, // stype
        nullptr,                                                     // pNext
        0,                                                           // argIndex
        nullptr,    // pProperties
        new_buffer, // hArgValue
    };

    ur_exp_command_buffer_update_kernel_launch_desc_t output_update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        1,                // numNewMemObjArgs
        0,                // numNewPointerArgs
        0,                // numNewValueArgs
        0,                // numNewExecInfos
        0,                // newWorkDim
        &new_output_desc, // pNewMemObjArgList
        nullptr,          // pNewPointerArgList
        nullptr,          // pNewValueArgList
        nullptr,          // pNewExecInfoList
        nullptr,          // pNewGlobalWorkOffset
        nullptr,          // pNewGlobalWorkSize
        nullptr,          // pNewLocalWorkSize
    };
    ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(command_handle,
                                                        &output_update_desc));

    uint32_t new_val = 33;
    ur_exp_command_buffer_update_value_arg_desc_t new_input_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        2,                                                          // argIndex
        sizeof(new_val),                                            // argSize
        nullptr,  // pProperties
        &new_val, // hArgValue
    };

    ur_exp_command_buffer_update_kernel_launch_desc_t input_update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        0,               // numNewMemObjArgs
        0,               // numNewPointerArgs
        1,               // numNewValueArgs
        0,               // numNewExecInfos
        0,               // newWorkDim
        nullptr,         // pNewMemObjArgList
        nullptr,         // pNewPointerArgList
        &new_input_desc, // pNewValueArgList
        nullptr,         // pNewExecInfoList
        nullptr,         // pNewGlobalWorkOffset
        nullptr,         // pNewGlobalWorkSize
        nullptr,         // pNewLocalWorkSize
    };
    ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(command_handle,
                                                        &input_update_desc));

    ur_exp_command_buffer_update_kernel_launch_desc_t global_size_update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        0,                // numNewMemObjArgs
        0,                // numNewPointerArgs
        0,                // numNewValueArgs
        0,                // numNewExecInfos
        0,                // newWorkDim
        nullptr,          // pNewMemObjArgList
        nullptr,          // pNewPointerArgList
        nullptr,          // pNewValueArgList
        nullptr,          // pNewExecInfoList
        nullptr,          // pNewGlobalWorkOffset
        &new_global_size, // pNewGlobalWorkSize
        nullptr,          // pNewLocalWorkSize
    };

    ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(
        command_handle, &global_size_update_desc));

    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    ValidateBuffer(new_buffer, new_buffer_size, new_val);
}

// Test calling update twice on the same command-handle updating the
// input value, and verifying that it's the second call which persists.
TEST_P(BufferFillCommandTest, OverrideUpdate) {
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    ValidateBuffer(buffer, sizeof(val) * global_size, val);

    uint32_t first_val = 33;
    ur_exp_command_buffer_update_value_arg_desc_t first_input_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        2,                                                          // argIndex
        sizeof(first_val),                                          // argSize
        nullptr,    // pProperties
        &first_val, // hArgValue
    };

    ur_exp_command_buffer_update_kernel_launch_desc_t first_update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        0,                 // numNewMemObjArgs
        0,                 // numNewPointerArgs
        1,                 // numNewValueArgs
        0,                 // numNewExecInfos
        0,                 // newWorkDim
        nullptr,           // pNewMemObjArgList
        nullptr,           // pNewPointerArgList
        &first_input_desc, // pNewValueArgList
        nullptr,           // pNewExecInfoList
        nullptr,           // pNewGlobalWorkOffset
        nullptr,           // pNewGlobalWorkSize
        nullptr,           // pNewLocalWorkSize
    };
    ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(command_handle,
                                                        &first_update_desc));

    uint32_t second_val = -99;
    ur_exp_command_buffer_update_value_arg_desc_t second_input_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        2,                                                          // argIndex
        sizeof(second_val),                                         // argSize
        nullptr,     // pProperties
        &second_val, // hArgValue
    };

    ur_exp_command_buffer_update_kernel_launch_desc_t second_update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        0,                  // numNewMemObjArgs
        0,                  // numNewPointerArgs
        1,                  // numNewValueArgs
        0,                  // numNewExecInfos
        0,                  // newWorkDim
        nullptr,            // pNewMemObjArgList
        nullptr,            // pNewPointerArgList
        &second_input_desc, // pNewValueArgList
        nullptr,            // pNewExecInfoList
        nullptr,            // pNewGlobalWorkOffset
        nullptr,            // pNewGlobalWorkSize
        nullptr,            // pNewLocalWorkSize
    };

    ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(command_handle,
                                                        &second_update_desc));

    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    ValidateBuffer(buffer, sizeof(val) * global_size, second_val);
}

// Test calling update with multiple ur_exp_command_buffer_update_value_arg_desc_t
// instances updating the same argument, and checking that the last one in the
// list persists.
TEST_P(BufferFillCommandTest, OverrideArgList) {
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    ValidateBuffer(buffer, sizeof(val) * global_size, val);

    ur_exp_command_buffer_update_value_arg_desc_t input_descs[2];
    uint32_t first_val = 33;
    input_descs[0] = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        2,                                                          // argIndex
        sizeof(first_val),                                          // argSize
        nullptr,    // pProperties
        &first_val, // hArgValue
    };

    uint32_t second_val = -99;
    input_descs[1] = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        2,                                                          // argIndex
        sizeof(second_val),                                         // argSize
        nullptr,     // pProperties
        &second_val, // hArgValue
    };

    ur_exp_command_buffer_update_kernel_launch_desc_t second_update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        0,           // numNewMemObjArgs
        0,           // numNewPointerArgs
        2,           // numNewValueArgs
        0,           // numNewExecInfos
        0,           // newWorkDim
        nullptr,     // pNewMemObjArgList
        nullptr,     // pNewPointerArgList
        input_descs, // pNewValueArgList
        nullptr,     // pNewExecInfoList
        nullptr,     // pNewGlobalWorkOffset
        nullptr,     // pNewGlobalWorkSize
        nullptr,     // pNewLocalWorkSize
    };

    ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(command_handle,
                                                        &second_update_desc));

    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    ValidateBuffer(buffer, sizeof(val) * global_size, second_val);
}
