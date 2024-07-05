// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include <cstring>

// Negative tests that correct error codes are thrown on invalid update usage.
struct InvalidUpdateTest
    : uur::command_buffer::urUpdatableCommandBufferExpExecutionTest {
    void SetUp() override {
        program_name = "fill_usm";
        UUR_RETURN_ON_FATAL_FAILURE(
            urUpdatableCommandBufferExpExecutionTest::SetUp());

        ur_device_usm_access_capability_flags_t shared_usm_flags;
        ASSERT_SUCCESS(
            uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
        if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
            GTEST_SKIP() << "Shared USM is not supported.";
        }

        // Allocate USM pointer to fill
        ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                        allocation_size, &shared_ptr));
        ASSERT_NE(shared_ptr, nullptr);
        std::memset(shared_ptr, 0, allocation_size);

        // Index 0 is output
        ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, shared_ptr));
        // Index 1 is input scalar
        ASSERT_SUCCESS(
            urKernelSetArgValue(kernel, 1, sizeof(val), nullptr, &val));

        // Append kernel command to command-buffer
        ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
            updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
            &global_size, &local_size, 0, nullptr, nullptr, &command_handle));
        ASSERT_NE(command_handle, nullptr);
    }

    void TearDown() override {
        // Workaround an issue with the OpenCL adapter implementing urUsmFree
        // using a blocking free where hangs
        if (updatable_cmd_buf_handle && !finalized) {
            EXPECT_SUCCESS(
                urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
        }

        if (shared_ptr) {
            EXPECT_SUCCESS(urUSMFree(context, shared_ptr));
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
    static constexpr size_t allocation_size = sizeof(val) * global_size;
    void *shared_ptr = nullptr;
    ur_exp_command_buffer_command_handle_t command_handle = nullptr;
    bool finalized = false;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(InvalidUpdateTest);

// Test error code is returned if command-buffer not finalized
TEST_P(InvalidUpdateTest, NotFinalizedCommandBuffer) {
    // Set new value to use for fill at kernel index 1
    uint32_t new_val = 33;
    ur_exp_command_buffer_update_value_arg_desc_t new_input_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        1,                                                          // argIndex
        sizeof(new_val),                                            // argSize
        nullptr,  // pProperties
        &new_val, // hArgValue
    };

    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        0,               // numNewMemObjArgs
        0,               // numNewPointerArgs
        1,               // numNewValueArgs
        0,               // newWorkDim
        nullptr,         // pNewMemObjArgList
        nullptr,         // pNewPointerArgList
        &new_input_desc, // pNewValueArgList
        nullptr,         // pNewGlobalWorkOffset
        nullptr,         // pNewGlobalWorkSize
        nullptr,         // pNewLocalWorkSize
    };

    // Update command to command-buffer that has not been finalized
    ur_result_t result =
        urCommandBufferUpdateKernelLaunchExp(command_handle, &update_desc);
    ASSERT_EQ(UR_RESULT_ERROR_INVALID_OPERATION, result);
}

// Test error code is returned if command-buffer not created with isUpdatable
TEST_P(InvalidUpdateTest, NotUpdatableCommandBuffer) {
    // Create a command-buffer without isUpdatable
    ur_exp_command_buffer_handle_t test_cmd_buf_handle = nullptr;
    ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, nullptr,
                                            &test_cmd_buf_handle));
    EXPECT_NE(test_cmd_buf_handle, nullptr);

    // Append a kernel commands to command-buffer and close command-buffer
    ur_exp_command_buffer_command_handle_t test_command_handle = nullptr;
    EXPECT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
        test_cmd_buf_handle, kernel, n_dimensions, &global_offset, &global_size,
        &local_size, 0, nullptr, nullptr, &test_command_handle));
    EXPECT_NE(test_command_handle, nullptr);

    EXPECT_SUCCESS(urCommandBufferFinalizeExp(test_cmd_buf_handle));
    finalized = true;

    // Set new value to use for fill at kernel index 1
    uint32_t new_val = 33;
    ur_exp_command_buffer_update_value_arg_desc_t new_input_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        1,                                                          // argIndex
        sizeof(new_val),                                            // argSize
        nullptr,  // pProperties
        &new_val, // hArgValue
    };

    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        0,               // numNewMemObjArgs
        0,               // numNewPointerArgs
        1,               // numNewValueArgs
        0,               // newWorkDim
        nullptr,         // pNewMemObjArgList
        nullptr,         // pNewPointerArgList
        &new_input_desc, // pNewValueArgList
        nullptr,         // pNewGlobalWorkOffset
        nullptr,         // pNewGlobalWorkSize
        nullptr,         // pNewLocalWorkSize
    };

    // Update command to command-buffer that doesn't have updatable set should
    // be an error
    ur_result_t result =
        urCommandBufferUpdateKernelLaunchExp(test_command_handle, &update_desc);
    EXPECT_EQ(UR_RESULT_ERROR_INVALID_OPERATION, result);

    if (test_command_handle) {
        EXPECT_SUCCESS(urCommandBufferReleaseCommandExp(test_command_handle));
    }
    if (test_cmd_buf_handle) {
        EXPECT_SUCCESS(urCommandBufferReleaseExp(test_cmd_buf_handle));
    }
}

// Test setting `pNewLocalWorkSize` to a non-NULL value and `pNewGlobalWorkSize`
// to NULL gives the correct error.
TEST_P(InvalidUpdateTest, GlobalLocalSizeMistach) {
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
    finalized = true;

    size_t new_local_size = 16;
    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        0,               // numNewMemObjArgs
        0,               // numNewPointerArgs
        0,               // numNewValueArgs
        n_dimensions,    // newWorkDim
        nullptr,         // pNewMemObjArgList
        nullptr,         // pNewPointerArgList
        nullptr,         // pNewValueArgList
        nullptr,         // pNewGlobalWorkOffset
        nullptr,         // pNewGlobalWorkSize
        &new_local_size, // pNewLocalWorkSize
    };

    // Update command local size but not global size
    ur_result_t result =
        urCommandBufferUpdateKernelLaunchExp(command_handle, &update_desc);
    ASSERT_EQ(UR_RESULT_ERROR_INVALID_OPERATION, result);
}

// Test setting `pNewLocalWorkSize` to a non-NULL value when the command was
// created with a NULL local work size gives the correct error.
TEST_P(InvalidUpdateTest, ImplToUserDefinedLocalSize) {
    // Append kernel command to command-buffer using NULL local work size
    ur_exp_command_buffer_command_handle_t second_command_handle = nullptr;
    ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
        updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
        &global_size, nullptr, 0, nullptr, nullptr, &second_command_handle));
    ASSERT_NE(second_command_handle, nullptr);

    EXPECT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
    finalized = true;

    size_t new_global_size = 64;
    size_t new_local_size = 16;
    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        0,                // numNewMemObjArgs
        0,                // numNewPointerArgs
        0,                // numNewValueArgs
        n_dimensions,     // newWorkDim
        nullptr,          // pNewMemObjArgList
        nullptr,          // pNewPointerArgList
        nullptr,          // pNewValueArgList
        nullptr,          // pNewGlobalWorkOffset
        &new_global_size, // pNewGlobalWorkSize
        &new_local_size,  // pNewLocalWorkSize
    };

    // Update command local size to non-NULL when created with NULL value
    ur_result_t result = urCommandBufferUpdateKernelLaunchExp(
        second_command_handle, &update_desc);
    EXPECT_EQ(UR_RESULT_ERROR_INVALID_OPERATION, result);

    if (second_command_handle) {
        EXPECT_SUCCESS(urCommandBufferReleaseCommandExp(second_command_handle));
    }
}

// Test setting `pNewLocalWorkSize` to a NULL value when the command was
// created with a non-NULL local work size gives the correct error.
TEST_P(InvalidUpdateTest, UserToImplDefinedLocalSize) {
    ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
    finalized = true;

    size_t new_global_size = 64;
    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        0,                // numNewMemObjArgs
        0,                // numNewPointerArgs
        0,                // numNewValueArgs
        n_dimensions,     // newWorkDim
        nullptr,          // pNewMemObjArgList
        nullptr,          // pNewPointerArgList
        nullptr,          // pNewValueArgList
        nullptr,          // pNewGlobalWorkOffset
        &new_global_size, // pNewGlobalWorkSize
        nullptr,          // pNewLocalWorkSize
    };

    // Update command local size to NULL when created with non-NULL value
    ur_result_t result =
        urCommandBufferUpdateKernelLaunchExp(command_handle, &update_desc);
    ASSERT_EQ(UR_RESULT_ERROR_INVALID_OPERATION, result);
}
