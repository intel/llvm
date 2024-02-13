// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include <cstring>

// Test that updating a command-buffer with a single kernel command
// taking USM arguments works correctly.
struct USMFillCommandTest
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
        ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, &shared_ptr));
        // Index 1 is input scalar
        ASSERT_SUCCESS(
            urKernelSetArgValue(kernel, 1, sizeof(val), nullptr, &val));

        // Append kernel command to command-buffer and close command-buffer
        ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
            updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
            &global_size, &local_size, 0, nullptr, nullptr, &command_handle));
        ASSERT_NE(command_handle, nullptr);

        ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
    }

    void Validate(uint32_t *pointer, size_t length, uint32_t val) {
        for (size_t i = 0; i < length; i++) {
            ASSERT_EQ(pointer[i], val);
        }
    }

    void TearDown() override {
        if (shared_ptr) {
            EXPECT_SUCCESS(urUSMFree(context, shared_ptr));
        }

        if (new_shared_ptr) {
            EXPECT_SUCCESS(urUSMFree(context, new_shared_ptr));
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
    void *new_shared_ptr = nullptr;
    ur_exp_command_buffer_command_handle_t command_handle = nullptr;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(USMFillCommandTest);

// Test using a different global size to fill and larger USM output buffer
TEST_P(USMFillCommandTest, UpdateParameters) {
    // Run command-buffer prior to update an verify output
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    Validate((uint32_t *)shared_ptr, global_size, val);

    // Allocate a new USM pointer of larger size
    size_t new_global_size = 64;
    const size_t new_allocation_size = sizeof(val) * new_global_size;
    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    new_allocation_size, &new_shared_ptr));
    ASSERT_NE(new_shared_ptr, nullptr);
    std::memset(new_shared_ptr, 0, new_allocation_size);

    // Set new USM pointer as kernel output at index 0
    ur_exp_command_buffer_update_pointer_arg_desc_t new_output_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
        nullptr,                                                      // pNext
        0,               // argIndex
        nullptr,         // pProperties
        &new_shared_ptr, // pArgValue
    };

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
        0,                // numNewMemObjArgs
        1,                // numNewPointerArgs
        1,                // numNewValueArgs
        0,                // numNewExecInfos
        0,                // newWorkDim
        nullptr,          // pNewMemObjArgList
        &new_output_desc, // pNewPointerArgList
        &new_input_desc,  // pNewValueArgList
        nullptr,          // pNewExecInfoList
        nullptr,          // pNewGlobalWorkOffset
        &new_global_size, // pNewGlobalWorkSize
        nullptr,          // pNewLocalWorkSize
    };

    // Update kernel and enqueue command-buffer again
    ASSERT_SUCCESS(
        urCommandBufferUpdateKernelLaunchExp(command_handle, &update_desc));
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    // Verify that update occurred correctly
    Validate((uint32_t *)new_shared_ptr, new_global_size, new_val);
}

// Test updating the kernel execution info
TEST_P(USMFillCommandTest, UpdateExecInfo) {
    // Run command-buffer prior to update an verify output
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    Validate((uint32_t *)shared_ptr, global_size, val);

    ur_exp_command_buffer_update_exec_info_desc_t new_exec_info_descs[3];

    // Update direct access flag
    bool indirect_access = false;
    new_exec_info_descs[0] = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_EXEC_INFO_DESC, // stype
        nullptr,                                                    // pNext
        UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS,                    // propName
        sizeof(indirect_access),                                    // propSize
        nullptr,          // pProperties
        &indirect_access, // pPropValue
    };

    // Update cache config
    ur_kernel_cache_config_t cache_config = UR_KERNEL_CACHE_CONFIG_DEFAULT;
    new_exec_info_descs[1] = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_EXEC_INFO_DESC, // stype
        nullptr,                                                    // pNext
        UR_KERNEL_EXEC_INFO_CACHE_CONFIG,                           // propName
        sizeof(cache_config),                                       // propSize
        nullptr,       // pProperties
        &cache_config, // pPropValue
    };

    // Create a new USM allocation to set indirect access for
    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &new_shared_ptr));
    ASSERT_NE(new_shared_ptr, nullptr);
    void *pointers = {new_shared_ptr};
    new_exec_info_descs[2] = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_EXEC_INFO_DESC, // stype
        nullptr,                                                    // pNext
        UR_KERNEL_EXEC_INFO_USM_PTRS,                               // propName
        sizeof(pointers),                                           // propSize
        nullptr,   // pProperties
        &pointers, // pPropValue
    };

    ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        0,                   // numNewMemObjArgs
        0,                   // numNewPointerArgs
        0,                   // numNewValueArgs
        3,                   // numNewExecInfos
        0,                   // newWorkDim
        nullptr,             // pNewMemObjArgList
        nullptr,             // pNewPointerArgList
        nullptr,             // pNewValueArgList
        new_exec_info_descs, // pNewExecInfoList
        nullptr,             // pNewGlobalWorkOffset
        nullptr,             // pNewGlobalWorkSize
        nullptr,             // pNewLocalWorkSize
    };

    // Update kernel and enqueue command-buffer again
    ASSERT_SUCCESS(
        urCommandBufferUpdateKernelLaunchExp(command_handle, &update_desc));
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    // Verify results are correct, although exec info modifications should
    // have no effect on output
    Validate((uint32_t *)shared_ptr, global_size, val);
}

// Test updating a command-buffer with multiple USM fill kernel commands
struct USMMultipleFillCommandTest
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

        // Create a single USM allocation which will be used by all kernels
        // by accessing at pointer offsets
        ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                        allocation_size, &shared_ptr));
        ASSERT_NE(shared_ptr, nullptr);
        std::memset(shared_ptr, 0, allocation_size);

        // Append multiple kernel commands to command-buffer
        for (size_t k = 0; k < num_kernels; k++) {
            // Calculate offset into output allocation, and set as
            // kernel output.
            void *offset_ptr = (uint32_t *)shared_ptr + (k * elements);
            ASSERT_SUCCESS(
                urKernelSetArgPointer(kernel, 0, nullptr, &offset_ptr));

            // Each kernel has a unique fill value
            uint32_t fill_val = val + k;
            ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(fill_val),
                                               nullptr, &fill_val));

            // Append kernel and store returned handle
            ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
                updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
                &elements, &local_size, 0, nullptr, nullptr,
                &command_handles[k]));
            ASSERT_NE(command_handles[k], nullptr);
        }

        ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
    }

    void Validate(uint32_t *pointer, size_t length, uint32_t val) {
        for (size_t i = 0; i < length; i++) {
            ASSERT_EQ(pointer[i], val);
        }
    }

    void TearDown() override {
        if (shared_ptr) {
            EXPECT_SUCCESS(urUSMFree(context, shared_ptr));
        }

        if (new_shared_ptr) {
            EXPECT_SUCCESS(urUSMFree(context, new_shared_ptr));
        }

        UUR_RETURN_ON_FATAL_FAILURE(
            urUpdatableCommandBufferExpExecutionTest::TearDown());
    }

    static constexpr uint32_t val = 42;
    static constexpr size_t local_size = 4;
    static constexpr size_t global_size = 64;
    static constexpr size_t global_offset = 0;
    static constexpr size_t n_dimensions = 1;
    static constexpr size_t allocation_size = sizeof(val) * global_size;
    static constexpr size_t num_kernels = 8;
    static constexpr size_t elements = global_size / num_kernels;

    void *shared_ptr = nullptr;
    void *new_shared_ptr = nullptr;
    std::array<ur_exp_command_buffer_command_handle_t, num_kernels>
        command_handles;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(USMMultipleFillCommandTest);

// Test updating all the kernels commands in the command-buffer
TEST_P(USMMultipleFillCommandTest, UpdateAllKernels) {
    // Run command-buffer prior to update an verify output
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    uint32_t *output = (uint32_t *)shared_ptr;
    for (size_t i = 0; i < global_size; i++) {
        const uint32_t expected = val + (i / elements);
        ASSERT_EQ(expected, output[i]);
    }

    // Create a new USM allocation to update kernel outputs to
    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &new_shared_ptr));
    ASSERT_NE(new_shared_ptr, nullptr);
    std::memset(new_shared_ptr, 0, allocation_size);

    // Update each kernel in the command-buffer.
    uint32_t new_val = 33;
    for (size_t k = 0; k < num_kernels; k++) {
        // Update output pointer to an offset into new USM allocation
        void *offset_ptr = (uint32_t *)new_shared_ptr + (k * elements);
        ur_exp_command_buffer_update_pointer_arg_desc_t new_output_desc = {
            UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
            nullptr,     // pNext
            0,           // argIndex
            nullptr,     // pProperties
            &offset_ptr, // pArgValue
        };

        // Update fill value
        uint32_t new_fill_val = new_val + k;
        ur_exp_command_buffer_update_value_arg_desc_t new_input_desc = {
            UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
            nullptr,                                                    // pNext
            1,             // argIndex
            sizeof(int),   // argSize
            nullptr,       // pProperties
            &new_fill_val, // hArgValue
        };

        ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
            UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
            nullptr,          // pNext
            0,                // numNewMemObjArgs
            1,                // numNewPointerArgs
            1,                // numNewValueArgs
            0,                // numNewExecInfos
            0,                // newWorkDim
            nullptr,          // pNewMemObjArgList
            &new_output_desc, // pNewPointerArgList
            &new_input_desc,  // pNewValueArgList
            nullptr,          // pNewExecInfoList
            nullptr,          // pNewGlobalWorkOffset
            nullptr,          // pNewGlobalWorkSize
            nullptr,          // pNewLocalWorkSize
        };

        ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(command_handles[k],
                                                            &update_desc));
    }

    // Update kernel and enqueue command-buffer again
    ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    // Verify that update occurred correctly
    uint32_t *updated_output = (uint32_t *)new_shared_ptr;
    for (size_t i = 0; i < global_size; i++) {
        uint32_t expected = new_val + (i / elements);
        ASSERT_EQ(expected, updated_output[i]) << i;
    }
}
