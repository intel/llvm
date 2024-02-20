// Copyright (C) 2022-2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_COMMAND_BUFFER_FIXTURES_H_INCLUDED
#define UR_CONFORMANCE_COMMAND_BUFFER_FIXTURES_H_INCLUDED

#include <uur/fixtures.h>

namespace uur {
namespace command_buffer {

struct urCommandBufferExpTest : uur::urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urContextTest::SetUp());

        size_t returned_size;
        ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_EXTENSIONS, 0,
                                       nullptr, &returned_size));

        std::unique_ptr<char[]> returned_extensions(new char[returned_size]);

        ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_EXTENSIONS,
                                       returned_size, returned_extensions.get(),
                                       nullptr));

        std::string_view extensions_string(returned_extensions.get());
        bool command_buffer_support =
            extensions_string.find(UR_COMMAND_BUFFER_EXTENSION_STRING_EXP) !=
            std::string::npos;

        if (!command_buffer_support) {
            GTEST_SKIP() << "EXP command-buffer feature is not supported.";
        }

        ASSERT_SUCCESS(urDeviceGetInfo(
            device, UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_SUPPORT_EXP,
            sizeof(ur_bool_t), &updatable_command_buffer_support, nullptr));

        // Create a command-buffer
        ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, nullptr,
                                                &cmd_buf_handle));
        ASSERT_NE(cmd_buf_handle, nullptr);
    }

    void TearDown() override {
        if (cmd_buf_handle) {
            EXPECT_SUCCESS(urCommandBufferReleaseExp(cmd_buf_handle));
        }
        UUR_RETURN_ON_FATAL_FAILURE(uur::urContextTest::TearDown());
    }

    ur_exp_command_buffer_handle_t cmd_buf_handle = nullptr;
    ur_bool_t updatable_command_buffer_support = false;
};

struct urCommandBufferExpExecutionTest : uur::urKernelExecutionTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTest::SetUp());

        size_t returned_size;
        ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_EXTENSIONS, 0,
                                       nullptr, &returned_size));

        std::unique_ptr<char[]> returned_extensions(new char[returned_size]);

        ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_EXTENSIONS,
                                       returned_size, returned_extensions.get(),
                                       nullptr));

        std::string_view extensions_string(returned_extensions.get());
        bool command_buffer_support =
            extensions_string.find(UR_COMMAND_BUFFER_EXTENSION_STRING_EXP) !=
            std::string::npos;

        if (!command_buffer_support) {
            GTEST_SKIP() << "EXP command-buffer feature is not supported.";
        }

        ASSERT_SUCCESS(urDeviceGetInfo(
            device, UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_SUPPORT_EXP,
            sizeof(ur_bool_t), &updatable_command_buffer_support, nullptr));

        // Create a command-buffer
        ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, nullptr,
                                                &cmd_buf_handle));
        ASSERT_NE(cmd_buf_handle, nullptr);
    }

    void TearDown() override {
        if (cmd_buf_handle) {
            EXPECT_SUCCESS(urCommandBufferReleaseExp(cmd_buf_handle));
        }
        UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTest::TearDown());
    }

    ur_exp_command_buffer_handle_t cmd_buf_handle = nullptr;
    ur_bool_t updatable_command_buffer_support = false;
};

struct urUpdatableCommandBufferExpExecutionTest
    : urCommandBufferExpExecutionTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urCommandBufferExpExecutionTest ::SetUp());

        if (!updatable_command_buffer_support) {
            GTEST_SKIP() << "Updating EXP command-buffers is not supported.";
        }

        // Create a command-buffer with update enabled.
        ur_exp_command_buffer_desc_t desc{
            UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC, nullptr, true};

        ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, &desc,
                                                &updatable_cmd_buf_handle));
        ASSERT_NE(updatable_cmd_buf_handle, nullptr);
    }

    void TearDown() override {
        if (updatable_cmd_buf_handle) {
            EXPECT_SUCCESS(urCommandBufferReleaseExp(updatable_cmd_buf_handle));
        }
        UUR_RETURN_ON_FATAL_FAILURE(
            urCommandBufferExpExecutionTest::TearDown());
    }

    ur_exp_command_buffer_handle_t updatable_cmd_buf_handle = nullptr;
};

struct urCommandBufferCommandExpTest
    : urUpdatableCommandBufferExpExecutionTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            urUpdatableCommandBufferExpExecutionTest::SetUp());

        // Append 2 kernel commands to command-buffer and close command-buffer
        ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
            updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
            &global_size, &local_size, 0, nullptr, nullptr, &command_handle));
        ASSERT_NE(command_handle, nullptr);

        ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
            updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
            &global_size, &local_size, 0, nullptr, nullptr, &command_handle_2));
        ASSERT_NE(command_handle_2, nullptr);

        ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
    }

    void TearDown() override {
        if (command_handle) {
            EXPECT_SUCCESS(urCommandBufferReleaseCommandExp(command_handle));
        }

        if (command_handle_2) {
            EXPECT_SUCCESS(urCommandBufferReleaseCommandExp(command_handle_2));
        }

        UUR_RETURN_ON_FATAL_FAILURE(
            urUpdatableCommandBufferExpExecutionTest::TearDown());
    }

    static constexpr size_t local_size = 4;
    static constexpr size_t global_size = 32;
    static constexpr size_t global_offset = 0;
    static constexpr size_t n_dimensions = 1;

    ur_exp_command_buffer_command_handle_t command_handle = nullptr;
    ur_exp_command_buffer_command_handle_t command_handle_2 = nullptr;
};
} // namespace command_buffer
} // namespace uur

#endif // UR_CONFORMANCE_EVENT_COMMAND_BUFFER_H_INCLUDED
