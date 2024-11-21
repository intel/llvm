// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urProgramGetGlobalVariablePointerTest = uur::urGlobalVariableTest;

UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramGetGlobalVariablePointerTest);

TEST_P(urProgramGetGlobalVariablePointerTest, Success) {
    size_t global_variable_size = 0;
    void *global_variable_pointer;
    ASSERT_SUCCESS(urProgramGetGlobalVariablePointer(
        device, program, global_var.name.c_str(), &global_variable_size,
        &global_variable_pointer));
    ASSERT_GT(global_variable_size, 0);
    ASSERT_NE(global_variable_pointer, nullptr);
}

TEST_P(urProgramGetGlobalVariablePointerTest, InvalidNullHandleDevice) {
    void *global_variable_pointer;
    ASSERT_EQ_RESULT(urProgramGetGlobalVariablePointer(
                         nullptr, program, global_var.name.c_str(), nullptr,
                         &global_variable_pointer),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urProgramGetGlobalVariablePointerTest, InvalidNullHandleProgram) {
    void *global_variable_pointer;
    ASSERT_EQ_RESULT(urProgramGetGlobalVariablePointer(
                         device, nullptr, global_var.name.c_str(), nullptr,
                         &global_variable_pointer),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urProgramGetGlobalVariablePointerTest, InvalidVariableName) {
    void *global_variable_pointer;
    ASSERT_EQ_RESULT(
        urProgramGetGlobalVariablePointer(device, program, "foo", nullptr,
                                          &global_variable_pointer),
        UR_RESULT_ERROR_INVALID_VALUE);
}

TEST_P(urProgramGetGlobalVariablePointerTest, InvalidNullPointerVariableName) {
    void *global_variable_pointer;
    ASSERT_EQ_RESULT(
        urProgramGetGlobalVariablePointer(device, program, nullptr, nullptr,
                                          &global_variable_pointer),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urProgramGetGlobalVariablePointerTest,
       InvalidNullPointerVariablePointer) {
    size_t global_variable_size = 0;
    ASSERT_EQ_RESULT(urProgramGetGlobalVariablePointer(
                         device, program, global_var.name.c_str(),
                         &global_variable_size, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urProgramGetGlobalVariablePointerTest, InvalidProgramExecutable) {
    ur_platform_backend_t backend;
    ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(ur_platform_backend_t), &backend,
                                     nullptr));
    if (backend != UR_PLATFORM_BACKEND_LEVEL_ZERO) {
        GTEST_SKIP();
    }
    // Get IL from the compiled program.
    size_t il_size = 0;
    ASSERT_SUCCESS(
        urProgramGetInfo(program, UR_PROGRAM_INFO_IL, 0, nullptr, &il_size));
    ASSERT_GT(il_size, 0);
    std::vector<char> il(il_size);
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_IL, il_size,
                                    il.data(), nullptr));
    // Create program with IL.
    ur_program_handle_t program_with_il;
    ASSERT_SUCCESS(urProgramCreateWithIL(context, il.data(), il.size(), nullptr,
                                         &program_with_il));
    // Expect error when trying to get global variable pointer from a program which is not in exe state.
    size_t global_variable_size = 0;
    void *global_variable_pointer;
    ASSERT_EQ_RESULT(urProgramGetGlobalVariablePointer(
                         device, program_with_il, global_var.name.c_str(),
                         &global_variable_size, &global_variable_pointer),
                     UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE);
}
