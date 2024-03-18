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
