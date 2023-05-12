// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urProgramCompileTest = uur::urProgramTest;
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramCompileTest);

TEST_P(urProgramCompileTest, Success) {
    ASSERT_SUCCESS(urProgramCompile(context, program, nullptr));
}

TEST_P(urProgramCompileTest, InvalidNullHandleContext) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramCompile(nullptr, program, nullptr));
}

TEST_P(urProgramCompileTest, InvalidNullHandleProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramCompile(context, nullptr, nullptr));
}
