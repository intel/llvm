// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urProgramBuildTest = uur::urProgramTest;
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramBuildTest);

TEST_P(urProgramBuildTest, Success) {
    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
}

TEST_P(urProgramBuildTest, InvalidNullHandleContext) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramBuild(nullptr, program, nullptr));
}

TEST_P(urProgramBuildTest, InvalidNullHandleProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramBuild(context, nullptr, nullptr));
}
