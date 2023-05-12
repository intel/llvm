// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urProgramRetainTest = uur::urProgramTest;
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramRetainTest);

TEST_P(urProgramRetainTest, Success) {
    ASSERT_SUCCESS(urProgramRetain(program));
    EXPECT_SUCCESS(urProgramRelease(program));
}

TEST_P(urProgramRetainTest, InvalidNullHandleProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramRetain(nullptr));
}
