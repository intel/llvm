// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urProgramReleaseTest = uur::urProgramTest;
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramReleaseTest);

TEST_P(urProgramReleaseTest, Success) {
    ASSERT_SUCCESS(urProgramRetain(program));
    ASSERT_SUCCESS(urProgramRelease(program));
}

TEST_P(urProgramReleaseTest, InvalidNullHandleProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramRelease(nullptr));
}
