// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

struct urProgramSetSpecializationConstantsTest : uur::urProgramTest {
    uint32_t spec_value = 42;
    ur_specialization_constant_info_t info = {0, sizeof(spec_value),
                                              &spec_value};
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramSetSpecializationConstantsTest);

TEST_P(urProgramSetSpecializationConstantsTest, Success) {
    ASSERT_SUCCESS(urProgramSetSpecializationConstants(program, 1, &info));
    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
    // TODO: Run the program to verify the spec constant was set.
}

TEST_P(urProgramSetSpecializationConstantsTest, InvalidNullHandleProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramSetSpecializationConstants(nullptr, 1, &info));
}

TEST_P(urProgramSetSpecializationConstantsTest,
       InvalidNullPointerSpecConstants) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramSetSpecializationConstants(program, 1, nullptr));
}

TEST_P(urProgramSetSpecializationConstantsTest, InvalidSizeCount) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urProgramSetSpecializationConstants(program, 0, &info));
}
