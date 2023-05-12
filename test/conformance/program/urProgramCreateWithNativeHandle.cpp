// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

struct urProgramCreateWithNativeHandleTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(
            urProgramGetNativeHandle(program, &native_program_handle));
    }

    void TearDown() override {
        if (native_program) {
            EXPECT_SUCCESS(urProgramRelease(native_program));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::TearDown());
    }

    ur_native_handle_t native_program_handle = nullptr;
    ur_program_handle_t native_program = nullptr;
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramCreateWithNativeHandleTest);

TEST_P(urProgramCreateWithNativeHandleTest, Success) {
    ASSERT_SUCCESS(urProgramCreateWithNativeHandle(native_program_handle,
                                                   context, &native_program));

    uint32_t ref_count = 0;
    ASSERT_SUCCESS(urProgramGetInfo(native_program,
                                    UR_PROGRAM_INFO_REFERENCE_COUNT,
                                    sizeof(ref_count), &ref_count, nullptr));

    ASSERT_NE(ref_count, 0);
}

TEST_P(urProgramCreateWithNativeHandleTest, InvalidNullHandleContext) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramCreateWithNativeHandle(native_program_handle,
                                                     context, &native_program));
}

TEST_P(urProgramCreateWithNativeHandleTest, InvalidNullHandleNativeProgram) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urProgramCreateWithNativeHandle(nullptr, context, &native_program));
}

TEST_P(urProgramCreateWithNativeHandleTest, InvalidNullPointerProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramCreateWithNativeHandle(native_program_handle,
                                                     context, nullptr));
}
