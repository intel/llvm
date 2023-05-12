// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urProgramGetNativeHandleTest = uur::urProgramTest;
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramGetNativeHandleTest);

TEST_P(urProgramGetNativeHandleTest, Success) {
    ur_native_handle_t native_program_handle = nullptr;
    ASSERT_SUCCESS(urProgramGetNativeHandle(program, &native_program_handle));

    ur_program_handle_t native_program = nullptr;
    ASSERT_SUCCESS(urProgramCreateWithNativeHandle(native_program_handle,
                                                   context, &native_program));

    uint32_t ref_count = 0;
    ASSERT_SUCCESS(urProgramGetInfo(native_program,
                                    UR_PROGRAM_INFO_REFERENCE_COUNT,
                                    sizeof(ref_count), &ref_count, nullptr));
    ASSERT_NE(ref_count, 0);

    ASSERT_SUCCESS(urProgramRelease(native_program));
}

TEST_P(urProgramGetNativeHandleTest, InvalidNullHandleProgram) {
    ur_native_handle_t native_program_handle = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramGetNativeHandle(nullptr, &native_program_handle));
}

TEST_P(urProgramGetNativeHandleTest, InvalidNullPointerNativeProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramGetNativeHandle(program, nullptr));
}
