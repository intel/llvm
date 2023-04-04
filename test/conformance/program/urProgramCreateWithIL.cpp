// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
//
#include <uur/fixtures.h>

using urProgramCreateWithILTest = uur::urProgramILBinaryTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urProgramCreateWithILTest);

TEST_P(urProgramCreateWithILTest, Success) {
    ur_program_handle_t program = nullptr;
    ASSERT_SUCCESS(urProgramCreateWithIL(context, il_binary->data(),
                                         il_binary->size(), nullptr, &program));
    ASSERT_NE(nullptr, program);
    ASSERT_SUCCESS(urProgramRelease(program));
}

TEST_P(urProgramCreateWithILTest, InvalidNullHandle) {
    ur_program_handle_t program = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramCreateWithIL(nullptr, il_binary->data(),
                                           il_binary->size(), nullptr,
                                           &program));
}

TEST_P(urProgramCreateWithILTest, InvalidNullPointerSource) {
    ur_program_handle_t program = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramCreateWithIL(context, nullptr, il_binary->size(),
                                           nullptr, &program));
}

TEST_P(urProgramCreateWithILTest, InvalidSizeLength) {
    ur_program_handle_t program = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urProgramCreateWithIL(context, il_binary->data(), 0,
                                           nullptr, &program));
}

TEST_P(urProgramCreateWithILTest, InvalidNullPointerProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramCreateWithIL(context, il_binary->data(),
                                           il_binary->size(), nullptr,
                                           nullptr));
}
