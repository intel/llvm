// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urProgramGetNativeHandleTest = uur::urProgramTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urProgramGetNativeHandleTest);

TEST_P(urProgramGetNativeHandleTest, Success) {
    ur_platform_backend_t backend;
    ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(backend), &backend, nullptr));
    // For Level Zero we have to build the program to have the native handle.
    if (backend == UR_PLATFORM_BACKEND_LEVEL_ZERO) {
        ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
    }
    ur_native_handle_t native_program_handle = 0;
    if (auto error =
            urProgramGetNativeHandle(program, &native_program_handle)) {
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, error);
    }
}

TEST_P(urProgramGetNativeHandleTest, InvalidNullHandleProgram) {
    ur_native_handle_t native_program_handle = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramGetNativeHandle(nullptr, &native_program_handle));
}

TEST_P(urProgramGetNativeHandleTest, InvalidNullPointerNativeProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramGetNativeHandle(program, nullptr));
}
