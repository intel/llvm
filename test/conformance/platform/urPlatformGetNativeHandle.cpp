// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urPlatformGetNativeHandleTest = uur::urPlatformTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urPlatformGetNativeHandleTest);

TEST_P(urPlatformGetNativeHandleTest, Success) {
    ur_native_handle_t native_handle = 0;
    if (auto error = urPlatformGetNativeHandle(platform, &native_handle)) {
        ASSERT_EQ(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, error);
    }
}

TEST_P(urPlatformGetNativeHandleTest, InvalidNullHandlePlatform) {
    ur_native_handle_t native_handle = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urPlatformGetNativeHandle(nullptr, &native_handle));
}

TEST_P(urPlatformGetNativeHandleTest, InvalidNullPointerNativePlatform) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urPlatformGetNativeHandle(platform, nullptr));
}
