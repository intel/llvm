// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urPlatformGetNativeHandleTest = uur::platform::urPlatformsTest;

TEST_F(urPlatformGetNativeHandleTest, Success) {
    for (auto platform : platforms) {
        ur_native_handle_t native_handle = nullptr;
        if (auto error = urPlatformGetNativeHandle(platform, &native_handle)) {
            ASSERT_EQ(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, error);
        }
    }
}

TEST_F(urPlatformGetNativeHandleTest, InvalidNullHandlePlatform) {
    ur_native_handle_t native_handle = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urPlatformGetNativeHandle(nullptr, &native_handle));
}

TEST_F(urPlatformGetNativeHandleTest, InvalidNullPointerNativePlatform) {
    for (auto platform : platforms) {
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                         urPlatformGetNativeHandle(platform, nullptr));
    }
}
