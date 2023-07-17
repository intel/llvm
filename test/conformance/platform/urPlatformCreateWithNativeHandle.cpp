// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urPlatformCreateWithNativeHandleTest = uur::platform::urPlatformTest;

TEST_F(urPlatformCreateWithNativeHandleTest, Success) {
    for (auto platform : platforms) {
        ur_native_handle_t native_handle = nullptr;
        if (urPlatformGetNativeHandle(platform, &native_handle)) {
            continue;
        };

        // We cannot assume anything about a native_handle, not even if it's
        // `nullptr` since this could be a valid representation within a backend.
        // We can however convert the native_handle back into a unified-runtime
        // handle and perform some query on it to verify that it works.
        ur_platform_handle_t plat = nullptr;
        ASSERT_SUCCESS(
            urPlatformCreateWithNativeHandle(native_handle, nullptr, &plat));
        ASSERT_NE(plat, nullptr);

        ur_platform_backend_t backend;
        ASSERT_SUCCESS(urPlatformGetInfo(plat, UR_PLATFORM_INFO_BACKEND,
                                         sizeof(ur_platform_backend_t),
                                         &backend, nullptr));
    }
}

TEST_F(urPlatformCreateWithNativeHandleTest, InvalidNullPointerPlatform) {
    for (auto platform : platforms) {
        ur_native_handle_t native_handle = nullptr;
        if (urPlatformGetNativeHandle(platform, &native_handle)) {
            continue;
        }
        ASSERT_EQ_RESULT(
            UR_RESULT_ERROR_INVALID_NULL_POINTER,
            urPlatformCreateWithNativeHandle(native_handle, nullptr, nullptr));
    }
}
