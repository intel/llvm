// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urPlatformCreateWithNativeHandleTest = uur::platform::urPlatformTest;

TEST_F(urPlatformCreateWithNativeHandleTest, Success) {
    for (auto platform : platforms) {
        ur_native_handle_t native_handle = 0;

        UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
            urPlatformGetNativeHandle(platform, &native_handle));

        // We cannot assume anything about a native_handle, not even if it's
        // `nullptr` since this could be a valid representation within a backend.
        // We can however convert the native_handle back into a unified-runtime
        // handle and perform some query on it to verify that it works.
        ur_platform_handle_t plat = nullptr;
        UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urPlatformCreateWithNativeHandle(
            native_handle, adapters[0], nullptr, &plat));
        ASSERT_NE(plat, nullptr);

        std::string input_platform_name = uur::GetPlatformName(platform);
        std::string created_platform_name = uur::GetPlatformName(plat);
        ASSERT_EQ(input_platform_name, created_platform_name);
    }
}

TEST_F(urPlatformCreateWithNativeHandleTest, SuccessWithOwnedNativeHandle) {
    for (auto platform : platforms) {
        ur_native_handle_t native_handle = 0;

        UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
            urPlatformGetNativeHandle(platform, &native_handle));

        // We cannot assume anything about a native_handle, not even if it's
        // `nullptr` since this could be a valid representation within a backend.
        // We can however convert the native_handle back into a unified-runtime
        // handle and perform some query on it to verify that it works.
        ur_platform_native_properties_t props = {
            UR_STRUCTURE_TYPE_PLATFORM_NATIVE_PROPERTIES, nullptr, true};
        ur_platform_handle_t plat = nullptr;
        UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urPlatformCreateWithNativeHandle(
            native_handle, adapters[0], &props, &plat));
        ASSERT_NE(plat, nullptr);

        std::string input_platform_name = uur::GetPlatformName(platform);
        std::string created_platform_name = uur::GetPlatformName(plat);
        ASSERT_EQ(input_platform_name, created_platform_name);
    }
}

TEST_F(urPlatformCreateWithNativeHandleTest, SuccessWithUnOwnedNativeHandle) {
    for (auto platform : platforms) {
        ur_native_handle_t native_handle = 0;

        UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
            urPlatformGetNativeHandle(platform, &native_handle));

        // We cannot assume anything about a native_handle, not even if it's
        // `nullptr` since this could be a valid representation within a backend.
        // We can however convert the native_handle back into a unified-runtime
        // handle and perform some query on it to verify that it works.
        ur_platform_native_properties_t props = {
            UR_STRUCTURE_TYPE_PLATFORM_NATIVE_PROPERTIES, nullptr, false};
        ur_platform_handle_t plat = nullptr;
        UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urPlatformCreateWithNativeHandle(
            native_handle, adapters[0], &props, &plat));
        ASSERT_NE(plat, nullptr);

        std::string input_platform_name = uur::GetPlatformName(platform);
        std::string created_platform_name = uur::GetPlatformName(plat);
        ASSERT_EQ(input_platform_name, created_platform_name);
    }
}

TEST_F(urPlatformCreateWithNativeHandleTest, InvalidNullPointerPlatform) {
    for (auto platform : platforms) {
        ur_native_handle_t native_handle = 0;
        UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
            urPlatformGetNativeHandle(platform, &native_handle));
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                         urPlatformCreateWithNativeHandle(
                             native_handle, adapters[0], nullptr, nullptr));
    }
}
