// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urPlatformCreateWithNativeHandleTest : uur::urPlatformTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urPlatformTest::SetUp());
        ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_ADAPTER,
                                         sizeof(ur_adapter_handle_t), &adapter,
                                         nullptr));
    }
    ur_adapter_handle_t adapter = nullptr;
};
UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urPlatformCreateWithNativeHandleTest);

TEST_P(urPlatformCreateWithNativeHandleTest, Success) {
    ur_native_handle_t native_handle = 0;

    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urPlatformGetNativeHandle(platform, &native_handle));

    // We cannot assume anything about a native_handle, not even if it's
    // `nullptr` since this could be a valid representation within a backend.
    // We can however convert the native_handle back into a unified-runtime
    // handle and perform some query on it to verify that it works.
    ur_platform_handle_t plat = nullptr;
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urPlatformCreateWithNativeHandle(
        native_handle, adapter, nullptr, &plat));
    ASSERT_NE(plat, nullptr);

    std::string input_platform_name = uur::GetPlatformName(platform);
    std::string created_platform_name = uur::GetPlatformName(plat);
    ASSERT_EQ(input_platform_name, created_platform_name);
}

TEST_P(urPlatformCreateWithNativeHandleTest, SuccessWithOwnedNativeHandle) {
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
        native_handle, adapter, &props, &plat));
    ASSERT_NE(plat, nullptr);

    std::string input_platform_name = uur::GetPlatformName(platform);
    std::string created_platform_name = uur::GetPlatformName(plat);
    ASSERT_EQ(input_platform_name, created_platform_name);
}

TEST_P(urPlatformCreateWithNativeHandleTest, SuccessWithUnOwnedNativeHandle) {
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
        native_handle, adapter, &props, &plat));
    ASSERT_NE(plat, nullptr);

    std::string input_platform_name = uur::GetPlatformName(platform);
    std::string created_platform_name = uur::GetPlatformName(plat);
    ASSERT_EQ(input_platform_name, created_platform_name);
}

TEST_P(urPlatformCreateWithNativeHandleTest, InvalidNullPointerPlatform) {
    ur_native_handle_t native_handle = 0;
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urPlatformGetNativeHandle(platform, &native_handle));
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urPlatformCreateWithNativeHandle(native_handle, adapter,
                                                      nullptr, nullptr));
}
