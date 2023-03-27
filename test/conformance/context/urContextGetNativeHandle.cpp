// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urContextGetNativeHandleTest = uur::urContextTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextGetNativeHandleTest);

TEST_P(urContextGetNativeHandleTest, Success) {
    ur_native_handle_t native_context = nullptr;
    ASSERT_SUCCESS(urContextGetNativeHandle(context, &native_context));

    // We cannot assume anything about a native_handle, not even if it's
    // `nullptr` since this could be a valid representation within a backend.
    // We can however convert the native_handle back into a unified-runtime handle
    // and perform some query on it to verify that it works.
    ur_context_handle_t ctx = nullptr;
    ASSERT_SUCCESS(urContextCreateWithNativeHandle(native_context, &ctx));
    ASSERT_NE(ctx, nullptr);

    uint32_t n_devices = 0;
    ASSERT_SUCCESS(urContextGetInfo(ctx, UR_CONTEXT_INFO_NUM_DEVICES,
                                    sizeof(uint32_t), &n_devices, nullptr));

    ASSERT_SUCCESS(urContextRelease(ctx));
}

TEST_P(urContextGetNativeHandleTest, InvalidNullHandleContext) {
    ur_native_handle_t native_handle = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urContextGetNativeHandle(nullptr, &native_handle));
}

TEST_P(urContextGetNativeHandleTest, InvalidNullPointerNativeHandle) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urContextGetNativeHandle(context, nullptr));
}
