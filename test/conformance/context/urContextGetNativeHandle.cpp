// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urContextGetNativeHandleTest = uur::urContextTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextGetNativeHandleTest);

TEST_P(urContextGetNativeHandleTest, Success) {
    ur_native_handle_t native_handle = nullptr;
    ASSERT_SUCCESS(urContextGetNativeHandle(context, &native_handle));
    ASSERT_NE(native_handle, nullptr);
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
