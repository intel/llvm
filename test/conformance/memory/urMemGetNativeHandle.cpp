// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urMemGetNativeHandleTest = uur::urMemBufferTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemGetNativeHandleTest);

TEST_P(urMemGetNativeHandleTest, Success) {
    ur_native_handle_t hNativeMem = nullptr;
    if (auto error = urMemGetNativeHandle(buffer, device, &hNativeMem)) {
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, error);
    }
}

TEST_P(urMemGetNativeHandleTest, InvalidNullHandleMem) {
    ur_native_handle_t phNativeMem;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urMemGetNativeHandle(nullptr, device, &phNativeMem));
}

TEST_P(urMemGetNativeHandleTest, InvalidNullHandleDevice) {
    ur_native_handle_t phNativeMem;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urMemGetNativeHandle(buffer, nullptr, &phNativeMem));
}

TEST_P(urMemGetNativeHandleTest, InvalidNullPointerNativeMem) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urMemGetNativeHandle(buffer, device, nullptr));
}
