// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urQueueGetNativeHandleTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueGetNativeHandleTest);

TEST_P(urQueueGetNativeHandleTest, Success) {
    ur_native_handle_t native_handle = nullptr;
    ASSERT_SUCCESS(urQueueGetNativeHandle(queue, &native_handle));
    ASSERT_NE(native_handle, nullptr);
}

TEST_P(urQueueGetNativeHandleTest, InvalidNullHandleQueue) {
    ur_native_handle_t native_handle = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueGetNativeHandle(nullptr, &native_handle));
}

TEST_P(urQueueGetNativeHandleTest, InvalidNullPointerNativeHandle) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urQueueGetNativeHandle(queue, nullptr));
}
