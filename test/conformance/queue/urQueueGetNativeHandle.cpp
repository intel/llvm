// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urQueueGetNativeHandleTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueGetNativeHandleTest);

TEST_P(urQueueGetNativeHandleTest, Success) {
    ur_native_handle_t native_handle = nullptr;
    ASSERT_SUCCESS(urQueueGetNativeHandle(queue, &native_handle));

    // We cannot assume anything about a native_handle, not even if it's
    // `nullptr` since this could be a valid representation within a backend.
    // We can however convert the native_handle back into a unified-runtime handle
    // and perform some query on it to verify that it works.
    ur_queue_handle_t q = nullptr;
    ASSERT_SUCCESS(urQueueCreateWithNativeHandle(native_handle, context, &q));
    ASSERT_NE(q, nullptr);

    uint32_t q_size = 0;
    ASSERT_SUCCESS(urQueueGetInfo(q, UR_QUEUE_INFO_SIZE, sizeof(uint32_t),
                                  &q_size, nullptr));

    ASSERT_SUCCESS(urQueueRelease(q));
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
