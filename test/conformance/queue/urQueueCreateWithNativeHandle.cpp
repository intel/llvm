// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urQueueCreateWithNativeHandleTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueCreateWithNativeHandleTest);

TEST_P(urQueueCreateWithNativeHandleTest, Success) {
    ur_native_handle_t native_handle = 0;

    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urQueueGetNativeHandle(queue, nullptr, &native_handle));

    // We cannot assume anything about a native_handle, not even if it's
    // `nullptr` since this could be a valid representation within a backend.
    // We can however convert the native_handle back into a unified-runtime handle
    // and perform some query on it to verify that it works.
    ur_queue_handle_t q = nullptr;
    ur_queue_native_properties_t properties{};
    ASSERT_SUCCESS(urQueueCreateWithNativeHandle(native_handle, context, device,
                                                 &properties, &q));
    ASSERT_NE(q, nullptr);

    ur_context_handle_t q_context = nullptr;
    ASSERT_SUCCESS(urQueueGetInfo(q, UR_QUEUE_INFO_CONTEXT, sizeof(q_context),
                                  &q_context, nullptr));
    ASSERT_EQ(q_context, context);
    ASSERT_SUCCESS(urQueueRelease(q));
}

TEST_P(urQueueCreateWithNativeHandleTest, InvalidNullHandle) {
    ur_native_handle_t native_handle = 0;
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urQueueGetNativeHandle(queue, nullptr, &native_handle));

    ur_queue_handle_t q = nullptr;
    ASSERT_EQ(urQueueCreateWithNativeHandle(native_handle, nullptr, device,
                                            nullptr, &q),
              UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urQueueCreateWithNativeHandleTest, InvalidNullPointer) {
    ur_native_handle_t native_handle = 0;
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urQueueGetNativeHandle(queue, nullptr, &native_handle));

    ASSERT_EQ(urQueueCreateWithNativeHandle(native_handle, context, device,
                                            nullptr, nullptr),
              UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
