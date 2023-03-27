// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "fixtures.h"

using urEventGetNativeHandleTest = uur::event::urEventTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventGetNativeHandleTest);

TEST_P(urEventGetNativeHandleTest, Success) {
    ur_native_handle_t native_event = nullptr;
    ASSERT_SUCCESS(urEventGetNativeHandle(event, &native_event));

    // We cannot assume anything about a native_handle, not even if it's
    // `nullptr` since this could be a valid representation within a backend.
    // We can however convert the native_handle back into a unified-runtime handle
    // and perform some query on it to verify that it works.
    ur_event_handle_t evt = nullptr;
    ASSERT_SUCCESS(urEventCreateWithNativeHandle(native_event, context, &evt));
    ASSERT_NE(evt, nullptr);

    ur_execution_info_t exec_info;
    ASSERT_SUCCESS(urEventGetInfo(evt, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                  sizeof(ur_execution_info_t), &exec_info,
                                  nullptr));

    ASSERT_SUCCESS(urEventRelease(evt));
}

TEST_P(urEventGetNativeHandleTest, InvalidNullHandleEvent) {
    ur_native_handle_t native_event = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEventGetNativeHandle(nullptr, &native_event));
}

TEST_P(urEventGetNativeHandleTest, InvalidNullPointerNativeEvent) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEventGetNativeHandle(event, nullptr));
}
