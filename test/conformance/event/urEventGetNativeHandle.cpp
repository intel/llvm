// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "fixtures.h"

using urEventGetNativeHandleTest = uur::event::urEventTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventGetNativeHandleTest);

TEST_P(urEventGetNativeHandleTest, Success) {
    ur_native_handle_t native_event = nullptr;
    ASSERT_SUCCESS(urEventGetNativeHandle(event, &native_event));
    ASSERT_NE(native_event, nullptr);
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
