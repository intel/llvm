// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urEventCreateWithNativeHandleTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventCreateWithNativeHandleTest);

TEST_P(urEventCreateWithNativeHandleTest, InvalidNullHandleNativeEvent) {
    ur_event_handle_t event = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEventCreateWithNativeHandle(nullptr, context, &event));
}
