// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urEventCreateWithNativeHandleTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventCreateWithNativeHandleTest);

TEST_P(urEventCreateWithNativeHandleTest, InvalidNullHandleNativeEvent) {
    ur_event_handle_t event = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urEventCreateWithNativeHandle(nullptr, context, nullptr, &event));
}
