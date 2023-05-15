// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urDeviceCreateWithNativeHandleTest = uur::urPlatformTest;

TEST_F(urDeviceCreateWithNativeHandleTest, InvalidNullHandleNativeDevice) {
    ur_device_handle_t device = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urDeviceCreateWithNativeHandle(nullptr, platform, nullptr, &device));
}
