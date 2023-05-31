// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urDeviceCreateWithNativeHandleTest = uur::urPlatformTest;

TEST_F(urDeviceCreateWithNativeHandleTest, InvalidNullHandleNativeDevice) {
    ur_device_handle_t device = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urDeviceCreateWithNativeHandle(nullptr, platform, nullptr, &device));
}
