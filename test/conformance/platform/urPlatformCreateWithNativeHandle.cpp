// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urPlatformCreateWithNativeHandleTest = uur::platform::urPlatformTest;

TEST_F(urPlatformCreateWithNativeHandleTest, InvalidNullPointerPlatform) {
    ur_native_handle_t native_handle = nullptr;
    ASSERT_SUCCESS(urPlatformGetNativeHandle(platform, &native_handle));
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urPlatformCreateWithNativeHandle(native_handle, nullptr, nullptr));
}
