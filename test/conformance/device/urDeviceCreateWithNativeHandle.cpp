// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urDeviceCreateWithNativeHandleTest = uur::urPlatformTest;

// TODO - Should not use nullptr for the native handle - #165
TEST_F(urDeviceCreateWithNativeHandleTest, DISABLED_Success) {
  ur_native_handle_t native_handle = nullptr;
  ur_device_handle_t device_handle = nullptr;
  ASSERT_SUCCESS(
      urDeviceCreateWithNativeHandle(native_handle, platform, &device_handle));
  ASSERT_NE(device_handle, nullptr);
}
