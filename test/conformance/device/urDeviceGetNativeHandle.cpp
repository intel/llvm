// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception
#include <uur/fixtures.h>

using urDeviceGetNativeHandleTest = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urDeviceGetNativeHandleTest);

TEST_P(urDeviceGetNativeHandleTest, Success) {
  ur_native_handle_t native_handle = 0;
  if (auto error = urDeviceGetNativeHandle(device, &native_handle)) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, error);
  }
}

TEST_P(urDeviceGetNativeHandleTest, InvalidNullHandleDevice) {
  ur_native_handle_t native_handle = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urDeviceGetNativeHandle(nullptr, &native_handle));
}

TEST_P(urDeviceGetNativeHandleTest, InvalidNullPointerNativeDevice) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urDeviceGetNativeHandle(device, nullptr));
}
