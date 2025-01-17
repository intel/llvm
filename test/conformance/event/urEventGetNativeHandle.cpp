// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "fixtures.h"

using urEventGetNativeHandleTest = uur::event::urEventTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventGetNativeHandleTest);

TEST_P(urEventGetNativeHandleTest, Success) {
  ur_native_handle_t native_event = 0;
  if (auto error = urEventGetNativeHandle(event, &native_event)) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, error);
  }
}

TEST_P(urEventGetNativeHandleTest, InvalidNullHandleEvent) {
  ur_native_handle_t native_event = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEventGetNativeHandle(nullptr, &native_event));
}

TEST_P(urEventGetNativeHandleTest, InvalidNullPointerNativeEvent) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEventGetNativeHandle(event, nullptr));
}
