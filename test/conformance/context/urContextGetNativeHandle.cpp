// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urContextGetNativeHandleTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextGetNativeHandleTest);

TEST_P(urContextGetNativeHandleTest, Success) {
  ur_native_handle_t native_context = 0;
  if (auto error = urContextGetNativeHandle(context, &native_context)) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, error);
  }
}

TEST_P(urContextGetNativeHandleTest, InvalidNullHandleContext) {
  ur_native_handle_t native_handle = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urContextGetNativeHandle(nullptr, &native_handle));
}

TEST_P(urContextGetNativeHandleTest, InvalidNullPointerNativeHandle) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urContextGetNativeHandle(context, nullptr));
}
