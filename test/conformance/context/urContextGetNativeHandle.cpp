// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urContextGetNativeHandleTest = uur::urContextTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextGetNativeHandleTest);

TEST_P(urContextGetNativeHandleTest, Success) {
  ur_native_handle_t native_handle = nullptr;
  ASSERT_SUCCESS(urContextGetNativeHandle(context, &native_handle));
  ASSERT_NE(native_handle, nullptr);
}
