// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urContextCreateWithNativeHandleTest = uur::urDeviceTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextCreateWithNativeHandleTest);

// TODO - Should not use nullptr for the native handle - #178
TEST_P(urContextCreateWithNativeHandleTest, DISABLED_Success) {
    ur_native_handle_t native_handle = nullptr;
    ur_context_handle_t context = nullptr;
    ASSERT_SUCCESS(urContextCreateWithNativeHandle(native_handle, &context));
    ASSERT_NE(context, nullptr);
}
