// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urContextCreateWithNativeHandleTest = uur::urDeviceTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextCreateWithNativeHandleTest);

TEST_P(urContextCreateWithNativeHandleTest, InvalidNullHandleNativeHandle) {
    ur_context_handle_t context = nullptr;
    ur_context_native_desc_t desc{};
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urContextCreateWithNativeHandle(nullptr, 0u, nullptr, &desc, &context));
}
