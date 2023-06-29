// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urContextCreateWithNativeHandleTest = uur::urDeviceTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextCreateWithNativeHandleTest);

TEST_P(urContextCreateWithNativeHandleTest, InvalidNullHandleNativeHandle) {
    ur_context_handle_t context = nullptr;
    ur_context_native_properties_t props{};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urContextCreateWithNativeHandle(nullptr, 0u, nullptr,
                                                     &props, &context));
}
