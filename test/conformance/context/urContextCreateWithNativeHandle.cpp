// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/environment.h"
#include <uur/fixtures.h>

using urContextCreateWithNativeHandleTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextCreateWithNativeHandleTest);

TEST_P(urContextCreateWithNativeHandleTest, Success) {
    ur_native_handle_t native_context = 0;
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urContextGetNativeHandle(context, &native_context));

    // We cannot assume anything about a native_handle, not even if it's
    // `nullptr` since this could be a valid representation within a backend.
    // We can however convert the native_handle back into a unified-runtime handle
    // and perform some query on it to verify that it works.
    ur_context_handle_t ctx = nullptr;
    ur_context_native_properties_t props{};
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urContextCreateWithNativeHandle(
        native_context, adapter, 1, &device, &props, &ctx));
    ASSERT_NE(ctx, nullptr);

    uint32_t n_devices = 0;
    ASSERT_SUCCESS(urContextGetInfo(ctx, UR_CONTEXT_INFO_NUM_DEVICES,
                                    sizeof(uint32_t), &n_devices, nullptr));

    ASSERT_SUCCESS(urContextRelease(ctx));
}

TEST_P(urContextCreateWithNativeHandleTest, SuccessWithOwnedNativeHandle) {
    ur_native_handle_t native_context = 0;

    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urContextGetNativeHandle(context, &native_context));

    ur_context_handle_t ctx = nullptr;
    ur_context_native_properties_t props{
        UR_STRUCTURE_TYPE_CONTEXT_NATIVE_PROPERTIES, nullptr, true};
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urContextCreateWithNativeHandle(
        native_context, adapter, 1, &device, &props, &ctx));
    ASSERT_NE(ctx, nullptr);
}

TEST_P(urContextCreateWithNativeHandleTest, SuccessWithUnOwnedNativeHandle) {
    ur_native_handle_t native_context = 0;

    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urContextGetNativeHandle(context, &native_context));

    ur_context_handle_t ctx = nullptr;
    ur_context_native_properties_t props{
        UR_STRUCTURE_TYPE_CONTEXT_NATIVE_PROPERTIES, nullptr, false};
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urContextCreateWithNativeHandle(
        native_context, adapter, 1, &device, &props, &ctx));
    ASSERT_NE(ctx, nullptr);
}

TEST_P(urContextCreateWithNativeHandleTest, InvalidNullHandleAdapter) {
    ur_native_handle_t native_context = 0;
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urContextGetNativeHandle(context, &native_context));

    ur_context_handle_t ctx = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urContextCreateWithNativeHandle(native_context, nullptr, 1,
                                                     nullptr, nullptr, &ctx));
}

TEST_P(urContextCreateWithNativeHandleTest, InvalidNullPointerContext) {
    ur_native_handle_t native_context = 0;
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urContextGetNativeHandle(context, &native_context));

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urContextCreateWithNativeHandle(native_context, adapter, 1,
                                                     &device, nullptr,
                                                     nullptr));
}
