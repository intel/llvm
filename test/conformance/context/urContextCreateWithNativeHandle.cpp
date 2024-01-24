// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urContextCreateWithNativeHandleTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextCreateWithNativeHandleTest);

TEST_P(urContextCreateWithNativeHandleTest, Success) {
    ur_native_handle_t native_context = nullptr;
    if (urContextGetNativeHandle(context, &native_context)) {
        GTEST_SKIP();
    }

    // We cannot assume anything about a native_handle, not even if it's
    // `nullptr` since this could be a valid representation within a backend.
    // We can however convert the native_handle back into a unified-runtime handle
    // and perform some query on it to verify that it works.
    ur_context_handle_t ctx = nullptr;
    ur_context_native_properties_t props{};
    ASSERT_SUCCESS(urContextCreateWithNativeHandle(native_context, 1, &device,
                                                   &props, &ctx));
    ASSERT_NE(ctx, nullptr);

    uint32_t n_devices = 0;
    ASSERT_SUCCESS(urContextGetInfo(ctx, UR_CONTEXT_INFO_NUM_DEVICES,
                                    sizeof(uint32_t), &n_devices, nullptr));

    ASSERT_SUCCESS(urContextRelease(ctx));
}
