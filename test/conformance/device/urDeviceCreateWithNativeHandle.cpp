// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urDeviceCreateWithNativeHandleTest = uur::urAllDevicesTest;

TEST_F(urDeviceCreateWithNativeHandleTest, Success) {
    for (auto device : devices) {
        ur_native_handle_t native_handle = nullptr;
        if (urDeviceGetNativeHandle(device, &native_handle)) {
            continue;
        }

        // We cannot assume anything about a native_handle, not even if it's
        // `nullptr` since this could be a valid representation within a backend.
        // We can however convert the native_handle back into a unified-runtime handle
        // and perform some query on it to verify that it works.
        ur_device_handle_t dev = nullptr;
        ASSERT_SUCCESS(urDeviceCreateWithNativeHandle(native_handle, platform,
                                                      nullptr, &dev));
        ASSERT_NE(dev, nullptr);

        uint32_t dev_id = 0;
        ASSERT_SUCCESS(urDeviceGetInfo(dev, UR_DEVICE_INFO_TYPE,
                                       sizeof(uint32_t), &dev_id, nullptr));
    }
}
