// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urDeviceGetNativeHandleTest = uur::urAllDevicesTest;

TEST_F(urDeviceGetNativeHandleTest, Success) {
    for (auto device : devices) {
        ur_native_handle_t native_handle = nullptr;
        ASSERT_SUCCESS(urDeviceGetNativeHandle(device, &native_handle));

        // We cannot assume anything about a native_handle, not even if it's
        // `nullptr` since this could be a valid representation within a backend.
        // We can however convert the native_handle back into a unified-runtime handle
        // and perform some query on it to verify that it works.
        ur_device_handle_t dev = nullptr;
        ASSERT_SUCCESS(
            urDeviceCreateWithNativeHandle(native_handle, platform, &dev));
        ASSERT_NE(dev, nullptr);

        uint32_t dev_id = 0;
        ASSERT_SUCCESS(urDeviceGetInfo(dev, UR_DEVICE_INFO_DEVICE_ID,
                                       sizeof(uint32_t), &dev_id, nullptr));

        ASSERT_SUCCESS(urDeviceRelease(dev));
    }
}

TEST_F(urDeviceGetNativeHandleTest, InvalidNullHandleDevice) {
    ur_native_handle_t native_handle = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urDeviceGetNativeHandle(nullptr, &native_handle));
}

TEST_F(urDeviceGetNativeHandleTest, InvalidNullPointerNativeDevice) {
    for (auto device : devices) {
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                         urDeviceGetNativeHandle(device, nullptr));
    }
}
