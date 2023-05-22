// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urDeviceRetainTest = uur::urAllDevicesTest;

TEST_F(urDeviceRetainTest, Success) {
    for (auto device : devices) {
        uint32_t prevRefCount = 0;
        ASSERT_SUCCESS(uur::GetObjectReferenceCount(device, prevRefCount));

        ASSERT_SUCCESS(urDeviceRetain(device));

        uint32_t refCount = 0;
        ASSERT_SUCCESS(uur::GetObjectReferenceCount(device, refCount));

        /* If device is a root level device, the device reference counts should
         * remain unchanged */
        ASSERT_EQ(prevRefCount, refCount);

        EXPECT_SUCCESS(urDeviceRelease(device));
    }
}

TEST_F(urDeviceRetainTest, SuccessSubdevices) {
    for (auto device : devices) {

        ur_device_partition_property_t properties[] = {
            UR_DEVICE_PARTITION_BY_COUNTS, 1, 0};

        ur_device_handle_t sub_device;
        ASSERT_SUCCESS(
            urDevicePartition(device, properties, 1, &sub_device, nullptr));

        uint32_t prevRefCount = 0;
        ASSERT_SUCCESS(uur::GetObjectReferenceCount(sub_device, prevRefCount));

        ASSERT_SUCCESS(urDeviceRetain(sub_device));

        uint32_t refCount = 0;
        ASSERT_SUCCESS(uur::GetObjectReferenceCount(sub_device, refCount));

        ASSERT_LT(prevRefCount, refCount);

        EXPECT_SUCCESS(urDeviceRelease(sub_device));
    }
}

TEST_F(urDeviceRetainTest, InvalidNullHandle) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urDeviceRetain(nullptr));
}
