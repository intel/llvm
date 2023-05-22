// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

struct urDeviceReleaseTest : uur::urAllDevicesTest {};

TEST_F(urDeviceReleaseTest, Success) {
    for (auto device : devices) {
        ASSERT_SUCCESS(urDeviceRetain(device));

        uint32_t prevRefCount = 0;
        ASSERT_SUCCESS(uur::GetObjectReferenceCount(device, prevRefCount));

        EXPECT_SUCCESS(urDeviceRelease(device));

        uint32_t refCount = 0;
        ASSERT_SUCCESS(uur::GetObjectReferenceCount(device, refCount));

        /* If device is a root level device, the device reference counts should
         * remain unchanged */
        ASSERT_EQ(prevRefCount, refCount);
    }
}

TEST_F(urDeviceReleaseTest, SuccessSubdevices) {
    for (auto device : devices) {

        ur_device_partition_property_t properties[] = {
            UR_DEVICE_PARTITION_BY_COUNTS, 1, 0};

        ur_device_handle_t sub_device;
        ASSERT_SUCCESS(
            urDevicePartition(device, properties, 1, &sub_device, nullptr));

        ASSERT_SUCCESS(urDeviceRetain(sub_device));

        uint32_t prevRefCount = 0;
        ASSERT_SUCCESS(uur::GetObjectReferenceCount(sub_device, prevRefCount));

        EXPECT_SUCCESS(urDeviceRelease(sub_device));

        uint32_t refCount = 0;
        ASSERT_SUCCESS(uur::GetObjectReferenceCount(sub_device, refCount));

        ASSERT_GT(prevRefCount, refCount);
    }
}

TEST_F(urDeviceReleaseTest, InvalidNullHandle) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urDeviceRelease(nullptr));
}
