// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    }
}

TEST_F(urDeviceRetainTest, SuccessSubdevices) {
    for (auto device : devices) {
        if (!uur::hasDevicePartitionSupport(device,
                                            UR_DEVICE_PARTITION_EQUALLY)) {
            ::testing::Message() << "Device: \'" << device
                                 << "\' does not support partitioning equally.";
            continue;
        }

        ur_device_partition_property_t prop = uur::makePartitionEquallyDesc(1);
        ur_device_partition_properties_t properties{
            UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES,
            nullptr,
            &prop,
            1,
        };

        ur_device_handle_t sub_device = nullptr;
        ASSERT_SUCCESS(
            urDevicePartition(device, &properties, 1, &sub_device, nullptr));

        uint32_t prevRefCount = 0;
        ASSERT_SUCCESS(uur::GetObjectReferenceCount(sub_device, prevRefCount));

        ASSERT_SUCCESS(urDeviceRetain(sub_device));

        uint32_t refCount = 0;
        ASSERT_SUCCESS(uur::GetObjectReferenceCount(sub_device, refCount));

        ASSERT_LT(prevRefCount, refCount);

        EXPECT_SUCCESS(urDeviceRelease(sub_device));
        EXPECT_SUCCESS(urDeviceRelease(sub_device));
    }
}

TEST_F(urDeviceRetainTest, InvalidNullHandle) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urDeviceRetain(nullptr));
}
