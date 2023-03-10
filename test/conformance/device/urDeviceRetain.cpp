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

        ASSERT_LT(prevRefCount, refCount);

        EXPECT_SUCCESS(urDeviceRelease(device));
    }
}

TEST_F(urDeviceRetainTest, InvalidNullHandle) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urDeviceRetain(nullptr));
}
