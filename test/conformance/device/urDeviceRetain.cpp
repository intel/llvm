// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urDeviceRetainTest = uur::urAllDevicesTest;

TEST_F(urDeviceRetainTest, Success) {
    for (auto device : devices) {
        const auto prevRefCount = uur::GetObjectReferenceCount(device);
        ASSERT_TRUE(prevRefCount.has_value());

        ASSERT_SUCCESS(urDeviceRetain(device));

        const auto refCount = uur::GetObjectReferenceCount(device);
        ASSERT_TRUE(refCount.has_value());

        ASSERT_LT(prevRefCount.value(), refCount.value());

        EXPECT_SUCCESS(urDeviceRelease(device));
    }
}

TEST_F(urDeviceRetainTest, InvalidNullHandle) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urDeviceRetain(nullptr));
}
