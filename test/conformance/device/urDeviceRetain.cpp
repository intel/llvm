// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urDeviceRetainTest = uur::urAllDevicesTest;

TEST_F(urDeviceRetainTest, Success) {
    for (auto device : devices) {
        const auto prevRefCount = uur::urDeviceGetReferenceCount(device);
        ASSERT_TRUE(prevRefCount.second);

        ASSERT_SUCCESS(urDeviceRetain(device));

        const auto refCount = uur::urDeviceGetReferenceCount(device);
        ASSERT_TRUE(refCount.second);

        ASSERT_LT(prevRefCount.first, refCount.first);

        EXPECT_SUCCESS(urDeviceRelease(device));
    }
}

TEST_F(urDeviceRetainTest, InvalidNullHandle) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urDeviceRetain(nullptr));
}
