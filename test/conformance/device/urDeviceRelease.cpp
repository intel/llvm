// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

struct urDeviceReleaseTest : uur::urAllDevicesTest {};

TEST_F(urDeviceReleaseTest, Success) {
    for (auto device : devices) {
        ASSERT_SUCCESS(urDeviceRetain(device));

        const auto prevRefCount = uur::urDeviceGetReferenceCount(device);
        ASSERT_TRUE(prevRefCount.second);

        EXPECT_SUCCESS(urDeviceRelease(device));

        const auto refCount = uur::urDeviceGetReferenceCount(device);
        ASSERT_TRUE(refCount.second);

        ASSERT_GT(prevRefCount.first, refCount.first);
    }
}

TEST_F(urDeviceReleaseTest, InvalidNullHandle) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urDeviceRelease(nullptr));
}
