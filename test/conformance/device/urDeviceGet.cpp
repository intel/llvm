// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urDeviceGetTest = uur::urPlatformTest;

TEST_F(urDeviceGetTest, Success) {
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_NE(count, 0);
    std::vector<ur_device_handle_t> devices(count);
    ASSERT_SUCCESS(urDeviceGet(platform, UR_DEVICE_TYPE_ALL, count,
                               devices.data(), nullptr));
    for (auto device : devices) {
        ASSERT_NE(nullptr, device);
    }
}

TEST_F(urDeviceGetTest, SuccessSubsetOfDevices) {
    uint32_t count;
    ASSERT_SUCCESS(
        urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    if (count < 2) {
        GTEST_SKIP();
    }
    std::vector<ur_device_handle_t> devices(count - 1);
    ASSERT_SUCCESS(urDeviceGet(platform, UR_DEVICE_TYPE_ALL, count - 1,
                               devices.data(), nullptr));
    for (auto device : devices) {
        ASSERT_NE(nullptr, device);
    }
}

TEST_F(urDeviceGetTest, InvalidNullHandlePlatform) {
    uint32_t count;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urDeviceGet(nullptr, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
}

TEST_F(urDeviceGetTest, InvalidEnumerationDevicesType) {
    uint32_t count;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_ENUMERATION,
        urDeviceGet(platform, UR_DEVICE_TYPE_FORCE_UINT32, 0, nullptr, &count));
}

TEST_F(urDeviceGetTest, InvalidSizeNumEntries) {
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_NE(count, 0);
    std::vector<ur_device_handle_t> devices(count);
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_SIZE,
        urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, devices.data(), nullptr));
}

TEST_F(urDeviceGetTest, InvalidNullPointerDevices) {
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_NE(count, 0);
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urDeviceGet(platform, UR_DEVICE_TYPE_ALL, count, nullptr, nullptr));
}
