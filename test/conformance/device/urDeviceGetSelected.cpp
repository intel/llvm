// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urDeviceGetSelectedTest = uur::urPlatformTest;

TEST_F(urDeviceGetSelectedTest, Success) {
    uint32_t count = 0;
    ur_result_t res1 =
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count);
    ASSERT_EQ_RESULT(res1, UR_RESULT_SUCCESS);
    ASSERT_NE(count, 0);
    std::vector<ur_device_handle_t> devices(count);
    ASSERT_NE(devices.size(), 0);
    ASSERT_NE(devices.data(), nullptr);
    //FAIL() << "devices.size() = " << devices.size() << " whereas count = " << count;
    ur_result_t res2 = urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, count,
                                           devices.data(), nullptr);
    ASSERT_EQ_RESULT(res2, UR_RESULT_SUCCESS);
    for (auto &device : devices) {
        ASSERT_NE(nullptr, device);
    }
}

TEST_F(urDeviceGetSelectedTest, SuccessSubsetOfDevices) {
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    if (count < 2) {
        GTEST_SKIP() << "There are fewer than two devices in total for the "
                        "platform so the subset test is impossible";
    }
    std::vector<ur_device_handle_t> devices(count - 1);
    ASSERT_SUCCESS(urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, count - 1,
                                       devices.data(), nullptr));
    for (auto device : devices) {
        ASSERT_NE(nullptr, device);
    }
}

TEST_F(urDeviceGetSelectedTest, InvalidNullHandlePlatform) {
    uint32_t count = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urDeviceGetSelected(nullptr, UR_DEVICE_TYPE_ALL,
                                         0, nullptr, &count));
}

TEST_F(urDeviceGetSelectedTest, InvalidEnumerationDevicesType) {
    uint32_t count = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urDeviceGetSelected(platform, UR_DEVICE_TYPE_FORCE_UINT32,
                                         0, nullptr, &count));
}

TEST_F(urDeviceGetSelectedTest, InvalidValueNumEntries) {
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_NE(count, 0);
    std::vector<ur_device_handle_t> devices(count);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0,
                                         devices.data(), nullptr));
}
