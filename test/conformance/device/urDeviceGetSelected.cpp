// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include "helpers.h"

using urDeviceGetSelectedTest = uur::urPlatformTest;

/* adpater agnostic tests -- none assume the existence or support of any specific adapter */

TEST_F(urDeviceGetSelectedTest, Success) {
    unsetenv("ONEAPI_DEVICE_SELECTOR");
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_NE(count, 0);
    std::vector<ur_device_handle_t> devices(count);
    ASSERT_SUCCESS(urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, count,
                                       devices.data(), nullptr));
    for (auto &device : devices) {
        ASSERT_NE(nullptr, device);
    }
}

TEST_F(urDeviceGetSelectedTest, SuccessSubsetOfDevices) {
    unsetenv("ONEAPI_DEVICE_SELECTOR");
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

TEST_F(urDeviceGetSelectedTest, SuccessSelected_StarColonStar) {
    setenv("ONEAPI_DEVICE_SELECTOR", "*:*", 1);
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_NE(count, 0);
    std::vector<ur_device_handle_t> devices(count);
    ASSERT_SUCCESS(urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, count,
                                       devices.data(), nullptr));
    for (auto &device : devices) {
        ASSERT_NE(nullptr, device);
    }

    uint32_t countAll = 0;
    ASSERT_SUCCESS(
        urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &countAll));
    ASSERT_NE(countAll, 0);
    ASSERT_EQ(countAll, count);
    std::vector<ur_device_handle_t> devicesAll(countAll);
    ASSERT_SUCCESS(urDeviceGet(platform, UR_DEVICE_TYPE_ALL, countAll,
                                       devicesAll.data(), nullptr));
    for (auto &device : devicesAll) {
        ASSERT_NE(nullptr, device);
    }

    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(devices[i], devicesAll[i]);
    }
}

TEST_F(urDeviceGetSelectedTest, SuccessSelected_StarColonZero) {
    setenv("ONEAPI_DEVICE_SELECTOR", "*:0", 1);
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_NE(count, 0);
    std::vector<ur_device_handle_t> devices(count);
    ASSERT_SUCCESS(urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, count,
                                       devices.data(), nullptr));
    for (auto &device : devices) {
        ASSERT_NE(nullptr, device);
    }
}

TEST_F(urDeviceGetSelectedTest, SuccessSelected_StarColonZeroCommaStar) {
    setenv("ONEAPI_DEVICE_SELECTOR", "*:0,*", 1);
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_NE(count, 0);
    std::vector<ur_device_handle_t> devices(count);
    ASSERT_SUCCESS(urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, count,
                                       devices.data(), nullptr));
    for (auto &device : devices) {
        ASSERT_NE(nullptr, device);
    }
}

TEST_F(urDeviceGetSelectedTest, SuccessSelected_DiscardStarColonStar) {
    setenv("ONEAPI_DEVICE_SELECTOR", "!*:*", 1);
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_EQ(count, 0);
}

TEST_F(urDeviceGetSelectedTest, SuccessSelected_SelectAndDiscard) {
    setenv("ONEAPI_DEVICE_SELECTOR", "*:0;!*:*", 1);
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_EQ(count, 0);
}

TEST_F(urDeviceGetSelectedTest, SuccessSelected_SelectSomethingAndDiscardSomethingElse) {
    setenv("ONEAPI_DEVICE_SELECTOR", "*:0;!*:1", 1);
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_NE(count, 0);
    std::vector<ur_device_handle_t> devices(count);
    ASSERT_SUCCESS(urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, count,
                                       devices.data(), nullptr));
    for (auto &device : devices) {
        ASSERT_NE(nullptr, device);
    }
}

TEST_F(urDeviceGetSelectedTest, InvalidNullHandlePlatform) {
    unsetenv("ONEAPI_DEVICE_SELECTOR");
    uint32_t count = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urDeviceGetSelected(nullptr, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
}

TEST_F(urDeviceGetSelectedTest, InvalidEnumerationDevicesType) {
    unsetenv("ONEAPI_DEVICE_SELECTOR");
    uint32_t count = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urDeviceGetSelected(platform, UR_DEVICE_TYPE_FORCE_UINT32,
                                         0, nullptr, &count));
}

TEST_F(urDeviceGetSelectedTest, InvalidValueNumEntries) {
    unsetenv("ONEAPI_DEVICE_SELECTOR");
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_NE(count, 0);
    std::vector<ur_device_handle_t> devices(count);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0,
                                         devices.data(), nullptr));
}

TEST_F(urDeviceGetSelectedTest, InvalidMissingBackend) {
    setenv("ONEAPI_DEVICE_SELECTOR", ":garbage", 1);
    uint32_t count = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_UNKNOWN,
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_EQ(count, 0);
}

TEST_F(urDeviceGetSelectedTest, InvalidGarbageBackendString) {
    setenv("ONEAPI_DEVICE_SELECTOR", "garbage:0", 1);
    uint32_t count = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_VALUE,
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_EQ(count, 0);
}

TEST_F(urDeviceGetSelectedTest, InvalidMissingFilterStrings) {
    setenv("ONEAPI_DEVICE_SELECTOR", "*", 1);
    uint32_t count = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_EQ(count, 0);
    setenv("ONEAPI_DEVICE_SELECTOR", "*:", 1);
    uint32_t count2 = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_VALUE,
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count2));
    ASSERT_EQ(count2, 0);
}

TEST_F(urDeviceGetSelectedTest, InvalidMissingFilterString) {
    setenv("ONEAPI_DEVICE_SELECTOR", "*:0,,2", 1);
    uint32_t count = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_UNKNOWN,
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_EQ(count, 0);
}

TEST_F(urDeviceGetSelectedTest, InvalidTooManyDotsInFilterString) {
    setenv("ONEAPI_DEVICE_SELECTOR", "*:0.1.2.3", 1);
    uint32_t count = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_VALUE,
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_EQ(count, 0);
}

TEST_F(urDeviceGetSelectedTest, InvalidBadWildardInFilterString) {
    setenv("ONEAPI_DEVICE_SELECTOR", "*:*.", 1);
    uint32_t count = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_VALUE,
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_EQ(count, 0);
    setenv("ONEAPI_DEVICE_SELECTOR", "*:*.0", 1);
    uint32_t count2 = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_VALUE,
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count2));
    ASSERT_EQ(count2, 0);
}

TEST_F(urDeviceGetSelectedTest, InvalidSelectingNonexistentDevice) {
    setenv("ONEAPI_DEVICE_SELECTOR", "*:4321", 1);
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_EQ(count, 0);
}

TEST_F(urDeviceGetSelectedTest, InvalidSelectingNonexistentSubDevice) {
    setenv("ONEAPI_DEVICE_SELECTOR", "*:0.4321", 1);
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_EQ(count, 0);
}

TEST_F(urDeviceGetSelectedTest, InvalidSelectingNonexistentSubSubDevice) {
    setenv("ONEAPI_DEVICE_SELECTOR", "*:0.0.4321", 1);
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_EQ(count, 0);
}
