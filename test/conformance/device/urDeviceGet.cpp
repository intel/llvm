// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urDeviceGetTest = uur::urPlatformTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urDeviceGetTest);

TEST_P(urDeviceGetTest, Success) {
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

TEST_P(urDeviceGetTest, SuccessSubsetOfDevices) {
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

TEST_P(urDeviceGetTest, InvalidNullHandlePlatform) {
    uint32_t count;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urDeviceGet(nullptr, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
}

TEST_P(urDeviceGetTest, InvalidEnumerationDevicesType) {
    uint32_t count;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_ENUMERATION,
        urDeviceGet(platform, UR_DEVICE_TYPE_FORCE_UINT32, 0, nullptr, &count));
}

TEST_P(urDeviceGetTest, InvalidSizeNumEntries) {
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_NE(count, 0);
    std::vector<ur_device_handle_t> devices(count);
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_SIZE,
        urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, devices.data(), nullptr));
}

TEST_P(urDeviceGetTest, InvalidNullPointerDevices) {
    uint32_t count = 0;
    ASSERT_SUCCESS(
        urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count));
    ASSERT_NE(count, 0);
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urDeviceGet(platform, UR_DEVICE_TYPE_ALL, count, nullptr, nullptr));
}

struct urDeviceGetTestWithDeviceTypeParam
    : uur::urAllDevicesTest,
      ::testing::WithParamInterface<ur_device_type_t> {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urAllDevicesTest::SetUp());
    }
};

INSTANTIATE_TEST_SUITE_P(
    , urDeviceGetTestWithDeviceTypeParam,
    ::testing::Values(UR_DEVICE_TYPE_DEFAULT, UR_DEVICE_TYPE_GPU,
                      UR_DEVICE_TYPE_CPU, UR_DEVICE_TYPE_FPGA,
                      UR_DEVICE_TYPE_MCA, UR_DEVICE_TYPE_VPU),
    [](const ::testing::TestParamInfo<ur_device_type_t> &info) {
        std::stringstream ss;
        ss << info.param;
        return ss.str();
    });

TEST_P(urDeviceGetTestWithDeviceTypeParam, Success) {
    ur_device_type_t device_type = GetParam();
    uint32_t count = 0;
    ASSERT_SUCCESS(urDeviceGet(platform, device_type, 0, nullptr, &count));
    ASSERT_GE(devices.size(), count);

    if (count > 0) {
        std::vector<ur_device_handle_t> devices(count);
        ASSERT_SUCCESS(
            urDeviceGet(platform, device_type, count, devices.data(), nullptr));
        for (auto device : devices) {
            ASSERT_NE(nullptr, device);
        }
    }
}
