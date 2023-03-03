// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_VALIDATION_TEST_HELPERS_H
#define UR_VALIDATION_TEST_HELPERS_H

#include <gtest/gtest.h>
#include <ur_api.h>

struct urTest : ::testing::Test {

    void SetUp() override {
        ur_device_init_flags_t device_flags = 0;
        ASSERT_EQ(urInit(device_flags), UR_RESULT_SUCCESS);
    }

    void TearDown() override {
        ur_tear_down_params_t tear_down_params{};
        ASSERT_EQ(urTearDown(&tear_down_params), UR_RESULT_SUCCESS);
    }
};

struct valPlatformsTest : urTest {

    void SetUp() override {
        urTest::SetUp();
        uint32_t count;
        ASSERT_EQ(urPlatformGet(0, nullptr, &count), UR_RESULT_SUCCESS);
        ASSERT_NE(count, 0);
        platforms.resize(count);
        ASSERT_EQ(urPlatformGet(count, platforms.data(), nullptr),
                  UR_RESULT_SUCCESS);
    }

    std::vector<ur_platform_handle_t> platforms;
};

struct valPlatformTest : valPlatformsTest {

    void SetUp() override {
        valPlatformsTest::SetUp();
        ASSERT_GE(platforms.size(), 1);
        platform = platforms[0]; // TODO - which to choose?
    }

    ur_platform_handle_t platform;
};

struct valAllDevicesTest : valPlatformTest {

    void SetUp() override {
        valPlatformTest::SetUp();

        uint32_t count = 0;
        if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count) ||
            count == 0) {
            FAIL() << "Failed to get devices";
        }

        devices.resize(count);
        if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, count, devices.data(),
                        nullptr)) {
            FAIL() << "Failed to get devices";
        }
    }
    std::vector<ur_device_handle_t> devices;
};

struct valDeviceTest : valAllDevicesTest {

    void SetUp() override {
        valAllDevicesTest::SetUp();
        ASSERT_GE(devices.size(), 1);
        device = devices[0];
    }
    ur_device_handle_t device;
};

struct valDeviceTestMultithreaded : valDeviceTest,
                                    public ::testing::WithParamInterface<int> {

    void SetUp() override {
        valDeviceTest::SetUp();
        threadCount = GetParam();
    }
    int threadCount;
};

#endif // UR_VALIDATION_TEST_HELPERS_H
