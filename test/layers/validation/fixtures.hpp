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

#endif // UR_VALIDATION_TEST_HELPERS_H
