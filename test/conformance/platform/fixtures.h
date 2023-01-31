// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_CONFORMANCE_PLATFORM_FIXTURES_H_INCLUDED
#define UR_CONFORMANCE_PLATFORM_FIXTURES_H_INCLUDED

#include <uur/fixtures.h>
namespace uur {
namespace platform {

struct urTest : ::testing::Test {

    void SetUp() override {
        ur_platform_init_flags_t platform_flags = 0;
        ur_device_init_flags_t device_flags = 0;
        ASSERT_SUCCESS(urInit(platform_flags, device_flags));
    }

    void TearDown() override {
        ur_tear_down_params_t tear_down_params{};
        ASSERT_SUCCESS(urTearDown(&tear_down_params));
    }
};

struct urPlatformsTest : urTest {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urTest::SetUp());
        uint32_t count;
        ASSERT_SUCCESS(urPlatformGet(0, nullptr, &count));
        ASSERT_NE(count, 0);
        platforms.resize(count);
        ASSERT_SUCCESS(urPlatformGet(count, platforms.data(), nullptr));
    }

    std::vector<ur_platform_handle_t> platforms;
};

struct urPlatformTest : urPlatformsTest {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformsTest::SetUp());
        ASSERT_GE(platforms.size(), 1);
        platform = platforms[0]; // TODO - which to choose?
    }

    ur_platform_handle_t platform;
};

} // namespace platform
} // namespace uur

#endif // UR_CONFORMANCE_PLATFORM_FIXTURES_H_INCLUDED
