// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_PLATFORM_FIXTURES_H_INCLUDED
#define UR_CONFORMANCE_PLATFORM_FIXTURES_H_INCLUDED

#include <uur/fixtures.h>
namespace uur {
namespace platform {

struct urTest : ::testing::Test {

    void SetUp() override {
        ur_device_init_flags_t device_flags = 0;
        ASSERT_SUCCESS(urLoaderConfigCreate(&loader_config));
        ASSERT_SUCCESS(urLoaderConfigEnableLayer(loader_config,
                                                 "UR_LAYER_FULL_VALIDATION"));
        ASSERT_SUCCESS(urLoaderInit(device_flags, loader_config));

        uint32_t adapter_count;
        ASSERT_SUCCESS(urAdapterGet(0, nullptr, &adapter_count));
        adapters.resize(adapter_count);
        ASSERT_SUCCESS(urAdapterGet(adapter_count, adapters.data(), nullptr));
    }

    void TearDown() override {
        for (auto adapter : adapters) {
            ASSERT_SUCCESS(urAdapterRelease(adapter));
        }
        if (loader_config) {
            ASSERT_SUCCESS(urLoaderConfigRelease(loader_config));
        }
        ASSERT_SUCCESS(urLoaderTearDown());
    }

    ur_loader_config_handle_t loader_config = nullptr;
    std::vector<ur_adapter_handle_t> adapters;
};

struct urPlatformsTest : urTest {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urTest::SetUp());
        uint32_t count;
        ASSERT_SUCCESS(urPlatformGet(adapters.data(),
                                     static_cast<uint32_t>(adapters.size()), 0,
                                     nullptr, &count));
        ASSERT_NE(count, 0);
        platforms.resize(count);
        ASSERT_SUCCESS(urPlatformGet(adapters.data(),
                                     static_cast<uint32_t>(adapters.size()),
                                     count, platforms.data(), nullptr));
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

#define UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(FIXTURE)                         \
    INSTANTIATE_TEST_SUITE_P(                                                  \
        , FIXTURE,                                                             \
        ::testing::ValuesIn(uur::DevicesEnvironment::instance->devices),       \
        [](const ::testing::TestParamInfo<ur_device_handle_t> &info) {         \
            return uur::GetPlatformAndDeviceName(info.param);                  \
        })

} // namespace platform
} // namespace uur

#endif // UR_CONFORMANCE_PLATFORM_FIXTURES_H_INCLUDED
