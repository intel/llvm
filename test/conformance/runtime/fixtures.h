// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
namespace uur {
namespace runtime {

struct urTest : ::testing::Test {

    void SetUp() override {
        ur_device_init_flags_t device_flags = 0;
        ur_loader_config_handle_t config;
        ASSERT_SUCCESS(urLoaderConfigCreate(&config));
        ASSERT_SUCCESS(
            urLoaderConfigEnableLayer(config, "UR_LAYER_FULL_VALIDATION"));
        ASSERT_SUCCESS(urInit(device_flags, config));
        ASSERT_SUCCESS(urLoaderConfigRelease(config));
    }

    void TearDown() override {
        ur_tear_down_params_t tear_down_params{};
        ASSERT_SUCCESS(urTearDown(&tear_down_params));
    }
};

struct urAdapterTest : urTest {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urTest::SetUp());

        uint32_t adapter_count;
        ASSERT_SUCCESS(urAdapterGet(0, nullptr, &adapter_count));
        adapters.resize(adapter_count);
        ASSERT_SUCCESS(urAdapterGet(adapter_count, adapters.data(), nullptr));
    }

    void TearDown() override {
        for (auto adapter : adapters) {
            ASSERT_SUCCESS(urAdapterRelease(adapter));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urTest::TearDown());
    }

    std::vector<ur_adapter_handle_t> adapters;
};

} // namespace runtime
} // namespace uur
